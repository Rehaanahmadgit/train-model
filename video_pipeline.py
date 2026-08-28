#!/usr/bin/env python3
"""
Full video inference + tracking pipeline for the "bundle" segmentation model.

What it does:
  1. Reads a video frame by frame.
  2. Runs YOLOv8-seg tracking (ByteTrack) -> each bundle gets a stable track_id.
  3. Renders an output video with semi-transparent mask overlay + box
     label (track_id, confidence), same fps/resolution as input.
  4. Writes a CSV log (one row per detection; "absent" rows when empty).
  5. Prints a track-level summary (unique tracks, visible duration,
     absent gaps).

Usage:
    python video_pipeline.py --input /path/to/video.mp4 \
                             --output out.mp4 \
                             --log track_log.csv \
                             --conf 0.5

    Add --stride 2 to process every 2nd frame (faster, lower-res tracking).
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# Quiet the FFmpeg backend logs (avoids noisy v4l2m2m errors when H.264
# hardware encoder is not present on this machine).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BEST_PT = "runs/segment/runs/bundle_seg_train/weights/best.pt"
TRACKER = "bytetrack.yaml"   # ByteTrack tracker (ultralytics built-in)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def get_color(track_id: int):
    """Deterministic color for a track id (so it stays same across frames)."""
    palette = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
        (128, 0, 255),  # purple
        (255, 128, 0),  # orange
    ]
    return palette[int(track_id) % len(palette)]


def draw_masks(frame, masks, track_ids):
    """Overlay semi-transparent colored masks on the frame (in-place).

    Masks arrive at the inference resolution (e.g. 384x640), so each mask
    is resized to the original frame size before blending.
    """
    if masks is None:
        return frame

    H, W = frame.shape[:2]
    overlay = frame.copy()
    for mask, tid in zip(masks, track_ids):
        m = mask.astype(bool)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        color = get_color(tid)
        overlay[m] = color
    # blend: 40% overlay color, 60% original frame
    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)


def draw_boxes(frame, boxes):
    """Draw bounding box + label (track_id, conf) for each detection."""
    if boxes is None:
        return frame

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0]) if box.conf is not None else -1.0
        tid = int(box.id[0]) if box.id is not None else -1
        color = get_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id:{tid} conf:{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def mask_areas(masks, frame_shape):
    """Mask area in ORIGINAL-image pixels for each mask.

    Masks are at inference resolution, so the pixel count is scaled up by
    (origH/maskH)*(origW/maskW) to report area in real image pixels.
    """
    if masks is None:
        return []
    H, W = frame_shape
    areas = []
    for m in masks:
        mH, mW = m.shape[:2]
        scale = (H / mH) * (W / mW)
        areas.append(int(m.astype(bool).sum() * scale))
    return areas


def create_writer(path, fps, size):
    """Create VideoWriter.

    Tries H.264 ('avc1') first; if that encoder is unavailable (common on
    machines without a working H.264 encoder in OpenCV), falls back to
    'mp4v' (MPEG-4 Part 2), which plays in virtually every player.
    """
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("Could not create VideoWriter for: " + path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Bundle tracking video pipeline")
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", default="output_tracked.mp4", help="output video")
    ap.add_argument("--log", default="track_log.csv", help="CSV log path")
    ap.add_argument("--model", default=BEST_PT, help="trained weights path")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference resolution")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (2 = skip every other frame)")
    ap.add_argument("--device", default="0", help="device: 0=cuda, cpu=cpu")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] input not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] model not found: {args.model}")
        sys.exit(1)

    # --- load model + open video ---
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # with frame-skipping the output plays back at src_fps/stride so timing
    # stays correct (1 second of video = 1 second of real time)
    out_fps = src_fps / args.stride
    writer = create_writer(args.output, out_fps, (width, height))

    print(f"Input : {args.input}")
    print(f"       {width}x{height} @ {src_fps:.2f} fps, {total_frames} frames")
    print(f"Stride: {args.stride}  -> output @ {out_fps:.2f} fps")
    print(f"Model : {args.model}  conf={args.conf}  imgsz={args.imgsz}  device={args.device}")
    print("-" * 60)

    # --- open log file ---
    log_f = open(args.log, "w", newline="")
    writer_csv = csv.writer(log_f)
    writer_csv.writerow(["frame_number", "timestamp_sec", "track_id",
                         "status", "confidence", "mask_area_px"])

    # --- tracking state ---
    tracks = {}                 # track_id -> {start_frame, end_frame, n_frames, confs}
    absent_gaps = []            # list of (start_frame, end_frame, n_frames)
    gap_start = None            # current absent run start frame
    frame_idx = 0               # source frame counter (0-based)
    processed = 0               # frames actually fed to the model

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame skipping
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        # --- run tracking on this frame ---
        res = model.track(
            source=frame,
            persist=True,           # keep tracker state across frames
            tracker=TRACKER,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        boxes = res.boxes
        masks = res.masks
        timestamp = frame_idx / src_fps

        # --- extract per-detection info ---
        det_tids = []
        if boxes is not None and len(boxes) > 0:
            det_tids = [int(b.id[0]) if b.id is not None else -1
                        for b in boxes]
            confs = [float(b.conf[0]) if b.conf is not None else -1.0
                     for b in boxes]
            mask_np = masks.data.cpu().numpy() if masks is not None else None
            areas = mask_areas(mask_np, frame.shape[:2])  # area in real pixels

            # update track summary
            for i, tid in enumerate(det_tids):
                if tid not in tracks:
                    tracks[tid] = {"start_frame": frame_idx, "end_frame": frame_idx,
                                   "n_frames": 0, "confs": []}
                t = tracks[tid]
                t["end_frame"] = frame_idx
                t["n_frames"] += 1
                t["confs"].append(confs[i])

                # CSV row for this detection
                writer_csv.writerow([frame_idx, f"{timestamp:.3f}", tid,
                                     "present", f"{confs[i]:.3f}", areas[i]])

            # render masks + boxes
            frame = draw_masks(frame, mask_np, det_tids)
            frame = draw_boxes(frame, boxes)

            # close any open absent gap
            if gap_start is not None:
                absent_gaps.append((gap_start, frame_idx - 1,
                                    frame_idx - gap_start))
                gap_start = None
        else:
            # no bundle in this frame
            writer_csv.writerow([frame_idx, f"{timestamp:.3f}", "",
                                 "absent", "", ""])
            if gap_start is None:
                gap_start = frame_idx

        writer.write(frame)
        processed += 1

        if processed % 50 == 0:
            print(f"  processed {processed} frames ...")

        frame_idx += 1

    # close final gap if video ended while absent
    if gap_start is not None:
        absent_gaps.append((gap_start, frame_idx - 1, frame_idx - gap_start))

    cap.release()
    writer.release()
    log_f.close()

    # ------------------------------------------------------------------
    # Track-level summary
    # ------------------------------------------------------------------
    print("-" * 60)
    print("TRACK-LEVEL SUMMARY")
    print(f"Total unique tracks (bundles) : {len(tracks)}")
    for tid in sorted(tracks):
        t = tracks[tid]
        avg_conf = sum(t["confs"]) / len(t["confs"]) if t["confs"] else 0.0
        print(f"  track {tid:2d}: start_frame={t['start_frame']:4d} "
              f"end_frame={t['end_frame']:4d} visible={t['n_frames']:4d} frames "
              f"avg_conf={avg_conf:.2f}")

    total_absent = sum(n for _, _, n in absent_gaps)
    print(f"Absent gaps (continuous frames, no bundle): {len(absent_gaps)}")
    for s, e, n in absent_gaps:
        print(f"  gap: start={s:4d} end={e:4d} length={n:4d} frames "
              f"({n/src_fps:.1f}s)")
    print(f"Total frames with no bundle : {total_absent}")

    print("-" * 60)
    print(f"Output video : {args.output}")
    print(f"Log CSV      : {args.log}")
    print("Done.")


if __name__ == "__main__":
    main()
