#!/usr/bin/env python3
"""
Quick presence/absence test using the trained YOLOv8n-seg model.

Usage:
    python test.py                      # uses a default sample image
    python test.py /path/to/image.png   # test your own image

It prints whether a "bundle" is present or absent and saves an
annotated copy next to the input image.
"""

import os
import sys

from ultralytics import YOLO

# --- config ---
# Prefer the committed model (best.pt in repo root); fall back to the
# training output path if it exists locally (e.g. right after retraining).
BEST_PT = "best.pt"
if not os.path.exists(BEST_PT):
    BEST_PT = "runs/segment/runs/bundle_seg_train/weights/best.pt"
CONF_THRESHOLD = 0.25      # only count detections above this confidence
DEFAULT_IMAGE = "images/val/8919efc6-frame_000008.png"

# --- pick input image ---
image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

if not os.path.exists(image_path):
    print(f"[ERROR] image not found: {image_path}")
    sys.exit(1)

# --- load model & predict ---
model = YOLO(BEST_PT)
results = model.predict(
    source=image_path,
    conf=CONF_THRESHOLD,
    imgsz=640,
    device=0,
    save=True,          # save annotated image to runs/segment/predict
)

# --- decide present / absent ---
res = results[0]
boxes = res.boxes          # detection boxes (None if no detections)
masks = res.masks          # segmentation masks (None if no detections)

n = 0 if boxes is None else len(boxes)

if n == 0:
    print(f"\nRESULT: bundle ABSENT  (no detection above conf {CONF_THRESHOLD})")
else:
    confs = boxes.conf.tolist() if boxes is not None else []
    top_conf = max(confs)
    print(f"\nRESULT: bundle PRESENT")
    print(f"  detections : {n}")
    print(f"  confidences: {[round(c, 3) for c in confs]}")
    print(f"  best conf  : {top_conf:.3f}")
    if masks is not None:
        print(f"  masks      : {len(masks)}")

print(f"\nAnnotated image saved under: runs/segment/predict/")
