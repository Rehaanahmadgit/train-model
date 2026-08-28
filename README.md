# Bundle Detection (YOLOv8 Segmentation + Tracking)

Detect whether a **"bundle"** is **present or absent** in images/videos using a
fine-tuned **YOLOv8n-seg** (nano segmentation) model, with ByteTrack tracking
for videos.

> Status: **Proof-of-Concept** — trained on a very small dataset (14 images).
> Works for testing the full pipeline; for production accuracy, label more data.

---

## Table of Contents
1. [What this repo contains](#what-this-repo-contains)
2. [Prerequisites](#prerequisites)
3. [Step-by-step setup](#step-by-step-setup)
4. [Dataset format](#dataset-format)
5. [Train the model](#train-the-model)
6. [Test on a single image (present/absent)](#test-on-a-single-image)
7. [Run on a video (tracking)](#run-on-a-video)
8. [Project structure](#project-structure)

---

## What this repo contains

- COCO → YOLO-seg conversion script
- Train/val split + `data.yaml`
- Training script (YOLOv8n-seg fine-tuning)
- Single-image presence/absence test script
- Full video tracking pipeline (ByteTrack + mask overlay + CSV log)

---

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **GPU (recommended)**: NVIDIA with CUDA. CPU also works but is slow.
- `git`, `pip`

---

## Step-by-step setup

```bash
# 1. Clone the repo
git clone https://github.com/Rehaanahmadgit/train-model.git
cd train-model

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 4. Install PyTorch with CUDA (GPU)  -- for CPU-only, skip the --index-url part
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. Install the rest
pip install ultralytics pycocotools opencv-python

# 6. Verify GPU is visible (optional but recommended)
python -c "import torch; print(torch.cuda.is_available())"
# should print: True
```

---

## Dataset format

Dataset is already converted and split in this repo:

```
images/train/   12 images
images/val/      2 images
labels/train/    YOLO-seg label files (.txt)
labels/val/
```

Each label line looks like (class-id + normalized polygon points):

```
0 0.483333 0.363889 0.482812 0.364815 ...
```

If you get a **new COCO JSON** (e.g. from Label Studio), re-run:

```bash
python convert_coco_to_yolo.py
```

> Edit the paths at the top of `convert_coco_to_yolo.py` first.
> Note: the original JSON had 2 classes (`bundal backup`, `bundal uses`);
> the script merges them into a single class `bundle`.

---

## Train the model

```bash
python train.py
```

This downloads `yolov8n-seg.pt` (COCO pretrained) on first run, fine-tunes it,
and saves results under `runs/segment/runs/bundle_seg_train/`.

**Most important output:**
```
runs/segment/runs/bundle_seg_train/weights/best.pt   <- the trained model
```

Other training outputs (optional, for analysis): `results.csv` (per-epoch
metrics), `results.png` (loss curves), confusion matrix, PR curves.

---

## Test on a single image

```bash
python test.py                          # uses a default sample image
python test.py /path/to/image.png       # test your own image
```

Prints either `bundle PRESENT` (with count + confidence) or `bundle ABSENT`.

---

## Run on a video (tracking)

```bash
python video_pipeline.py \
    --input /path/to/video.mp4 \
    --output output_tracked.mp4 \
    --log track_log.csv \
    --conf 0.5
```

Options:
| Flag | Default | Meaning |
|---|---|---|
| `--input` | (required) | input video |
| `--output` | `output_tracked.mp4` | rendered video |
| `--log` | `track_log.csv` | CSV log |
| `--conf` | `0.5` | confidence threshold |
| `--stride` | `1` | process every Nth frame (2 = 2x faster) |
| `--imgsz` | `640` | inference resolution |
| `--device` | `0` | `0` = GPU, `cpu` = CPU |

**Outputs:**
- Rendered video with semi-transparent masks + boxes labeled `id` + `conf`.
- `track_log.csv` — per frame: `frame_number, timestamp_sec, track_id, status, confidence, mask_area_px`.
- Console summary: unique tracks, visibility duration, absent gaps.

---

## Project structure

```
.
├── convert_coco_to_yolo.py   # COCO JSON -> YOLO-seg format + train/val split
├── data.yaml                 # dataset config (class: bundle)
├── train.py                  # YOLOv8n-seg fine-tuning script
├── test.py                   # single-image present/absent check
├── video_pipeline.py         # video tracking + render + log
├── images/                   # train/val images
├── labels/                   # train/val YOLO-seg labels
└── runs/                     # training output (gitignored) -> best.pt lives here
```

---

## Notes

- **H.264 output**: if your OpenCV build has no H.264 encoder, the script
  automatically falls back to `mp4v` (plays everywhere). To force H.264,
  re-encode with ffmpeg:
  ```bash
  ffmpeg -i output_tracked.mp4 -c:v libx264 -crf 23 final.mp4
  ```
- **Model reliability**: trained on only 14 images. For a reliable model,
  label **200–500+ images** with varied angles, lighting, occlusion, and
  **empty frames** (no bundle present).
