#!/usr/bin/env python3
"""
Fine-tune YOLOv8n-seg (nano) on the "bundle" presence/absence dataset.

Dataset is tiny (12 train / 2 val images), so we:
  - start from COCO-pretrained weights (transfer learning)
  - use a low learning rate (fine-tuning, not training from scratch)
  - train many epochs (200)
  - enable heavy augmentation (flip, rotation, scale, HSV, mosaic,
    mixup, copy-paste) to reduce overfitting

Runs on GPU (device=0). For a 4GB RTX 2050 we use imgsz=640 + AMP.
"""

from ultralytics import YOLO

# 1) Load nano segmentation model with COCO pretrained weights.
#    First call downloads yolov8n-seg.pt (~7 MB) automatically.
model = YOLO("yolov8n-seg.pt")

# 2) Train
results = model.train(
    data="data.yaml",        # dataset config (created earlier)
    epochs=200,              # many epochs: tiny data needs many passes
    imgsz=640,               # train resolution (input images are 1920x1080)
    batch=8,                 # fits in 4GB VRAM for nano model
    device=0,                # GPU (RTX 2050)
    workers=4,
    amp=True,                # mixed precision -> less VRAM, faster

    # --- fine-tuning friendly optimizer/lr ---
    optimizer="AdamW",
    lr0=0.002,               # low base LR (default 0.01 is too aggressive here)
    lrf=0.01,                # final LR factor
    patience=50,             # early-stop patience (epochs without improvement)

    # --- heavy augmentation (only 12 train images!) ---
    fliplr=0.5,              # horizontal flip
    degrees=30,              # random rotation +/- 30 deg
    translate=0.2,           # random shift
    scale=0.5,               # random zoom
    hsv_h=0.015,             # hue jitter
    hsv_s=0.7,               # saturation jitter
    hsv_v=0.4,               # value jitter
    mosaic=1.0,              # mosaic-4 augmentation on every batch
    mixup=0.15,              # mixup blending
    copy_paste=0.1,          # copy-paste (segmentation) augmentation
    erasing=0.2,             # random erasing
    close_mosaic=10,         # disable mosaic for last 10 epochs (stabilizes)

    # --- output ---
    project="runs",          # results go to runs/segment/...
    name="bundle_seg_train",
    exist_ok=True,           # overwrite previous run of same name
)

# 3) Print where the best weights were saved
print("\nBest weights:", model.trainer.best)
