#!/usr/bin/env python3
"""
COCO segmentation JSON -> YOLO segmentation format converter.

Input:
  - COCO JSON (Label Studio "Brush labels to COCO" export) with polygon
    segmentation annotations.
Output (created on disk):
  - images/train/, images/val/   (image files copied here)
  - labels/train/, labels/val/   (one .txt per image, YOLO-seg format)

YOLO segmentation label line format:
  class_id x1 y1 x2 y2 ... xN yN
  (all coordinates normalized to 0..1)

NOTE: The original JSON has 2 categories ("bundal backup", "bundal uses").
For the "bundle present / absent" task we merge BOTH categories into a
single class -> "bundle" (class id 0).
"""

import json
import os
import random
import shutil
from collections import Counter

# ---------------------------------------------------------------------------
# Paths (edit only these if needed)
# ---------------------------------------------------------------------------
COCO_JSON = "/home/rehaan/Desktop/yolo /result_coco.json"   # the COCO export
SRC_IMAGES = "/home/rehaan/Desktop/yolo/images"             # folder with the 14 PNGs
OUT_DIR = "/home/rehaan/Desktop/yolo/bundle_seg"            # output dataset root

VAL_RATIO = 0.15          # 15% -> 2 images out of 14
SEED = 42                 # fixed seed so the split is reproducible
CLASS_NAMES = ["bundle"]  # single merged class


def main():
    # --- 1. load COCO JSON ---
    with open(COCO_JSON) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"JSON loaded: {len(images)} images, {len(annotations)} annotations")
    print(f"Source categories: {[(c['id'], c['name']) for c in categories]}")
    print(f"Merging all categories into single class: {CLASS_NAMES}")

    # --- 2. group annotations by image ---
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # --- 3. deterministic train/val split over images ---
    rng = random.Random(SEED)
    image_list = list(images)
    rng.shuffle(image_list)
    n_val = max(1, round(len(image_list) * VAL_RATIO))  # 14 * 0.15 = 2
    val_ids = {im["id"] for im in image_list[:n_val]}
    train_ids = {im["id"] for im in image_list[n_val:]}

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val (seed={SEED})")

    # --- 4. build folder structure ---
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    # --- 5. convert each image ---
    stats = Counter()
    for im in images:
        im_id = im["id"]
        split = "train" if im_id in train_ids else "val"

        # copy the image file (match by filename, ignoring the old server path)
        src = os.path.join(SRC_IMAGES, im["file_name"])
        dst = os.path.join(OUT_DIR, "images", split, im["file_name"])
        if not os.path.exists(src):
            print(f"  [WARN] image not found, skipping: {im['file_name']}")
            continue
        shutil.copy2(src, dst)

        # build the label file
        w, h = im["width"], im["height"]
        lines = []
        for ann in ann_by_image.get(im_id, []):
            # merge both categories -> class 0
            cls = 0
            segs = ann["segmentation"]
            # segmentation can be a list of polygons (list of lists)
            for poly in segs:
                if len(poly) < 6:          # need at least 3 points
                    continue
                # normalize every coordinate to 0..1
                coords = []
                for i in range(0, len(poly), 2):
                    x = poly[i] / w
                    y = poly[i + 1] / h
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])
                lines.append(f"{cls} " + " ".join(coords))
                stats["polygons"] += 1

        # write YOLO label file
        label_name = os.path.splitext(im["file_name"])[0] + ".txt"
        label_path = os.path.join(OUT_DIR, "labels", split, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        stats[f"images_{split}"] += 1
        if not lines:
            stats["images_empty"] += 1

    print("\n--- Conversion summary ---")
    print(f"  train images : {stats['images_train']}")
    print(f"  val images   : {stats['images_val']}")
    print(f"  total polygons written: {stats['polygons']}")
    print(f"  images with no label   : {stats['images_empty']}")
    print(f"  Output dataset: {OUT_DIR}")


if __name__ == "__main__":
    main()
