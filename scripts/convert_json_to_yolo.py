#!/usr/bin/env python3
"""
Convert AnyLabeling / LabelMe JSON annotations to YOLOv11 Instance Segmentation (.txt) format.
Automatically manages train/val split so both have images and labels.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def convert_annotations():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "datasets" / "tray_nails"
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"

    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(train_img_dir.glob("*.json")))
    if not json_files:
        print("[!] No JSON files found in train_img_dir.")
        return

    print(f"[*] Found {len(json_files)} annotated JSON files.")

    # We designate 1 tray (48.png - the challenging sheer set) as validation
    val_stem = "48"

    total_instances = 0
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = float(data.get("imageWidth", 1))
        img_h = float(data.get("imageHeight", 1))
        shapes = data.get("shapes", [])

        stem = jf.stem
        is_val = (stem == val_stem)
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        out_txt = target_lbl_dir / f"{stem}.txt"
        lines = []

        for shape in shapes:
            label = shape.get("label", "").strip().lower()
            if label not in ("nail", "nailf"):  # handle slight typo 'nailf' if present
                continue

            pts = shape.get("points", [])
            if len(pts) < 3:
                continue

            # YOLO polygon format: class_idx x1 y1 x2 y2 ... xn yn (normalized)
            coords = []
            for pt in pts:
                nx = max(0.0, min(1.0, float(pt[0]) / img_w))
                ny = max(0.0, min(1.0, float(pt[1]) / img_h))
                coords.append(f"{nx:.6f} {ny:.6f}")

            line = f"0 {' '.join(coords)}"
            lines.append(line)
            total_instances += 1

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"  [+] {stem}: {len(lines)} nail instances -> {out_txt.name}")

        # If it's validation, ensure image is also in val_img_dir
        if is_val:
            for ext in (".jpg", ".png", ".jpeg"):
                src_img = train_img_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, val_img_dir / src_img.name)
                    print(f"  [+] Copied validation image {src_img.name} -> images/val/")

    # Remove unannotated 37.jpg from val if present
    unannotated = val_img_dir / "37.jpg"
    if unannotated.exists() and not (val_lbl_dir / "37.txt").exists():
        unannotated.unlink()

    print(f"\n[OK] Converted {total_instances} nail polygon instances successfully!")
    print(f"Train labels: {len(list(train_lbl_dir.glob('*.txt')))} files")
    print(f"Val labels:   {len(list(val_lbl_dir.glob('*.txt')))} files")

if __name__ == "__main__":
    convert_annotations()
