#!/usr/bin/env python3
"""
Dataset Preparation & Organization Script for Try-On Studio.
Scans test_assets and user folders, organizes images into train/val splits,
and verifies annotation status for X-AnyLabeling.
"""

import os
import sys
import shutil
import random
from pathlib import Path

def prepare_splits(source_dir: Path, target_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"[!] No image files found in {source_dir}")
        return

    random.shuffle(images)
    n_val = max(1, int(len(images) * val_ratio)) if len(images) > 3 else 0
    val_set = set(images[:n_val])
    train_set = set(images[n_val:])

    train_img_dir = target_dir / "images" / "train"
    val_img_dir = target_dir / "images" / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Splitting {len(images)} images -> {len(train_set)} train, {len(val_set)} val")

    for p in train_set:
        shutil.copy2(p, train_img_dir / p.name)
    for p in val_set:
        shutil.copy2(p, val_img_dir / p.name)

    print(f"[+] Images populated into: {target_dir / 'images'}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    tray_source = repo_root / "test_assets" / "test_nails"
    tray_target = repo_root / "datasets" / "tray_nails"

    if tray_source.exists():
        print("[*] Organizing packaging tray dataset from test_assets/test_nails...")
        prepare_splits(tray_source, tray_target)
    else:
        print(f"[!] Tray source {tray_source} not found.")
