#!/usr/bin/env python3
"""
Train YOLO11 Nano Instance Segmentation for Press-On Nail Packaging Trays.
Auto-detects GPU (NVIDIA RTX 3060) or multi-core CPU.
Exports production ONNX model to models/yolo11n_tray_1024.onnx.
"""

import os
import sys
import shutil
from pathlib import Path

def train_tray_segmentation():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Missing dependency: {e}. Install via: pip install ultralytics")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    data_yaml = repo_root / "datasets" / "tray_nails" / "data.yaml"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    batch_size = 8 if has_cuda else 4
    epochs = 80

    print(f"[*] Training dataset manifest: {data_yaml}")
    print(f"[*] Execution device: {'NVIDIA GPU (CUDA)' if has_cuda else 'Multi-Core CPU'}")
    print(f"[*] Batch size: {batch_size}, Epochs: {epochs}")
    print("[*] Initializing pre-trained YOLO11n-seg backbone...")
    model = YOLO("yolo11n-seg.pt")

    print("[*] Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=1024,          # High-resolution input preserves thin tips & sharp corners
        batch=batch_size,
        device=device,
        retina_masks=True,   # Full-resolution mask supervision
        overlap_mask=False,  # Nails in trays are disjoint slots; prevents mask bleeding
        degrees=15.0,        # Rotational augmentation for slightly tilted trays
        perspective=0.0005,  # Perspective tilt augmentation
        flipud=0.5,          # Handles inverted vs upright nails
        fliplr=0.5,
        mosaic=0.5,          # Retain high pixel density per nail
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        project=str(repo_root / "runs" / "tray_seg"),
        name="yolo11n_tray_1024",
    )

    best_pt = repo_root / "runs" / "tray_seg" / "yolo11n_tray_1024" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[!] Warning: Could not find {best_pt}, checking last.pt...")
        best_pt = repo_root / "runs" / "tray_seg" / "yolo11n_tray_1024" / "weights" / "last.pt"

    print(f"[*] Exporting {best_pt} to ONNX format (opset=17, simplify=True)...")
    trained_model = YOLO(str(best_pt))
    exported_onnx = trained_model.export(
        format="onnx",
        imgsz=1024,
        simplify=True,
        opset=17,
        dynamic=False,
    )

    dest_onnx = models_dir / "yolo11n_tray_1024.onnx"
    shutil.copyfile(exported_onnx, dest_onnx)
    print(f"[+] Successfully exported production model to: {dest_onnx}")

if __name__ == "__main__":
    train_tray_segmentation()
