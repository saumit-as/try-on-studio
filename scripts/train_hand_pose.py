#!/usr/bin/env python3
"""
Train YOLO11 Nano Keypoint Pose Estimation for 5-Point Hand Nail-Bed Anchors.
Optimized for local training on NVIDIA GeForce RTX 3060 (12 GB VRAM).
Exports production ONNX model to models/yolo11n_nailbed_pose.onnx.
"""

import os
import sys
import shutil
from pathlib import Path

def train_nail_bed_pose():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Install via: pip install ultralytics")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    data_yaml = repo_root / "datasets" / "hand_nail_keypoints" / "hand_nails_pose.yaml"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Training dataset config: {data_yaml}")
    print("[*] Initializing pre-trained YOLO11n-pose backbone...")
    model = YOLO("yolo11n-pose.pt")

    print("[*] Launching training on GPU (RTX 3060, imgsz=256, batch=16)...")
    results = model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=256,           # Upright MediaPipe fingertip crops
        batch=16,            # High batch size for fast convergence
        device=0,            # Force NVIDIA RTX 3060
        pose=12.0,           # High weight on keypoint localization loss (OKS)
        degrees=25.0,
        fliplr=0.5,          # Automatically swaps flip_idx: [2, 1, 0, 4, 3]
        hsv_h=0.03,          # Skin-tone diversity augmentation
        hsv_s=0.5,
        hsv_v=0.5,
        project=str(repo_root / "runs" / "hand_pose"),
        name="yolo11n_nailbed_pose",
    )

    best_pt = repo_root / "runs" / "hand_pose" / "yolo11n_nailbed_pose" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[!] Warning: Could not find {best_pt}, checking last.pt...")
        best_pt = repo_root / "runs" / "hand_pose" / "yolo11n_nailbed_pose" / "weights" / "last.pt"

    print(f"[*] Exporting {best_pt} to ONNX format (opset=17, simplify=True)...")
    trained_model = YOLO(str(best_pt))
    exported_onnx = trained_model.export(
        format="onnx",
        imgsz=256,
        simplify=True,
        opset=17,
        dynamic=False,
    )

    dest_onnx = models_dir / "yolo11n_nailbed_pose.onnx"
    shutil.copyfile(exported_onnx, dest_onnx)
    print(f"[+] Successfully exported production model to: {dest_onnx}")

if __name__ == "__main__":
    train_nail_bed_pose()
