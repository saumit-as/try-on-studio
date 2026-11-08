"""
ML Model Inference Engine for Try-On Studio.
Provides high-performance, deterministic ONNX Runtime inference for:
  1. Packaging Tray Nail Instance Segmentation (YOLO11n-seg)
  2. Hand Fingertip 5-Point Anatomical Keypoint Estimation (YOLO11n-pose)

Features:
  - Zero PyTorch dependency in production (pure onnxruntime CPU/GPU).
  - Graceful fallback to classical computer vision when models are not yet trained.
  - Drop-in compatibility with segmentation.py and template_generator.py schemas.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Optional ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TRAY_MODEL_PATH = MODELS_DIR / "yolo11n_tray_1024.onnx"
POSE_MODEL_PATH = MODELS_DIR / "yolo11n_nailbed_pose.onnx"


class MLModelManager:
    """Singleton model manager for loading and running ONNX vision models."""
    _instance: Optional["MLModelManager"] = None

    def __init__(self):
        self.tray_session: Optional[Any] = None
        self.pose_session: Optional[Any] = None
        self._init_sessions()

    @classmethod
    def get_instance(cls) -> "MLModelManager":
        if cls._instance is None:
            cls._instance = MLModelManager()
        return cls._instance

    def _init_sessions(self):
        if not HAS_ONNX:
            return

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        if TRAY_MODEL_PATH.exists():
            try:
                self.tray_session = ort.InferenceSession(str(TRAY_MODEL_PATH), sess_options=opts, providers=providers)
            except Exception as e:
                print(f"[!] Warning: Failed to load {TRAY_MODEL_PATH}: {e}")

        if POSE_MODEL_PATH.exists():
            try:
                self.pose_session = ort.InferenceSession(str(POSE_MODEL_PATH), sess_options=opts, providers=providers)
            except Exception as e:
                print(f"[!] Warning: Failed to load {POSE_MODEL_PATH}: {e}")

    def has_tray_model(self) -> bool:
        return self.tray_session is not None

    def has_pose_model(self) -> bool:
        return self.pose_session is not None

    def segment_tray_ml(
        self,
        image_bgr: np.ndarray,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO11n-seg inference on rectified tray image.
        Returns extracted nail slots matching segmentation.py dictionary schema.
        """
        if not self.has_tray_model():
            raise RuntimeError("Tray ONNX model is not loaded.")

        orig_h, orig_w = image_bgr.shape[:2]
        target_size = 1024

        # Letterbox image to 1024x1024
        scale = min(target_size / orig_w, target_size / orig_h)
        nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        # Preprocess: BGR -> RGB, HWC -> CHW, normalize [0, 1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

        # Run inference
        inputs = {self.tray_session.get_inputs()[0].name: blob}
        preds, protos = self.tray_session.run(None, inputs)

        # Output format:
        # preds: [1, 32 + num_classes + 4, num_anchors]
        # protos: [1, 32, 256, 256]
        preds = np.squeeze(preds, axis=0)  # [features, anchors]
        protos = np.squeeze(protos, axis=0)  # [32, 256, 256]

        boxes = []
        confidences = []
        mask_coeffs = []

        num_anchors = preds.shape[1]
        for i in range(num_anchors):
            col = preds[:, i]
            cx, cy, bw, bh = col[:4]
            cls_conf = col[4]  # class 0: nail
            if cls_conf < conf_thresh:
                continue

            x0 = cx - bw / 2.0
            y0 = cy - bh / 2.0
            boxes.append([float(x0), float(y0), float(bw), float(bh)])
            confidences.append(float(cls_conf))
            mask_coeffs.append(col[5:])

        if not boxes:
            return []

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()

        detected_slots = []
        c_protos, proto_h, proto_w = protos.shape
        protos_flat = protos.reshape(c_protos, -1)

        for idx in indices:
            bx, by, bw, bh = boxes[idx]
            coeff = np.array(mask_coeffs[idx], dtype=np.float32)

            # Reconstruct mask at prototype resolution (256x256)
            mask_proto = 1.0 / (1.0 + np.exp(-np.dot(coeff, protos_flat)))
            mask_proto = mask_proto.reshape(proto_h, proto_w)

            # Rescale mask to 1024x1024 padded space
            mask_pad = cv2.resize(mask_proto, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            # Crop mask to original image bounding box
            mask_orig = mask_pad[pad_y : pad_y + nh, pad_x : pad_x + nw]
            mask_full = cv2.resize((mask_orig > 0.5).astype(np.uint8) * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Crop local nail box
            nx0 = max(0, int(round((bx - pad_x) / scale)))
            ny0 = max(0, int(round((by - pad_y) / scale)))
            nw_px = max(1, int(round(bw / scale)))
            nh_px = max(1, int(round(bh / scale)))
            nx1 = min(orig_w, nx0 + nw_px)
            ny1 = min(orig_h, ny0 + nh_px)

            box_mask = mask_full[ny0:ny1, nx0:nx1]
            cnts, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            nail_cnt = max(cnts, key=cv2.contourArea)
            nail_cnt = nail_cnt + np.array([[[nx0, ny0]]])

            detected_slots.append({
                "bbox": [nx0, ny0, nx1 - nx0, ny1 - ny0],
                "contour": nail_cnt.reshape(-1, 2).tolist(),
                "area": float(cv2.contourArea(nail_cnt)),
                "aspect_ratio": float((ny1 - ny0) / max(1, nx1 - nx0)),
                "centroid": (float(nx0 + (nx1 - nx0) / 2.0), float(ny0 + (ny1 - ny0) / 2.0)),
            })

        # Sort detected nails into horizontal rows and left-to-right columns
        detected_slots.sort(key=lambda s: s["centroid"][1])
        rows: List[List[Dict[str, Any]]] = []
        row_y_threshold = orig_h * 0.12

        for slot in detected_slots:
            cy = slot["centroid"][1]
            placed = False
            for r in rows:
                r_mean_y = np.mean([s["centroid"][1] for s in r])
                if abs(cy - r_mean_y) < row_y_threshold:
                    r.append(slot)
                    placed = True
                    break
            if not placed:
                rows.append([slot])

        final_slots: List[Dict[str, Any]] = []
        slot_id = 1
        for r in rows:
            r.sort(key=lambda s: s["centroid"][0])
            for s in r:
                s["slot_id"] = slot_id
                s.pop("centroid", None)
                final_slots.append(s)
                slot_id += 1

        return final_slots


def get_model_manager() -> MLModelManager:
    return MLModelManager.get_instance()
