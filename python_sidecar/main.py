#!/usr/bin/env python3
"""
Press-On Nail Mockup Generator - Python Sidecar
Persistent JSON-RPC stdio daemon handling:
  - Nail contour segmentation & PCA deskewing
  - Thin-Plate Spline (TPS) warping & C-curve cylindrical distortion
  - Multi-pass PBR lighting compositing
  - High-DPI and multi-layer PSD exports
"""

import sys
import os
import json
import base64
import traceback
import cv2
import numpy as np

# Ensure local imports work regardless of cwd
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from segmentation import (
    segment_packaging_tray,
    extract_nail_rgba,
    deskew_nail_pca,
    classify_nail_shape,
    refit_slot_contour,
)
from warping import (
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
    build_cuticle_pocket_mask,
)


def handle_ping(params):
    return {"status": "pong", "version": "0.1.0"}


def handle_echo(params):
    return params


def _load_image(params):
    if "image_path" in params and os.path.exists(params["image_path"]):
        img = cv2.imread(params["image_path"])
        if img is None:
            raise ValueError(f"Failed to read image at {params['image_path']}")
        return img
    elif "image_base64" in params:
        raw_b64 = params["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image data")
        return img
    else:
        raise ValueError("Missing 'image_path' or 'image_base64' parameter")


def handle_segment_tray(params):
    img = _load_image(params)
    max_slots = params.get("max_slots", 10)
    slots = segment_packaging_tray(img, max_slots=max_slots)
    return {
        "status": "ok",
        "slot_count": len(slots),
        "slots": slots,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
    }


def handle_extract_nail(params):
    img = _load_image(params)
    contour = np.array(params["contour"], dtype=np.int32)
    feather_px = float(params.get("feather_px", 1.2))
    deskew = bool(params.get("deskew", True))
    auto_orient = bool(params.get("auto_orient_cuticle_down", True))
    exact_mask = bool(params.get("exact_mask", False))
    homography = None
    if "homography" in params and params["homography"] is not None:
        homography = np.array(params["homography"], dtype=np.float64)
    rgba = extract_nail_rgba(
        img,
        contour,
        feather_px=feather_px,
        deskew=deskew,
        auto_orient_cuticle_down=auto_orient,
        homography=homography,
        exact_mask=exact_mask,
    )

    # Encode as PNG base64
    success, buffer = cv2.imencode(".png", rgba)
    if not success:
        raise ValueError("Failed to encode extracted nail to PNG")
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
        "image_base64": f"data:image/png;base64,{b64_str}",
    }


def handle_refit_slot_contour(params):
    """
    Re-fits and extracts a localized nail contour within a user-nudged or resized bounding box.
    """
    img = _load_image(params)
    bbox = params.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError("Missing or invalid 'bbox' parameter [x, y, w, h]")

    res = refit_slot_contour(img, bbox)
    return {
        "status": "ok",
        "bbox": res["bbox"],
        "contour": res["contour"],
        "pca_angle": res["pca_angle"],
        "shape_info": res["shape_info"],
    }


def handle_warp_nail(params):
    """
    Warps a nail texture onto destination finger landmarks using 3-tier TPS
    and transverse cylindrical C-curve distortion.
    """
    nail_rgba = None
    if "nail_image_path" in params and os.path.exists(params["nail_image_path"]):
        nail_rgba = cv2.imread(params["nail_image_path"], cv2.IMREAD_UNCHANGED)
    elif "nail_base64" in params:
        raw_b64 = params["nail_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        nail_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if nail_rgba is None:
        raise ValueError("Missing or invalid 'nail_image_path' / 'nail_base64'")

    if nail_rgba.shape[2] == 3:
        b, g, r = cv2.split(nail_rgba)
        alpha = np.full((nail_rgba.shape[0], nail_rgba.shape[1]), 255, dtype=np.uint8)
        nail_rgba = cv2.merge([b, g, r, alpha])

    c_curve = float(params.get("c_curve_intensity", 0.65))
    roll_offset = float(params.get("roll_offset", 0.0))
    length_scale = float(params.get("length_scale", 1.15))
    output_shape = tuple(params.get("output_shape", [1000, 800]))

    # 1. Apply transverse cylindrical barrel distortion
    curved_nail = apply_cylindrical_c_curve(nail_rgba, c_curve_intensity=c_curve, roll_offset=roll_offset)

    # 2. Build 3-tier control grid
    src_h, src_w = curved_nail.shape[:2]
    src_pts, dst_pts = build_3tier_control_grid(
        src_w=src_w,
        src_h=src_h,
        cuticle_left=params["cuticle_left"],
        cuticle_center=params["cuticle_center"],
        cuticle_right=params["cuticle_right"],
        distal_left=params["distal_left"],
        distal_right=params["distal_right"],
        length_scale=length_scale,
        tip_taper=float(params.get("tip_taper", 1.02)),
        roll_offset=roll_offset,
    )

    # 3. Thin-Plate Spline deformation
    warped_roi, bbox = warp_nail_tps(curved_nail, src_pts, dst_pts, output_shape)

    # 4. Apply anatomical cuticle & sidewall skin-fold tuck mask
    rx, ry, rw, rh = bbox
    if rw > 0 and rh > 0:
        pocket_allow, cuticle_ao = build_cuticle_pocket_mask(
            roi_bbox=bbox,
            cuticle_left=params["cuticle_left"],
            cuticle_center=params["cuticle_center"],
            cuticle_right=params["cuticle_right"],
            distal_left=params["distal_left"],
            distal_right=params["distal_right"],
            feather_sigma=1.6,
        )
        warped_roi[:, :, :3] = np.clip(
            warped_roi[:, :, :3].astype(np.float32) * cuticle_ao[:, :, None], 0, 255
        ).astype(np.uint8)
        warped_roi[:, :, 3] = np.clip(
            warped_roi[:, :, 3].astype(np.float32) * pocket_allow, 0, 255
        ).astype(np.uint8)

    # Encode warped ROI as PNG base64
    success, buffer = cv2.imencode(".png", warped_roi)
    if not success:
        raise ValueError("Failed to encode warped nail to PNG")
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "bbox": list(bbox),
        "warped_roi_base64": f"data:image/png;base64,{b64_str}",
    }


HANDLERS = {
    "ping": handle_ping,
    "echo": handle_echo,
    "segment_tray": handle_segment_tray,
    "extract_nail": handle_extract_nail,
    "refit_slot_contour": handle_refit_slot_contour,
    "warp_nail": handle_warp_nail,
}


def register_handler(name, func):
    HANDLERS[name] = func


def main():
    # Unbuffered stdout for line-delimited JSON-RPC
    sys.stdout.reconfigure(line_buffering=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method not in HANDLERS:
                response = {
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }
            else:
                try:
                    result = HANDLERS[method](params)
                    response = {"id": req_id, "result": result}
                except Exception as ex:
                    response = {
                        "id": req_id,
                        "error": {
                            "code": -32000,
                            "message": str(ex),
                            "traceback": traceback.format_exc()
                        }
                    }
        except Exception as e:
            response = {
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
