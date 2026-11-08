#!/usr/bin/env python3
"""
Press-On Nail Mockup Generator - Python Sidecar
Persistent JSON-RPC stdio daemon handling:
  - Nail contour segmentation & PCA deskewing
  - Thin-Plate Spline (TPS) warping & C-curve cylindrical distortion
  - Hand template rigging & procedural map generation (cuticle fold, AO shadow, specular gloss)
  - Multi-pass PBR lighting compositing
  - High-DPI and multi-layer PSD exports
"""

import sys
import os
import time
import json
import base64
import traceback
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Ensure local imports work regardless of cwd
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

REPO_ROOT = os.path.abspath(os.path.join(current_dir, ".."))

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
from template_generator import (
    DEFAULT_STUDIO_LANDMARKS,
    generate_all_rig_maps,
    save_nailtemplate_bundle,
    load_nailtemplate_bundle,
    init_default_studio_template,
    auto_detect_hand_landmarks,
    refit_finger_landmarks,
)
from compositor import render_full_hand_composite
from export import export_composite_package


def handle_ping(params):
    return {"status": "pong", "version": "0.1.0"}


def handle_echo(params):
    return params


def _encode_png_base64(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def _encode_jpg_base64(img: np.ndarray, quality: int = 92) -> str:
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode image to JPG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


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
    t0 = time.time()
    img = _load_image(params)
    max_slots = params.get("max_slots", 10)
    refine_with_sam = bool(params.get("refine_with_sam", True))
    slots = segment_packaging_tray(img, max_slots=max_slots, refine_with_sam=refine_with_sam)
    elapsed_ms = round((time.time() - t0) * 1000.0, 1)
    sys.stderr.write(
        f"[segment_tray] Detected {len(slots)} slots in {elapsed_ms}ms "
        f"(image size={img.shape[1]}x{img.shape[0]}, refine_with_sam={refine_with_sam})\n"
    )
    return {
        "status": "ok",
        "slot_count": len(slots),
        "slots": slots,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "execution_time_ms": elapsed_ms,
    }


def handle_extract_nail(params):
    t0 = time.time()
    img = _load_image(params)
    contour = np.array(params["contour"], dtype=np.int32)
    feather_px = float(params.get("feather_px", 1.2))
    deskew = bool(params.get("deskew", True))
    auto_orient = bool(params.get("auto_orient_cuticle_down", True))
    exact_mask = bool(params.get("exact_mask", False))
    slot_id = int(params.get("slot_id", 1))
    debug_dump = bool(params.get("debug_dump", False) or os.environ.get("TRYON_DEBUG") == "1")

    homography = None
    if "homography" in params and params["homography"] is not None:
        homography = np.array(params["homography"], dtype=np.float64)

    metrics: Dict[str, Any] = {}
    rgba = extract_nail_rgba(
        img,
        contour,
        feather_px=feather_px,
        deskew=deskew,
        auto_orient_cuticle_down=auto_orient,
        homography=homography,
        exact_mask=exact_mask,
        metrics_out=metrics,
    )

    # Encode as PNG base64
    success, buffer = cv2.imencode(".png", rgba)
    if not success:
        raise ValueError("Failed to encode extracted nail to PNG")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    elapsed_ms = round((time.time() - t0) * 1000.0, 1)

    dump_path = None
    if debug_dump:
        try:
            dump_dir = os.path.join(REPO_ROOT, "test_output", "debug_extractions", f"slot_{slot_id}_{int(time.time()*1000)}")
            os.makedirs(dump_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dump_dir, "extracted_rgba.png"), rgba)
            debug_meta = {
                "slot_id": slot_id,
                "width": int(rgba.shape[1]),
                "height": int(rgba.shape[0]),
                "metrics": metrics,
                "contour_pts": len(contour),
                "exact_mask": exact_mask,
                "deskew": deskew,
                "elapsed_ms": elapsed_ms,
            }
            with open(os.path.join(dump_dir, "debug_meta.json"), "w") as f:
                json.dump(debug_meta, f, indent=2)
            dump_path = dump_dir
            sys.stderr.write(f"[extract_nail] Dumped debug assets to: {dump_dir}\n")
        except Exception as e:
            sys.stderr.write(f"[extract_nail] Warning: Failed to dump debug assets: {e}\n")

    debug_info = {
        "slot_id": slot_id,
        "method": "exact_mask" if exact_mask else "classical_adaptive",
        "contour_points": len(contour),
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
        "deskew_angle": round(float(metrics.get("deskew_angle", 0.0)), 2),
        "is_inverted": bool(metrics.get("is_inverted", False)),
        "dematting_metrics": {
            "choke_px": int(metrics.get("choke_px", 0)),
            "area_removed_pct": float(metrics.get("area_removed_pct", 0.0)),
            "core_px": int(metrics.get("core_px", 0)),
        },
        "crop_box": metrics.get("crop_box"),
        "duration_ms": elapsed_ms,
        "debug_dump_path": dump_path,
    }

    sys.stderr.write(
        f"[extract_nail] slot={slot_id} -> ok: size=({rgba.shape[1]}x{rgba.shape[0]}), "
        f"mode={'exact' if exact_mask else 'adaptive'}, deskew={debug_info['deskew_angle']}°, "
        f"inv={debug_info['is_inverted']}, choke={debug_info['dematting_metrics']['choke_px']}px ({elapsed_ms}ms)\n"
    )

    return {
        "status": "ok",
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
        "image_base64": f"data:image/png;base64,{b64_str}",
        "debug_info": debug_info,
    }


def handle_segment_prompt(params):
    img = _load_image(params)
    bbox = params.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError("Parameter 'bbox' must be a 4-element list [x0, y0, x1, y1] in original-image coordinates.")

    points = params.get("points")
    prev_mask_id = params.get("prev_mask_id")
    image_id = params.get("image_id")
    slot_id = params.get("slot_id", 1)

    from ml_models import get_sam_runner
    runner = get_sam_runner()
    result = runner.predict_prompt(
        img,
        bbox=bbox,
        points=points,
        prev_mask_id=prev_mask_id,
        image_id=image_id,
        slot_id=slot_id,
    )
    return result


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


def handle_get_default_hand_template(params):
    """
    Returns the bundled Default Studio Hand template metadata, landmarks,
    and base64 pre-rendered maps and image.
    """
    template_dir = os.path.join(REPO_ROOT, "templates", "default_studio_hand")
    if not os.path.exists(template_dir) or not os.path.exists(os.path.join(template_dir, "template.json")):
        init_default_studio_template(REPO_ROOT)

    data = load_nailtemplate_bundle(template_dir)
    include_images = params.get("include_images", True)

    res = {
        "status": "ok",
        "metadata": data["metadata"],
        "landmarks": data["metadata"].get("landmarks", DEFAULT_STUDIO_LANDMARKS),
        "image_size": data["metadata"].get("image_size", [1536, 2752]),
    }

    if include_images:
        if data["hand_bgr"] is not None:
            res["hand_image_base64"] = _encode_jpg_base64(data["hand_bgr"], quality=90)
        if data["cuticle_mask"] is not None:
            res["cuticle_mask_base64"] = _encode_png_base64(data["cuticle_mask"])
        if data["ao_map"] is not None:
            res["ao_map_base64"] = _encode_png_base64(data["ao_map"])
        if data["specular_map"] is not None:
            res["specular_map_base64"] = _encode_png_base64(data["specular_map"])

    return res


def handle_generate_hand_maps(params):
    """
    Generates procedural cuticle occlusion, contact AO, and specular reflection maps
    given an arbitrary bare hand image and 5-point finger landmarks.
    """
    img = _load_image(params)
    landmarks = params.get("landmarks", DEFAULT_STUDIO_LANDMARKS)
    maps = generate_all_rig_maps(img, landmarks)

    return {
        "status": "ok",
        "cuticle_mask_base64": _encode_png_base64(maps["cuticle_mask"]),
        "ao_map_base64": _encode_png_base64(maps["ao_map"]),
        "specular_map_base64": _encode_png_base64(maps["specular_map"]),
    }


def handle_save_hand_template(params):
    """
    Saves a .nailtemplate file or bundle.
    """
    target_path = params.get("target_path")
    template_name = params.get("name", "Custom Hand Template")
    landmarks = params.get("landmarks", {})
    img = _load_image(params)

    if not target_path:
        raise ValueError("Missing 'target_path' parameter")

    if target_path.endswith(".nailtemplate") or target_path.endswith(".json"):
        maps = generate_all_rig_maps(img, landmarks)
        h, w = img.shape[:2]
        bundle = {
            "format": "nailtemplate",
            "version": "1.0.0",
            "name": template_name,
            "image_size": [int(w), int(h)],
            "landmarks": landmarks,
            "hand_image_base64": _encode_jpg_base64(img),
            "cuticle_mask_base64": _encode_png_base64(maps["cuticle_mask"]),
            "ao_map_base64": _encode_png_base64(maps["ao_map"]),
            "specular_map_base64": _encode_png_base64(maps["specular_map"]),
        }
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f)
        return {"status": "ok", "saved_path": target_path}
    else:
        meta_path = save_nailtemplate_bundle(target_path, template_name, img, landmarks)
        return {"status": "ok", "saved_path": meta_path}


def handle_load_hand_template(params):
    """
    Loads a .nailtemplate file or directory bundle.
    """
    source_path = params.get("source_path")
    raw_content = params.get("content_json")

    if raw_content:
        data = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
        return {"status": "ok", "template": data}
    elif source_path:
        if os.path.isfile(source_path):
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "ok", "template": data}
        elif os.path.isdir(source_path):
            data = load_nailtemplate_bundle(source_path)
            res = {
                "format": "nailtemplate",
                "version": "1.0.0",
                "name": data["metadata"].get("name", "Loaded Template"),
                "image_size": data["metadata"].get("image_size", [1536, 2752]),
                "landmarks": data["metadata"].get("landmarks", {}),
            }
            if data["hand_bgr"] is not None:
                res["hand_image_base64"] = _encode_jpg_base64(data["hand_bgr"])
            if data["cuticle_mask"] is not None:
                res["cuticle_mask_base64"] = _encode_png_base64(data["cuticle_mask"])
            if data["ao_map"] is not None:
                res["ao_map_base64"] = _encode_png_base64(data["ao_map"])
            if data["specular_map"] is not None:
                res["specular_map_base64"] = _encode_png_base64(data["specular_map"])
            return {"status": "ok", "template": res}
        else:
            raise FileNotFoundError(f"Source path not found: {source_path}")
    else:
        raise ValueError("Missing 'source_path' or 'content_json'")


def handle_auto_detect_hand(params):
    """
    Analyzes an arbitrary user hand photo using Computer Vision,
    automatically detects fingertips, palm center, handedness, and the 5-point nail landmarks.
    """
    img = _load_image(params)
    handedness_hint = params.get("handedness", "auto")
    res = auto_detect_hand_landmarks(img, handedness_hint=handedness_hint)
    return {
        "status": "ok",
        "landmarks": res["landmarks"],
        "handedness": res["handedness"],
        "palm_center": res["palm_center"],
        "palm_radius": res["palm_radius"],
    }


def handle_refit_finger(params):
    """
    Computes accurate 5-point nail landmarks within a user-drawn or detected
    bounding box [x, y, w, h] over any specific fingertip.
    """
    img = _load_image(params)
    finger_name = params.get("finger_name") or params.get("finger")
    if not finger_name:
        raise ValueError("Missing 'finger_name' parameter")
    bbox = params.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError("Missing or invalid 'bbox' parameter [x, y, w, h]")
    handedness = params.get("handedness", "left")

    res = refit_finger_landmarks(img, finger_name, bbox, handedness=handedness)
    return {
        "status": "ok",
        "finger_name": res["finger_name"],
        "landmarks": res["landmarks"],
    }


def _load_rgba_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode base64 RGBA image")
    return img


def handle_render_composite(params):
    import time
    t0 = time.time()

    # 1. Load Hand Base Image
    hand_src = params.get("hand_image_base64") or params.get("image_base64")
    if hand_src and isinstance(hand_src, str) and hand_src.startswith("data:"):
        hand_bgr = _load_image({"image_base64": hand_src})
    elif "image_path" in params and os.path.exists(params["image_path"]):
        hand_bgr = cv2.imread(params["image_path"])
    else:
        default_path = os.path.join(REPO_ROOT, "templates", "default_studio_hand", "hand.jpg")
        if not os.path.exists(default_path):
            default_path = os.path.join(REPO_ROOT, "test_assets", "hand-template.jpg")
        hand_bgr = cv2.imread(default_path)
        if hand_bgr is None:
            raise FileNotFoundError("Master studio hand template not found")

    # 2. Landmarks
    landmarks = params.get("landmarks") or DEFAULT_STUDIO_LANDMARKS

    # 3. Mapped Nails (dict mapping finger name to RGBA base64)
    raw_mapped = params.get("mapped_nails", {})
    mapped_nails_rgba = {}
    for fname, b64 in raw_mapped.items():
        if b64 and isinstance(b64, str):
            try:
                mapped_nails_rgba[fname] = _load_rgba_image(b64)
            except Exception as e:
                print(f"Warning: Failed to decode nail for {fname}: {e}")

    # 4. Lighting Options & Downscale
    lighting_options = params.get("lighting_options", {})
    downscale = float(params.get("downscale", 1.0))
    downscale = max(0.2, min(1.0, downscale))

    # 5. Render Full Hand Composite
    composite_bgr = render_full_hand_composite(
        hand_bgr=hand_bgr,
        landmarks=landmarks,
        mapped_nails_rgba=mapped_nails_rgba,
        lighting_options=lighting_options,
        downscale=downscale,
    )

    t_render = int((time.time() - t0) * 1000)

    quality = 90 if downscale < 1.0 else 95
    return {
        "status": "ok",
        "composite_base64": _encode_jpg_base64(composite_bgr, quality=quality),
        "image_size": [int(composite_bgr.shape[1]), int(composite_bgr.shape[0])],
        "render_time_ms": t_render,
    }


def handle_export_composite(params):
    import time
    t0 = time.time()

    # 1. Load Hand Base Image
    hand_src = params.get("hand_image_base64") or params.get("image_base64")
    if hand_src and isinstance(hand_src, str) and hand_src.startswith("data:"):
        hand_bgr = _load_image({"image_base64": hand_src})
    elif "image_path" in params and os.path.exists(params["image_path"]):
        hand_bgr = cv2.imread(params["image_path"])
    else:
        default_path = os.path.join(REPO_ROOT, "templates", "default_studio_hand", "hand.jpg")
        if not os.path.exists(default_path):
            default_path = os.path.join(REPO_ROOT, "test_assets", "hand-template.jpg")
        hand_bgr = cv2.imread(default_path)
        if hand_bgr is None:
            raise FileNotFoundError("Master studio hand template not found")

    # 2. Landmarks
    landmarks = params.get("landmarks") or DEFAULT_STUDIO_LANDMARKS

    # 3. Mapped Nails
    raw_mapped = params.get("mapped_nails", {})
    mapped_nails_rgba = {}
    for fname, b64 in raw_mapped.items():
        if b64 and isinstance(b64, str):
            try:
                mapped_nails_rgba[fname] = _load_rgba_image(b64)
            except Exception as e:
                print(f"Warning: Failed to decode nail for {fname}: {e}")

    # 4. Lighting & Export Options
    lighting_options = params.get("lighting_options", {})
    export_options = params.get("export_options", {})

    out_dir = export_options.get("output_dir", "exports")
    if not os.path.isabs(out_dir):
        export_options["output_dir"] = os.path.join(REPO_ROOT, out_dir)

    res = export_composite_package(
        hand_bgr=hand_bgr,
        landmarks=landmarks,
        mapped_nails_rgba=mapped_nails_rgba,
        lighting_options=lighting_options,
        export_options=export_options,
    )
    res["execution_time_ms"] = int((time.time() - t0) * 1000)
    return res


HANDLERS = {
    "ping": handle_ping,
    "echo": handle_echo,
    "segment_tray": handle_segment_tray,
    "segment_prompt": handle_segment_prompt,
    "extract_nail": handle_extract_nail,
    "refit_slot_contour": handle_refit_slot_contour,
    "warp_nail": handle_warp_nail,
    "get_default_hand_template": handle_get_default_hand_template,
    "generate_hand_maps": handle_generate_hand_maps,
    "save_hand_template": handle_save_hand_template,
    "load_hand_template": handle_load_hand_template,
    "auto_detect_hand": handle_auto_detect_hand,
    "refit_finger": handle_refit_finger,
    "render_composite": handle_render_composite,
    "export_composite": handle_export_composite,
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
                sys.stderr.write(f"[Sidecar RPC ERROR] Unknown method '{method}'\n")
                response = {
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }
            else:
                t0_req = time.time()
                try:
                    result = HANDLERS[method](params)
                    elapsed_req = (time.time() - t0_req) * 1000.0
                    if method not in ("ping", "echo"):
                        sys.stderr.write(f"[Sidecar RPC] {method} -> OK ({elapsed_req:.1f}ms)\n")
                    response = {"id": req_id, "result": result}
                except Exception as ex:
                    elapsed_req = (time.time() - t0_req) * 1000.0
                    tb = traceback.format_exc()
                    sys.stderr.write(f"[Sidecar RPC EXCEPTION] {method} ({elapsed_req:.1f}ms): {ex}\n{tb}\n")
                    response = {
                        "id": req_id,
                        "error": {
                            "code": -32000,
                            "message": str(ex),
                            "traceback": tb
                        }
                    }
        except Exception as e:
            sys.stderr.write(f"[Sidecar JSON Parse ERROR]: {e}\n")
            response = {
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
