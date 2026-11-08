#!/usr/bin/env python3
"""
Press-On Nail Studio - Template Generator & Rigging Engine
Implements:
  - Procedural Eponychium Skin-Fold Cuticle Occlusion Mask (0.5px-0.7px Gaussian feather)
  - Contact Ambient Occlusion (AO) shadow map along the proximal nail fold and lateral grooves
  - Specular gloss reflection vector and highlight map aligned with finger axes
  - .nailtemplate serialization and default studio hand bundling
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
REPO_ROOT = os.path.abspath(os.path.join(current_dir, ".."))

_MP_DETECTOR = None

def _get_hand_landmarker_path() -> Optional[str]:
    """Resolves the MediaPipe hand_landmarker.task model path across dev and packaged distributions."""
    candidates = [
        os.path.join(REPO_ROOT, "models", "hand_landmarker.task"),
        os.path.join(current_dir, "..", "models", "hand_landmarker.task"),
        os.path.join(current_dir, "models", "hand_landmarker.task"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "models", "hand_landmarker.task") if hasattr(sys, "_MEIPASS") else "",
        os.path.join(os.path.dirname(sys.executable), "models", "hand_landmarker.task"),
        os.path.join(os.path.dirname(sys.executable), "hand_landmarker.task"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None

def get_mediapipe_hand_detector():
    """Returns a singleton MediaPipe HandLandmarker instance, or None if unavailable."""
    global _MP_DETECTOR
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR

    try:
        from mediapipe.tasks.python import vision, BaseOptions
    except ImportError:
        return None

    model_path = _get_hand_landmarker_path()
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        _MP_DETECTOR = vision.HandLandmarker.create_from_options(options)
        return _MP_DETECTOR
    except Exception as ex:
        print(f"[mediapipe] Failed to initialize HandLandmarker: {ex}", file=sys.stderr)
        return None

from warping import build_cuticle_pocket_mask

# Standard default calibrated landmarks for the bundled studio hand (test_assets/hand-template.jpg: 1536 x 2752)
DEFAULT_STUDIO_LANDMARKS: Dict[str, Dict[str, Any]] = {
    "Thumb": {
        "c_l": [204.0, 1418.0],
        "c_c": [232.0, 1432.0],
        "c_r": [264.0, 1412.0],
        "d_l": [196.0, 1278.0],
        "d_r": [258.0, 1282.0],
        "c_curve": 0.52,
        "roll_offset": -0.45,
        "length_scale": 0.95,
        "tip_taper": 0.92,
    },
    "Index": {
        "c_l": [412.0, 714.0],
        "c_c": [466.0, 738.0],
        "c_r": [519.0, 712.0],
        "d_l": [402.0, 562.0],
        "d_r": [515.0, 560.0],
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    "Middle": {
        "c_l": [693.0, 566.0],
        "c_c": [747.0, 592.0],
        "c_r": [799.0, 566.0],
        "d_l": [684.0, 410.0],
        "d_r": [795.0, 410.0],
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    "Ring": {
        "c_l": [948.0, 686.0],
        "c_c": [1000.0, 710.0],
        "c_r": [1051.0, 686.0],
        "d_l": [939.0, 540.0],
        "d_r": [1045.0, 540.0],
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    "Pinky": {
        "c_l": [1222.0, 990.0],
        "c_c": [1262.0, 1012.0],
        "c_r": [1301.0, 988.0],
        "d_l": [1217.0, 866.0],
        "d_r": [1298.0, 864.0],
        "c_curve": 0.32,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
}


def compute_finger_geometry(landmarks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes directional axis, orthogonal perpendicular unit, bed dimensions, and angle
    from the 5 primary landmarks.
    """
    c_l = np.array(landmarks["c_l"], dtype=np.float64)
    c_c = np.array(landmarks["c_c"], dtype=np.float64)
    c_r = np.array(landmarks["c_r"], dtype=np.float64)
    d_l = np.array(landmarks["d_l"], dtype=np.float64)
    d_r = np.array(landmarks["d_r"], dtype=np.float64)

    distal_mid = (d_l + d_r) * 0.5
    axis_vector = distal_mid - c_c
    bed_length = float(np.linalg.norm(axis_vector))
    if bed_length < 1e-4:
        axis_unit = np.array([0.0, -1.0], dtype=np.float64)
        bed_length = 50.0
    else:
        axis_unit = axis_vector / bed_length

    # Orthogonal perpendicular unit pointing towards the right lateral sidewall
    perp_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    if np.dot(perp_unit, c_r - c_l) < 0:
        perp_unit = -perp_unit

    cuticle_width = float(np.linalg.norm(c_r - c_l))
    distal_width = float(np.linalg.norm(d_r - d_l))
    bed_width = max(cuticle_width, distal_width)

    # Angle in degrees relative to vertical (0 deg = straight up)
    angle_rad = np.arctan2(axis_unit[0], -axis_unit[1])
    angle_deg = float(np.degrees(angle_rad))

    return {
        "c_l": c_l,
        "c_c": c_c,
        "c_r": c_r,
        "d_l": d_l,
        "d_r": d_r,
        "distal_mid": distal_mid,
        "axis_unit": axis_unit,
        "perp_unit": perp_unit,
        "bed_length": bed_length,
        "bed_width": bed_width,
        "cuticle_width": cuticle_width,
        "distal_width": distal_width,
        "angle_deg": angle_deg,
    }


def generate_cuticle_occlusion_mask(
    h: int,
    w: int,
    landmarks_dict: Dict[str, Dict[str, Any]],
    feather_sigma: float = 0.65,
) -> np.ndarray:
    """
    Generates a full-resolution 8-bit alpha mask (0-255) representing the anatomical
    eponychium skin fold that overlaps the base of the press-on nail.
    White (255) = skin fold covering the cuticle base; Black (0) = exposed nail bed / tip.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for finger_name, lm in landmarks_dict.items():
        geom = compute_finger_geometry(lm)
        c_l, c_c, c_r = geom["c_l"], geom["c_c"], geom["c_r"]
        axis_unit, perp_unit = geom["axis_unit"], geom["perp_unit"]
        bed_len, bed_w = geom["bed_length"], geom["bed_width"]

        # Cuticle pocket curvature parameters
        s_eq = bed_len * 0.22
        pocket_half_w = max(geom["cuticle_width"] * 0.52, 10.0)

        thetas = np.linspace(np.pi, 0.0, 45, dtype=np.float64)
        eq_center = c_c + axis_unit * s_eq
        u_cup_pts = (
            eq_center[None, :]
            + (np.cos(thetas) * pocket_half_w)[:, None] * perp_unit[None, :]
            - (np.sin(thetas) * s_eq)[:, None] * axis_unit[None, :]
        )

        # Proximal skin boundary wrapping around the finger knuckle
        knuckle_base_1 = c_c - axis_unit * (bed_len * 0.90) - perp_unit * (pocket_half_w * 1.35)
        knuckle_base_2 = c_c - axis_unit * (bed_len * 0.90) + perp_unit * (pocket_half_w * 1.35)

        skin_poly = np.vstack([
            u_cup_pts,
            [knuckle_base_2, knuckle_base_1],
        ]).astype(np.int32)

        # Draw skin flap on temporary supersampled canvas for sub-pixel anti-aliasing
        min_x = max(0, int(np.min(skin_poly[:, 0]) - 20))
        max_x = min(w, int(np.max(skin_poly[:, 0]) + 20))
        min_y = max(0, int(np.min(skin_poly[:, 1]) - 20))
        max_y = min(h, int(np.max(skin_poly[:, 1]) + 20))

        roi_w = max_x - min_x
        roi_h = max_y - min_y
        if roi_w <= 0 or roi_h <= 0:
            continue

        ss = 4
        poly_ss = (skin_poly - np.array([min_x, min_y])) * ss
        temp_ss = np.zeros((roi_h * ss, roi_w * ss), dtype=np.uint8)
        cv2.fillPoly(temp_ss, [poly_ss.reshape(-1, 1, 2)], 255)
        temp_roi = cv2.resize(temp_ss, (roi_w, roi_h), interpolation=cv2.INTER_AREA)

        mask[min_y:max_y, min_x:max_x] = np.maximum(mask[min_y:max_y, min_x:max_x], temp_roi)

    if feather_sigma > 0:
        ksize = int(np.ceil(feather_sigma * 4)) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_sigma)

    return mask


def generate_contact_ao_map(
    h: int,
    w: int,
    landmarks_dict: Dict[str, Dict[str, Any]],
    ao_depth: float = 0.22,
    blur_radius: float = 3.2,
) -> np.ndarray:
    """
    Generates an Ambient Occlusion (AO) contact shadow map along the proximal nail fold
    and lateral grooves.
    Output: 8-bit grayscale (0-255), where 255 = fully illuminated, and contact crease
    darkens down to ~ (1.0 - ao_depth) * 255.
    """
    ao_map = np.full((h, w), 255, dtype=np.uint8)
    shadow_acc = np.zeros((h, w), dtype=np.float32)

    for finger_name, lm in landmarks_dict.items():
        geom = compute_finger_geometry(lm)
        c_l, c_c, c_r = geom["c_l"], geom["c_c"], geom["c_r"]
        d_l, d_r = geom["d_l"], geom["d_r"]
        axis_unit, perp_unit = geom["axis_unit"], geom["perp_unit"]
        bed_len, bed_w = geom["bed_length"], geom["bed_width"]

        # Proximal U-seam
        s_eq = bed_len * 0.22
        pocket_half_w = max(geom["cuticle_width"] * 0.52, 10.0)
        thetas = np.linspace(np.pi, 0.0, 35, dtype=np.float64)
        eq_center = c_c + axis_unit * s_eq
        u_cup_pts = (
            eq_center[None, :]
            + (np.cos(thetas) * pocket_half_w)[:, None] * perp_unit[None, :]
            - (np.sin(thetas) * s_eq)[:, None] * axis_unit[None, :]
        )

        # Extended lateral groove lines towards distal sidewalls
        groove_pts = np.vstack([
            [d_l * 0.35 + c_l * 0.65],
            u_cup_pts,
            [d_r * 0.35 + c_r * 0.65],
        ])

        min_x = max(0, int(np.min(groove_pts[:, 0]) - 30))
        max_x = min(w, int(np.max(groove_pts[:, 0]) + 30))
        min_y = max(0, int(np.min(groove_pts[:, 1]) - 30))
        max_y = min(h, int(np.max(groove_pts[:, 1]) + 30))

        roi_w = max_x - min_x
        roi_h = max_y - min_y
        if roi_w <= 0 or roi_h <= 0:
            continue

        ss = 4
        pts_ss = ((groove_pts - np.array([min_x, min_y])) * ss).astype(np.int32)
        line_ss = np.zeros((roi_h * ss, roi_w * ss), dtype=np.uint8)
        cv2.polylines(line_ss, [pts_ss.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=int(3.0 * ss), lineType=cv2.LINE_AA)
        line_roi = cv2.resize(line_ss, (roi_w, roi_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        ksize = int(np.ceil(blur_radius * 4)) | 1
        line_blur = cv2.GaussianBlur(line_roi, (ksize, ksize), blur_radius)

        shadow_acc[min_y:max_y, min_x:max_x] = np.maximum(shadow_acc[min_y:max_y, min_x:max_x], line_blur)

    ao_val = np.clip(1.0 - shadow_acc * ao_depth, 0.0, 1.0)
    ao_map = (ao_val * 255.0).astype(np.uint8)
    return ao_map


def generate_specular_gloss_map(
    h: int,
    w: int,
    landmarks_dict: Dict[str, Dict[str, Any]],
    ribbon_width_ratio: float = 0.16,
    gloss_intensity: float = 0.85,
) -> np.ndarray:
    """
    Generates a 3-channel BGR specular gloss reflection map aligned with the studio key light
    and each finger's longitudinal directional axis.
    Simulates the salon gel topcoat glossy reflection strip running down the center arch.
    """
    spec_map = np.zeros((h, w, 3), dtype=np.uint8)
    spec_acc = np.zeros((h, w), dtype=np.float32)

    for finger_name, lm in landmarks_dict.items():
        geom = compute_finger_geometry(lm)
        c_c, distal_mid = geom["c_c"], geom["distal_mid"]
        axis_unit, perp_unit = geom["axis_unit"], geom["perp_unit"]
        bed_len, bed_w = geom["bed_length"], geom["bed_width"]
        roll_offset = float(lm.get("roll_offset", 0.0))

        # The specular ribbon runs along the finger axis, shifted slightly by roll_offset
        ribbon_center_offset = perp_unit * (bed_w * (roll_offset * 0.35))
        p_start = c_c + ribbon_center_offset + axis_unit * (bed_len * 0.10)
        p_end = distal_mid + ribbon_center_offset + axis_unit * (bed_len * 0.65)

        ribbon_w = max(6.0, bed_w * ribbon_width_ratio)

        # Draw centered specular stroke
        pts = np.array([p_start, p_end])
        min_x = max(0, int(np.min(pts[:, 0]) - ribbon_w * 3))
        max_x = min(w, int(np.max(pts[:, 0]) + ribbon_w * 3))
        min_y = max(0, int(np.min(pts[:, 1]) - ribbon_w * 3))
        max_y = min(h, int(np.max(pts[:, 1]) + ribbon_w * 3))

        roi_w = max_x - min_x
        roi_h = max_y - min_y
        if roi_w <= 0 or roi_h <= 0:
            continue

        ss = 4
        p_start_ss = (p_start - np.array([min_x, min_y])) * ss
        p_end_ss = (p_end - np.array([min_x, min_y])) * ss

        line_ss = np.zeros((roi_h * ss, roi_w * ss), dtype=np.uint8)
        cv2.line(
            line_ss,
            (int(round(p_start_ss[0])), int(round(p_start_ss[1]))),
            (int(round(p_end_ss[0])), int(round(p_end_ss[1]))),
            255,
            thickness=int(ribbon_w * ss),
            lineType=cv2.LINE_AA,
        )
        stroke_roi = cv2.resize(line_ss, (roi_w, roi_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ksize = int(np.ceil(ribbon_w * 1.5)) | 1
        stroke_blur = cv2.GaussianBlur(stroke_roi, (ksize, ksize), ribbon_w * 0.45)

        spec_acc[min_y:max_y, min_x:max_x] = np.maximum(spec_acc[min_y:max_y, min_x:max_x], stroke_blur * gloss_intensity)

    spec_acc_clamped = np.clip(spec_acc, 0.0, 1.0)
    spec_val = (spec_acc_clamped * 255.0).astype(np.uint8)
    spec_map = cv2.merge([spec_val, spec_val, spec_val])
    return spec_map


def generate_all_rig_maps(
    hand_bgr: np.ndarray,
    landmarks_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Computes all 3 procedural maps for the hand template:
      - 'cuticle_mask': 8-bit alpha mask of eponychium skin fold (0-255)
      - 'ao_map': 8-bit contact ambient occlusion shadow (0-255)
      - 'specular_map': BGR specular reflection ribbon map (0-255)
    """
    h, w = hand_bgr.shape[:2]
    cuticle_mask = generate_cuticle_occlusion_mask(h, w, landmarks_dict)
    ao_map = generate_contact_ao_map(h, w, landmarks_dict)
    specular_map = generate_specular_gloss_map(h, w, landmarks_dict)

    return {
        "cuticle_mask": cuticle_mask,
        "ao_map": ao_map,
        "specular_map": specular_map,
    }


def save_nailtemplate_bundle(
    target_dir: str,
    template_name: str,
    hand_bgr: np.ndarray,
    landmarks_dict: Dict[str, Dict[str, Any]],
    maps: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    Saves a self-contained .nailtemplate directory bundle containing:
      - template.json: Metadata, image dimensions, and calibrated 5-point landmarks
      - hand.jpg: High-resolution bare hand image
      - cuticle_mask.png: Pre-baked eponychium skin fold occlusion mask
      - ao_map.png: Pre-baked contact ambient occlusion shadow map
      - specular_map.png: Pre-baked salon gel specular ribbon map
    """
    os.makedirs(target_dir, exist_ok=True)

    if maps is None:
        maps = generate_all_rig_maps(hand_bgr, landmarks_dict)

    h, w = hand_bgr.shape[:2]

    # Save images
    hand_path = os.path.join(target_dir, "hand.jpg")
    cv2.imwrite(hand_path, hand_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])

    cuticle_path = os.path.join(target_dir, "cuticle_mask.png")
    cv2.imwrite(cuticle_path, maps["cuticle_mask"])

    ao_path = os.path.join(target_dir, "ao_map.png")
    cv2.imwrite(ao_path, maps["ao_map"])

    specular_path = os.path.join(target_dir, "specular_map.png")
    cv2.imwrite(specular_path, maps["specular_map"])

    metadata = {
        "format": "nailtemplate",
        "version": "1.0.0",
        "name": template_name,
        "image_size": [int(w), int(h)],
        "files": {
            "hand_image": "hand.jpg",
            "cuticle_mask": "cuticle_mask.png",
            "ao_map": "ao_map.png",
            "specular_map": "specular_map.png",
        },
        "landmarks": landmarks_dict,
    }

    meta_path = os.path.join(target_dir, "template.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def load_nailtemplate_bundle(target_dir: str) -> Dict[str, Any]:
    """
    Loads a .nailtemplate bundle from disk.
    """
    meta_path = os.path.join(target_dir, "template.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"template.json not found in {target_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    files = metadata.get("files", {})
    hand_file = os.path.join(target_dir, files.get("hand_image", "hand.jpg"))
    hand_bgr = cv2.imread(hand_file) if os.path.exists(hand_file) else None

    cuticle_file = os.path.join(target_dir, files.get("cuticle_mask", "cuticle_mask.png"))
    cuticle_mask = cv2.imread(cuticle_file, cv2.IMREAD_GRAYSCALE) if os.path.exists(cuticle_file) else None

    ao_file = os.path.join(target_dir, files.get("ao_map", "ao_map.png"))
    ao_map = cv2.imread(ao_file, cv2.IMREAD_GRAYSCALE) if os.path.exists(ao_file) else None

    spec_file = os.path.join(target_dir, files.get("specular_map", "specular_map.png"))
    spec_map = cv2.imread(spec_file) if os.path.exists(spec_file) else None

    return {
        "metadata": metadata,
        "hand_bgr": hand_bgr,
        "cuticle_mask": cuticle_mask,
        "ao_map": ao_map,
        "specular_map": spec_map,
    }


def init_default_studio_template(repo_root: str):
    """
    Bundles and initializes the built-in Default Studio Hand template
    in both `templates/default_studio_hand/` and `public/templates/default_studio_hand/`.
    """
    assets_dir = os.path.join(repo_root, "test_assets")
    src_hand = os.path.join(assets_dir, "hand-template.jpg")
    if not os.path.exists(src_hand):
        raise FileNotFoundError(f"Source hand template not found at {src_hand}")

    hand_bgr = cv2.imread(src_hand)
    if hand_bgr is None:
        raise ValueError(f"Failed to read image at {src_hand}")

    # Generate maps
    maps = generate_all_rig_maps(hand_bgr, DEFAULT_STUDIO_LANDMARKS)

    # 1. Target directory: templates/default_studio_hand
    target_dir = os.path.join(repo_root, "templates", "default_studio_hand")
    save_nailtemplate_bundle(
        target_dir=target_dir,
        template_name="Default Studio Hand",
        hand_bgr=hand_bgr,
        landmarks_dict=DEFAULT_STUDIO_LANDMARKS,
        maps=maps,
    )
    print(f"Initialized template at: {target_dir}")

    # 2. Public directory for webview: public/templates/default_studio_hand
    pub_dir = os.path.join(repo_root, "public", "templates", "default_studio_hand")
    save_nailtemplate_bundle(
        target_dir=pub_dir,
        template_name="Default Studio Hand",
        hand_bgr=hand_bgr,
        landmarks_dict=DEFAULT_STUDIO_LANDMARKS,
        maps=maps,
    )
    print(f"Initialized webview template at: {pub_dir}")


def detect_hand_landmarks_mediapipe(
    hand_bgr: np.ndarray,
    handedness_hint: str = "auto",
) -> Optional[Dict[str, Any]]:
    """
    Detects hand 3D skeletal landmarks using MediaPipe HandLandmarker.
    Extracts precise 5-point nail landmarks for all 5 fingers along with
    handedness and palm center/radius.
    Returns None if detection fails or MediaPipe is unavailable.
    """
    detector = get_mediapipe_hand_detector()
    if detector is None:
        return None

    try:
        import mediapipe as mp
        h, w = hand_bgr.shape[:2]
        rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        if not res.hand_landmarks or len(res.hand_landmarks) == 0:
            return None

        lms = res.hand_landmarks[0]
        detected_handedness = "left"
        if res.handedness and len(res.handedness) > 0 and len(res.handedness[0]) > 0:
            detected_handedness = res.handedness[0][0].category_name.lower()

        handedness = handedness_hint if handedness_hint in ("left", "right") else detected_handedness

        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)

        finger_joints = {
            "Thumb": (3, 4),   # IP, TIP
            "Index": (7, 8),   # DIP, TIP
            "Middle": (11, 12),
            "Ring": (15, 16),
            "Pinky": (19, 20),
        }

        landmarks: Dict[str, Dict[str, Any]] = {}
        for fname, (dip_idx, tip_idx) in finger_joints.items():
            dip = pts[dip_idx]
            tip = pts[tip_idx]
            is_thumb = (fname == "Thumb")

            bone_v = tip - dip
            bone_len = float(np.linalg.norm(bone_v))
            bone_u = bone_v / max(1e-4, bone_len)

            perp_u = np.array([-bone_u[1], bone_u[0]], dtype=np.float64)
            if perp_u[0] < 0:
                perp_u = -perp_u

            if is_thumb:
                nail_w = bone_len * 0.28
                distal_mid = tip + bone_u * (bone_len * 0.38)
                c_c = tip - bone_u * (bone_len * 0.24)
                c_curve = 0.52
                roll_offset = -0.45 if handedness == "right" else 0.45
                tip_taper = 0.92
                length_scale = 0.95
            else:
                nail_w = bone_len * 0.50
                distal_mid = tip + bone_u * (bone_len * 0.40)
                c_c = tip - bone_u * (bone_len * 0.42)
                c_curve = 0.28
                roll_offset = 0.0
                tip_taper = 1.00
                length_scale = 1.00

            d_l = distal_mid - perp_u * (nail_w * 0.50)
            d_r = distal_mid + perp_u * (nail_w * 0.50)
            c_l = c_c - perp_u * (nail_w * 0.48)
            c_r = c_c + perp_u * (nail_w * 0.48)

            landmarks[fname] = {
                "c_l": [round(float(c_l[0]), 1), round(float(c_l[1]), 1)],
                "c_c": [round(float(c_c[0]), 1), round(float(c_c[1]), 1)],
                "c_r": [round(float(c_r[0]), 1), round(float(c_r[1]), 1)],
                "d_l": [round(float(d_l[0]), 1), round(float(d_l[1]), 1)],
                "d_r": [round(float(d_r[0]), 1), round(float(d_r[1]), 1)],
                "c_curve": c_curve,
                "roll_offset": roll_offset,
                "length_scale": length_scale,
                "tip_taper": tip_taper,
            }

        wrist = pts[0]
        mid_mcp = pts[9]
        index_mcp = pts[5]
        pinky_mcp = pts[17]
        palm_c = wrist * 0.40 + mid_mcp * 0.30 + index_mcp * 0.15 + pinky_mcp * 0.15
        palm_r = float(np.linalg.norm(mid_mcp - wrist) * 0.38)

        return {
            "landmarks": landmarks,
            "handedness": handedness,
            "palm_center": [int(palm_c[0]), int(palm_c[1])],
            "palm_radius": palm_r,
            "detection_method": "mediapipe",
        }
    except Exception as e:
        print(f"[mediapipe] Error during detection: {e}", file=sys.stderr)
        return None


def _auto_detect_hand_landmarks_cv(hand_bgr: np.ndarray, handedness_hint: str = "auto") -> Dict[str, Any]:
    """
    Classical CV fallback:
    Automatically detects the hand, palm center, handedness (left vs right hand),
    and all 5 individual fingernail landmarks (Thumb, Index, Middle, Ring, Pinky).
    Uses a robust 2-stage anatomical decomposition:
      1. Isolates upright fingers via upper envelope minima.
      2. Isolates lateral abducted thumb via horizontal prominence from the palm.
      3. Fits local phalanx spines and applies natural human nail aspect ratios.
    """
    h, w = hand_bgr.shape[:2]

    # 1. Skin & Hand Silhouette Segmentation
    ycrcb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2HSV)

    # Chrominance bounds
    skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 185, 135]))
    skin_hsv = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([32, 255, 255]))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)

    # Background contrast check from image border corners
    corner_samples = np.vstack([
        hand_bgr[:25, :25].reshape(-1, 3),
        hand_bgr[:25, -25:].reshape(-1, 3),
        hand_bgr[-25:, :25].reshape(-1, 3),
        hand_bgr[-25:, -25:].reshape(-1, 3),
    ])
    bg_color = np.median(corner_samples, axis=0)
    color_dist = np.linalg.norm(hand_bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
    fg_mask = (color_dist > 25).astype(np.uint8) * 255

    combined = cv2.bitwise_and(skin_mask, fg_mask)
    if cv2.countNonZero(combined) < (h * w * 0.03):
        combined = fg_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No hand detected in the uploaded image")

    hand_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand_cnt) < (h * w * 0.02):
        raise ValueError("Detected hand contour is too small")

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [hand_cnt], -1, 255, -1)

    # 2. Palm Center via Distance Transform
    dist_map = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist_search = dist_map.copy()
    dist_search[int(h * 0.78):, :] = 0  # ignore wrist area
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_search)
    palm_x, palm_y = max_loc
    palm_r = max(18.0, float(max_val))

    pts = hand_cnt.squeeze()
    if len(pts.shape) != 2:
        raise ValueError("Invalid contour coordinates")

    # 3. Detect 4 Upright Main Fingers (y < palm_y - palm_r * 0.22)
    # Ignore border pixels (within 3% of top edge or sides) to avoid contour artifacts
    y_min_cutoff = int(h * 0.04)
    x_min_cutoff = int(w * 0.03)
    x_max_cutoff = int(w * 0.97)

    col_tops: Dict[int, int] = {}
    for p in pts:
        x, y = int(p[0]), int(p[1])
        if y > y_min_cutoff and y < (palm_y - palm_r * 0.22) and x_min_cutoff < x < x_max_cutoff:
            if x not in col_tops or y < col_tops[x]:
                col_tops[x] = y

    xs = sorted(col_tops.keys())
    four_fingers = []

    if len(xs) >= 20:
        ys = np.array([col_tops[x] for x in xs], dtype=np.float32)
        k_smooth = max(5, (len(xs) // 35) | 1)
        ys_smooth = cv2.GaussianBlur(ys[:, None], (1, k_smooth), k_smooth / 3.0).ravel()

        minima = []
        win = max(6, len(xs) // 45)
        for i in range(win, len(ys_smooth) - win):
            val = ys_smooth[i]
            if val == np.min(ys_smooth[i - win : i + win + 1]):
                local_max = np.max(ys_smooth[max(0, i - win * 2) : min(len(ys_smooth), i + win * 2 + 1)])
                depth = local_max - val
                if depth > 12:  # Distinguish prominent finger apex from subtle ripples
                    minima.append((xs[i], int(ys[i]), depth))

        min_tip_spacing = max(24, int(palm_r * 0.24))
        # Select candidates, prioritizing fingers directly above the palm span
        valid_minima = [
            m for m in minima
            if abs(m[0] - palm_x) < (palm_r * 1.35)
        ]
        for m in sorted(valid_minima, key=lambda x: x[1]):  # highest first
            x_m, y_m, _ = m
            if not any(abs(x_m - f[0]) < min_tip_spacing for f in four_fingers):
                four_fingers.append((x_m, y_m))
            if len(four_fingers) == 4:
                break

    four_fingers.sort(key=lambda f: f[0])

    # 4. Detect Lateral Thumb (Check Right vs Left of Palm)
    right_pts = pts[(pts[:, 0] > palm_x + palm_r * 0.35) & (pts[:, 1] > palm_y - palm_r * 0.6) & (pts[:, 1] < palm_y + palm_r * 0.5)]
    left_pts = pts[(pts[:, 0] < palm_x - palm_r * 0.35) & (pts[:, 1] > palm_y - palm_r * 0.6) & (pts[:, 1] < palm_y + palm_r * 0.5)]

    right_extent = float(np.max(right_pts[:, 0]) - palm_x) if len(right_pts) > 0 else 0.0
    left_extent = float(palm_x - np.min(left_pts[:, 0])) if len(left_pts) > 0 else 0.0

    # Determine handedness:
    # On a dorsal (knuckles/nails facing camera) hand:
    # Thumb on left => Right Hand
    # Thumb on right => Left Hand
    if handedness_hint in ("left", "right"):
        handedness = handedness_hint
    else:
        # Check lateral prominence of thumb compared to hypothenar margin
        if left_extent > right_extent * 1.08:
            handedness = "right"
        elif right_extent > left_extent * 1.08:
            handedness = "left"
        else:
            # Fallback: check which side has a lower lateral protrusion
            left_lowest_y = float(np.max(left_pts[:, 1])) if len(left_pts) > 0 else 0.0
            right_lowest_y = float(np.max(right_pts[:, 1])) if len(right_pts) > 0 else 0.0
            handedness = "right" if left_extent >= right_extent else "left"

    if handedness == "left" and len(right_pts) > 0:
        dists_thumb = (right_pts[:, 0] - palm_x) * 1.0 - (right_pts[:, 1] - palm_y) * 0.5
        thumb_tip = (int(right_pts[np.argmax(dists_thumb)][0]), int(right_pts[np.argmax(dists_thumb)][1]))
        if len(four_fingers) == 4:
            finger_tips = {
                "Pinky": four_fingers[0],
                "Ring": four_fingers[1],
                "Middle": four_fingers[2],
                "Index": four_fingers[3],
                "Thumb": thumb_tip,
            }
        else:
            finger_tips = None
    elif handedness == "right" and len(left_pts) > 0:
        dists_thumb = (palm_x - left_pts[:, 0]) * 1.0 - (left_pts[:, 1] - palm_y) * 0.5
        thumb_tip = (int(left_pts[np.argmax(dists_thumb)][0]), int(left_pts[np.argmax(dists_thumb)][1]))
        if len(four_fingers) == 4:
            finger_tips = {
                "Thumb": thumb_tip,
                "Index": four_fingers[0],
                "Middle": four_fingers[1],
                "Ring": four_fingers[2],
                "Pinky": four_fingers[3],
            }
        else:
            finger_tips = None
    else:
        finger_tips = None

    # Fallback to polar peaks or default scale if detection was ambiguous
    if finger_tips is None:
        scale_x = w / 1536.0
        scale_y = h / 2752.0
        fallback = {}
        for fname, lm in DEFAULT_STUDIO_LANDMARKS.items():
            fallback[fname] = {
                "c_l": [round(lm["c_l"][0] * scale_x, 1), round(lm["c_l"][1] * scale_y, 1)],
                "c_c": [round(lm["c_c"][0] * scale_x, 1), round(lm["c_c"][1] * scale_y, 1)],
                "c_r": [round(lm["c_r"][0] * scale_x, 1), round(lm["c_r"][1] * scale_y, 1)],
                "d_l": [round(lm["d_l"][0] * scale_x, 1), round(lm["d_l"][1] * scale_y, 1)],
                "d_r": [round(lm["d_r"][0] * scale_x, 1), round(lm["d_r"][1] * scale_y, 1)],
                "c_curve": lm.get("c_curve", 0.28),
                "roll_offset": lm.get("roll_offset", 0.0),
                "length_scale": 1.0,
                "tip_taper": 1.0,
            }
        return {
            "landmarks": fallback,
            "handedness": handedness,
            "palm_center": [int(palm_x), int(palm_y)],
            "palm_radius": float(palm_r),
            "detection_method": "cv",
        }

    # 5. Local Phalanx Medial Spine Fitting & Anatomical Nail Sizing
    landmarks_dict = {}

    for fname, (tx, ty) in finger_tips.items():
        is_thumb = (fname == "Thumb")
        mids = []
        widths = []

        if not is_thumb:
            for dy in range(4, 35, 3):
                curr_y = ty + dy
                if curr_y >= h:
                    break
                row_span = filled[curr_y, max(0, tx - 35) : min(w, tx + 36)]
                row_indices = np.where(row_span > 0)[0]
                if len(row_indices) > 5:
                    xl = max(0, tx - 35) + row_indices[0]
                    xr = max(0, tx - 35) + row_indices[-1]
                    mids.append([(xl + xr) * 0.5, float(curr_y)])
                    widths.append(float(xr - xl))

            if len(mids) >= 3:
                mids_arr = np.array(mids)
                v = mids_arr[0] - mids_arr[-1]  # pointing UPWARDS towards tip
                v_len = np.linalg.norm(v)
                spine_unit = v / max(1e-4, v_len)
                fw = float(np.median(widths))
            else:
                spine_unit = np.array([0.0, -1.0])
                fw = 28.0
        else:
            thumb_sign = 1.0 if handedness == "left" else -1.0
            v_diag = np.array([thumb_sign * 0.9, -0.9], dtype=np.float64)
            spine_unit = v_diag / np.linalg.norm(v_diag)
            v_perp = np.array([-spine_unit[1], spine_unit[0]])

            for step in range(5, 35, 4):
                c_pt = np.array([tx, ty]) - spine_unit * step
                w_side = 0
                for sw in range(1, 30):
                    p = c_pt + v_perp * sw
                    if 0 <= int(p[1]) < h and 0 <= int(p[0]) < w and filled[int(p[1]), int(p[0])] > 0:
                        w_side = sw
                    else:
                        break
                if w_side > 4:
                    widths.append(float(w_side * 2))

            fw = float(np.median(widths)) if widths else 32.0

        perp_unit = np.array([-spine_unit[1], spine_unit[0]], dtype=np.float64)
        if perp_unit[0] < 0:
            perp_unit = -perp_unit

        # Natural human fingernail proportions
        nail_w = fw * 0.74
        nail_len = nail_w * 1.05

        distal_mid = np.array([tx, ty], dtype=np.float64) - spine_unit * 3.0
        d_l = distal_mid - perp_unit * (nail_w * 0.50)
        d_r = distal_mid + perp_unit * (nail_w * 0.50)

        c_c = distal_mid - spine_unit * nail_len
        c_l = c_c - perp_unit * (nail_w * 0.48)
        c_r = c_c + perp_unit * (nail_w * 0.48)

        c_curve = 0.42 if is_thumb else 0.28
        roll_offset = 0.0
        if is_thumb:
            roll_offset = -0.35 if handedness == "left" else 0.35

        landmarks_dict[fname] = {
            "c_l": [round(float(c_l[0]), 1), round(float(c_l[1]), 1)],
            "c_c": [round(float(c_c[0]), 1), round(float(c_c[1]), 1)],
            "c_r": [round(float(c_r[0]), 1), round(float(c_r[1]), 1)],
            "d_l": [round(float(d_l[0]), 1), round(float(d_l[1]), 1)],
            "d_r": [round(float(d_r[0]), 1), round(float(d_r[1]), 1)],
            "c_curve": c_curve,
            "roll_offset": roll_offset,
            "length_scale": 1.0,
            "tip_taper": 1.0,
        }

    return {
        "landmarks": landmarks_dict,
        "handedness": handedness,
        "palm_center": [int(palm_x), int(palm_y)],
        "palm_radius": float(palm_r),
        "detection_method": "cv",
    }


def auto_detect_hand_landmarks(hand_bgr: np.ndarray, handedness_hint: str = "auto") -> Dict[str, Any]:
    """
    Automatically detects the hand, palm center, handedness (left vs right hand),
    and all 5 individual fingernail landmarks (Thumb, Index, Middle, Ring, Pinky).
    Uses Google MediaPipe HandLandmarker for sub-pixel deep learning skeletal accuracy,
    and seamlessly falls back to 2-stage anatomical CV decomposition if MediaPipe is unavailable.
    """
    mp_res = detect_hand_landmarks_mediapipe(hand_bgr, handedness_hint=handedness_hint)
    if mp_res is not None:
        return mp_res

    return _auto_detect_hand_landmarks_cv(hand_bgr, handedness_hint=handedness_hint)


def refit_finger_landmarks(
    hand_bgr: np.ndarray,
    finger_name: str,
    bbox: List[int],
    handedness: str = "left",
) -> Dict[str, Any]:
    """
    Computes accurate 5-point nail landmarks within a user-drawn or detected
    bounding box [x, y, w, h] over any specific fingertip.
    Uses MediaPipe if the detected finger landmark is within the bounding region,
    falling back to contour morphology.
    """
    bx, by, bw, bh = bbox
    h, w = hand_bgr.shape[:2]
    bx = max(0, min(w - 1, int(bx)))
    by = max(0, min(h - 1, int(by)))
    bw = max(10, min(w - bx, int(bw)))
    bh = max(10, min(h - by, int(bh)))

    # Try MediaPipe detection first
    mp_res = detect_hand_landmarks_mediapipe(hand_bgr, handedness_hint=handedness)
    if mp_res and finger_name in mp_res.get("landmarks", {}):
        flm = mp_res["landmarks"][finger_name]
        cx, cy = flm["c_c"]
        dx, dy = (flm["d_l"][0] + flm["d_r"][0]) * 0.5, (flm["d_l"][1] + flm["d_r"][1]) * 0.5
        margin_x = bw * 0.75
        margin_y = bh * 0.75
        in_bbox = (
            (bx - margin_x <= cx <= bx + bw + margin_x and by - margin_y <= cy <= by + bh + margin_y) or
            (bx - margin_x <= dx <= bx + bw + margin_x and by - margin_y <= dy <= by + bh + margin_y)
        )
        if in_bbox:
            return {
                "finger_name": finger_name,
                "landmarks": flm,
                "detection_method": "mediapipe",
            }

    roi = hand_bgr[by : by + bh, bx : bx + bw]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([40, 133, 77]), np.array([255, 185, 135]))

    corner_bgr = np.median(np.vstack([roi[:5, :5], roi[:5, -5:]]), axis=(0, 1))
    diff = np.linalg.norm(roi.astype(np.float32) - corner_bgr[None, None, :], axis=2)
    fg = (diff > 25).astype(np.uint8) * 255
    combined = cv2.bitwise_and(skin, fg)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        tx, ty = bx + bw // 2, by + bh // 3
        spine_unit = np.array([0.0, -1.0])
        fw = float(bw * 0.70)
    else:
        best_cnt = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(clean)
        cv2.drawContours(filled, [best_cnt], -1, 255, -1)
        ys, xs = np.where(filled > 0)

        is_thumb = (finger_name == "Thumb")
        if not is_thumb:
            min_y = np.min(ys)
            top_xs = xs[ys < min_y + 6]
            tx = bx + int(np.median(top_xs))
            ty = by + int(min_y)

            # Find cross sections to determine local spine angle
            mids = []
            widths = []
            for dy in range(6, min(35, bh - min_y), 3):
                curr_y = min_y + dy
                row_xs = xs[ys == curr_y]
                if len(row_xs) > 4:
                    xl = np.min(row_xs)
                    xr = np.max(row_xs)
                    mids.append([(xl + xr) * 0.5 + bx, curr_y + by])
                    widths.append(float(xr - xl))

            if len(mids) >= 3:
                mids_arr = np.array(mids)
                v = mids_arr[0] - mids_arr[-1]  # points upwards to tip
                v_len = np.linalg.norm(v)
                spine_raw = v / max(1e-4, v_len)
                if spine_raw[1] > -0.70:  # clamp excessive horizontal tilt
                    spine_unit = np.array([0.0, -1.0])
                else:
                    spine_unit = spine_raw
                fw = float(np.median(widths))
            else:
                spine_unit = np.array([0.0, -1.0])
                fw = float(bw * 0.70)
        else:
            thumb_sign = 1.0 if handedness == "left" else -1.0
            scores = xs * thumb_sign * 1.0 - ys * 1.3
            best_i = np.argmax(scores)
            tx = bx + int(xs[best_i])
            ty = by + int(ys[best_i])

            v_thumb = np.array([thumb_sign * 0.65, -0.76])
            spine_unit = v_thumb / np.linalg.norm(v_thumb)
            fw = float(bw * 0.75)

    perp_unit = np.array([-spine_unit[1], spine_unit[0]], dtype=np.float64)
    if perp_unit[0] < 0:
        perp_unit = -perp_unit

    is_thumb = (finger_name == "Thumb")
    nail_w = max(16.0, fw * 0.74)
    nail_len = nail_w * (0.95 if is_thumb else 1.05)

    distal_mid = np.array([tx, ty], dtype=np.float64) - spine_unit * 3.0
    d_l = distal_mid - perp_unit * (nail_w * 0.50)
    d_r = distal_mid + perp_unit * (nail_w * 0.50)

    c_c = distal_mid - spine_unit * nail_len
    c_l = c_c - perp_unit * (nail_w * 0.48)
    c_r = c_c + perp_unit * (nail_w * 0.48)

    c_curve = 0.42 if is_thumb else 0.28
    roll_offset = 0.0
    if is_thumb:
        roll_offset = -0.35 if handedness == "left" else 0.35

    return {
        "finger_name": finger_name,
        "landmarks": {
            "c_l": [round(float(c_l[0]), 1), round(float(c_l[1]), 1)],
            "c_c": [round(float(c_c[0]), 1), round(float(c_c[1]), 1)],
            "c_r": [round(float(c_r[0]), 1), round(float(c_r[1]), 1)],
            "d_l": [round(float(d_l[0]), 1), round(float(d_l[1]), 1)],
            "d_r": [round(float(d_r[0]), 1), round(float(d_r[1]), 1)],
            "c_curve": c_curve,
            "roll_offset": roll_offset,
            "length_scale": 1.0,
            "tip_taper": 1.0,
        },
        "detection_method": "cv",
    }


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    init_default_studio_template(repo_root)
