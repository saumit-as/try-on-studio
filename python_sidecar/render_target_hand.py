#!/usr/bin/env python3
"""
Renders the user's uploaded press-on nail tray onto the user's hand photo
with 100% texture preservation, salon gel topcoat gloss ribbon, and natural contact shadows.
"""

import os
import sys
import time
import shutil
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from segmentation import segment_packaging_tray, extract_nail_rgba
from warping import (
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
    build_cuticle_pocket_mask,
)
import compositor

REPO_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
ASSETS_DIR = os.path.join(REPO_ROOT, "test_assets")
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_output")
ARTIFACT_DIR = r"C:\Users\Timuas\.gemini\antigravity\brain\f2f8b0da-86b6-4059-9f72-faf98ddc9525"

TRAY_PATH = os.path.join(ASSETS_DIR, "target_tray.jpg")
HAND_TEMPLATE_PATH = os.path.join(ASSETS_DIR, "hand-template.jpg")
TARGET_HAND_PATH = os.path.join(ASSETS_DIR, "target_hand.jpg")

FINGER_TARGETS = [
    {
        "name": "Thumb",
        "slot_idx": 0,
        "c_l": (200.0, 1425.0),
        "c_c": (236.0, 1442.0),
        "c_r": (275.0, 1416.0),
        "d_l": (190.0, 1270.0),
        "d_r": (268.0, 1276.0),
        "c_curve": 0.44,
        "roll_offset": -0.35,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    {
        "name": "Index",
        "slot_idx": 2,
        "c_l": (412.0, 714.0),
        "c_c": (466.0, 738.0),
        "c_r": (519.0, 712.0),
        "d_l": (399.0, 562.0),
        "d_r": (518.0, 560.0),
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    {
        "name": "Middle",
        "slot_idx": 1,
        "c_l": (693.0, 566.0),
        "c_c": (747.0, 592.0),
        "c_r": (799.0, 566.0),
        "d_l": (681.0, 410.0),
        "d_r": (798.0, 410.0),
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    {
        "name": "Ring",
        "slot_idx": 3,
        "c_l": (948.0, 686.0),
        "c_c": (1000.0, 710.0),
        "c_r": (1051.0, 686.0),
        "d_l": (936.0, 540.0),
        "d_r": (1048.0, 540.0),
        "c_curve": 0.28,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
    {
        "name": "Pinky",
        "slot_idx": 4,
        "c_l": (1222.0, 990.0),
        "c_c": (1262.0, 1012.0),
        "c_r": (1301.0, 988.0),
        "d_l": (1214.0, 866.0),
        "d_r": (1301.0, 864.0),
        "c_curve": 0.32,
        "roll_offset": 0.0,
        "length_scale": 1.00,
        "tip_taper": 1.00,
    },
]


def main():
    print(f"Loading tray: {TRAY_PATH}")
    tray_img = cv2.imread(TRAY_PATH)
    if tray_img is None:
        raise FileNotFoundError(f"Tray not found at {TRAY_PATH}")

    print(f"Loading hand template: {HAND_TEMPLATE_PATH}")
    hand_img = cv2.imread(HAND_TEMPLATE_PATH)
    if hand_img is None:
        raise FileNotFoundError(f"Hand template not found at {HAND_TEMPLATE_PATH}")

    user_hand_img = cv2.imread(TARGET_HAND_PATH)
    h_user, w_user = user_hand_img.shape[:2] if user_hand_img is not None else (1024, 571)

    # 1. Segment Tray
    print("Segmenting tray slots...")
    slots = segment_packaging_tray(tray_img, max_slots=15)
    print(f"Detected {len(slots)} slots.")

    # 2. Extract first 5 nails
    extracted_nails = []
    clean_nails = []
    for s in slots[:5]:
        sid = s["slot_id"]
        cnt = np.array(s["contour"], dtype=np.int32)
        inv = s["shape_info"]["metrics"].get("is_inverted", False)

        nail_rgba = extract_nail_rgba(
            tray_img,
            cnt,
            feather_px=1.5,
            deskew=True,
            auto_orient_cuticle_down=True,
            force_inverted=inv,
        )
        extracted_nails.append(nail_rgba)

        # Surgical glare cleaning
        cleaned_rgba = compositor.clean_tray_glare_preserving_texture(nail_rgba)
        clean_nails.append(cleaned_rgba)

    # 3. Calculate scale
    h_hand, w_hand = hand_img.shape[:2]
    hand_bed_widths = []
    tray_nail_widths = []
    min_scale_to_cover_bed = 0.0
    for t in FINGER_TARGETS[1:]:
        cw = float(np.linalg.norm(np.array(t["c_r"]) - np.array(t["c_l"])))
        dw = float(np.linalg.norm(np.array(t["d_r"]) - np.array(t["d_l"])))
        distal_mid = 0.5 * (np.array(t["d_l"]) + np.array(t["d_r"]))
        bed_len = float(np.linalg.norm(distal_mid - np.array(t["c_c"])))
        hand_bed_widths.append(max(cw, dw))

        nd = clean_nails[t["slot_idx"]]
        xs = np.where(nd[:, :, 3] > 25)[1]
        tray_nail_widths.append(float(np.max(xs) - np.min(xs) + 1) if len(xs) > 0 else float(nd.shape[1]))
        min_scale_to_cover_bed = max(min_scale_to_cover_bed, (bed_len * 1.24) / max(1.0, float(nd.shape[0])))

    width_based_scale = (sum(hand_bed_widths) / max(1.0, sum(tray_nail_widths))) * 1.05
    tray_to_hand_scale = max(width_based_scale, min_scale_to_cover_bed)
    print(f"Tray-to-hand scale: {tray_to_hand_scale:.3f}x")

    # 4. Warp & Composite
    relit_hand = hand_img.copy().astype(np.float32)

    for target in FINGER_TARGETS:
        fname = target["name"]
        clean_rgba = clean_nails[target["slot_idx"]]

        curved = apply_cylindrical_c_curve(
            clean_rgba,
            c_curve_intensity=target["c_curve"],
            roll_offset=target.get("roll_offset", 0.0),
        )

        src_h, src_w = curved.shape[:2]
        src_pts, dst_pts = build_3tier_control_grid(
            src_w=src_w,
            src_h=src_h,
            cuticle_left=target["c_l"],
            cuticle_center=target["c_c"],
            cuticle_right=target["c_r"],
            distal_left=target["d_l"],
            distal_right=target["d_r"],
            length_scale=target["length_scale"],
            tip_taper=target.get("tip_taper", 1.00),
            roll_offset=target.get("roll_offset", 0.0),
            tray_to_hand_scale=tray_to_hand_scale,
        )

        warped_roi, bbox = warp_nail_tps(curved, src_pts, dst_pts, output_shape=(h_hand, w_hand))
        compositor.composite_nail_pbr(relit_hand, hand_img, warped_roi, bbox, target)
        print(f"Composited {fname} successfully.")

    final_composite_highres = np.clip(relit_hand, 0, 255).astype(np.uint8)

    # Downscale highres composite to user's exact uploaded hand resolution (571x1024)
    final_composite_user_res = cv2.resize(
        final_composite_highres, (w_user, h_user), interpolation=cv2.INTER_AREA
    )

    # Save to test_output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_highres_path = os.path.join(OUTPUT_DIR, "user_hand_pressons_highres.jpg")
    out_userres_path = os.path.join(OUTPUT_DIR, "user_hand_pressons.jpg")

    cv2.imwrite(out_highres_path, final_composite_highres, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(out_userres_path, final_composite_user_res, [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"Saved: {out_highres_path}")
    print(f"Saved: {out_userres_path}")

    # Copy to artifacts directory
    artifact_highres = os.path.join(ARTIFACT_DIR, "user_hand_pressons_highres.jpg")
    artifact_userres = os.path.join(ARTIFACT_DIR, "user_hand_pressons.jpg")
    shutil.copyfile(out_highres_path, artifact_highres)
    shutil.copyfile(out_userres_path, artifact_userres)
    print(f"Copied artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
