import os
import cv2
import numpy as np
import sys
sys.path.append("python_sidecar")

from segmentation import segment_packaging_tray, extract_nail_rgba
from template_generator import DEFAULT_STUDIO_LANDMARKS
from compositor import clean_tray_glare_preserving_texture, composite_nail_pbr
from warping import apply_cylindrical_c_curve, build_3tier_control_grid, warp_nail_tps

def test_full_pipeline():
    tray_path = "test_assets/target_tray.jpg"
    hand_path = "templates/default_studio_hand/hand.jpg"
    
    tray_img = cv2.imread(tray_path)
    hand_img = cv2.imread(hand_path)
    print("Tray img:", tray_img.shape if tray_img is not None else None)
    print("Hand img:", hand_img.shape if hand_img is not None else None)

    # Segment tray
    slots = segment_packaging_tray(tray_img, max_slots=5)
    print(f"Segmented {len(slots)} slots")

    # Extract 5 nails
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    extracted = {}
    for i, s in enumerate(slots[:5]):
        cnt = np.array(s["contour"], dtype=np.int32)
        inv = s["shape_info"]["metrics"].get("is_inverted", False)
        rgba = extract_nail_rgba(tray_img, cnt, deskew=True, auto_orient_cuticle_down=True, force_inverted=inv)
        cleaned = clean_tray_glare_preserving_texture(rgba)
        extracted[fingers[i]] = cleaned
    print("Extracted 5 clean nails")

    # Downscale 0.5x for ultra-fast preview
    scale = 0.5
    h_hand, w_hand = hand_img.shape[:2]
    h_down, w_down = int(h_hand * scale), int(w_hand * scale)
    hand_down = cv2.resize(hand_img, (w_down, h_down), interpolation=cv2.INTER_AREA)

    scaled_landmarks = {}
    for fname, lm in DEFAULT_STUDIO_LANDMARKS.items():
        scaled_landmarks[fname] = {
            "c_l": [lm["c_l"][0] * scale, lm["c_l"][1] * scale],
            "c_c": [lm["c_c"][0] * scale, lm["c_c"][1] * scale],
            "c_r": [lm["c_r"][0] * scale, lm["c_r"][1] * scale],
            "d_l": [lm["d_l"][0] * scale, lm["d_l"][1] * scale],
            "d_r": [lm["d_r"][0] * scale, lm["d_r"][1] * scale],
            "c_curve": lm.get("c_curve", 0.28),
            "roll_offset": lm.get("roll_offset", 0.0),
            "length_scale": lm.get("length_scale", 1.0),
            "tip_taper": lm.get("tip_taper", 1.0),
        }

    relit_hand = hand_down.copy().astype(np.float32)

    for fname in fingers:
        clean_rgba = cv2.resize(extracted[fname], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        target = scaled_landmarks[fname]
        curved = apply_cylindrical_c_curve(clean_rgba, c_curve_intensity=target["c_curve"], roll_offset=target["roll_offset"])
        src_h, src_w = curved.shape[:2]
        src_pts, dst_pts = build_3tier_control_grid(
            src_w=src_w,
            src_h=src_h,
            cuticle_left=target["c_l"],
            cuticle_center=target["c_c"],
            cuticle_right=target["c_r"],
            distal_left=target["d_l"],
            distal_right=target["d_r"],
            length_scale=1.1,
            tip_taper=target["tip_taper"],
            roll_offset=target["roll_offset"],
            tray_to_hand_scale=1.0,
        )
        warped_roi, bbox = warp_nail_tps(curved, src_pts, dst_pts, output_shape=(h_down, w_down))
        composite_nail_pbr(relit_hand, hand_down, warped_roi, bbox, target)

    result = np.clip(relit_hand, 0, 255).astype(np.uint8)
    cv2.imwrite("test_output/phase4_fast_preview.jpg", result)
    print("Saved test_output/phase4_fast_preview.jpg with shape:", result.shape)

if __name__ == "__main__":
    test_full_pipeline()
