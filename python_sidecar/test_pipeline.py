#!/usr/bin/env python3
"""
Press-On Nail Studio - Phase 1 Verification Test Pipeline
Executes standalone end-to-end testing of:
  1. Tray contour segmentation & PCA deskewing
  2. Geometric nail shape classification
  3. Transverse cylindrical C-curve barrel distortion
  4. 3-tier Thin-Plate Spline (TPS) transformation onto 5-finger hand template
Saves visual verification assets to test_output/ directory.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure local modules are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from segmentation import segment_packaging_tray, extract_nail_rgba, deskew_nail_pca, classify_nail_shape
from warping import apply_cylindrical_c_curve, build_3tier_control_grid, warp_nail_tps


def main():
    print("=" * 70)
    print(" PRESS-ON NAIL STUDIO - PHASE 1 PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Output directory
    output_dir = os.path.join(current_dir, "..", "test_output")
    os.makedirs(output_dir, exist_ok=True)

    tray_path = os.path.join(current_dir, "..", "test_assets", "nails-sample-tray.jpg")
    hand_path = os.path.join(current_dir, "..", "test_assets", "hand-template.jpg")

    if not os.path.exists(tray_path):
        raise FileNotFoundError(f"Tray asset not found: {tray_path}")
    if not os.path.exists(hand_path):
        raise FileNotFoundError(f"Hand template asset not found: {hand_path}")

    # 2. Tray Segmentation & PCA Deskewing
    print("\n[Step 1] Loading packaging tray photo...")
    t0 = time.time()
    tray_img = cv2.imread(tray_path)
    h_tray, w_tray = tray_img.shape[:2]
    print(f"  Loaded tray image: {w_tray}x{h_tray} px")

    print("[Step 2] Executing hierarchical contour segmentation & PCA deskewing...")
    t_seg_start = time.time()
    slots = segment_packaging_tray(tray_img, max_slots=10)
    t_seg_end = time.time()
    print(f"  Completed in {(t_seg_end - t_seg_start)*1000:.1f} ms")
    print(f"  Detected {len(slots)} nail slots:")

    # Annotate tray visualization
    annotated_tray = tray_img.copy()
    slot_colors = [
        (255, 105, 180),  # Pink
        (0, 255, 255),    # Yellow
        (50, 205, 50),    # Lime
        (30, 144, 255),   # Dodger Blue
        (255, 140, 0),    # Dark Orange
        (186, 85, 211),   # Medium Orchid
        (0, 250, 154),    # Medium Spring Green
        (255, 20, 147),   # Deep Pink
        (255, 215, 0),    # Gold
        (0, 191, 255),    # Deep Sky Blue
    ]

    extracted_nails = []

    for s in slots:
        sid = s["slot_id"]
        x, y, bw, bh = s["bbox"]
        angle = s["pca_angle"]
        shape = s["shape_info"]["shape"]
        conf = s["shape_info"]["confidence"]
        cnt = np.array(s["contour"], dtype=np.int32)
        color = slot_colors[(sid - 1) % len(slot_colors)]

        print(f"    Slot {sid:2d}: Pos=({x}, {y}), Size=({bw}x{bh}), Angle={angle:+.2f}°, Shape={shape} (conf={conf})")

        # Draw contour & bounding box on debug image
        cv2.drawContours(annotated_tray, [cnt], -1, color, 3)
        cv2.rectangle(annotated_tray, (x, y), (x + bw, y + bh), color, 2)
        label = f"#{sid} {shape} ({angle:+.1f}deg)"
        cv2.putText(annotated_tray, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(annotated_tray, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Extract RGBA for the first 5 nails (Hand 1 display row)
        if sid <= 5:
            nail_rgba = extract_nail_rgba(tray_img, cnt, feather_px=0.7)
            # Deskew to vertical
            deskew_angle, _, deskewed_rgba = deskew_nail_pca(cnt, nail_rgba)
            if deskewed_rgba is not None:
                nail_rgba = deskewed_rgba

            extracted_nails.append({
                "slot_id": sid,
                "rgba": nail_rgba,
                "shape": shape,
            })

            # Save individual cutout
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            fname = finger_names[sid - 1]
            out_nail_path = os.path.join(output_dir, f"nail_{sid}_{fname}.png")
            cv2.imwrite(out_nail_path, nail_rgba)

    tray_annotated_path = os.path.join(output_dir, "01_tray_segmented.jpg")
    cv2.imwrite(tray_annotated_path, annotated_tray)
    print(f"  Saved annotated tray to: {tray_annotated_path}")

    # 3. Load Hand Template & TPS Warping
    print("\n[Step 3] Loading hand template photo...")
    hand_img = cv2.imread(hand_path)
    h_hand, w_hand = hand_img.shape[:2]
    print(f"  Loaded hand template: {w_hand}x{h_hand} px")

    # Defined 5-finger landmark targets calibrated for test_assets/hand-template.jpg
    # Format per finger: (cuticle_left, cuticle_center, cuticle_right, distal_left, distal_right)
    finger_targets = [
        # 1. Thumb (x ≈ 270, y ≈ 1230)
        {
            "name": "Thumb",
            "c_l": (205.0, 1370.0),
            "c_c": (268.0, 1385.0),
            "c_r": (330.0, 1370.0),
            "d_l": (215.0, 1220.0),
            "d_r": (320.0, 1220.0),
            "c_curve": 0.70,
            "length_scale": 1.25,
        },
        # 2. Index (x ≈ 450, y ≈ 560)
        {
            "name": "Index",
            "c_l": (405.0, 710.0),
            "c_c": (455.0, 725.0),
            "c_r": (505.0, 710.0),
            "d_l": (410.0, 565.0),
            "d_r": (500.0, 565.0),
            "c_curve": 0.65,
            "length_scale": 1.25,
        },
        # 3. Middle (x ≈ 740, y ≈ 430)
        {
            "name": "Middle",
            "c_l": (690.0, 580.0),
            "c_c": (740.0, 595.0),
            "c_r": (790.0, 580.0),
            "d_l": (695.0, 435.0),
            "d_r": (785.0, 435.0),
            "c_curve": 0.65,
            "length_scale": 1.25,
        },
        # 4. Ring (x ≈ 1000, y ≈ 560)
        {
            "name": "Ring",
            "c_l": (950.0, 705.0),
            "c_c": (1000.0, 720.0),
            "c_r": (1050.0, 705.0),
            "d_l": (955.0, 560.0),
            "d_r": (1045.0, 560.0),
            "c_curve": 0.65,
            "length_scale": 1.25,
        },
        # 5. Pinky (x ≈ 1260, y ≈ 890)
        {
            "name": "Pinky",
            "c_l": (1220.0, 1020.0),
            "c_c": (1260.0, 1032.0),
            "c_r": (1300.0, 1020.0),
            "d_l": (1225.0, 895.0),
            "d_r": (1295.0, 895.0),
            "c_curve": 0.70,
            "length_scale": 1.25,
        },
    ]

    print("\n[Step 4] Executing 3-tier TPS warping & C-curve distortion onto hand...")
    composite_hand = hand_img.copy().astype(np.float32)

    total_warp_time = 0.0

    for idx, target in enumerate(finger_targets):
        nail_data = extracted_nails[idx]
        nail_rgba = nail_data["rgba"]
        fname = target["name"]

        t_w0 = time.time()

        # 1. Apply transverse cylindrical barrel distortion
        curved_nail = apply_cylindrical_c_curve(
            nail_rgba,
            c_curve_intensity=target["c_curve"],
            roll_offset=0.0
        )

        # 2. Build 3-tier control grid
        src_h, src_w = curved_nail.shape[:2]
        src_pts, dst_pts = build_3tier_control_grid(
            src_w=src_w,
            src_h=src_h,
            cuticle_left=target["c_l"],
            cuticle_center=target["c_c"],
            cuticle_right=target["c_r"],
            distal_left=target["d_l"],
            distal_right=target["d_r"],
            length_scale=target["length_scale"],
        )

        # 3. SciPy Thin-Plate Spline deformation
        warped_roi, bbox = warp_nail_tps(
            curved_nail,
            src_pts,
            dst_pts,
            output_shape=(h_hand, w_hand),
        )

        t_w1 = time.time()
        warp_ms = (t_w1 - t_w0) * 1000
        total_warp_time += warp_ms

        rx, ry, rw, rh = bbox
        print(f"  Warped {fname:6s} nail: ROI size=({rw}x{rh}), render time={warp_ms:.1f} ms")

        # 4. Alpha composite warped nail onto hand
        if rw > 0 and rh > 0:
            warped_bgr = warped_roi[:, :, :3].astype(np.float32)
            alpha = (warped_roi[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

            hand_roi = composite_hand[ry:ry+rh, rx:rx+rw]
            composite_hand[ry:ry+rh, rx:rx+rw] = (warped_bgr * alpha) + (hand_roi * (1.0 - alpha))

    # Save final warped composite
    final_composite_uint8 = np.clip(composite_hand, 0, 255).astype(np.uint8)
    composite_out_path = os.path.join(output_dir, "02_tps_composited_hand.jpg")
    cv2.imwrite(composite_out_path, final_composite_uint8)

    t_total = time.time() - t0

    print("\n" + "=" * 70)
    print(" VERIFICATION RESULTS & BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total pipeline execution time: {t_total:.2f} s")
    print(f"  Segmentation & PCA time:       {(t_seg_end - t_seg_start)*1000:.1f} ms")
    print(f"  Average TPS warp time / nail:  {total_warp_time / len(finger_targets):.1f} ms")
    print(f"\n  Output Artifacts Generated:")
    print(f"    1. Annotated Tray:     {tray_annotated_path}")
    print(f"    2. 5 Extracted Nails:  {output_dir}/nail_1_thumb.png to nail_5_pinky.png")
    print(f"    3. Warped Composite:   {composite_out_path}")
    print("=" * 70)
    print(" PHASE 1 VERIFICATION COMPLETE - READY FOR USER FEEDBACK")
    print("=" * 70)


if __name__ == "__main__":
    main()
