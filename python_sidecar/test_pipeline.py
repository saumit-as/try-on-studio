#!/usr/bin/env python3
"""
Press-On Nail Studio - Phase 1 Verification Test Pipeline
Executes standalone end-to-end testing of:
  1. Tray contour segmentation & PCA deskewing
  2. Automatic 180-degree inversion normalization (cuticle down, tapered tip up)
  3. Periodic B-spline contour smoothing (zero polygonal corners)
  4. Geometric nail shape classification
  5. Transverse cylindrical C-curve barrel distortion
  6. 3-tier Thin-Plate Spline (TPS) transformation onto 5-finger hand template
  7. Anatomical U-curve Cuticle & Sidewall Skin-Fold Tuck Mask (places cuticle skin on top of nail base)
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

from segmentation import segment_packaging_tray, extract_nail_rgba
from warping import (
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
    build_cuticle_pocket_mask,
)


def blend_nail_naturally(
    composite_hand: np.ndarray,
    hand_orig: np.ndarray,
    warped_roi: np.ndarray,
    bbox: tuple,
    target: dict,
) -> None:
    """
    Blends the warped press-on nail onto the target finger ROI with:
      1. Anatomical Cuticle & Sidewall Skin-Fold Tuck Mask: cuts any nail overflow below
         the U-shaped cuticle fold and lateral nail grooves so the finger's natural cuticle
         skin sits cleanly ON TOP of the nail base.
      2. Cuticle Pocket Ambient Occlusion (AO): subtle 2px contact shadow where the nail
         tucks under the proximal skin lip.
      3. Natural 3D cylindrical nail-bed shading & smooth sub-pixel anti-aliased edges.
    """
    rx, ry, rw, rh = bbox
    if rw <= 0 or rh <= 0:
        return

    warped_bgr = warped_roi[:, :, :3].astype(np.float32)
    raw_alpha = warped_roi[:, :, 3].astype(np.float32) / 255.0

    # 1. Build the U-curve Cuticle & Sidewall Skin-Fold Tuck Mask
    pocket_allow, cuticle_ao = build_cuticle_pocket_mask(
        roi_bbox=bbox,
        cuticle_left=target["c_l"],
        cuticle_center=target["c_c"],
        cuticle_right=target["c_r"],
        distal_left=target["d_l"],
        distal_right=target["d_r"],
        feather_sigma=1.6,
    )

    # Multiply the nail's alpha by pocket_allow so the finger's cuticle & sidewall skin
    # remain 100% on top of the nail wherever the nail base would otherwise overflow!
    tucked_alpha = raw_alpha * pocket_allow
    alpha_smooth = cv2.GaussianBlur(tucked_alpha, (3, 3), 0.8)[:, :, None]

    hand_patch = hand_orig[ry:ry + rh, rx:rx + rw].astype(np.float32)
    hand_curr = composite_hand[ry:ry + rh, rx:rx + rw]

    # 2. Compute local 3D shading map from the underlying finger/nail-bed luminance
    hand_gray = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hand_shading_blur = cv2.GaussianBlur(hand_gray, (21, 21), 7.0)

    mask_bool = tucked_alpha > 0.15
    if np.any(mask_bool):
        mean_lum = float(np.mean(hand_shading_blur[mask_bool]))
    else:
        mean_lum = 170.0

    shading_ratio = np.clip(hand_shading_blur / max(1.0, mean_lum), 0.78, 1.12)[:, :, None]

    # Combine 3D cylinder shading + Cuticle Pocket AO shadow
    shaded_nail_bgr = np.clip(
        warped_bgr * (0.45 + 0.55 * shading_ratio) * cuticle_ao[:, :, None],
        0.0,
        255.0,
    )

    # 3. Under-Tip Fingertip Cast Shadow (Pass 3 preview):
    # Shift the nail alpha slightly downward (+2.5px in y, +1.0px in x) and blur it,
    # but apply the shadow ONLY onto finger skin pixels (where skin saturation > 25),
    # never onto the grey studio backdrop!
    M_sh = np.float32([[1.0, 0.0, 1.0], [0.0, 1.0, 3.0]])
    shifted_alpha = cv2.warpAffine(tucked_alpha, M_sh, (rw, rh), flags=cv2.INTER_LINEAR)
    shadow_blur = cv2.GaussianBlur(shifted_alpha, (7, 7), 2.2)
    # Detect skin pixels vs neutral grey background using HSV saturation
    patch_hsv = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
    skin_weight = np.clip((patch_hsv[:, :, 1].astype(np.float32) - 18.0) / 25.0, 0.0, 1.0)
    under_shadow_mult = 1.0 - 0.18 * shadow_blur * (1.0 - tucked_alpha) * skin_weight
    hand_with_shadow = hand_curr * under_shadow_mult[:, :, None]

    # 4. Composite onto hand
    composite_hand[ry:ry + rh, rx:rx + rw] = (
        shaded_nail_bgr * alpha_smooth + hand_with_shadow * (1.0 - alpha_smooth)
    )


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

    print("[Step 2] Executing B-spline smooth segmentation, 180° orientation check & PCA deskewing...")
    t_seg_start = time.time()
    slots = segment_packaging_tray(tray_img, max_slots=15)
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
        inv = s["shape_info"]["metrics"].get("is_inverted", False)
        cnt = np.array(s["contour"], dtype=np.int32)
        color = slot_colors[(sid - 1) % len(slot_colors)]

        print(
            f"    Slot {sid:2d}: Pos=({x:4d}, {y:4d}), Size=({bw:3d}x{bh:3d}), "
            f"Angle={angle:+5.2f}°, Inverted={str(inv):5s}, Shape={shape:8s} (conf={conf})"
        )

        # Draw contour & bounding box on debug image
        cv2.drawContours(annotated_tray, [cnt], -1, color, 3, lineType=cv2.LINE_AA)
        cv2.rectangle(annotated_tray, (x, y), (x + bw, y + bh), color, 2)
        label = f"#{sid} {shape} ({angle:+.1f}d)"
        cv2.putText(annotated_tray, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(annotated_tray, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        # Extract RGBA for the first 5 nails (Row 1: Thumb, Index, Middle, Ring, Pinky)
        if sid <= 5:
            nail_rgba = extract_nail_rgba(
                tray_img,
                cnt,
                feather_px=1.5,
                deskew=True,
                auto_orient_cuticle_down=True,
            )

            extracted_nails.append({
                "slot_id": sid,
                "rgba": nail_rgba,
                "shape": shape,
            })

            # Save individual upright, B-spline smoothed cutout
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

    # Precisely calibrated 5-finger anatomical nail-bed landmarks for test_assets/hand-template.jpg
    # c_l, c_c, c_r trace the exact proximal cuticle skin fold (eponychium)
    # d_l, d_r trace the exact lateral nail grooves at the free edge
    # Note on Tray Slot Mapping (see user diagram):
    # In the packaging tray, the 5 nails are ordered strictly by decreasing height/width:
    #   Slot 1 (264px) -> Thumb
    #   Slot 2 (237px) -> Middle finger (longest/widest of the 4 fingers!)
    #   Slot 3 (235px) -> Index finger
    #   Slot 4 (221px) -> Ring finger
    #   Slot 5 (204px) -> Pinky finger (shortest/narrowest)
    finger_targets = [
        # 1. Thumb (Slot 1 — largest, lateral 3/4 view rotated in 3D via roll_offset=-0.35)
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
        # 2. Index (Slot 3 — 3rd largest nail)
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
        # 3. Middle (Slot 2 — 2nd largest nail, for the longest/widest middle finger)
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
        # 4. Ring (Slot 4 — 4th largest nail)
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
        # 5. Pinky (Slot 5 — smallest nail)
        {
            "name": "Pinky",
            "slot_idx": 4,
            "c_l": (1222.0, 990.0),
            "c_c": (1262.0, 1012.0),
            "c_r": (1301.0, 988.0),
            "d_l": (1214.0, 866.0),
            "d_r": (1301.0, 864.0),
            "c_curve": 0.28,
            "roll_offset": 0.0,
            "length_scale": 1.00,
            "tip_taper": 1.00,
        },
    ]

    # Compute the unified Tray-to-Hand Scale Factor (`tray_to_hand_scale`) from the 4 frontal
    # fingers (Index, Middle, Ring, Pinky) so that every nail's rendered length on the hand is
    # strictly proportional to its extracted pixel height `src_h` in the tray image!
    hand_bed_widths = []
    tray_nail_widths = []
    for t in finger_targets[1:]:
        cw = np.linalg.norm(np.array(t["c_r"]) - np.array(t["c_l"]))
        dw = np.linalg.norm(np.array(t["d_r"]) - np.array(t["d_l"]))
        hand_bed_widths.append(max(cw, dw))
        nd = extracted_nails[t["slot_idx"]]["rgba"]
        tray_nail_widths.append(float(nd.shape[1]))
    # Unified scale factor mapping tray pixels -> hand pixels (calibrated to 1.15x so every nail
    # clears the bare nail bed AND strictly preserves the 264 : 237 : 235 : 221 : 204 height ratio!)
    tray_to_hand_scale = (sum(hand_bed_widths) / max(1.0, sum(tray_nail_widths))) * 1.15
    print(f"  Unified Tray-to-Hand Scale Factor: {tray_to_hand_scale:.3f}x")

    print("\n[Step 4] Executing 3-tier TPS warping, C-curve & Cuticle Skin-Fold Tuck...")
    composite_hand = hand_img.copy().astype(np.float32)

    total_warp_time = 0.0

    for target in finger_targets:
        nail_data = extracted_nails[target["slot_idx"]]
        nail_rgba = nail_data["rgba"]
        fname = target["name"]
        sid = nail_data["slot_id"]

        t_w0 = time.time()

        # 1. Apply transverse cylindrical barrel distortion
        curved_nail = apply_cylindrical_c_curve(
            nail_rgba,
            c_curve_intensity=target["c_curve"],
            roll_offset=target.get("roll_offset", 0.0),
        )

        # 2. Build 3-tier control grid preserving exact tray height graduation
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
            tip_taper=target.get("tip_taper", 1.00),
            roll_offset=target.get("roll_offset", 0.0),
            tray_to_hand_scale=tray_to_hand_scale,
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
        worn_len = float(np.linalg.norm(dst_pts[7] - dst_pts[1]))
        print(
            f"  Warped {fname:6s} (Slot #{sid}, tray_h={src_h}px -> worn_len={worn_len:.1f}px): "
            f"ROI=({rw}x{rh}), time={warp_ms:.1f} ms"
        )

        # 4. Blend warped nail naturally onto the hand with Cuticle Skin-Fold Tuck
        blend_nail_naturally(composite_hand, hand_img, warped_roi, bbox, target)

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
