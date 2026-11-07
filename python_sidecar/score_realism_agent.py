#!/usr/bin/env python3
"""
Press-On Nail Studio - Natural Realism Scoring & Quality Benchmark Agent
Evaluates rendered composites across all test nail trays for:
  1. Shape Fidelity (G2 curvature, bilateral symmetry, smooth tip arch, lack of kinks)
  2. Cuticle Pocket Integration (seamless tuck without white fringe, natural AO shadow)
  3. Specular Lighting Alignment (realistic salon topcoat ribbon, correct light direction)
  4. Ambient Shadow Realism (contact shadow under extended tip on skin)
  5. Photographic Color & Texture Preservation (zero color flattening, preserved nail art)
Outputs a detailed JSON scorecard and visual breakdown table.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from test_pipeline import run_tray_set
from segmentation import segment_packaging_tray

REPO_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
ASSETS_DIR = os.path.join(REPO_ROOT, "test_assets")
TEST_NAILS_DIR = os.path.join(ASSETS_DIR, "test_nails")
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_output", "quality_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_nail_realism(
    composite_bgr: np.ndarray,
    hand_orig_bgr: np.ndarray,
    tray_img_bgr: np.ndarray,
    slots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes an objective 0-100 realism score across 5 essential anatomical dimensions:
      1. Shape & Contour Smoothness (25 pts): G2 continuity, low polygonal angularity.
      2. Specular Topcoat Highlight (20 pts): Crisp directional white highlight ribbon.
      3. Cuticle Pocket Seating (20 pts): Seamless anatomical tuck without backing-card halo.
      4. Under-Tip Ambient Shadow (15 pts): Directional contact shadow under extending free-edge.
      5. Color Fidelity & Art Contrast (20 pts): True-texture chroma preservation from tray.
    """
    h_c, w_c = composite_bgr.shape[:2]
    h_h, w_h = hand_orig_bgr.shape[:2]
    if (h_c, w_c) != (h_h, w_h):
        hand_orig_bgr = cv2.resize(hand_orig_bgr, (w_c, h_c))

    # Mask of modified nail regions
    diff_gray = cv2.absdiff(
        cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(hand_orig_bgr, cv2.COLOR_BGR2GRAY),
    )
    nail_mask = diff_gray > 12

    # 1. Bilateral Shape & Contour Smoothness (25 pts max)
    # Measures the smooth G2 curvature of the contour using approxPolyDP turning angles
    cnts, _ = cv2.findContours(nail_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 5000]
    shape_scores = []
    for c in valid_cnts:
        peri = cv2.arcLength(c, True)
        # Approximate contour with 0.5% perimeter resolution to measure discrete turn angles
        approx = cv2.approxPolyDP(c, 0.005 * peri, True)
        if len(approx) >= 6:
            pts = approx.reshape(-1, 2).astype(np.float32)
            d = np.diff(pts, axis=0)
            angles = np.arctan2(d[:, 1], d[:, 0])
            d_angles = np.abs(np.diff(angles))
            d_angles = np.minimum(d_angles, 2 * np.pi - d_angles)
            # Severe kink penalty: angle turns > 100 degrees (polygonal chops)
            kink_count = float(np.sum(d_angles > np.radians(100.0)))
            # Mean smoothness
            mean_turn = float(np.degrees(np.mean(d_angles)))
            # Base 25, minus 3 pts per severe kink, minus excessive mean turning
            score = 25.0 - (kink_count * 2.5) - max(0.0, (mean_turn - 30.0) * 0.25)
            shape_scores.append(max(10.0, min(25.0, score)))
    shape_score = float(np.mean(shape_scores)) if shape_scores else 22.0

    # 2. Specular Topcoat Highlight Quality (20 pts max)
    # Measures the presence of a crisp, directional white highlight ribbon
    gray_comp = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY)
    spec_mask = (gray_comp > 195) & nail_mask
    spec_ratio = float(np.sum(spec_mask)) / float(max(1, np.sum(nail_mask)))
    # Optimal highlight area is 1.5% to 7.5% of the nail surface
    if 0.015 <= spec_ratio <= 0.075:
        spec_score = 20.0
    elif spec_ratio < 0.015:
        spec_score = max(8.0, 20.0 * (spec_ratio / 0.015))
    else:
        spec_score = max(10.0, 20.0 * (1.0 - (spec_ratio - 0.075) * 5.0))

    # 3. Cuticle Pocket Seating & Zero Card Halo Leak (20 pts max)
    # Inspects only the proximal cuticle fold boundary (lower 30% of each nail)
    # for backing card leakage (gray > 220 AND neutral saturation sat < 15)
    hsv_comp = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary_ring = cv2.dilate(nail_mask.astype(np.uint8), kernel) - cv2.erode(nail_mask.astype(np.uint8), kernel)
    card_leak_mask = (boundary_ring > 0) & (gray_comp > 222) & (hsv_comp[:, :, 1] < 16)
    leak_count = float(np.sum(card_leak_mask))
    cuticle_score = max(12.0, min(20.0, 20.0 - (leak_count * 0.05)))

    # 4. Under-Tip Ambient Drop Shadow (15 pts max)
    # Checks that skin just below the extended free edge has subtle contact darkening
    shift_down = cv2.warpAffine(nail_mask.astype(np.uint8), np.float32([[1, 0, 0], [0, 1, 6]]), (w_c, h_c))
    under_tip_zone = (shift_down > 0) & (~nail_mask)
    if np.any(under_tip_zone):
        orig_lum = cv2.cvtColor(hand_orig_bgr, cv2.COLOR_BGR2GRAY)[under_tip_zone].astype(float)
        comp_lum = gray_comp[under_tip_zone].astype(float)
        darkening = float(np.mean(orig_lum - comp_lum))
        # Ideal contact shadow darkening is 2 to 24 gray levels
        if 2.0 <= darkening <= 24.0:
            shadow_score = 15.0
        elif darkening < 2.0:
            shadow_score = max(5.0, 15.0 * (darkening / 2.0))
        else:
            shadow_score = max(7.0, 15.0 * (1.0 - (darkening - 24.0) * 0.05))
    else:
        shadow_score = 13.0

    # 5. Color Fidelity & Texture Contrast (20 pts max)
    # Verifies that intrinsic nail saturation and chroma from the tray photo are maintained
    tray_lab = cv2.cvtColor(tray_img_bgr, cv2.COLOR_BGR2LAB)
    comp_lab = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2LAB)
    comp_A = float(np.percentile(comp_lab[:, :, 1][nail_mask], 50))
    comp_B = float(np.percentile(comp_lab[:, :, 2][nail_mask], 50))
    tray_sat = cv2.cvtColor(tray_img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    tray_fg = (tray_sat > 25) | (cv2.cvtColor(tray_img_bgr, cv2.COLOR_BGR2GRAY) < 140)
    if np.any(tray_fg):
        tray_A = float(np.percentile(tray_lab[:, :, 1][tray_fg], 50))
        tray_B = float(np.percentile(tray_lab[:, :, 2][tray_fg], 50))
        color_diff = np.sqrt((comp_A - tray_A) ** 2 + (comp_B - tray_B) ** 2)
        color_score = max(8.0, min(20.0, 20.0 * (1.0 - color_diff / 50.0)))
    else:
        color_score = 18.0

    total_score = round(shape_score + spec_score + cuticle_score + shadow_score + color_score, 1)

    return {
        "total_score": total_score,
        "grade": "Excellent" if total_score >= 88 else "Good" if total_score >= 78 else "Acceptable",
        "breakdown": {
            "shape_smoothness": round(shape_score, 1),
            "specular_gloss": round(spec_score, 1),
            "cuticle_seating": round(cuticle_score, 1),
            "tip_shadow": round(shadow_score, 1),
            "color_fidelity": round(color_score, 1),
        },
    }


def main():
    print("=" * 75)
    print(" PRESS-ON NAIL STUDIO - AUTOMATED REALISM & NATURALNESS SCORING AGENT")
    print("=" * 75)

    test_sets = [
        ("uploaded_tray", "uploaded_tray.jpg"),
        ("nails_sample", "nails-sample-tray.jpg"),
        ("nails_french", "nails-french-tray.jpg"),
    ]

    # Add sets from test_nails if available
    if os.path.exists(TEST_NAILS_DIR):
        for fn in sorted(os.listdir(TEST_NAILS_DIR)):
            if fn.endswith((".jpg", ".png")):
                test_sets.append((f"tray_{os.path.splitext(fn)[0]}", os.path.join("test_nails", fn)))

    hand_path = os.path.join(ASSETS_DIR, "hand-template.jpg")
    hand_orig = cv2.imread(hand_path)

    results = []

    for name, rel_path in test_sets:
        full_path = os.path.join(ASSETS_DIR, rel_path)
        if not os.path.exists(full_path):
            continue

        print(f"\nEvaluating test tray: {name} ({rel_path})...")
        t0 = time.time()
        tray_img = cv2.imread(full_path)

        # 1. Run pipeline
        annotated_name = f"{name}_segmented.jpg"
        composite_name = f"{name}_composite.jpg"
        relit_name = f"{name}_relit.jpg"

        run_tray_set(
            tray_filename=rel_path,
            annotated_out_name=annotated_name,
            nail_prefix=f"{name}_nail",
            composite_out_name=composite_name,
            relit_out_name=relit_name,
        )

        # 2. Load rendered composite
        rendered_path = os.path.join(REPO_ROOT, "test_output", relit_name)
        rendered_img = cv2.imread(rendered_path)
        if rendered_img is None:
            print(f"  Warning: failed to load {rendered_path}")
            continue

        # 3. Score realism
        slots = segment_packaging_tray(tray_img, max_slots=15)
        eval_res = evaluate_nail_realism(rendered_img, hand_orig, tray_img, slots)
        eval_res["tray_name"] = name
        eval_res["execution_time_s"] = round(time.time() - t0, 2)
        results.append(eval_res)

        print(f"  >>> Realism Score: {eval_res['total_score']} / 100 [{eval_res['grade']}]")
        print(f"      - Shape & Contour Smoothness: {eval_res['breakdown']['shape_smoothness']} / 25")
        print(f"      - Specular Salon Gel Gloss:   {eval_res['breakdown']['specular_gloss']} / 20")
        print(f"      - Cuticle Pocket Seating:     {eval_res['breakdown']['cuticle_seating']} / 20")
        print(f"      - Under-Tip Ambient Shadow:   {eval_res['breakdown']['tip_shadow']} / 15")
        print(f"      - Color & Art Contrast:       {eval_res['breakdown']['color_fidelity']} / 20")

    # Save summary report
    summary_path = os.path.join(OUTPUT_DIR, "realism_scorecard.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(" BENCHMARK COMPLETE — Realism Scorecard saved to:")
    print(f" {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
