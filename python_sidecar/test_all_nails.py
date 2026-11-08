#!/usr/bin/env python3
"""
Press-On Nail Studio - True-Texture Batch Benchmark & Verification Harness
(docs/true-texture-composition-plan.md, Step 5)

Iterates over every tray set in test_assets/test_nails/ plus test_assets/nails-french-tray.jpg
and writes three diagnostic sheets per set into test_output/<out_subdir>/:

  <set>_segmented.jpg        Card homography quad + all slot contours + shape labels
  <set>_extracted_strip.png  The 5 extracted RGBA nails: top row = raw extraction,
                             bottom row = after surgical glare-tick cleanup
  <set>_composited_hand.jpg  Final studio-relit hand composite

plus a machine-readable summary.json (slot counts, shapes, timings, errors).

Usage:
  python python_sidecar/test_all_nails.py                       # all sets -> test_output/batch
  python python_sidecar/test_all_nails.py --out seg_agent       # -> test_output/seg_agent
  python python_sidecar/test_all_nails.py --sets 29 33 48       # subset (matches file stem)
  python python_sidecar/test_all_nails.py --no-composite        # segmentation + strips only
  python python_sidecar/test_all_nails.py --dump-raw test_output/fixtures   # also save raw RGBA nails
  python python_sidecar/test_all_nails.py --fixtures test_output/fixtures   # cleaner+compositor only

The harness is API-tolerant so it runs against both the legacy and the new pipeline:
  - If slots carry a "homography" key (3x3, original -> rectified plane) and
    extract_nail_rgba() accepts a `homography` kwarg, it is passed through.
  - If slots carry a "card_quad" key (4x2 in original image coords), it is drawn.
  - Uses compositor.clean_tray_glare_preserving_texture() when present, otherwise
    falls back to the legacy compositor.recover_clean_nail_albedo().
"""

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

current_dir: str = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import compositor  # noqa: E402
from segmentation import extract_nail_rgba, segment_packaging_tray  # noqa: E402
from warping import (  # noqa: E402
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
)

REPO_ROOT: str = os.path.abspath(os.path.join(current_dir, ".."))
ASSETS_DIR: str = os.path.join(REPO_ROOT, "test_assets")
TEST_NAILS_DIR: str = os.path.join(ASSETS_DIR, "test_nails")
HAND_PATH: str = os.path.join(ASSETS_DIR, "hand-template.jpg")

FINGER_NAMES: List[str] = ["thumb", "index", "middle", "ring", "pinky"]

SLOT_COLORS: List[Tuple[int, int, int]] = [
    (255, 105, 180),
    (0, 255, 255),
    (50, 205, 50),
    (30, 144, 255),
    (255, 140, 0),
    (186, 85, 211),
    (0, 250, 154),
    (255, 20, 147),
    (255, 215, 0),
    (0, 191, 255),
]

# Calibrated landmarks for test_assets/hand-template.jpg (identical to test_pipeline.py)
FINGER_TARGETS: List[Dict[str, Any]] = [
    {
        "name": "Thumb", "slot_idx": 0,
        "c_l": (200.0, 1425.0), "c_c": (236.0, 1442.0), "c_r": (275.0, 1416.0),
        "d_l": (190.0, 1270.0), "d_r": (268.0, 1276.0),
        "c_curve": 0.44, "roll_offset": -0.35, "length_scale": 1.00, "tip_taper": 1.00,
    },
    {
        "name": "Index", "slot_idx": 2,
        "c_l": (412.0, 714.0), "c_c": (466.0, 738.0), "c_r": (519.0, 712.0),
        "d_l": (399.0, 562.0), "d_r": (518.0, 560.0),
        "c_curve": 0.28, "roll_offset": 0.0, "length_scale": 1.00, "tip_taper": 1.00,
    },
    {
        "name": "Middle", "slot_idx": 1,
        "c_l": (693.0, 566.0), "c_c": (747.0, 592.0), "c_r": (799.0, 566.0),
        "d_l": (681.0, 410.0), "d_r": (798.0, 410.0),
        "c_curve": 0.28, "roll_offset": 0.0, "length_scale": 1.00, "tip_taper": 1.00,
    },
    {
        "name": "Ring", "slot_idx": 3,
        "c_l": (948.0, 686.0), "c_c": (1000.0, 710.0), "c_r": (1051.0, 686.0),
        "d_l": (936.0, 540.0), "d_r": (1048.0, 540.0),
        "c_curve": 0.28, "roll_offset": 0.0, "length_scale": 1.00, "tip_taper": 1.00,
    },
    {
        "name": "Pinky", "slot_idx": 4,
        "c_l": (1222.0, 990.0), "c_c": (1262.0, 1012.0), "c_r": (1301.0, 988.0),
        "d_l": (1214.0, 866.0), "d_r": (1301.0, 864.0),
        "c_curve": 0.28, "roll_offset": 0.0, "length_scale": 1.00, "tip_taper": 1.00,
    },
]


def discover_sets() -> List[Tuple[str, str]]:
    """Returns (set_name, absolute_path) for every tray image under test, sorted by name."""
    sets: List[Tuple[str, str]] = []
    if os.path.isdir(TEST_NAILS_DIR):
        for fname in sorted(os.listdir(TEST_NAILS_DIR)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                sets.append((stem, os.path.join(TEST_NAILS_DIR, fname)))
    french_path: str = os.path.join(ASSETS_DIR, "nails-french-tray.jpg")
    if os.path.exists(french_path):
        sets.append(("nails-french-tray", french_path))
    return sets


def get_cleaner() -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """Returns the glare cleanup function (new true-texture path preferred)."""
    fn: Optional[Callable[[np.ndarray], np.ndarray]] = getattr(
        compositor, "clean_tray_glare_preserving_texture", None
    )
    if fn is not None:
        return "clean_tray_glare_preserving_texture", fn
    return "recover_clean_nail_albedo", compositor.recover_clean_nail_albedo


def extract_supports_homography() -> bool:
    """True if segmentation.extract_nail_rgba() accepts a `homography` keyword argument."""
    try:
        return "homography" in inspect.signature(extract_nail_rgba).parameters
    except (TypeError, ValueError):
        return False


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscales so the longest side is <= max_side (never upscales)."""
    h, w = img.shape[:2]
    scale: float = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def draw_label(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Tuple[int, int, int]) -> None:
    """Draws outlined text for legibility on any background."""
    thick: int = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_segmentation_sheet(tray_img: np.ndarray, slots: List[Dict[str, Any]], set_name: str) -> np.ndarray:
    """Draws the card quad (if any), every slot contour, bbox and shape label."""
    sheet: np.ndarray = tray_img.copy()
    h, w = sheet.shape[:2]
    line_w: int = max(2, int(round(max(h, w) / 600.0)))
    font_scale: float = max(0.5, max(h, w) / 2200.0)

    card_quad: Optional[Any] = slots[0].get("card_quad") if slots else None
    if card_quad is not None:
        quad: np.ndarray = np.array(card_quad, dtype=np.float32).reshape(-1, 2)
        cv2.polylines(sheet, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), line_w * 2, cv2.LINE_AA)
        for i, (qx, qy) in enumerate(quad):
            cv2.circle(sheet, (int(qx), int(qy)), line_w * 4, (0, 0, 255), -1, cv2.LINE_AA)
            draw_label(sheet, ["TL", "TR", "BR", "BL"][i % 4], (int(qx) + 8, int(qy) - 8), font_scale, (0, 0, 255))

    for s in slots:
        sid: int = int(s["slot_id"])
        color: Tuple[int, int, int] = SLOT_COLORS[(sid - 1) % len(SLOT_COLORS)]
        cnt: np.ndarray = np.array(s["contour"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(sheet, [cnt], -1, color, line_w, lineType=cv2.LINE_AA)
        x, y, bw, bh = [int(v) for v in s["bbox"]]
        shape: str = str(s.get("shape_info", {}).get("shape", "?"))
        inv: bool = bool(s.get("shape_info", {}).get("metrics", {}).get("is_inverted", False))
        label: str = f"#{sid} {shape}{' inv' if inv else ''}"
        draw_label(sheet, label, (x, max(int(30 * font_scale), y - 10)), font_scale, color)

    header: str = f"{set_name}: {len(slots)} slots"
    if card_quad is not None:
        header += " | card homography applied"
    draw_label(sheet, header, (20, int(50 * font_scale) + 10), font_scale * 1.4, (255, 255, 255))
    return resize_max(sheet, 1800)


def rgba_on_checker(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    """Composites an RGBA nail over a light/dark grey checkerboard for inspection."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker: np.ndarray = np.where(((yy // cell) + (xx // cell)) % 2 == 0, 96, 64).astype(np.float32)
    bg: np.ndarray = np.repeat(checker[:, :, None], 3, axis=2)
    alpha: np.ndarray = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out: np.ndarray = rgba[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_extracted_strip(
    raw_nails: List[np.ndarray],
    clean_nails: List[np.ndarray],
    set_name: str,
    cleaner_name: str,
    cell_h: int = 420,
) -> np.ndarray:
    """Top row: raw extractions. Bottom row: after glare cleanup. Each nail scaled to cell_h."""
    pad: int = 16
    label_h: int = 34

    def fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale: float = float(cell_h) / float(max(1, h))
        return cv2.resize(img, (max(1, int(round(w * scale))), cell_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    raw_fit: List[np.ndarray] = [rgba_on_checker(fit(n)) for n in raw_nails]
    clean_fit: List[np.ndarray] = [rgba_on_checker(fit(n)) for n in clean_nails]
    col_ws: List[int] = [max(a.shape[1], b.shape[1]) for a, b in zip(raw_fit, clean_fit)]
    total_w: int = pad + sum(cw + pad for cw in col_ws)
    total_h: int = label_h + (cell_h + label_h + pad) * 2
    canvas: np.ndarray = np.full((total_h, max(total_w, 900), 3), 30, dtype=np.uint8)

    draw_label(canvas, f"{set_name}  |  top: raw extraction   bottom: {cleaner_name}", (pad, 24), 0.6, (255, 255, 255))
    x: int = pad
    for i, (a, b) in enumerate(zip(raw_fit, clean_fit)):
        y_top: int = label_h + label_h
        canvas[y_top:y_top + cell_h, x:x + a.shape[1]] = a
        draw_label(canvas, f"{FINGER_NAMES[i] if i < 5 else i} raw", (x, y_top - 8), 0.5, (200, 200, 200))
        y_bot: int = y_top + cell_h + pad + label_h
        canvas[y_bot:y_bot + cell_h, x:x + b.shape[1]] = b
        draw_label(canvas, "clean", (x, y_bot - 8), 0.5, (200, 255, 200))
        x += col_ws[i] + pad
    return canvas


def composite_on_hand(
    hand_img: np.ndarray,
    clean_nails: List[np.ndarray],
) -> np.ndarray:
    """Runs C-curve + 3-tier TPS + PBR compositing for the 5 fingers; returns BGR uint8."""
    h_hand, w_hand = hand_img.shape[:2]

    hand_bed_widths: List[float] = []
    tray_nail_widths: List[float] = []
    min_scale_to_cover_bed: float = 0.0
    for t in FINGER_TARGETS[1:]:
        cw: float = float(np.linalg.norm(np.array(t["c_r"]) - np.array(t["c_l"])))
        dw: float = float(np.linalg.norm(np.array(t["d_r"]) - np.array(t["d_l"])))
        distal_mid: np.ndarray = 0.5 * (np.array(t["d_l"]) + np.array(t["d_r"]))
        bed_len: float = float(np.linalg.norm(distal_mid - np.array(t["c_c"])))
        hand_bed_widths.append(max(cw, dw))
        nd: np.ndarray = clean_nails[t["slot_idx"]]
        xs: np.ndarray = np.where(nd[:, :, 3] > 25)[1]
        tray_nail_widths.append(float(np.max(xs) - np.min(xs) + 1) if len(xs) > 0 else float(nd.shape[1]))
        min_scale_to_cover_bed = max(min_scale_to_cover_bed, (bed_len * 1.24) / max(1.0, float(nd.shape[0])))

    width_based_scale: float = (sum(hand_bed_widths) / max(1.0, sum(tray_nail_widths))) * 1.05
    tray_to_hand_scale: float = max(width_based_scale, min_scale_to_cover_bed)

    relit_hand: np.ndarray = hand_img.copy().astype(np.float32)
    for target in FINGER_TARGETS:
        clean_rgba: np.ndarray = clean_nails[target["slot_idx"]]
        curved: np.ndarray = apply_cylindrical_c_curve(
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

    return np.clip(relit_hand, 0, 255).astype(np.uint8)


def run_set_from_fixtures(
    set_name: str,
    fixtures_dir: str,
    out_dir: str,
    hand_img: np.ndarray,
    do_composite: bool,
) -> Dict[str, Any]:
    """
    Skips segmentation/extraction: loads frozen raw RGBA nails <set>_<1..5>.png from fixtures_dir,
    runs the glare cleaner + compositor, and writes the strip and composite sheets. Never raises.
    """
    record: Dict[str, Any] = {"set": set_name, "path": fixtures_dir, "ok": False, "fixtures": True}
    cleaner_name, cleaner = get_cleaner()
    try:
        raw_nails: List[np.ndarray] = []
        for i in range(1, 6):
            p: str = os.path.join(fixtures_dir, f"{set_name}_{i}.png")
            img: Optional[np.ndarray] = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                raise FileNotFoundError(f"missing/invalid RGBA fixture: {p}")
            raw_nails.append(img)
        t1: float = time.time()
        clean_nails: List[np.ndarray] = [cleaner(n) for n in raw_nails]
        record["extract_clean_ms"] = round((time.time() - t1) * 1000.0, 1)
        cv2.imwrite(
            os.path.join(out_dir, f"{set_name}_extracted_strip.png"),
            render_extracted_strip(raw_nails, clean_nails, set_name, cleaner_name),
        )
        if do_composite:
            t2: float = time.time()
            hand_out: np.ndarray = composite_on_hand(hand_img, clean_nails)
            record["composite_ms"] = round((time.time() - t2) * 1000.0, 1)
            cv2.imwrite(os.path.join(out_dir, f"{set_name}_composited_hand.jpg"), hand_out, [cv2.IMWRITE_JPEG_QUALITY, 94])
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - harness must keep going across sets
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def run_set(
    set_name: str,
    tray_path: str,
    out_dir: str,
    hand_img: np.ndarray,
    do_composite: bool,
    dump_raw_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs one tray set end-to-end and writes its diagnostic sheets. Never raises."""
    record: Dict[str, Any] = {"set": set_name, "path": tray_path, "ok": False}
    cleaner_name, cleaner = get_cleaner()
    pass_h: bool = extract_supports_homography()
    try:
        tray_img: Optional[np.ndarray] = cv2.imread(tray_path, cv2.IMREAD_COLOR)
        if tray_img is None:
            raise FileNotFoundError(tray_path)
        record["image_size"] = [int(tray_img.shape[1]), int(tray_img.shape[0])]

        t0: float = time.time()
        slots: List[Dict[str, Any]] = segment_packaging_tray(tray_img, max_slots=15)
        record["segment_ms"] = round((time.time() - t0) * 1000.0, 1)
        record["slot_count"] = len(slots)
        record["shapes"] = [str(s.get("shape_info", {}).get("shape", "?")) for s in slots]
        record["homography_applied"] = bool(slots and slots[0].get("card_quad") is not None)

        cv2.imwrite(os.path.join(out_dir, f"{set_name}_segmented.jpg"), render_segmentation_sheet(tray_img, slots, set_name))

        if len(slots) < 5:
            raise RuntimeError(f"only {len(slots)} slots detected; need >= 5 for extraction")

        raw_nails: List[np.ndarray] = []
        clean_nails: List[np.ndarray] = []
        t1: float = time.time()
        for s in slots[:5]:
            kwargs: Dict[str, Any] = {
                "feather_px": 1.5,
                "deskew": True,
                "auto_orient_cuticle_down": True,
                "force_inverted": s.get("shape_info", {}).get("metrics", {}).get("is_inverted", None),
                "exact_mask": True,
            }
            if pass_h and s.get("homography") is not None:
                kwargs["homography"] = np.array(s["homography"], dtype=np.float64)
            rgba: np.ndarray = extract_nail_rgba(tray_img, np.array(s["contour"], dtype=np.int32), **kwargs)
            raw_nails.append(rgba)
            clean_nails.append(cleaner(rgba))
        record["extract_clean_ms"] = round((time.time() - t1) * 1000.0, 1)
        record["nail_sizes"] = [[int(n.shape[1]), int(n.shape[0])] for n in raw_nails]
        if dump_raw_dir is not None:
            os.makedirs(dump_raw_dir, exist_ok=True)
            for i, n in enumerate(raw_nails, start=1):
                cv2.imwrite(os.path.join(dump_raw_dir, f"{set_name}_{i}.png"), n)

        cv2.imwrite(
            os.path.join(out_dir, f"{set_name}_extracted_strip.png"),
            render_extracted_strip(raw_nails, clean_nails, set_name, cleaner_name),
        )

        if do_composite:
            t2: float = time.time()
            hand_out: np.ndarray = composite_on_hand(hand_img, clean_nails)
            record["composite_ms"] = round((time.time() - t2) * 1000.0, 1)
            cv2.imwrite(os.path.join(out_dir, f"{set_name}_composited_hand.jpg"), hand_out, [cv2.IMWRITE_JPEG_QUALITY, 94])

        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - harness must keep going across sets
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="True-texture batch benchmark harness")
    parser.add_argument("--out", default="batch", help="Output subdirectory under test_output/")
    parser.add_argument("--sets", nargs="*", default=None, help="Subset of set names (file stems) to run")
    parser.add_argument("--no-composite", action="store_true", help="Skip hand compositing")
    parser.add_argument(
        "--dump-raw",
        default=None,
        help="Directory to write raw extracted RGBA nails as <set>_<1..5>.png (fixtures)",
    )
    parser.add_argument(
        "--fixtures",
        default=None,
        help="Directory of frozen raw RGBA nails <set>_<1..5>.png; skips segmentation/extraction",
    )
    args = parser.parse_args()

    out_dir: str = os.path.join(REPO_ROOT, "test_output", args.out)
    os.makedirs(out_dir, exist_ok=True)

    hand_img: Optional[np.ndarray] = cv2.imread(HAND_PATH, cv2.IMREAD_COLOR)
    if hand_img is None:
        print(f"Hand template not found: {HAND_PATH}")
        return 2

    all_sets: List[Tuple[str, str]] = discover_sets()
    if args.sets:
        wanted = set(args.sets)
        all_sets = [s for s in all_sets if s[0] in wanted]

    cleaner_name, _ = get_cleaner()
    print(f"Cleaner: {cleaner_name} | extract homography kwarg: {extract_supports_homography()}")
    print(f"Output:  {out_dir}\n")

    records: List[Dict[str, Any]] = []
    dump_dir: Optional[str] = os.path.abspath(args.dump_raw) if args.dump_raw else None
    fixtures_dir: Optional[str] = os.path.abspath(args.fixtures) if args.fixtures else None
    if fixtures_dir is not None:
        stems: List[str] = sorted({f.rsplit("_", 1)[0] for f in os.listdir(fixtures_dir) if f.endswith("_1.png")})
        all_sets = [(s, fixtures_dir) for s in stems if not args.sets or s in set(args.sets)]

    for set_name, path in all_sets:
        rec: Dict[str, Any]
        if fixtures_dir is not None:
            rec = run_set_from_fixtures(set_name, fixtures_dir, out_dir, hand_img, not args.no_composite)
        else:
            rec = run_set(set_name, path, out_dir, hand_img, not args.no_composite, dump_raw_dir=dump_dir)
        records.append(rec)
        status: str = "OK  " if rec["ok"] else "FAIL"
        print(
            f"[{status}] {set_name:18s} slots={rec.get('slot_count', '-'):>3} "
            f"H={'y' if rec.get('homography_applied') else 'n'} "
            f"seg={rec.get('segment_ms', '-')}ms ext={rec.get('extract_clean_ms', '-')}ms "
            f"comp={rec.get('composite_ms', '-')}ms shapes={rec.get('shapes', [])[:5]}"
        )
        if not rec["ok"]:
            print(f"       -> {rec.get('error')}")

    summary_path: str = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    n_ok: int = sum(1 for r in records if r["ok"])
    print(f"\n{n_ok}/{len(records)} sets OK. Summary: {summary_path}")
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
