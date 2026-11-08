"""
Edge Quality Metrics Harness (hybrid-implementation-v2.md, Section 8).

Measures, for every extracted nail on the benchmark trays:

  1. Edge color purity:
       Alpha-weighted mean CIEDE2000 between each feather-band pixel (0 < alpha < 220)
       and the color of its nearest solid-core pixel (alpha >= 220).
       Card-contaminated (white) edges produce a large Delta E. Gate: <= 5 (target <= 3).

  2. Halo excess on 3 skin tones (light, medium, dark):
       The nail is alpha-composited over a flat skin color. In a band of +/- 2 px around the
       alpha >= 0.5 silhouette, L* of the actual composite is compared against an "ideal"
       composite in which every edge pixel carries its nearest-core color.
       Positive values mean the edge is brighter than the nail itself (milky halo).
       Comparing against the nail's own interior keeps pale/white French tips fair.
       Gate: <= 2 L*.

  3. Extraction method reporting:
       Every tray reports which segmentation path ran ("yolo" or "classical").

Usage:
    .venv\\Scripts\\python.exe python_sidecar\\edge_quality_metrics.py --label baseline
    .venv\\Scripts\\python.exe python_sidecar\\edge_quality_metrics.py --label after --compare baseline
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage

current_dir: str = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from segmentation import extract_nail_rgba, segment_packaging_tray  # noqa: E402

REPO_ROOT: str = os.path.abspath(os.path.join(current_dir, ".."))
ASSETS_DIR: str = os.path.join(REPO_ROOT, "test_assets")
TEST_NAILS_DIR: str = os.path.join(ASSETS_DIR, "test_nails")
OUT_DIR: str = os.path.join(REPO_ROOT, "test_output", "edge_quality")

CORE_ALPHA: int = 220

# Flat skin tones in BGR (sRGB): light, medium, dark
SKIN_TONES_BGR: Dict[str, Tuple[float, float, float]] = {
    "light": (167.0, 194.0, 241.0),
    "medium": (66.0, 134.0, 198.0),
    "dark": (38.0, 57.0, 92.0),
}


def discover_sets() -> List[Tuple[str, str]]:
    """Returns (set_name, absolute_path) for every benchmark tray image."""
    sets: List[Tuple[str, str]] = []
    if os.path.isdir(TEST_NAILS_DIR):
        for fname in sorted(os.listdir(TEST_NAILS_DIR)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                sets.append((stem, os.path.join(TEST_NAILS_DIR, fname)))
    for extra in ("nails-sample-tray.jpg", "nails-french-tray.jpg"):
        p: str = os.path.join(ASSETS_DIR, extra)
        if os.path.exists(p):
            sets.append((os.path.splitext(extra)[0], p))
    return sets


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Converts uint8/float BGR (0..255) to CIE L*a*b* (L in 0..100) as float32."""
    f: np.ndarray = np.clip(bgr.astype(np.float32) / 255.0, 0.0, 1.0)
    return cv2.cvtColor(f.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(bgr.shape[:-1] + (3,))


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 color difference between two (..., 3) L*a*b* arrays."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1: np.ndarray = np.sqrt(a1 ** 2 + b1 ** 2)
    C2: np.ndarray = np.sqrt(a2 ** 2 + b2 ** 2)
    C_bar: np.ndarray = 0.5 * (C1 + C2)
    G: np.ndarray = 0.5 * (1.0 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7)))
    a1p: np.ndarray = (1.0 + G) * a1
    a2p: np.ndarray = (1.0 + G) * a2
    C1p: np.ndarray = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p: np.ndarray = np.sqrt(a2p ** 2 + b2 ** 2)
    h1p: np.ndarray = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p: np.ndarray = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp: np.ndarray = L2 - L1
    dCp: np.ndarray = C2p - C1p
    dhp: np.ndarray = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((C1p * C2p) == 0.0, 0.0, dhp)
    dHp: np.ndarray = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar: np.ndarray = 0.5 * (L1 + L2)
    Cp_bar: np.ndarray = 0.5 * (C1p + C2p)
    hp_sum: np.ndarray = h1p + h2p
    hp_bar: np.ndarray = np.where(
        (C1p * C2p) == 0.0,
        hp_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            0.5 * hp_sum,
            np.where(hp_sum < 360.0, 0.5 * (hp_sum + 360.0), 0.5 * (hp_sum - 360.0)),
        ),
    )
    T: np.ndarray = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta: np.ndarray = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    R_C: np.ndarray = 2.0 * np.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25.0 ** 7))
    S_L: np.ndarray = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C: np.ndarray = 1.0 + 0.045 * Cp_bar
    S_H: np.ndarray = 1.0 + 0.015 * Cp_bar * T
    R_T: np.ndarray = -np.sin(np.radians(2.0 * d_theta)) * R_C

    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


def nearest_core_bgr(rgba: np.ndarray) -> Optional[np.ndarray]:
    """Returns an image where every pixel holds the BGR of its nearest solid-core pixel."""
    core: np.ndarray = rgba[:, :, 3] >= CORE_ALPHA
    if not np.any(core):
        return None
    _, (iy, ix) = scipy.ndimage.distance_transform_edt(~core, return_indices=True)
    return rgba[:, :, :3][iy, ix]


def edge_purity_delta_e(rgba: np.ndarray) -> Optional[float]:
    """Alpha-weighted mean CIEDE2000 between feather-band pixels and their nearest core color."""
    alpha: np.ndarray = rgba[:, :, 3].astype(np.float32)
    band: np.ndarray = (alpha > 0) & (alpha < CORE_ALPHA)
    if not np.any(band):
        return 0.0
    ref: Optional[np.ndarray] = nearest_core_bgr(rgba)
    if ref is None:
        return None
    de: np.ndarray = ciede2000(bgr_to_lab(rgba[:, :, :3][band]), bgr_to_lab(ref[band]))
    w: np.ndarray = alpha[band] / 255.0
    return float(np.sum(de * w) / max(1e-6, float(np.sum(w))))


def halo_excess_lstar(rgba: np.ndarray, skin_bgr: Tuple[float, float, float]) -> Optional[float]:
    """Mean L*(actual composite) - L*(ideal composite) in a +/- 2 px band around the silhouette."""
    ref: Optional[np.ndarray] = nearest_core_bgr(rgba)
    if ref is None:
        return None
    a: np.ndarray = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    skin: np.ndarray = np.array(skin_bgr, dtype=np.float32)[None, None, :]
    actual: np.ndarray = rgba[:, :, :3].astype(np.float32) * a + skin * (1.0 - a)
    ideal: np.ndarray = ref.astype(np.float32) * a + skin * (1.0 - a)

    sil: np.ndarray = (rgba[:, :, 3] >= 128).astype(np.uint8)
    k: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    band: np.ndarray = (cv2.dilate(sil, k) > 0) & (cv2.erode(sil, k) == 0)
    if not np.any(band):
        return 0.0
    dL: np.ndarray = bgr_to_lab(actual[band])[..., 0] - bgr_to_lab(ideal[band])[..., 0]
    return float(np.mean(dL))


def evaluate_set(set_name: str, path: str) -> Dict[str, Any]:
    """Segments a tray, extracts the first 5 nails, and measures edge quality."""
    record: Dict[str, Any] = {"set": set_name, "ok": False}
    img: Optional[np.ndarray] = cv2.imread(path)
    if img is None:
        record["error"] = "could not read image"
        return record

    slots: List[Dict[str, Any]] = segment_packaging_tray(img, max_slots=15)
    record["slot_count"] = len(slots)
    record["methods"] = sorted({str(s.get("method", "unreported")) for s in slots})
    if len(slots) < 5:
        record["error"] = f"only {len(slots)} slots"
        return record

    nails: List[Dict[str, Any]] = []
    for i, s in enumerate(slots[:5], start=1):
        kwargs: Dict[str, Any] = {
            "feather_px": 1.5,
            "deskew": True,
            "auto_orient_cuticle_down": True,
            "force_inverted": s.get("shape_info", {}).get("metrics", {}).get("is_inverted", None),
        }
        if s.get("homography") is not None:
            kwargs["homography"] = np.array(s["homography"], dtype=np.float64)
        rgba: np.ndarray = extract_nail_rgba(img, np.array(s["contour"], dtype=np.int32), **kwargs)
        cv2.imwrite(os.path.join(OUT_DIR, f"{set_name}_{i}.png"), rgba)

        entry: Dict[str, Any] = {
            "idx": i,
            "size": [int(rgba.shape[1]), int(rgba.shape[0])],
            "edge_delta_e": edge_purity_delta_e(rgba),
        }
        for tone, bgr in SKIN_TONES_BGR.items():
            entry[f"halo_{tone}"] = halo_excess_lstar(rgba, bgr)
        nails.append(entry)

    def _mean(key: str) -> Optional[float]:
        vals: List[float] = [float(n[key]) for n in nails if n.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def _max(key: str) -> Optional[float]:
        vals: List[float] = [float(n[key]) for n in nails if n.get(key) is not None]
        return round(float(np.max(vals)), 3) if vals else None

    record["nails"] = nails
    record["edge_delta_e_mean"] = _mean("edge_delta_e")
    record["edge_delta_e_max"] = _max("edge_delta_e")
    for tone in SKIN_TONES_BGR:
        record[f"halo_{tone}_mean"] = _mean(f"halo_{tone}")
        record[f"halo_{tone}_max"] = _max(f"halo_{tone}")
    record["ok"] = True
    return record


def print_table(records: List[Dict[str, Any]], baseline: Optional[Dict[str, Dict[str, Any]]]) -> None:
    """Prints a per-set summary table, with baseline deltas when available."""
    header: str = f"{'set':<20}{'method':<12}{'dE mean':>9}{'dE max':>9}{'halo L':>9}{'halo M':>9}{'halo D':>9}"
    print(header)
    print("-" * len(header))
    for r in records:
        if not r.get("ok"):
            print(f"{r['set']:<20}ERROR: {r.get('error')}")
            continue
        method: str = ",".join(r.get("methods", []))
        line: str = (
            f"{r['set']:<20}{method:<12}"
            f"{r['edge_delta_e_mean']:>9.2f}{r['edge_delta_e_max']:>9.2f}"
            f"{r['halo_light_max']:>9.2f}{r['halo_medium_max']:>9.2f}{r['halo_dark_max']:>9.2f}"
        )
        print(line)
        if baseline and r["set"] in baseline and baseline[r["set"]].get("ok"):
            b: Dict[str, Any] = baseline[r["set"]]
            print(
                f"{'  (baseline)':<32}"
                f"{b['edge_delta_e_mean']:>9.2f}{b['edge_delta_e_max']:>9.2f}"
                f"{b['halo_light_max']:>9.2f}{b['halo_medium_max']:>9.2f}{b['halo_dark_max']:>9.2f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Edge color purity and halo metrics for extracted nails.")
    parser.add_argument("--label", default="run", help="Name for this run's JSON output.")
    parser.add_argument("--compare", default=None, help="Label of a previous run to compare against.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    records: List[Dict[str, Any]] = [evaluate_set(n, p) for n, p in discover_sets()]

    out_path: str = os.path.join(OUT_DIR, f"metrics_{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    baseline: Optional[Dict[str, Dict[str, Any]]] = None
    if args.compare:
        cmp_path: str = os.path.join(OUT_DIR, f"metrics_{args.compare}.json")
        if os.path.exists(cmp_path):
            with open(cmp_path, "r", encoding="utf-8") as f:
                baseline = {r["set"]: r for r in json.load(f)}

    print_table(records, baseline)

    ok_records: List[Dict[str, Any]] = [r for r in records if r.get("ok")]
    de_max: float = max((float(r["edge_delta_e_mean"]) for r in ok_records), default=0.0)
    halo_max: float = max(
        (float(r[f"halo_{t}_max"]) for r in ok_records for t in SKIN_TONES_BGR),
        default=0.0,
    )
    print(f"\nWorst per-set mean edge dE2000: {de_max:.2f}  (gate <= 5.0, target <= 3.0)")
    print(f"Worst per-nail halo excess:     {halo_max:.2f} L*  (gate <= 2.0)")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
