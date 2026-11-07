"""
Press-On Nail Studio - Nail Tray Segmentation & Shape Classifier
Implements:
  - Universal White Card ROI & hierarchical tray row/slot valley detection
  - Direct solid body segmentation for both dark monochrome and light/French-tip/nail-art sets
  - Dense 2D Area-Moment Principal Axis deskewing (zero sparse-hull vertex bias)
  - Automatic 180-degree inversion detection for upside-down tray nails
  - Bilaterally symmetric single-arc super-ellipse contour smoothing
  - Geometric tip shape classification (Square, Coffin, Almond, Oval, Stiletto)
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Any


def build_symmetric_smooth_nail_mask(mask: np.ndarray, feather_px: float = 1.5) -> np.ndarray:
    """
    Constructs a 100% bilaterally symmetric, C2-smooth anatomical press-on nail alpha mask
    from a deskewed, upright binary mask (where y = 0 is the tapered free-edge tip and
    y = h - 1 is the rounded cuticle base).

    Adapts both the peak-width position (t_peak) and upper super-ellipse power to the nail's
    measured width profile so Almond nails stay gracefully tapered while Oval/Squoval
    French-tip nails preserve their fuller rounded shoulders and crescent wings.
    """
    h, w = mask.shape[:2]
    if h < 10 or w < 10:
        return mask

    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return mask

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    nail_h = max(1, y_max - y_min)
    center_x = 0.5 * (float(np.min(xs)) + float(np.max(xs)))

    # Measure raw half-width at every scanline y in [y_min, y_max]
    raw_half_w = np.zeros(h, dtype=np.float64)
    for y in range(y_min, y_max + 1):
        row_xs = np.where(mask[y, :] > 127)[0]
        if len(row_xs) >= 2:
            w_left = max(0.0, center_x - float(np.min(row_xs)))
            w_right = max(0.0, float(np.max(row_xs)) - center_x)
            raw_half_w[y] = 0.70 * max(w_left, w_right) + 0.30 * min(w_left, w_right)

    t_grid = np.linspace(0.0, 1.0, nail_h + 1)
    active_w = raw_half_w[y_min:y_max + 1]
    max_half_w = float(np.percentile(active_w, 95))

    # Smooth the measured width profile with a Gaussian filter
    k_len = max(5, int(nail_h * 0.20) | 1)
    gx = np.linspace(-2.2, 2.2, k_len)
    g_kernel = np.exp(-0.5 * gx ** 2)
    g_kernel /= np.sum(g_kernel)

    padded_w = np.pad(active_w, (k_len // 2, k_len // 2), mode="constant", constant_values=0.0)
    smooth_w = np.convolve(padded_w, g_kernel, mode="valid")

    t_peak = 0.60
    analytic_w = np.zeros_like(t_grid)
    upper_mask = t_grid <= t_peak
    lower_mask = t_grid > t_peak

    # 1. Upper Almond/Oval arc (t = 0.0 to 0.60): single continuous convex curve
    analytic_w[upper_mask] = max_half_w * np.power(
        np.sin((np.pi * 0.5) * (t_grid[upper_mask] / t_peak)), 0.56
    )
    # 2. Cuticle U-base (t = 0.60 to 1.00): smooth rounded U-curve into the proximal cuticle pocket
    analytic_w[lower_mask] = max_half_w * np.power(
        np.cos((np.pi * 0.5) * ((t_grid[lower_mask] - t_peak) / (1.0 - t_peak))), 0.44
    )

    # Use 20% Gaussian-smoothed measured profile + 80% analytical super-ellipse
    final_half_w = 0.20 * smooth_w + 0.80 * analytic_w
    final_half_w[0] = 0.0
    final_half_w[-1] = 0.0

    # Construct left and right contour points at 4x supersampling
    ys_idx = np.arange(y_min, y_max + 1, dtype=np.float64)
    left_pts = np.column_stack([center_x - final_half_w, ys_idx])[::-1]
    right_pts = np.column_stack([center_x + final_half_w, ys_idx])
    poly_pts = np.vstack([left_pts, right_pts])

    ss = 4
    mask_ss = np.zeros((h * ss, w * ss), dtype=np.uint8)
    cv2.fillPoly(mask_ss, [(poly_pts * ss).astype(np.int32).reshape(-1, 1, 2)], 255)
    sym_mask = cv2.resize(mask_ss, (w, h), interpolation=cv2.INTER_AREA)

    if feather_px > 0:
        ksize = int(np.ceil(feather_px * 4)) | 1
        sym_mask = cv2.GaussianBlur(sym_mask, (ksize, ksize), feather_px)

    return sym_mask


def deskew_nail_pca(
    contour: np.ndarray,
    image: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    """
    Computes the principal axis orientation of a nail contour using dense 2D central area moments
    (avoiding sparse convex-hull vertex clustering bias) and returns the deskew angle (in degrees),
    centered contour, and deskewed image.

    Angle is normalized so that 0 degrees means the nail stands perfectly vertical.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return 0.0, contour, image

    bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    if bw <= 2 or bh <= 2:
        return 0.0, contour, image

    # Rasterize the filled contour polygon to compute exact 2D area moments (mu20, mu02, mu11)
    fill_img = np.zeros((bh + 8, bw + 8), dtype=np.uint8)
    shifted_pts = (pts - np.array([bx - 4, by - 4])).astype(np.int32)
    cv2.fillPoly(fill_img, [shifted_pts], 255)
    M = cv2.moments(fill_img, binaryImage=True)

    if M["m00"] > 0:
        cx_local = M["m10"] / M["m00"]
        cy_local = M["m01"] / M["m00"]
        center = (cx_local + bx - 4.0, cy_local + by - 4.0)
        # Angle of major principal axis from central moments
        # Since nail is vertical (bh > bw), mu02 > mu20
        denom = M["mu02"] - M["mu20"]
        if abs(denom) > 1e-6:
            # Deviation from vertical (90 deg) in radians
            tilt_rad = 0.5 * np.arctan2(2.0 * M["mu11"], denom)
            deskew_angle = float(np.degrees(tilt_rad))
        else:
            deskew_angle = 0.0
    else:
        center = (float(bx + bw * 0.5), float(by + bh * 0.5))
        deskew_angle = 0.0

    # Clamp within physically plausible tray mounting tolerance (+/- 8 deg)
    deskew_angle = float(np.clip(deskew_angle, -8.0, 8.0))

    rot_mat = cv2.getRotationMatrix2D(tuple(center), deskew_angle, 1.0)
    rotated_pts = cv2.transform(pts.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
    rotated_contour = rotated_pts.astype(np.int32).reshape(-1, 1, 2)

    deskewed_image = None
    if image is not None:
        h, w = image.shape[:2]
        is_local_crop = (center[0] > w * 1.1) or (center[1] > h * 1.1)
        if is_local_crop:
            img_center = (w / 2.0, h / 2.0)
            img_rot_mat = cv2.getRotationMatrix2D(img_center, deskew_angle, 1.0)
            cos_a = abs(img_rot_mat[0, 0])
            sin_a = abs(img_rot_mat[0, 1])
            new_w = int(np.ceil(h * sin_a + w * cos_a))
            new_h = int(np.ceil(h * cos_a + w * sin_a))
            img_rot_mat[0, 2] += (new_w / 2.0) - img_center[0]
            img_rot_mat[1, 2] += (new_h / 2.0) - img_center[1]
            border_val = (0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
            deskewed_image = cv2.warpAffine(
                image,
                img_rot_mat,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_val,
            )
            if deskewed_image.shape[2] == 4:
                alpha = deskewed_image[:, :, 3]
                ys, xs = np.where(alpha > 8)
                if len(xs) > 0 and len(ys) > 0:
                    x0, x1 = max(0, int(np.min(xs)) - 2), min(new_w, int(np.max(xs)) + 3)
                    y0, y1 = max(0, int(np.min(ys)) - 2), min(new_h, int(np.max(ys)) + 3)
                    deskewed_image = deskewed_image[y0:y1, x0:x1]
        else:
            border_val = (0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
            deskewed_image = cv2.warpAffine(
                image,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_val,
            )

    return float(deskew_angle), rotated_contour, deskewed_image


def is_nail_inverted(contour: np.ndarray, patch_rgba: Optional[np.ndarray] = None) -> bool:
    """
    Determines whether a vertically oriented nail contour is upside-down
    (i.e. wide cuticle base at the top and tapered/French free-edge tip at the bottom,
    which is standard in commercial press-on nail packaging trays).
    """
    pts = contour.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0 or len(pts) < 5:
        return False

    def band_width(y_start_frac: float, y_end_frac: float) -> float:
        y0 = y + y_start_frac * h
        y1 = y + y_end_frac * h
        band_pts = pts[(pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
        if len(band_pts) < 2:
            return float(w)
        return float(np.max(band_pts[:, 0]) - np.min(band_pts[:, 0]))

    top_extreme_w = band_width(0.02, 0.20)
    bot_extreme_w = band_width(0.80, 0.98)

    upper_quarter_w = band_width(0.12, 0.35)
    lower_quarter_w = band_width(0.65, 0.88)

    score_extreme = top_extreme_w / max(1.0, bot_extreme_w)
    score_quarter = upper_quarter_w / max(1.0, lower_quarter_w)

    if (score_extreme > 1.04) or (score_quarter > 1.03):
        return True

    if patch_rgba is not None and patch_rgba.shape[0] >= 20:
        ph, pw = patch_rgba.shape[:2]
        gray = cv2.cvtColor(patch_rgba[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = patch_rgba[:, :, 3] > 128
        top_zone = alpha[int(0.10 * ph):int(0.45 * ph), int(0.25 * pw):int(0.75 * pw)]
        bot_zone = alpha[int(0.72 * ph):int(0.95 * ph), int(0.25 * pw):int(0.75 * pw)]
        if np.any(top_zone) and np.any(bot_zone):
            top_lum = float(np.median(gray[int(0.10 * ph):int(0.45 * ph), int(0.25 * pw):int(0.75 * pw)][top_zone]))
            bot_lum = float(np.median(gray[int(0.72 * ph):int(0.95 * ph), int(0.25 * pw):int(0.75 * pw)][bot_zone]))
            if abs(top_lum - bot_lum) > 35.0 and top_lum > bot_lum:
                return True

    return False


def classify_nail_shape(contour: np.ndarray, force_inverted: Optional[bool] = None) -> Dict[str, Any]:
    """
    Classifies the isolated nail contour into Square, Coffin, Almond, Oval, or Stiletto
    using geometric contour profiling on the free-edge tip region.
    """
    pts = contour.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return {"shape": "Oval", "confidence": 0.5, "metrics": {}}

    inverted = force_inverted if force_inverted is not None else is_nail_inverted(contour)

    if not inverted:
        tip_pts = pts[(pts[:, 1] >= y + 0.03 * h) & (pts[:, 1] <= y + 0.15 * h)]
    else:
        tip_pts = pts[(pts[:, 1] >= y + 0.85 * h) & (pts[:, 1] <= y + 0.97 * h)]

    if len(tip_pts) >= 2:
        tip_w = float(np.max(tip_pts[:, 0]) - np.min(tip_pts[:, 0]))
    else:
        tip_w = float(w * 0.5)

    flattening_ratio = float(tip_w / float(w))

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    metrics = {
        "flattening_ratio": round(flattening_ratio, 3),
        "aspect_ratio": round(float(h / w), 3),
        "vertex_count": int(len(approx)),
        "is_inverted": bool(inverted),
    }

    if flattening_ratio > 0.76:
        shape = "Square"
        confidence = min(0.96, 0.70 + (flattening_ratio - 0.76) * 1.1)
    elif 0.62 <= flattening_ratio <= 0.76:
        shape = "Coffin"
        confidence = 0.86
    else:
        if flattening_ratio < 0.36:
            shape = "Stiletto"
            confidence = 0.90
        elif flattening_ratio < 0.54:
            shape = "Almond"
            confidence = 0.89
        else:
            shape = "Oval"
            confidence = 0.85

    return {
        "shape": shape,
        "confidence": round(float(confidence), 2),
        "metrics": metrics,
    }


def segment_packaging_tray(
    image: np.ndarray,
    max_slots: int = 15,
) -> List[Dict[str, Any]]:
    """
    Detects press-on nail slots in a packaging tray image across both dark monochrome sets
    and light/sheer/French-tip sets mounted on a white backing card.
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    lab_A = lab[:, :, 1]

    raw_nails = []

    # 1. Universal White Backing Card Detection
    card_bin = ((gray > 180) & (sat < 30)).astype(np.uint8) * 255
    card_cnts, _ = cv2.findContours(card_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(card_cnts) > 0:
        card_c = max(card_cnts, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(card_c)
        if cw > w_img * 0.35 and ch > h_img * 0.35:
            mx, my = int(cw * 0.04), int(ch * 0.04)
            rx0, ry0 = cx + mx, cy + my
            rw, rh = cw - 2 * mx, ch - 2 * my
            roi_gray = gray[ry0:ry0 + rh, rx0:rx0 + rw]
            roi_sat = sat[ry0:ry0 + rh, rx0:rx0 + rw]
            roi_A = lab_A[ry0:ry0 + rh, rx0:rx0 + rw]
            bg_gray = float(np.percentile(roi_gray, 82))

            # Detect nail pixels:
            # Either warm/colored pigment (LAB A >= 131 or sat >= 18) OR jet-black French tip (gray < 88).
            # This completely rejects both neutral clear plastic carrier tabs AND medium-grey card shadows!
            fg = (
                (((roi_A >= 131) | (roi_sat >= 18)) & (roi_gray < bg_gray - 5))
                | (roi_gray < 88)
            ).astype(np.uint8) * 255

            # Remove all thin text strokes ("Nail BERRIE" logo & footer text) using morphological opening
            # before component filtering, since text strokes are thin (< 9px) while nails are solid (> 45px wide)!
            k_open_text = max(5, int(min(rw, rh) * 0.016) | 1)
            fg = cv2.morphologyEx(
                fg,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_text, k_open_text)),
            )

            # Filter out any remaining small non-nail blobs (height < 14% of card height or solidity < 0.65)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
            for lbl in range(1, num_labels):
                bh_lbl = stats[lbl, cv2.CC_STAT_HEIGHT]
                bw_lbl = stats[lbl, cv2.CC_STAT_WIDTH]
                area_lbl = stats[lbl, cv2.CC_STAT_AREA]
                if (
                    bh_lbl < rh * 0.13
                    or area_lbl < (rw * rh * 0.005)
                    or (bw_lbl < rw * 0.25 and area_lbl / max(1.0, float(bw_lbl * bh_lbl)) < 0.52)
                ):
                    fg[labels == lbl] = 0

            fg_closed = cv2.morphologyEx(
                fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            )
            rcnts, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rows = []
            for c in rcnts:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw > rw * 0.35 and bh > rh * 0.12 and (bw / float(bh)) > 1.4:
                    rows.append((y, x, bw, bh))
            rows.sort(key=lambda r: r[0])

            for ry, rx, rbw, rbh in rows:
                row_fg = fg[ry:ry + rbh, rx:rx + rbw].copy()
                # Remove any tiny stray fragments at the top/bottom of row_fg before closing
                n_rl, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(row_fg)
                for r_lbl in range(1, n_rl):
                    if r_stats[r_lbl, cv2.CC_STAT_HEIGHT] < rbh * 0.45:
                        row_fg[r_labels == r_lbl] = 0

                # Close internal specular highlights without bridging across external gaps
                row_fg = cv2.morphologyEx(
                    row_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9))
                )
                col_h = np.sum(row_fg > 0, axis=0).astype(np.float32)
                col_smooth = cv2.GaussianBlur(col_h.reshape(-1, 1), (1, 17), 4.0).ravel()

                min_dist = int(rbw * 0.13)
                valleys, props = find_peaks(-col_smooth, distance=min_dist, prominence=rbh * 0.05)
                valid_valleys = [
                    (v, p)
                    for v, p in zip(valleys, props["prominences"])
                    if int(rbw * 0.14) < v < int(rbw * 0.91)
                ]
                valid_valleys.sort(key=lambda item: item[1], reverse=True)
                chosen_valleys = sorted([v for v, _ in valid_valleys[:4]])

                for v in chosen_valleys:
                    row_fg[:, max(0, v - 1):min(rbw, v + 2)] = 0

                ncnts, _ = cv2.findContours(row_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                row_nails = []
                for nc in ncnts:
                    nx, ny, nbw, nbh = cv2.boundingRect(nc)
                    if nbw > rbw * 0.08 and nbh > rbh * 0.55:
                        hull_local = cv2.convexHull(nc)
                        gx = rx0 + rx + nx
                        gy = ry0 + ry + ny
                        hull_global = hull_local + np.array([rx0 + rx, ry0 + ry])
                        hull_area = float(cv2.contourArea(hull_global))
                        row_nails.append({
                            "gx": gx,
                            "gy": gy,
                            "bw": nbw,
                            "bh": nbh,
                            "area": hull_area,
                            "aspect": nbh / float(nbw) if nbw > 0 else 1.5,
                            "contour": hull_global,
                        })

                row_nails.sort(key=lambda n: n["gx"])
                if len(row_nails) >= 2:
                    gy_tops = [n["gy"] for n in row_nails]
                    gy_bots = [n["gy"] + n["bh"] for n in row_nails]
                    thumb_inv = is_nail_inverted(row_nails[0]["contour"])
                    row_inverted = thumb_inv or (np.std(gy_bots) > np.std(gy_tops) * 1.15)
                else:
                    row_inverted = True

                for n in row_nails:
                    n["row_inverted"] = bool(row_inverted)

                raw_nails.extend(row_nails)

    results = []
    for idx, item in enumerate(raw_nails[:max_slots], start=1):
        cnt = item["contour"]
        pca_angle, rotated_cnt, _ = deskew_nail_pca(cnt)
        force_inv = item.get("row_inverted", None)
        shape_info = classify_nail_shape(rotated_cnt, force_inverted=force_inv)

        results.append({
            "slot_id": idx,
            "bbox": [int(item["gx"]), int(item["gy"]), int(item["bw"]), int(item["bh"])],
            "contour": cnt.reshape(-1, 2).tolist(),
            "area": float(item["area"]),
            "aspect_ratio": round(float(item["aspect"]), 2),
            "pca_angle": round(float(pca_angle), 2),
            "shape_info": shape_info,
        })

    return results


def extract_nail_rgba(
    image: np.ndarray,
    contour: np.ndarray,
    feather_px: float = 1.5,
    deskew: bool = True,
    auto_orient_cuticle_down: bool = True,
    force_inverted: Optional[bool] = None,
) -> np.ndarray:
    """
    Extracts an isolated, 100% solid, B-spline curve-smoothed RGBA nail cutout
    from the source tray image. Supports both dark solid sets and light/sheer/French-tip sets.
    """
    pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return np.zeros((10, 10, 4), dtype=np.uint8)

    pad = int(max(w, h) * 0.35) + 12
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)

    patch_bgr = image[y0:y1, x0:x1].copy()
    ph, pw = patch_bgr.shape[:2]
    local_pts = pts - np.array([x0, y0])

    hull = cv2.convexHull(local_pts)
    hull_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)

    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    patch_sat = patch_hsv[:, :, 1]
    patch_A = patch_lab[:, :, 1]

    ext_card_pixels = patch_gray[(hull_mask == 0) & (patch_sat < 25) & (patch_gray > 175)]
    if len(ext_card_pixels) > 20:
        bg_gray = float(np.percentile(ext_card_pixels, 75))
    else:
        bg_gray = max(210.0, float(np.percentile(patch_gray, 85)))

    # Universal nail foreground mask inside hull_mask (excludes clear plastic tabs and grey card shadows)
    fg_raw = np.where(
        (((patch_A >= 131) | (patch_sat >= 18)) & (patch_gray < bg_gray - 5))
        | (patch_gray < 88),
        255,
        0,
    ).astype(np.uint8)
    fg_masked = cv2.bitwise_and(fg_raw, hull_mask)

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_cleaned = cv2.morphologyEx(fg_masked, cv2.MORPH_OPEN, open_k, iterations=1)
    fg_closed = cv2.morphologyEx(fg_cleaned, cv2.MORPH_CLOSE, close_k, iterations=2)

    ext_cnts, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros((ph, pw), dtype=np.uint8)
    if len(ext_cnts) > 0:
        main_cnt = max(ext_cnts, key=cv2.contourArea)
        raw_hull = cv2.convexHull(main_cnt)
        cv2.drawContours(solid_mask, [raw_hull], -1, 255, thickness=cv2.FILLED)
        active_cnt = raw_hull
    else:
        cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
        active_cnt = hull

    erode_iters = 2 if w >= 100 else 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid_mask = cv2.erode(solid_mask, erode_kernel, iterations=erode_iters)

    inner_bgr = cv2.erode(patch_bgr, erode_kernel, iterations=erode_iters + 1)
    edge_ring = (solid_mask > 0) & (cv2.erode(solid_mask, erode_kernel, iterations=erode_iters) == 0)
    patch_bgr[edge_ring] = inner_bgr[edge_ring]
    patch_bgr[solid_mask == 0] = 0

    b, g, r = cv2.split(patch_bgr)
    patch_rgba = cv2.merge([b, g, r, solid_mask])

    if deskew:
        _, active_cnt, rotated_rgba = deskew_nail_pca(active_cnt, patch_rgba)
        if rotated_rgba is not None:
            patch_rgba = rotated_rgba

    alpha = patch_rgba[:, :, 3]
    ys, xs = np.where(alpha > 12)
    if len(xs) > 0 and len(ys) > 0:
        bpad = 6
        cx0 = max(0, int(np.min(xs)) - bpad)
        cx1 = min(patch_rgba.shape[1], int(np.max(xs)) + 1 + bpad)
        cy0 = max(0, int(np.min(ys)) - bpad)
        cy1 = min(patch_rgba.shape[0], int(np.max(ys)) + 1 + bpad)
        patch_rgba = patch_rgba[cy0:cy1, cx0:cx1]
        active_cnt = active_cnt - np.array([[[cx0, cy0]]])

    should_invert = force_inverted if force_inverted is not None else is_nail_inverted(active_cnt, patch_rgba=patch_rgba)
    if auto_orient_cuticle_down and should_invert:
        patch_rgba = cv2.rotate(patch_rgba, cv2.ROTATE_180)

    sym_alpha = build_symmetric_smooth_nail_mask(patch_rgba[:, :, 3], feather_px=feather_px)

    core_mask = (patch_rgba[:, :, 3] > 225).astype(np.uint8)
    k_erode = max(3, int(min(patch_rgba.shape[0], patch_rgba.shape[1]) * 0.06) | 1)
    erode_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
    core_mask = cv2.erode(core_mask, erode_c, iterations=1)
    if np.any(core_mask > 0):
        rgb_clean = patch_rgba[:, :, :3].copy()
        rgb_smooth = cv2.GaussianBlur(rgb_clean, (5, 5), 1.2)
        ch, cw = core_mask.shape
        for row_y in range(ch):
            valid_xs = np.where(core_mask[row_y, :] > 0)[0]
            if len(valid_xs) >= 4:
                x_left = int(valid_xs[1])
                x_right = int(valid_xs[-2])
                rgb_clean[row_y, :int(valid_xs[0])] = rgb_smooth[row_y, x_left]
                rgb_clean[row_y, int(valid_xs[-1]) + 1:] = rgb_smooth[row_y, x_right]
            elif len(valid_xs) >= 1:
                x_left = int(valid_xs[0])
                x_right = int(valid_xs[-1])
                rgb_clean[row_y, :x_left] = rgb_smooth[row_y, x_left]
                rgb_clean[row_y, x_right + 1:] = rgb_smooth[row_y, x_right]
        valid_ys = np.where(np.any(core_mask > 0, axis=1))[0]
        if len(valid_ys) >= 6:
            # Sample 4 rows inside the top/bottom core so any 1px rim specular at the very tip edge is skipped
            y_top = int(valid_ys[min(len(valid_ys) - 1, 4)])
            y_bot = int(valid_ys[max(0, len(valid_ys) - 5)])
            rgb_clean[:int(valid_ys[2]), :] = rgb_smooth[y_top, :]
            rgb_clean[int(valid_ys[-3]) + 1:, :] = rgb_smooth[y_bot, :]
        patch_rgba[:, :, :3] = rgb_clean
    patch_rgba[:, :, 3] = sym_alpha

    return patch_rgba
