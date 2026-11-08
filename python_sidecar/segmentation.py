"""
Press-On Nail Studio - Nail Tray Segmentation, Rectification & Shape Classifier
(docs/true-texture-composition-plan.md, Steps 1 & 2)

Implements:
  - Step 1: Backing card perspective rectification (rectify_backing_card)
  - Step 2.1: Multi-cue foreground detection for white French tips, sheer cuticles & angled tray
  - Step 2.2: Data-driven bilateral profile smoothing with monotonic envelopes (build_symmetric_smooth_nail_mask)
  - Extra A: Edge de-matting (adaptive inward choke + soft feather + radial color bleed)
    so card-contaminated bevel colors never reach the composite (dematte_nail_edges)
  - Extra B: Geometric shape classification including Squoval (classify_nail_shape)
  - Coordinate contract: homography transformation and card_quad preservation
  - Observability: every slot reports "method" ("yolo" | "classical") and "method_note"
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.ndimage
import scipy.signal


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as [top_left, top_right, bottom_right, bottom_left] in clockwise order.
    Handles arbitrary 2D quadrilateral rotations and camera perspective tilts.
    """
    pts = pts.reshape(4, 2).astype(np.float64)
    center = np.mean(pts, axis=0)
    # Sort counter-clockwise by polar angle from center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ccw_order = pts[np.argsort(angles)]
    cw_order = ccw_order[::-1]

    # Find corner index corresponding to Top-Left:
    # In standard image coordinates (x right, y down), the top edge vector p0 -> p1
    # points left-to-right (angle between -45 deg and +45 deg), with minimal (x + y).
    best_tl_idx = 0
    min_dist_to_tl = float("inf")
    for i in range(4):
        p0 = cw_order[i]
        p1 = cw_order[(i + 1) % 4]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        ang_deg = np.degrees(np.arctan2(dy, dx))
        if -45.0 <= ang_deg <= 45.0:
            dist = float(p0[0] + p0[1])
            if dist < min_dist_to_tl:
                min_dist_to_tl = dist
                best_tl_idx = i

    ordered = np.roll(cw_order, -best_tl_idx, axis=0)
    return ordered


def rectify_backing_card(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Detects the white presentation backing card inside the packaging box, checks for camera
    perspective tilt (> 1.5 deg) or perspective foreshortening (> 3%), and computes the
    fronto-parallel perspective rectification homography H (mapping ORIGINAL -> rectified coords).

    Returns:
        (rectified_img, H, card_quad)
        where H is 3x3 float64, and card_quad is (4, 2) float64 [TL, TR, BR, BL] in original coordinates.
        When no rectification is needed, returns (image, np.eye(3), None).
    """
    h_img, w_img = image.shape[:2]
    # Work on scale-normalized copy for fast, noise-robust card contour detection
    scale: float = min(1.0, 1000.0 / float(max(h_img, w_img)))
    sw: int = int(round(w_img * scale))
    sh: int = int(round(h_img * scale))
    small: np.ndarray = (
        cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    )

    gray: np.ndarray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv: np.ndarray = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat: np.ndarray = hsv[:, :, 1]

    # White backing card binary mask
    card_bin: np.ndarray = ((gray > 165) & (sat < 40)).astype(np.uint8) * 255
    k_close: int = max(5, int(min(sw, sh) * 0.008) | 1)
    card_closed: np.ndarray = cv2.morphologyEx(
        card_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    )

    cnts, _ = cv2.findContours(card_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, np.ndarray]] = []
    total_area: float = float(sw * sh)
    for c in cnts:
        a: float = float(cv2.contourArea(c))
        if a > total_area * 0.15:
            M = cv2.moments(c)
            cy: float = (M["m01"] / M["m00"]) if M["m00"] > 0 else sh / 2.0
            # Reject/penalize mirror reflections in the upper box lid (cy / sh < 0.35)
            score: float = a
            if (cy / float(sh)) < 0.35:
                score *= 0.1
            candidates.append((score, c))

    if not candidates:
        return image, np.eye(3, dtype=np.float64), None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_c: np.ndarray = candidates[0][1]

    # Fit 4 corners: approxPolyDP with escalating epsilon (0.015 to 0.065)
    peri: float = float(cv2.arcLength(best_c, True))
    poly4: Optional[np.ndarray] = None
    for eps_frac in np.linspace(0.015, 0.065, 20):
        approx = cv2.approxPolyDP(best_c, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            poly4 = approx.reshape(4, 2).astype(np.float64) / scale
            break

    # Fallback to minAreaRect if approxPolyDP did not find a 4-vertex convex quad
    rect = cv2.minAreaRect(best_c)
    box = cv2.boxPoints(rect).astype(np.float64) / scale
    if poly4 is None:
        poly4 = box

    # Order quad points as [TL, TR, BR, BL]
    quad_orig = order_quad_points(poly4)
    tl, tr, br, bl = quad_orig

    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))

    top_dx = tr[0] - tl[0]
    top_dy = tr[1] - tl[1]
    top_angle_deg = abs(float(np.degrees(np.arctan2(top_dy, top_dx))))

    w_diff = abs(w_top - w_bot) / max(1.0, max(w_top, w_bot))
    h_diff = abs(h_left - h_right) / max(1.0, max(h_left, h_right))

    # Tilt angle from minAreaRect
    angle = rect[2]
    tilt = min(abs(angle), abs(angle + 90.0), abs(angle - 90.0))

    # Rectification criterion: tilt > 1.5 deg or foreshortening > 3% with non-zero tilt
    needs_rectification = (top_angle_deg > 1.5) or (tilt > 1.5) or (w_diff > 0.03 and top_angle_deg > 0.8) or (h_diff > 0.03 and top_angle_deg > 0.8)

    if not needs_rectification or (top_angle_deg < 1.5 and tilt < 1.5):
        return image, np.eye(3, dtype=np.float64), None

    # Compute fronto-parallel rectified dimensions
    dst_w: int = int(round(max(w_top, w_bot)))
    dst_h: int = int(round(max(h_left, h_right)))

    # Ensure orientation is upright (Nail BERRIE cards have 3 horizontal rows of 5 nails)
    if dst_w < dst_h * 0.75:
        quad_orig = np.roll(quad_orig, -1, axis=0)
        tl, tr, br, bl = quad_orig
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        dst_w = int(round(max(w_top, w_bot)))
        dst_h = int(round(max(h_left, h_right)))

    dst_pts = np.array(
        [[0.0, 0.0], [float(dst_w - 1), 0.0], [float(dst_w - 1), float(dst_h - 1)], [0.0, float(dst_h - 1)]],
        dtype=np.float32,
    )
    H: np.ndarray = cv2.getPerspectiveTransform(quad_orig.astype(np.float32), dst_pts)
    rectified_img: np.ndarray = cv2.warpPerspective(
        image, H, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rectified_img, H.astype(np.float64), quad_orig


def build_symmetric_smooth_nail_mask(mask: np.ndarray, feather_px: float = 1.5) -> np.ndarray:
    """
    Constructs a 100% bilaterally symmetric, smooth anatomical press-on nail alpha mask
    from a deskewed, upright binary mask (where y = 0 is the free-edge tip and
    y = h - 1 is the cuticle base).

    Preserves exact salon silhouette geometry:
      - Square, Squoval, and Coffin shapes retain wide, flat distal tips and sharp/squoval shoulders.
      - Almond, Oval, and Stiletto shapes retain continuous G2 super-elliptical arches.
      - Cuticle base (y = h - 1) always terminates with a true horizontal tangent U-cup.
    """
    h, w = mask.shape[:2]
    if h < 10 or w < 10:
        return mask

    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return mask

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    nail_h = max(1, y_max - y_min + 1)

    # Compute robust horizontal center_x from the middle 50% of the nail body
    mid_y0 = y_min + int(0.25 * nail_h)
    mid_y1 = max(mid_y0 + 1, y_min + int(0.75 * nail_h))
    mid_centers: List[float] = []
    for y in range(mid_y0, min(y_max + 1, mid_y1 + 1)):
        r_xs = np.where(mask[y, :] > 127)[0]
        if len(r_xs) >= 2:
            mid_centers.append(0.5 * (float(np.min(r_xs)) + float(np.max(r_xs))))
    if mid_centers:
        center_x = float(np.median(mid_centers))
    else:
        center_x = 0.5 * (float(np.min(xs)) + float(np.max(xs)))

    # Measure raw half-width at every scanline y in [y_min, y_max]
    measured_half_w = np.zeros(nail_h, dtype=np.float64)
    for i, y in enumerate(range(y_min, y_max + 1)):
        row_xs = np.where(mask[y, :] > 127)[0]
        if len(row_xs) >= 2:
            w_left = max(0.0, center_x - float(np.min(row_xs)))
            w_right = max(0.0, float(np.max(row_xs)) - center_x)
            measured_half_w[i] = 0.52 * max(w_left, w_right) + 0.48 * min(w_left, w_right)
        elif len(row_xs) == 1:
            measured_half_w[i] = abs(float(row_xs[0]) - center_x)
        else:
            measured_half_w[i] = measured_half_w[i - 1] if i > 0 else 0.0

    # Pre-smooth measured_half_w before finding the equator and monotonic envelope
    win_pre = max(5, int(nail_h * 0.09) | 1)
    if 5 <= win_pre < nail_h:
        w_pre = scipy.signal.savgol_filter(measured_half_w, window_length=win_pre, polyorder=min(2, win_pre - 1))
        w_pre = np.maximum(0.0, w_pre)
    else:
        w_pre = measured_half_w.copy()

    # Locate the anatomical equator in the mid-lower zone [0.45 * nail_h, 0.72 * nail_h]
    eq_start = int(0.45 * nail_h)
    eq_end = max(eq_start + 1, int(0.72 * nail_h))
    eq_zone = w_pre[eq_start:eq_end]
    i_peak = eq_start + int(np.argmax(eq_zone)) if len(eq_zone) > 0 else int(0.58 * nail_h)
    i_peak = int(np.clip(i_peak, max(4, int(0.45 * nail_h)), min(nail_h - 5, int(0.72 * nail_h))))
    w_eq = max(2.0, float(w_pre[i_peak]))

    # Analyze tip flattening ratio in the distal 3% - 15% zone
    tip_zone_start = max(1, int(0.03 * nail_h))
    tip_zone_end = max(tip_zone_start + 1, int(0.15 * nail_h))
    w_tip_measured = float(np.median(w_pre[tip_zone_start:tip_zone_end]))
    flat_ratio = w_tip_measured / w_eq
    is_flat_tip = flat_ratio >= 0.55  # Square, Squoval, Coffin

    if not is_flat_tip:
        # -------------------------------------------------------------
        # ALMOND / OVAL / STILETTO: Slender Taper & True Tip Preservation
        # -------------------------------------------------------------
        w_mono = w_pre.copy()
        for i in range(1, i_peak + 1):
            if w_mono[i] < w_mono[i - 1]:
                w_mono[i] = w_mono[i - 1]
        for i in range(nail_h - 2, i_peak - 1, -1):
            if w_mono[i] < w_mono[i + 1]:
                w_mono[i] = w_mono[i + 1]

        # Top tip: only cap the top 4% (matching local slope) so Almond/Stiletto stay slender
        tip_cap_len = max(2, int(0.04 * float(nail_h)))
        if tip_cap_len < nail_h:
            t_cap = np.linspace(0.0, 1.0, tip_cap_len, dtype=np.float64)
            ellip_tip = np.sqrt(np.clip(1.0 - (1.0 - t_cap) ** 2, 0.0, 1.0))
            w_mono[:tip_cap_len] = np.minimum(w_mono[:tip_cap_len], w_mono[tip_cap_len] * ellip_tip)
        w_mono[0] = 0.0

        # Bottom cuticle: anatomical U-cup rounding on the bottom 14%
        cut_cap_len = max(4, int(0.14 * float(nail_h)))
        if cut_cap_len < nail_h:
            t_cut = np.linspace(0.0, 1.0, cut_cap_len, dtype=np.float64)
            ellip_cut = np.sqrt(np.clip(1.0 - (1.0 - t_cut) ** 2, 0.0, 1.0))
            w_mono[nail_h - cut_cap_len :] = np.minimum(
                w_mono[nail_h - cut_cap_len :],
                w_mono[nail_h - cut_cap_len] * ellip_cut[::-1],
            )

        ys_scan = np.arange(y_min, y_max + 1, dtype=np.float64)
        left_pts = np.column_stack([center_x - w_mono, ys_scan])[::-1]
        right_pts = np.column_stack([center_x + w_mono, ys_scan])
        poly_pts = np.vstack([left_pts, right_pts])

    else:
        # -------------------------------------------------------------
        # SQUARE / SQUOVAL / COFFIN: Flat Distal Free-Edge Preservation
        # -------------------------------------------------------------
        win_len = max(7, int(nail_h * 0.22) | 1)
        if 7 <= win_len < nail_h:
            w_smooth = scipy.signal.savgol_filter(w_pre, window_length=win_len, polyorder=min(3, win_len - 1))
            w_smooth = np.maximum(0.0, w_smooth)
        else:
            w_smooth = w_pre.copy()

        w_mono = w_smooth.copy()
        for i in range(1, i_peak + 1):
            if w_mono[i] < w_mono[i - 1]:
                w_mono[i] = w_mono[i - 1]
        for i in range(nail_h - 2, i_peak - 1, -1):
            if w_mono[i] < w_mono[i + 1]:
                w_mono[i] = w_mono[i + 1]

        # Lower cuticle U-cup (bottom 16% anatomical rounding)
        cut_cap_len = max(5, int(0.16 * float(nail_h)))
        if cut_cap_len < nail_h:
            t_cut = np.linspace(0.0, 1.0, cut_cap_len, dtype=np.float64)
            ellip_cut = np.sqrt(np.clip(1.0 - (1.0 - t_cut) ** 2, 0.0, 1.0))
            w_mono[nail_h - cut_cap_len :] = np.minimum(
                w_mono[nail_h - cut_cap_len :],
                w_mono[nail_h - cut_cap_len] * ellip_cut[::-1],
            )

        # Distal tip: preserve flat edge width and shoulder corners
        tip_corner_len = max(2, int(0.035 * float(nail_h)))
        if flat_ratio < 0.78:
            # Squoval / Coffin: slight shoulder radius
            t_corn = np.linspace(0.0, 1.0, tip_corner_len, dtype=np.float64)
            corn_round = np.sqrt(np.clip(1.0 - (1.0 - t_corn) ** 2, 0.0, 1.0))
            w_mono[:tip_corner_len] = np.minimum(w_mono[:tip_corner_len], w_tip_measured * corn_round)
        else:
            # Square: sharp orthogonal tip edge
            w_mono[:tip_corner_len] = np.maximum(w_mono[:tip_corner_len], w_tip_measured * 0.96)

        ys_scan = np.arange(y_min, y_max + 1, dtype=np.float64)
        left_pts = np.column_stack([center_x - w_mono, ys_scan])[::-1]
        right_pts = np.column_stack([center_x + w_mono, ys_scan])

        # Insert horizontal top flat edge vertices
        if w_mono[0] > 1.0:
            top_left = [center_x - w_mono[0], float(y_min)]
            top_right = [center_x + w_mono[0], float(y_min)]
            poly_pts = np.vstack([left_pts, [top_left, top_right], right_pts])
        else:
            poly_pts = np.vstack([left_pts, right_pts])

    # Rasterize contour at 4x supersampling
    ss = 4
    mask_ss = np.zeros((h * ss, w * ss), dtype=np.uint8)
    cv2.fillPoly(mask_ss, [np.round(poly_pts * ss).astype(np.int32).reshape(-1, 1, 2)], 255)
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
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return 0.0, contour, image

    bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    if bw <= 2 or bh <= 2:
        return 0.0, contour, image

    fill_img = np.zeros((bh + 8, bw + 8), dtype=np.uint8)
    shifted_pts = (pts - np.array([bx - 4, by - 4])).astype(np.int32)
    cv2.fillPoly(fill_img, [shifted_pts], 255)
    M = cv2.moments(fill_img, binaryImage=True)

    if M["m00"] > 0:
        cx_local = M["m10"] / M["m00"]
        cy_local = M["m01"] / M["m00"]
        center = (cx_local + bx - 4.0, cy_local + by - 4.0)
        denom = M["mu02"] - M["mu20"]
        if abs(denom) > 1e-6:
            tilt_rad = 0.5 * np.arctan2(2.0 * M["mu11"], denom)
            deskew_angle = float(np.degrees(tilt_rad))
        else:
            deskew_angle = 0.0
    else:
        center = (float(bx + bw * 0.5), float(by + bh * 0.5))
        deskew_angle = 0.0

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
    Classifies the isolated nail contour into Square, Squoval, Coffin, Almond, Oval, or Stiletto
    using geometric contour profiling on the free-edge tip region and aspect ratio.
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
    aspect_ratio = float(h / float(w))

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = int(len(approx))

    metrics = {
        "flattening_ratio": round(flattening_ratio, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "vertex_count": vertex_count,
        "is_inverted": bool(inverted),
    }

    # Shape taxonomy:
    # Stiletto: pointy tip (< 0.36)
    # Almond: tapered convex tip (0.36 <= flattening < 0.54)
    # Square: wide sharp flat tip (> 0.80)
    # Squoval: flat-ish tip with rounded shoulders (0.58 <= flattening <= 0.80, aspect < 1.70 or rounded corners)
    # Coffin: long tapered flat-tip (0.60 <= flattening <= 0.78, aspect >= 1.70)
    # Oval: smooth curved tip (0.50 <= flattening < 0.60)
    if flattening_ratio < 0.36:
        shape = "Stiletto"
        confidence = 0.90
    elif flattening_ratio < 0.54:
        shape = "Almond"
        confidence = 0.89
    elif flattening_ratio > 0.80:
        shape = "Square"
        confidence = min(0.96, 0.70 + (flattening_ratio - 0.80) * 1.1)
    elif 0.58 <= flattening_ratio <= 0.80:
        if aspect_ratio >= 1.70 and vertex_count <= 7 and flattening_ratio <= 0.76:
            shape = "Coffin"
            confidence = 0.86
        elif flattening_ratio >= 0.60:
            shape = "Squoval"
            confidence = 0.88
        else:
            shape = "Oval"
            confidence = 0.85
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
    Detects press-on nail slots in a packaging tray image across both dark monochrome sets,
    sheer sets, white French tips, and tilted/angled camera poses.
    
    Rectifies perspective internally when tilted (> 1.5 deg) or foreshortened (> 3%),
    performs multi-cue foreground segmentation in the fronto-parallel plane, cuts row
    valleys to prevent row merging, and maps all contour coordinates and bboxes back to
    ORIGINAL image coordinates via H_inv.
    """
    # Step 1: Perspective rectification
    rect_img, H, card_quad = rectify_backing_card(image)
    h_img, w_img = rect_img.shape[:2]

    # Check for fine-tuned ML segmentation model.
    # No silent fallback: whichever path runs, every slot carries "method" and "method_note"
    # so the UI and tests can see exactly why the ML path was or was not used.
    ml_note: str = "ml model not loaded"
    try:
        from ml_models import get_model_manager
        ml_mgr = get_model_manager()
        if ml_mgr.has_tray_model():
            ml_slots = ml_mgr.segment_tray_ml(rect_img)
            n_ml: int = len(ml_slots) if ml_slots else 0
            ml_note = f"yolo returned {n_ml} slots (< 5 required); using classical"
            if ml_slots and n_ml >= 5:
                H_inv = np.linalg.inv(H) if H is not None else None
                final_slots = []
                for s in ml_slots[:max_slots]:
                    cnt_rect = np.array(s["contour"], dtype=np.float32).reshape(-1, 1, 2)
                    if H_inv is not None:
                        cnt_orig = cv2.perspectiveTransform(cnt_rect, H_inv).reshape(-1, 2)
                    else:
                        cnt_orig = cnt_rect.reshape(-1, 2)

                    cnt_int = np.round(cnt_orig).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(cnt_int)
                    area = float(cv2.contourArea(cnt_int))
                    aspect = float(h / max(1, w))
                    shape_info = classify_nail_shape(cnt_int)
                    pca_angle, _, _ = deskew_nail_pca(cnt_int)

                    final_slots.append({
                        "slot_id": s["slot_id"],
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "contour": cnt_int.tolist(),
                        "area": round(area, 1),
                        "aspect_ratio": round(aspect, 3),
                        "pca_angle": round(float(pca_angle), 2),
                        "shape_info": shape_info,
                        "homography": H.tolist() if H is not None else None,
                        "card_quad": card_quad.tolist() if card_quad is not None else None,
                        "method": "yolo",
                        "method_note": f"yolo returned {n_ml} slots",
                    })
                if len(final_slots) >= 5:
                    return final_slots
    except Exception as e:
        ml_note = f"ml error: {type(e).__name__}: {e}; using classical"
        print(f"[segmentation] {ml_note}", file=sys.stderr)

    classical_slots: List[Dict[str, Any]] = _segment_packaging_tray_classical(
        rect_img, H, card_quad, max_slots
    )
    for s in classical_slots:
        s["method"] = "classical"
        s["method_note"] = ml_note
    return classical_slots


def _segment_packaging_tray_classical(
    rect_img: np.ndarray,
    H: np.ndarray,
    card_quad: Optional[np.ndarray],
    max_slots: int,
) -> List[Dict[str, Any]]:
    """
    Classical multi-cue foreground segmentation on the rectified tray image.
    Returns slots with contours and bboxes mapped back to ORIGINAL image coordinates.
    """
    h_img, w_img = rect_img.shape[:2]
    gray: np.ndarray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    hsv: np.ndarray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2HSV)
    lab: np.ndarray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2LAB)
    sat: np.ndarray = hsv[:, :, 1]
    lab_L: np.ndarray = lab[:, :, 0]
    lab_A: np.ndarray = lab[:, :, 1]

    # Card ROI in rectified image
    if card_quad is not None:
        rx0, ry0 = 0, 0
        rw, rh = w_img, h_img
    else:
        card_bin = ((gray > 165) & (sat < 40)).astype(np.uint8) * 255
        k_close = max(5, int(min(w_img, h_img) * 0.008) | 1)
        card_closed = cv2.morphologyEx(
            card_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        )
        cnts, _ = cv2.findContours(card_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) > w_img * h_img * 0.15]
        candidates = []
        for c in valid_cnts:
            M = cv2.moments(c)
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else h_img / 2.0
            score = cv2.contourArea(c)
            if cy / float(h_img) < 0.35:
                score *= 0.1
            candidates.append((score, c))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            card_c = candidates[0][1]
            cx, cy, cw, ch = cv2.boundingRect(card_c)
            mx, my = int(cw * 0.04), int(ch * 0.04)
            rx0, ry0 = cx + mx, cy + my
            rw, rh = cw - 2 * mx, ch - 2 * my
        else:
            rx0, ry0 = int(w_img * 0.05), int(h_img * 0.05)
            rw, rh = int(w_img * 0.90), int(h_img * 0.90)

    roi_gray = gray[ry0:ry0 + rh, rx0:rx0 + rw]
    roi_sat = sat[ry0:ry0 + rh, rx0:rx0 + rw]
    roi_A = lab_A[ry0:ry0 + rh, rx0:rx0 + rw]
    roi_L = lab_L[ry0:ry0 + rh, rx0:rx0 + rw]
    roi_bgr = rect_img[ry0:ry0 + rh, rx0:rx0 + rw]
    roi_b = roi_bgr[:, :, 0].astype(np.float32)
    roi_g = roi_bgr[:, :, 1].astype(np.float32)
    roi_r = roi_bgr[:, :, 2].astype(np.float32)
    roi_warmth = (roi_r - roi_b) + (roi_r - roi_g)

    bg_gray = float(np.percentile(roi_gray, 82))
    bg_sat = float(np.percentile(roi_sat, 50))
    bg_A = float(np.percentile(roi_A, 50))
    bg_warmth = float(np.percentile(roi_warmth, 50))
    sat_thresh = max(12.0, bg_sat + 10.0)
    A_thresh = max(130.5, bg_A + 2.0)
    warmth_thresh = max(14.0, bg_warmth + 12.0)

    # Step 2.1: Multi-cue foreground detection
    # 1. Chrominance & Warmth cue (warm pigment, peach, pink, terracotta, gold, sheer)
    # Requires roi_gray >= 35 to reject black cast shadows where HSV saturation numerically spikes
    cue_chroma = (
        (roi_gray >= 35)
        & ((roi_A >= A_thresh) | (roi_sat >= sat_thresh) | (roi_warmth >= warmth_thresh))
        & (roi_gray < bg_gray - 5)
    )
    # 2. Dark pigment cue (jet-black French tips, cat-eye crimson, dark art)
    # Requires chromatic or saturation separation from the white card so achromatic cast shadows are rejected
    cue_dark = (
        (roi_gray < bg_gray - 32.0)
        & ((roi_sat >= bg_sat + 12.0) | (roi_A >= bg_A + 2.5) | (roi_warmth >= bg_warmth + 14.0))
    )
    # Suppress dark cue in the header logo zone where "Nail BERRIE" is printed
    logo_mask = np.zeros((rh, rw), dtype=bool)
    logo_mask[:int(rh * 0.165), int(rw * 0.22):int(rw * 0.78)] = True
    cue_dark = cue_dark & (~logo_mask)
    # 3. White French tip rim edge gradient and rim shadow (33.jpg):
    gx_L = cv2.Scharr(roi_L, cv2.CV_32F, 1, 0)
    gy_L = cv2.Scharr(roi_L, cv2.CV_32F, 0, 1)
    mag_L = np.sqrt(gx_L**2 + gy_L**2)

    k_bg = max(15, int(min(rw, rh) * 0.04) | 1)
    local_bg = cv2.boxFilter(roi_gray, cv2.CV_32F, (k_bg, k_bg))
    rim_shadow = (roi_gray.astype(np.float32) < local_bg - 6.0)

    pink_base = cue_chroma.copy()
    k_down = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(rh * 0.08))))
    pink_col_span = cv2.dilate(pink_base.astype(np.uint8), k_down)

    cue_white_tip = (mag_L > np.percentile(mag_L, 88)) & rim_shadow & (pink_col_span > 0)

    fg_raw = (cue_chroma | cue_dark | cue_white_tip).astype(np.uint8) * 255

    # Remove thin text strokes ("Nail BERRIE" logo and footer text)
    k_open_text = max(5, int(min(rw, rh) * 0.016) | 1)
    fg_opened = cv2.morphologyEx(
        fg_raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_text, k_open_text)),
    )

    # Step 2.1b: Row valley detection via horizontal projection (y-profile)
    row_h = np.sum(fg_opened > 0, axis=1).astype(np.float32)
    k_smooth = max(21, int(rh * 0.02) | 1)
    row_smooth = cv2.GaussianBlur(row_h.reshape(-1, 1), (1, k_smooth), k_smooth / 4.0).ravel()

    min_dist_row = int(rh * 0.20)
    valleys, props = scipy.signal.find_peaks(-row_smooth, distance=min_dist_row, prominence=rw * 0.02)
    valid_valleys = [
        (v, p) for v, p in zip(valleys, props["prominences"])
        if int(rh * 0.24) < v < int(rh * 0.85)
    ]
    valid_valleys.sort(key=lambda item: item[1], reverse=True)
    chosen_valleys = sorted([v for v, _ in valid_valleys[:2]])

    # Cut fg at row valleys to prevent rows from merging vertically
    fg_cut = fg_opened.copy()
    cut_half = max(2, int(rh * 0.005))
    for v in chosen_valleys:
        fg_cut[max(0, v - cut_half):min(rh, v + cut_half + 1), :] = 0

    fg_closed = cv2.morphologyEx(
        fg_cut, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    rcnts, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: List[Tuple[int, int, int, int]] = []
    for c in rcnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > rw * 0.35 and bh > rh * 0.10:
            rows.append((y, x, bw, bh))
    rows.sort(key=lambda r: r[0])

    raw_nails = []
    for ry, rx, rbw, rbh in rows:
        row_fg = fg_cut[ry:ry + rbh, rx:rx + rbw].copy()
        n_rl, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(row_fg)
        for r_lbl in range(1, n_rl):
            if r_stats[r_lbl, cv2.CC_STAT_HEIGHT] < rbh * 0.35:
                row_fg[r_labels == r_lbl] = 0

        row_fg = cv2.morphologyEx(
            row_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9))
        )
        col_h = np.sum(row_fg > 0, axis=0).astype(np.float32)
        col_smooth = cv2.GaussianBlur(col_h.reshape(-1, 1), (1, 17), 4.0).ravel()

        min_dist_col = max(10, int(rbw * 0.08))
        c_valleys, props = scipy.signal.find_peaks(-col_smooth, distance=min_dist_col, prominence=rbh * 0.02)
        valid_c_valleys = [
            (v, p) for v, p in zip(c_valleys, props["prominences"])
            if int(rbw * 0.08) < v < int(rbw * 0.94)
        ]
        valid_c_valleys.sort(key=lambda item: item[1], reverse=True)
        chosen_c_valleys = sorted([v for v, _ in valid_c_valleys[:4]])

        for v in chosen_c_valleys:
            row_fg[:, max(0, v - 1):min(rbw, v + 2)] = 0

        ncnts, _ = cv2.findContours(row_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        row_nails = []
        for nc in ncnts:
            nx, ny, nbw, nbh = cv2.boundingRect(nc)
            if nbw > rbw * 0.05 and nbh > rbh * 0.35:
                hull_local = cv2.convexHull(nc)
                gx = rx0 + rx + nx
                gy = ry0 + ry + ny
                hull_rect = hull_local + np.array([rx0 + rx, ry0 + ry])
                hull_area = float(cv2.contourArea(hull_rect))
                row_nails.append({
                    "gx": gx,
                    "gy": gy,
                    "bw": nbw,
                    "bh": nbh,
                    "area": hull_area,
                    "aspect": nbh / float(nbw) if nbw > 0 else 1.5,
                    "contour_rect": hull_rect,
                })

        row_nails.sort(key=lambda n: n["gx"])
        if len(row_nails) >= 2:
            gy_tops = [n["gy"] for n in row_nails]
            gy_bots = [n["gy"] + n["bh"] for n in row_nails]
            thumb_inv = is_nail_inverted(row_nails[0]["contour_rect"])
            row_inverted = thumb_inv or (np.std(gy_bots) > np.std(gy_tops) * 1.15)
        else:
            row_inverted = True

        for n in row_nails:
            n["row_inverted"] = bool(row_inverted)

        raw_nails.extend(row_nails)

    # Step 2.1c: Map coordinates back to ORIGINAL image plane via H_inv
    H_inv = np.linalg.inv(H)
    is_identity = np.allclose(H, np.eye(3))

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_nails[:max_slots], start=1):
        cnt_rect = item["contour_rect"]
        force_inv = item.get("row_inverted", None)

        # Refine slot contour in rect_img against directional flash cast shadows
        sx, sy, sbw, sbh = cv2.boundingRect(cnt_rect)
        spad = 8
        sx0 = max(0, sx - spad)
        sy0 = max(0, sy - spad)
        sx1 = min(w_img, sx + sbw + spad)
        sy1 = min(h_img, sy + sbh + spad)
        sp_gray = gray[sy0:sy1, sx0:sx1]
        sp_sat = sat[sy0:sy1, sx0:sx1]
        sp_A = lab_A[sy0:sy1, sx0:sx1]
        sp_bgr = rect_img[sy0:sy1, sx0:sx1]
        sp_b = sp_bgr[:, :, 0].astype(np.float32)
        sp_g = sp_bgr[:, :, 1].astype(np.float32)
        sp_r = sp_bgr[:, :, 2].astype(np.float32)
        sp_warmth = (sp_r - sp_b) + (sp_r - sp_g)

        sp_hull = np.zeros_like(sp_gray, dtype=np.uint8)
        cv2.fillPoly(sp_hull, [cnt_rect.reshape(-1, 2) - np.array([sx0, sy0])], 255)
        sp_ek = max(5, int(min(sbw, sbh) * 0.22) | 1)
        sp_core = cv2.erode(sp_hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sp_ek, sp_ek))) > 0
        if np.sum(sp_core) >= 20:
            c_gray_m = float(np.percentile(sp_gray[sp_core], 50))
            c_sat_p75 = float(np.percentile(sp_sat[sp_core], 75))
            c_A_p75 = float(np.percentile(sp_A[sp_core], 75))
            c_warmth_p75 = float(np.percentile(sp_warmth[sp_core], 75))
            if c_gray_m < 115.0 and c_sat_p75 > 65.0:
                eff_s = max(sat_thresh, bg_sat + 0.72 * (c_sat_p75 - bg_sat))
                eff_a = max(A_thresh, bg_A + 0.72 * (c_A_p75 - bg_A))
                sp_fg = ((sp_A >= eff_a) | (sp_sat >= eff_s)) & (sp_gray < bg_gray - 12)
                sp_fg_u8 = cv2.bitwise_and(sp_fg.astype(np.uint8) * 255, sp_hull)
                sp_fg_u8 = cv2.morphologyEx(
                    sp_fg_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                )
                sp_fg_u8 = cv2.morphologyEx(
                    sp_fg_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                )
                sp_cnts, _ = cv2.findContours(sp_fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if sp_cnts:
                    best_sp = max(sp_cnts, key=cv2.contourArea)
                    if cv2.contourArea(best_sp) > 0.40 * float(sbw * sbh):
                        cnt_rect = cv2.convexHull(best_sp) + np.array([[[sx0, sy0]]])
                        rx_n, ry_n, rbw_n, rbh_n = cv2.boundingRect(cnt_rect)
                        item["gx"], item["gy"], item["bw"], item["bh"] = rx_n, ry_n, rbw_n, rbh_n
                        item["area"] = float(cv2.contourArea(cnt_rect))
                        item["aspect"] = rbh_n / float(rbw_n) if rbw_n > 0 else 1.5
            elif c_warmth_p75 > 15.0 or c_A_p75 > 131.0 or (c_gray_m >= 115.0 and c_sat_p75 > 15.0):
                eff_w = max(16.0, bg_warmth + 10.0)
                eff_a = max(130.5, bg_A + 2.0)
                eff_s = max(12.0, bg_sat + 10.0)
                sp_fg = (
                    (sp_gray >= 50)
                    & ((sp_warmth >= eff_w) | (sp_A >= eff_a) | (sp_sat >= eff_s))
                    & (sp_gray < bg_gray - 5)
                )
                sp_fg_u8 = cv2.bitwise_and(sp_fg.astype(np.uint8) * 255, sp_hull)
                k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(sbw * 0.12) | 1), 3))
                sp_fg_u8 = cv2.morphologyEx(sp_fg_u8, cv2.MORPH_CLOSE, k_bridge)
                sp_fg_u8 = cv2.morphologyEx(
                    sp_fg_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                )
                sp_fg_u8 = cv2.morphologyEx(
                    sp_fg_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                )
                sp_cnts, _ = cv2.findContours(sp_fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if sp_cnts:
                    best_sp = max(sp_cnts, key=cv2.contourArea)
                    if cv2.contourArea(best_sp) > 0.35 * float(sbw * sbh):
                        cnt_rect = cv2.convexHull(best_sp) + np.array([[[sx0, sy0]]])
                        rx_n, ry_n, rbw_n, rbh_n = cv2.boundingRect(cnt_rect)
                        item["gx"], item["gy"], item["bw"], item["bh"] = rx_n, ry_n, rbw_n, rbh_n
                        item["area"] = float(cv2.contourArea(cnt_rect))
                        item["aspect"] = rbh_n / float(rbw_n) if rbw_n > 0 else 1.5

        pca_angle, rotated_cnt, _ = deskew_nail_pca(cnt_rect)
        shape_info = classify_nail_shape(rotated_cnt, force_inverted=force_inv)

        if not is_identity:
            pts_rect = cnt_rect.reshape(-1, 2).astype(np.float64)
            pts_orig = cv2.perspectiveTransform(pts_rect.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
            cnt_final = pts_orig.astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(cnt_final)
            area_final = float(cv2.contourArea(cnt_final))
        else:
            cnt_final = cnt_rect.reshape(-1, 2).astype(np.int32)
            bx, by, bw, bh = int(item["gx"]), int(item["gy"]), int(item["bw"]), int(item["bh"])
            area_final = float(item["area"])

        results.append({
            "slot_id": int(idx),
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "contour": cnt_final.reshape(-1, 2).tolist(),
            "area": float(area_final),
            "aspect_ratio": round(float(item["aspect"]), 2),
            "pca_angle": round(float(pca_angle), 2),
            "shape_info": shape_info,
            "homography": H.tolist(),
            "card_quad": card_quad.tolist() if card_quad is not None else None,
        })

    return results


def dematte_nail_edges(
    patch_rgba: np.ndarray,
    metrics_out: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Removes white backing-card contamination from the nail edge (hybrid-implementation-v2.md, Pillar 1).

    The milky halo is a COLOR problem: the curved plastic bevel mirrors the white card, so edge
    pixels carry card-white RGB that alpha feathering blends over skin. This function:

      1a. Adaptive inward choke: r = clip(round(0.012 * nail_w), 1, 3) px grayscale erosion of alpha.
          Area guard: if the choke removes > 4% of the alpha >= 128 silhouette, r is reduced by 1.
      1b. Soft feather: Gaussian (sigma 0.8 px) on the choked alpha, clamped to never exceed the
          original alpha, so the edge stays soft (needed by the compositor's specular clamp).
      1c. Radial color bleed: every non-core pixel (alpha < 220, INCLUDING alpha == 0) receives the
          color of its nearest solid-core pixel (Euclidean distance transform indices), lightly
          blurred (sigma 1.0 px) to avoid flat bands on glitter/chrome.
          Filling alpha == 0 pixels too means later resampling (deskew rotation, TPS warp, resize)
          can never pull card-white colors back into the edge.

    Solid-core pixels (final alpha >= 220) remain byte-identical to the source photo.

    Args:
        patch_rgba: uint8 BGRA nail cutout.
        metrics_out: optional dict that receives choke_px, area_removed_pct and core_px.

    Returns:
        New uint8 BGRA array with de-matted edges.
    """
    out: np.ndarray = patch_rgba.copy()
    alpha: np.ndarray = out[:, :, 3].astype(np.float32)
    silhouette: np.ndarray = alpha >= 128.0
    area0: int = int(np.sum(silhouette))
    if area0 < 16:
        if metrics_out is not None:
            metrics_out.update({"choke_px": 0, "area_removed_pct": 0.0, "core_px": 0})
        return out

    xs: np.ndarray = np.where(silhouette)[1]
    nail_w: float = float(xs.max() - xs.min() + 1)

    # 1a. Adaptive inward choke with 4% area guard
    choke_px: int = int(np.clip(round(0.012 * nail_w), 1, 3))
    choked: np.ndarray = alpha
    area_removed: float = 0.0
    while choke_px >= 1:
        k_size: int = 2 * choke_px + 1
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        candidate: np.ndarray = cv2.erode(alpha, kernel)
        candidate_removed: float = 1.0 - float(np.sum(candidate >= 128.0)) / float(area0)
        if candidate_removed <= 0.04:
            choked = candidate
            area_removed = candidate_removed
            break
        choke_px -= 1
    if choke_px < 1:
        choke_px = 0
        choked = alpha
        area_removed = 0.0

    # 1b. Soft feather, never exceeding the original alpha
    feathered: np.ndarray = cv2.GaussianBlur(choked, (0, 0), sigmaX=0.8, sigmaY=0.8)
    final_alpha: np.ndarray = np.minimum(feathered, alpha)

    # 1c. Radial color bleed from the solid core
    core: np.ndarray = final_alpha >= 220.0
    if not np.any(core):
        core = final_alpha >= 128.0
    if np.any(core):
        _, (iy, ix) = scipy.ndimage.distance_transform_edt(~core, return_indices=True)
        bled: np.ndarray = out[:, :, :3][iy, ix].astype(np.float32)
        bled_blur: np.ndarray = cv2.GaussianBlur(bled, (0, 0), sigmaX=1.0, sigmaY=1.0)
        non_core: np.ndarray = ~core
        out[:, :, :3][non_core] = np.clip(bled_blur[non_core], 0.0, 255.0).astype(np.uint8)

    out[:, :, 3] = np.clip(final_alpha, 0.0, 255.0).astype(np.uint8)

    if metrics_out is not None:
        metrics_out.update({
            "choke_px": int(choke_px),
            "area_removed_pct": round(100.0 * float(area_removed), 2),
            "core_px": int(np.sum(core)),
        })
    return out


def extract_nail_rgba(
    image: np.ndarray,
    contour: np.ndarray,
    feather_px: float = 1.5,
    deskew: bool = True,
    auto_orient_cuticle_down: bool = True,
    force_inverted: Optional[bool] = None,
    homography: Optional[np.ndarray] = None,
    exact_mask: bool = False,
    metrics_out: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Extracts an isolated, 100% solid, B-spline curve-smoothed RGBA nail cutout
    from the source tray image.
    
    When homography is provided (from perspective rectification), warps the image
    into the rectified fronto-parallel plane at full source resolution and extracts
    the nail there, ensuring true aspect ratio and zero distortion.
    
    Uses data-driven bilateral profile smoothing with monotonic envelopes, then de-mattes
    the edge (dematte_nail_edges) so zero white card background leaks into anti-aliased
    edges, keeping solid interior RGB 100% byte-identical.

    metrics_out (optional) receives the de-matting metrics (choke_px, area_removed_pct, core_px).
    """
    pts = np.array(contour, dtype=np.float64).reshape(-1, 2)

    # If homography is provided and not identity, warp to rectified plane
    if homography is not None and not np.allclose(homography, np.eye(3)):
        h_src, w_src = image.shape[:2]
        corners = np.array(
            [[0.0, 0.0], [float(w_src), 0.0], [float(w_src), float(h_src)], [0.0, float(h_src)]],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        c_rect = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        dst_w = max(int(np.ceil(np.max(c_rect[:, 0]))), 1)
        dst_h = max(int(np.ceil(np.max(c_rect[:, 1]))), 1)
        warped_img = cv2.warpPerspective(
            image, homography, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        pts_rect = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), homography).reshape(-1, 2)
        working_img = warped_img
        working_pts = pts_rect.astype(np.int32)
    else:
        working_img = image
        working_pts = pts.astype(np.int32)

    x, y, w, h = cv2.boundingRect(working_pts)
    if w <= 0 or h <= 0:
        return np.zeros((10, 10, 4), dtype=np.uint8)

    pad = int(max(w, h) * 0.35) + 12
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(working_img.shape[1], x + w + pad)
    y1 = min(working_img.shape[0], y + h + pad)

    patch_bgr = working_img[y0:y1, x0:x1].copy()
    ph, pw = patch_bgr.shape[:2]
    local_pts = working_pts - np.array([x0, y0])

    if exact_mask:
        solid_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(solid_mask, [local_pts], 255)
        active_cnt = local_pts

        b, g, r = cv2.split(patch_bgr)
        patch_rgba = cv2.merge([b, g, r, solid_mask])

        if deskew:
            _, active_cnt, rotated_rgba = deskew_nail_pca(active_cnt, patch_rgba)
            if rotated_rgba is not None:
                patch_rgba = rotated_rgba

        alpha = patch_rgba[:, :, 3]
        ys, xs = np.where(alpha > 12)
        if len(xs) > 0 and len(ys) > 0:
            bpad = 4
            cx0 = max(0, int(np.min(xs)) - bpad)
            cx1 = min(patch_rgba.shape[1], int(np.max(xs)) + 1 + bpad)
            cy0 = max(0, int(np.min(ys)) - bpad)
            cy1 = min(patch_rgba.shape[0], int(np.max(ys)) + 1 + bpad)
            patch_rgba = patch_rgba[cy0:cy1, cx0:cx1]

        should_invert = force_inverted if force_inverted is not None else is_nail_inverted(active_cnt, patch_rgba=patch_rgba)
        if auto_orient_cuticle_down and should_invert:
            patch_rgba = cv2.rotate(patch_rgba, cv2.ROTATE_180)
        elif should_invert and not auto_orient_cuticle_down:
            patch_rgba = cv2.rotate(patch_rgba, cv2.ROTATE_180)

        if feather_px > 0:
            alpha_ch = patch_rgba[:, :, 3].astype(np.float32)
            blur_alpha = cv2.GaussianBlur(alpha_ch, (0, 0), sigmaX=feather_px, sigmaY=feather_px)
            core_mask = cv2.erode((alpha_ch > 200).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            final_alpha = np.where(core_mask > 0, 255.0, blur_alpha)
            patch_rgba[:, :, 3] = np.clip(final_alpha, 0, 255).astype(np.uint8)

        return dematte_nail_edges(patch_rgba, metrics_out=metrics_out)

    hull = cv2.convexHull(local_pts)
    hull_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    hull_mask = cv2.dilate(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    patch_sat = patch_hsv[:, :, 1]
    patch_A = patch_lab[:, :, 1]
    patch_B = patch_lab[:, :, 2]
    patch_L = patch_lab[:, :, 0]

    patch_b = patch_bgr[:, :, 0].astype(np.float32)
    patch_g = patch_bgr[:, :, 1].astype(np.float32)
    patch_r = patch_bgr[:, :, 2].astype(np.float32)
    patch_warmth = (patch_r - patch_b) + (patch_r - patch_g)

    ext_card_pixels = (hull_mask == 0) & (patch_sat < 40) & (patch_gray > 165)
    if np.sum(ext_card_pixels) > 20:
        bg_gray = float(np.percentile(patch_gray[ext_card_pixels], 75))
        bg_sat = float(np.percentile(patch_sat[ext_card_pixels], 50))
        bg_A = float(np.percentile(patch_A[ext_card_pixels], 50))
        bg_warmth = float(np.percentile(patch_warmth[ext_card_pixels], 50))
    else:
        bg_gray = max(210.0, float(np.percentile(patch_gray, 85)))
        bg_sat = 10.0
        bg_A = 128.0
        bg_warmth = 0.5

    sat_thresh = max(18.0, bg_sat + 16.0)
    A_thresh = max(132.0, bg_A + 2.5)
    warmth_thresh = max(14.0, bg_warmth + 10.0)

    # Measure core nail properties inside the central region of hull_mask
    # When the nail is a dark, richly pigmented nail (e.g. burgundy/espresso with core_gray < 110
    # and core_sat > 60), tighten chroma & dark thresholds relative to the nail core so directional
    # flash cast shadows on the white backing card (gray ~ 80-150, sat ~ 40-65) are 100% rejected.
    core_erode_k = max(5, int(min(w, h) * 0.22) | 1)
    core_region = cv2.erode(
        hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_erode_k, core_erode_k))
    ) > 0
    if np.sum(core_region) < 25:
        core_region = hull_mask > 0

    core_gray_med = float(np.percentile(patch_gray[core_region], 50))
    core_sat_p75 = float(np.percentile(patch_sat[core_region], 75))
    core_A_p75 = float(np.percentile(patch_A[core_region], 75))
    core_warmth_p75 = float(np.percentile(patch_warmth[core_region], 75))

    if core_gray_med < 115.0 and core_sat_p75 > 65.0:
        eff_sat_thresh = max(sat_thresh, bg_sat + 0.72 * (core_sat_p75 - bg_sat))
        eff_A_thresh = max(A_thresh, bg_A + 0.72 * (core_A_p75 - bg_A))
        cue_chroma = ((patch_A >= eff_A_thresh) | (patch_sat >= eff_sat_thresh)) & (patch_gray < bg_gray - 12)
        cue_dark = np.zeros_like(cue_chroma, dtype=bool)
    elif core_warmth_p75 > 15.0 or core_A_p75 > 131.0 or (core_gray_med >= 115.0 and core_sat_p75 > 15.0):
        eff_w = max(16.0, bg_warmth + 10.0)
        eff_a = max(130.5, bg_A + 2.0)
        eff_s = max(12.0, bg_sat + 10.0)
        cue_chroma = (
            (patch_gray >= 50)
            & ((patch_warmth >= eff_w) | (patch_A >= eff_a) | (patch_sat >= eff_s))
            & (patch_gray < bg_gray - 5)
        )
        cue_dark = np.zeros_like(cue_chroma, dtype=bool)
    else:
        cue_chroma = ((patch_A >= A_thresh) | (patch_sat >= sat_thresh)) & (patch_gray < bg_gray - 5)
        is_dark_nail = np.percentile(patch_gray[core_region], 25) < 95.0
        if is_dark_nail:
            cue_dark = (
                (patch_gray < bg_gray - 32.0)
                & ((patch_sat >= bg_sat + 12.0) | (patch_A >= bg_A + 2.5) | (patch_warmth >= bg_warmth + 14.0))
            )
        else:
            cue_dark = np.zeros_like(cue_chroma, dtype=bool)

    # Edge gradient and rim shadow for white French tips
    gx_L = cv2.Scharr(patch_L, cv2.CV_32F, 1, 0)
    gy_L = cv2.Scharr(patch_L, cv2.CV_32F, 0, 1)
    mag_L = np.sqrt(gx_L**2 + gy_L**2)

    k_bg = max(9, int(min(pw, ph) * 0.05) | 1)
    local_bg = cv2.boxFilter(patch_gray, cv2.CV_32F, (k_bg, k_bg))
    rim_shadow = patch_gray.astype(np.float32) < local_bg - 6.0

    pink_base = ((patch_A >= A_thresh) | (patch_sat >= sat_thresh)) & (patch_gray < bg_gray - 5)
    k_down = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(ph * 0.08))))
    pink_col_span = cv2.dilate(pink_base.astype(np.uint8), k_down)

    cue_edge = (mag_L > np.percentile(mag_L, 88)) & rim_shadow & (pink_col_span > 0)

    fg_raw = np.where((cue_chroma | cue_dark | cue_edge), 255, 0).astype(np.uint8)
    fg_masked = cv2.bitwise_and(fg_raw, hull_mask)

    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.12) | 1), 3))
    fg_bridged = cv2.morphologyEx(fg_masked, cv2.MORPH_CLOSE, k_bridge)

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_cleaned = cv2.morphologyEx(fg_bridged, cv2.MORPH_OPEN, open_k, iterations=1)
    fg_closed = cv2.morphologyEx(fg_cleaned, cv2.MORPH_CLOSE, close_k, iterations=1)

    ext_cnts, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros((ph, pw), dtype=np.uint8)
    if len(ext_cnts) > 0:
        main_cnt = max(ext_cnts, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
        active_cnt = main_cnt
    else:
        cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
        active_cnt = hull

    erode_iters = 2 if w >= 100 else 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid_mask = cv2.erode(solid_mask, erode_kernel, iterations=erode_iters)

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
    elif should_invert and not auto_orient_cuticle_down:
        temp_mask = cv2.rotate(patch_rgba[:, :, 3], cv2.ROTATE_180)
        temp_sym = build_symmetric_smooth_nail_mask(temp_mask, feather_px=feather_px)
        sym_alpha = cv2.rotate(temp_sym, cv2.ROTATE_180)
    else:
        sym_alpha = build_symmetric_smooth_nail_mask(patch_rgba[:, :, 3], feather_px=feather_px)

    # Step Extra A: Edge de-matting (adaptive choke + soft feather + radial color bleed).
    # Runs after deskew/orientation, and fills every non-core pixel with nearest-core color,
    # so no later resampling can reintroduce card-white edge colors.
    # Solid interior core pixels remain 100% byte-identical to the source photo.
    patch_rgba[:, :, 3] = sym_alpha
    return dematte_nail_edges(patch_rgba, metrics_out=metrics_out)


def refit_slot_contour(image: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
    """
    Refines and extracts the optimal G2-smooth nail contour within a user-specified
    or manually adjusted bounding box [x, y, w, h].
    Allows interactive manual adjustment and localized computer vision re-snapping.
    """
    h_img, w_img = image.shape[:2]
    bx, by, bw, bh = [int(v) for v in bbox]
    pad = 8
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w_img, bx + bw + pad)
    y1 = min(h_img, by + bh + pad)
    roi = image[y0:y1, x0:x1]

    if roi.size == 0 or bw < 10 or bh < 10:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        pts = [[int(cx + (bw / 2.0) * np.cos(t)), int(cy + (bh / 2.0) * np.sin(t))] for t in theta]
        return {
            "bbox": [bx, by, bw, bh],
            "contour": pts,
            "pca_angle": 0.0,
            "shape_info": {"shape": "Oval", "confidence": 0.8, "metrics": {}},
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab_A = lab[:, :, 1]
    b = roi[:, :, 0].astype(np.float32)
    g = roi[:, :, 1].astype(np.float32)
    r = roi[:, :, 2].astype(np.float32)
    warmth = (r - b) + (r - g)

    cue_chroma = (
        (gray >= 50)
        & ((warmth >= 15.0) | (lab_A >= 130.5) | (sat >= 15.0))
        & (gray < np.percentile(gray, 92) - 4)
    )
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(bw * 0.12) | 1), 3))
    cue_b = cv2.morphologyEx(cue_chroma.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_bridge)
    fg_closed = cv2.morphologyEx(cue_b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    cnts, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best_c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_c) > 0.20 * float(bw * bh):
            hull = cv2.convexHull(best_c)
            global_cnt = hull + np.array([[[x0, y0]]])
            bx_n, by_n, bw_n, bh_n = cv2.boundingRect(global_cnt)
            pca_angle, rot_cnt, _ = deskew_nail_pca(global_cnt)
            shape_info = classify_nail_shape(rot_cnt)
            return {
                "bbox": [int(bx_n), int(by_n), int(bw_n), int(bh_n)],
                "contour": global_cnt.reshape(-1, 2).tolist(),
                "pca_angle": round(float(pca_angle), 2),
                "shape_info": shape_info,
            }

    cx, cy = bx + bw / 2.0, by + bh / 2.0
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    pts = [[int(cx + (bw * 0.48) * np.cos(t)), int(cy + (bh * 0.49) * np.sin(t))] for t in theta]
    cnt_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    pca_angle, rot_cnt, _ = deskew_nail_pca(cnt_arr)
    shape_info = classify_nail_shape(rot_cnt)
    return {
        "bbox": [bx, by, bw, bh],
        "contour": pts,
        "pca_angle": round(float(pca_angle), 2),
        "shape_info": shape_info,
    }
