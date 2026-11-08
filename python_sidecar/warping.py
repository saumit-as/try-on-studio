"""
Press-On Nail Studio - Warping Engine
Implements:
  - 3-tier SciPy Thin-Plate Spline (TPS) transformation
  - Transverse cylindrical C-curve barrel distortion
  - Aspect-ratio & tip-taper preserving extension along the finger axis vector
  - Anatomical U-shaped Nail-Bed & Cuticle Skin-Fold Tuck Mask
"""

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Dict, Any, Optional


def apply_cylindrical_c_curve(
    image: np.ndarray,
    c_curve_intensity: float = 0.65,
    roll_offset: float = 0.0,
) -> np.ndarray:
    """
    Applies transverse non-linear barrel foreshortening across the nail's interior width
    to simulate 3D curvature wrapping around the cylindrical nail plate, plus
    3D lateral yaw/roll perspective foreshortening when roll_offset != 0
    (e.g. for a thumb viewed in 3/4 side profile).

    Crucially, for frontal nails (roll_offset == 0), the cylindrical mapping is normalized
    per-scanline across each row's active nail span [x_left(y), x_right(y)]. Because
    arcsin(u * sin(max_angle)) / max_angle maps -1 -> -1 and +1 -> +1 at the scanline
    boundaries, the outer 2D silhouette of the nail remains 100% invariant (zero capsule
    bloating of tapered almond/oval sidewalls), while interior nail art curves in 3D.
    """
    h, w = image.shape[:2]
    if w < 4 or h < 4 or (c_curve_intensity <= 0.01 and abs(roll_offset) <= 0.01):
        return image.copy()

    max_angle = (np.pi / 2.0) * min(0.95, max(0.05, c_curve_intensity))
    sin_max = float(np.sin(max_angle))

    # Per-scanline silhouette-preserving cylindrical C-curve mapping
    x_grid = np.arange(w, dtype=np.float32)[None, :]
    map_x = np.tile(x_grid, (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    has_alpha = image.shape[2] == 4
    if has_alpha:
        alpha = image[:, :, 3]
        centers = np.full(h, (w - 1) * 0.5, dtype=np.float32)
        radii = np.full(h, (w - 1) * 0.5, dtype=np.float32)
        valid_rows = np.zeros(h, dtype=bool)
        for y in range(h):
            xs = np.where(alpha[y] > 12)[0]
            if len(xs) >= 2:
                x0 = float(xs[0])
                x1 = float(xs[-1])
                centers[y] = 0.5 * (x0 + x1)
                radii[y] = max(2.0, 0.5 * (x1 - x0))
                valid_rows[y] = True

        if np.any(valid_rows):
            # Smooth centers and radii vertically so interior UV lines remain C2 smooth
            centers = cv2.GaussianBlur(centers.reshape(-1, 1), (1, 15), 3.0).ravel()
            radii = cv2.GaussianBlur(radii.reshape(-1, 1), (1, 15), 3.0).ravel()

            u = (x_grid - centers[:, None]) / radii[:, None]
            inside = np.abs(u) < 1.0
            u_clamped = np.clip(u, -0.9999, 0.9999)
            u_tex = np.arcsin(u_clamped * sin_max) / max_angle
            map_x = np.where(inside, centers[:, None] + u_tex * radii[:, None], x_grid)

    # Global 3D lateral yaw/roll perspective foreshortening for 3/4 profile thumb
    if abs(roll_offset) > 0.10:
        # 3D Horizon Occlusion: thumbnail turned in lateral profile has its far 24% occluded behind nail barrel
        u_occ = float(np.clip(0.24 * (abs(roll_offset) / 0.35), 0.08, 0.32))
        theta_max = 0.5 * np.pi * max(0.6, min(1.0, c_curve_intensity * 1.3))
        x_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)
        if roll_offset < 0:
            # Left side is far horizon, turning away from camera
            theta = (1.0 - x_lin) * theta_max
            x_proj = (1.0 - np.cos(theta)) / max(1e-5, (1.0 - np.cos(theta_max)))
            x_vis = 1.0 - x_proj
            src_u = u_occ + x_vis * (1.0 - u_occ)
        else:
            # Right side is far horizon
            theta = x_lin * theta_max
            x_proj = (1.0 - np.cos(theta)) / max(1e-5, (1.0 - np.cos(theta_max)))
            src_u = x_proj * (1.0 - u_occ)

        map_x = np.clip(src_u * float(w - 1), 0.0, float(w - 1))
        map_x = np.tile(map_x[None, :], (h, 1)).astype(np.float32)
    elif abs(roll_offset) > 0.01:
        x_norm = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        gamma = float(np.power(2.0, -roll_offset * 1.35))
        x_rolled = np.power(np.clip(x_norm, 0.0, 1.0), gamma) * (w - 1)
        # Re-interpolate map_x through the lateral roll curve
        map_x = np.clip(x_rolled, 0.0, float(w - 1))
        if c_curve_intensity > 0.01:
            u_r = (map_x / float(w - 1)) * 2.0 - 1.0
            u_r_tex = np.arcsin(np.clip(u_r, -0.9999, 0.9999) * sin_max) / max_angle
            map_x = (u_r_tex + 1.0) * 0.5 * float(w - 1)
        map_x = np.tile(map_x, (h, 1)).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if has_alpha else (0, 0, 0),
    )
    if has_alpha and abs(roll_offset) > 0.10:
        # Crisp 1.0px anti-aliased drop-off at the 3D silhouette horizon
        if roll_offset < 0:
            warped[:, 0, 3] = np.clip(warped[:, 0, 3].astype(np.float32) * 0.70, 0, 255).astype(np.uint8)
        else:
            warped[:, -1, 3] = np.clip(warped[:, -1, 3].astype(np.float32) * 0.70, 0, 255).astype(np.uint8)
    return warped


def build_3tier_control_grid(
    src_w: float,
    src_h: float,
    cuticle_left: Tuple[float, float],
    cuticle_center: Tuple[float, float],
    cuticle_right: Tuple[float, float],
    distal_left: Tuple[float, float],
    distal_right: Tuple[float, float],
    length_scale: float = 1.0,
    tip_taper: float = 1.0,
    roll_offset: float = 0.0,
    tray_to_hand_scale: Optional[float] = None,
    width_scale: float = 1.0,
    axial_shift: float = 0.0,
    angle_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs an Orthogonal, Silhouette-Preserving 3-tier (9-point) control grid
    calibrated against real worn press-on nail references:
      1. Strictly Orthogonal Frame (`perp_unit` perpendicular to `axis_unit`):
         - Eliminates parallelogram shear/skew when `d_r - d_l` is slightly angled,
           so the nail's bilateral symmetry around the finger axis is 100% preserved.
      2. Linear Longitudinal Mapping (`Row 1` at `y = src_h` -> `0%`, `Row 2` at `0.50 * src_h`
         -> `50%`, `Row 3` at `0.0` -> `100%`):
         - Keeps the upper nail body rigid so the true ovoid/almond equator placement and
           smooth tip taper from the extracted nail are never inflated into a capsule.
      3. Preserves Each Nail's True Relative Length from the Tray Image (`tray_to_hand_scale`).
      4. Per-Finger Custom Fitting:
         - `length_scale`: Scales longitudinal tip extension length.
         - `width_scale`: Scales transverse nail width / bed coverage.
         - `axial_shift`: Longitudinal nudge along finger axis (deeper cuticle tuck or tip shift).
         - `angle_offset`: Rotational angle alignment around cuticle center.

    Returns:
      (src_points [9, 2], dst_points [9, 2])
    """
    pad_x = min(6.0, src_w * 0.05)
    pad_y = min(5.5, src_h * 0.025)
    src_pts = np.array([
        # Row 1: Cuticle base frame (aligned to the true active bounds of the nail mask inside the 6px border)
        [pad_x, src_h - pad_y],
        [src_w * 0.5, src_h - pad_y],
        [src_w - pad_x, src_h - pad_y],
        # Row 2: Mid-body line (y = 0.50 * src_h maps to 50% along extension_dist — zero vertical stretch!)
        [pad_x, src_h * 0.50],
        [src_w * 0.5, src_h * 0.50],
        [src_w - pad_x, src_h * 0.50],
        # Row 3: Free-edge tip frame (aligned to the true top of the nail mask inside the 6px border)
        [pad_x, pad_y],
        [src_w * 0.5, pad_y],
        [src_w - pad_x, pad_y],
    ], dtype=np.float64)

    c_l = np.array(cuticle_left, dtype=np.float64)
    c_c = np.array(cuticle_center, dtype=np.float64)
    c_r = np.array(cuticle_right, dtype=np.float64)

    d_l = np.array(distal_left, dtype=np.float64)
    d_r = np.array(distal_right, dtype=np.float64)

    # Finger centerline axis vector (from cuticle center through distal midpoint)
    distal_mid = (d_l + d_r) * 0.5
    axis_vector = distal_mid - c_c
    bed_length = float(np.linalg.norm(axis_vector))
    if bed_length > 0:
        axis_unit = axis_vector / bed_length
    else:
        axis_unit = np.array([0.0, -1.0], dtype=np.float64)

    # Strictly orthogonal transverse unit vector (eliminates shear distortion)
    perp_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    if np.dot(perp_unit, d_r - d_l) < 0:
        perp_unit = -perp_unit

    # Apply rotational angle offset (degrees) around c_c if non-zero
    if abs(angle_offset) > 0.01:
        rad = np.deg2rad(float(angle_offset))
        cos_t = np.cos(rad)
        sin_t = np.sin(rad)
        rot_axis = cos_t * axis_unit + sin_t * perp_unit
        axis_unit = rot_axis / max(1e-6, np.linalg.norm(rot_axis))
        perp_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
        if np.dot(perp_unit, d_r - d_l) < 0:
            perp_unit = -perp_unit

    # Apply longitudinal axial shift along axis_unit
    if abs(axial_shift) > 0.01:
        shift_vec = axis_unit * float(axial_shift)
        c_c = c_c + shift_vec
        c_l = c_l + shift_vec
        c_r = c_r + shift_vec

    distal_width = float(np.linalg.norm(d_r - d_l))
    cuticle_width = float(np.linalg.norm(c_r - c_l))
    bed_width = max(distal_width, cuticle_width)

    # Nail-bed half-width adjusted by width_scale
    w_mult = max(0.4, min(2.5, float(width_scale)))
    frame_half_w = bed_width * 0.495 * w_mult
    distal_half_w = min(frame_half_w, distal_width * 0.50 * w_mult)

    # Preserve each nail's true length from the tray image
    user_mult = max(0.2, min(3.0, float(length_scale))) if length_scale is not None else 1.0
    if tray_to_hand_scale is not None and tray_to_hand_scale > 0:
        extension_dist = float(src_h) * tray_to_hand_scale * user_mult
    else:
        nail_aspect = float(src_h) / max(1.0, float(src_w))
        auto_total_len = (2.0 * frame_half_w) * nail_aspect
        if length_scale is not None and length_scale > 1.25:
            extension_dist = max(bed_length * user_mult, auto_total_len)
        else:
            extension_dist = max(bed_length * 1.34 * user_mult, auto_total_len)

    # Row 2 placed at distal nail-bed crease (s = 0.42 * extension_dist) matching distal_width * 0.50
    apex_center = c_c + axis_unit * (extension_dist * 0.42)
    tip_center = c_c + axis_unit * extension_dist
    tip_half_w = frame_half_w * tip_taper

    if abs(roll_offset) > 0.10:
        # 3/4 lateral profile thumb: horizon side shifted inward by 0.72, near side by 0.95
        if roll_offset < 0:
            # Horizon on left, near skin on right
            base_left = c_c - perp_unit * (frame_half_w * 0.72)
            base_right = c_c + perp_unit * (frame_half_w * 0.95)
            apex_left = apex_center - perp_unit * (distal_half_w * 0.72)
            apex_right = apex_center + perp_unit * (distal_half_w * 0.95)
            tip_left = tip_center - perp_unit * (tip_half_w * 0.72)
            tip_right = tip_center + perp_unit * (tip_half_w * 0.95)
        else:
            # Horizon on right, near skin on left
            base_left = c_c - perp_unit * (frame_half_w * 0.95)
            base_right = c_c + perp_unit * (frame_half_w * 0.72)
            apex_left = apex_center - perp_unit * (distal_half_w * 0.95)
            apex_right = apex_center + perp_unit * (distal_half_w * 0.72)
            tip_left = tip_center - perp_unit * (tip_half_w * 0.95)
            tip_right = tip_center + perp_unit * (tip_half_w * 0.72)
    else:
        # Strictly orthogonal rectangle aligned to (axis_unit, perp_unit)
        base_left = c_c - perp_unit * frame_half_w
        base_right = c_c + perp_unit * frame_half_w
        apex_left = apex_center - perp_unit * distal_half_w
        apex_right = apex_center + perp_unit * distal_half_w
        tip_left = tip_center - perp_unit * tip_half_w
        tip_right = tip_center + perp_unit * tip_half_w

    dst_pts = np.array([
        # Row 1: Cuticle base frame (s = 0.0)
        base_left,
        c_c,
        base_right,
        # Row 2: Mid-body line (s = 0.50)
        apex_left,
        apex_center,
        apex_right,
        # Row 3: Extended Tip Frame (s = 1.00)
        tip_left,
        tip_center,
        tip_right,
    ], dtype=np.float64)

    return src_pts, dst_pts


def build_cuticle_pocket_mask(
    roi_bbox: Tuple[int, int, int, int],
    cuticle_left: Tuple[float, float],
    cuticle_center: Tuple[float, float],
    cuticle_right: Tuple[float, float],
    distal_left: Tuple[float, float],
    distal_right: Tuple[float, float],
    feather_sigma: float = 1.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a G2-smooth anatomical U-curve mask along the proximal cuticle fold
    and lateral nail grooves (peronychium) so that:
      1. The cuticle U-cup transitions with a strictly parallel tangent into the left
         and right sidewalls (zero polygonal corners or 45-degree chops at c_l / c_r).
      2. A subtle 1.5px-2.5px contact shadow (ambient occlusion) is generated right at
         the cuticle pocket where the nail tucks under the proximal skin lip.

    Returns:
      (pocket_allow_mask [roi_h, roi_w] float32 in [0, 1],
       cuticle_ao_map [roi_h, roi_w] float32 in [0.75, 1.0])
    """
    rx, ry, rw, rh = roi_bbox
    if rw <= 0 or rh <= 0:
        return np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)

    # Local coordinates relative to the ROI origin (rx, ry)
    c_l = np.array([cuticle_left[0] - rx, cuticle_left[1] - ry], dtype=np.float64)
    c_c = np.array([cuticle_center[0] - rx, cuticle_center[1] - ry], dtype=np.float64)
    c_r = np.array([cuticle_right[0] - rx, cuticle_right[1] - ry], dtype=np.float64)
    d_l = np.array([distal_left[0] - rx, distal_left[1] - ry], dtype=np.float64)
    d_r = np.array([distal_right[0] - rx, distal_right[1] - ry], dtype=np.float64)

    distal_mid = (d_l + d_r) * 0.5
    axis_vec = distal_mid - c_c
    bed_len = max(1.0, float(np.linalg.norm(axis_vec)))
    axis_unit = axis_vec / bed_len
    perp_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    if np.dot(perp_unit, d_r - d_l) < 0:
        perp_unit = -perp_unit

    distal_w = float(np.linalg.norm(d_r - d_l))
    cuticle_w = float(np.linalg.norm(c_r - c_l))
    bed_w = max(distal_w, cuticle_w)

    # Flatter squoval U-cup with s_eq = 0.18 * bed_len matching natural squoval cuticle contours
    pocket_half_w = cuticle_w * 0.505
    s_eq = 0.18 * bed_len

    # Squoval U-cup from left equator (theta = pi) through c_c (theta = pi/2) to right equator (theta = 0)
    thetas = np.linspace(np.pi, 0.0, 45, dtype=np.float64)
    eq_center = c_c + axis_unit * s_eq
    sin_term = np.power(np.maximum(0.0, np.sin(thetas)), 0.75)
    u_cup_pts = (
        eq_center[None, :]
        + (np.cos(thetas) * pocket_half_w)[:, None] * perp_unit[None, :]
        - (sin_term * s_eq)[:, None] * axis_unit[None, :]
    )

    # Above s_eq, flare smoothly outward so the upper nail body and free-edge tip are 100% unclipped
    far_right_1 = c_c + axis_unit * (bed_len * 0.85) + perp_unit * (bed_w * 0.65)
    far_right_2 = c_c + axis_unit * (bed_len * 2.20) + perp_unit * (bed_w * 0.80)
    far_left_2 = c_c + axis_unit * (bed_len * 2.20) - perp_unit * (bed_w * 0.80)
    far_left_1 = c_c + axis_unit * (bed_len * 0.85) - perp_unit * (bed_w * 0.65)

    poly_pts = np.vstack([
        u_cup_pts,
        [far_right_1, far_right_2, far_left_2, far_left_1],
    ]).astype(np.int32)

    # Render at 4x supersampling for sub-pixel smooth cuticle & sidewall boundaries
    ss = 4
    mask_ss = np.zeros((rh * ss, rw * ss), dtype=np.uint8)
    cv2.fillPoly(mask_ss, [(poly_pts * ss).reshape(-1, 1, 2)], 255)
    pocket_mask_u8 = cv2.resize(mask_ss, (rw, rh), interpolation=cv2.INTER_AREA)

    pocket_allow = pocket_mask_u8.astype(np.float32) / 255.0
    if feather_sigma > 0:
        ksize = int(np.ceil(feather_sigma * 4)) | 1
        pocket_allow = cv2.GaussianBlur(pocket_allow, (ksize, ksize), feather_sigma)

    # Generate subtle Cuticle & Sidewall Pocket Ambient Occlusion (AO) shadow
    # along the lower U-cup curve so the nail looks naturally tucked into the proximal fold
    cuticle_line_ss = np.zeros((rh * ss, rw * ss), dtype=np.uint8)
    groove_pts = u_cup_pts[6:-6]
    cv2.polylines(
        cuticle_line_ss,
        [(groove_pts * ss).astype(np.int32).reshape(-1, 1, 2)],
        isClosed=False,
        color=255,
        thickness=int(2.5 * ss),
        lineType=cv2.LINE_AA,
    )
    groove_u8 = cv2.resize(cuticle_line_ss, (rw, rh), interpolation=cv2.INTER_AREA)
    groove_blur = cv2.GaussianBlur(groove_u8.astype(np.float32) / 255.0, (9, 9), 2.2)

    # Darken by up to 18% right at the cuticle/sidewall tuck line
    cuticle_ao = np.clip(1.0 - 0.18 * groove_blur, 0.80, 1.0)

    return pocket_allow, cuticle_ao


def warp_nail_tps(
    nail_rgba: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Deforms a nail texture onto destination coordinates using SciPy's Thin-Plate Spline
    (RBFInterpolator with kernel='thin_plate_spline').

    Optimized for high performance:
      - Computes destination bounding box with generous margin so curved edges never clip.
      - Evaluates inverse TPS only within the local target ROI.
      - Uses cv2.remap with INTER_CUBIC for smooth, anti-aliased edges.

    Returns:
      (warped_rgba_roi, (x0, y0, roi_w, roi_h))
    """
    h_out, w_out = output_shape[:2]

    # Destination bounding box with 16px padding so curved cuticle and sidewalls are never clipped
    pad = 16
    min_x = max(0, int(np.floor(np.min(dst_pts[:, 0]))) - pad)
    max_x = min(w_out, int(np.ceil(np.max(dst_pts[:, 0]))) + pad)
    min_y = max(0, int(np.floor(np.min(dst_pts[:, 1]))) - pad)
    max_y = min(h_out, int(np.ceil(np.max(dst_pts[:, 1]))) + pad)

    roi_w = max_x - min_x
    roi_h = max_y - min_y

    if roi_w <= 0 or roi_h <= 0:
        return np.zeros((10, 10, 4), dtype=np.uint8), (0, 0, 0, 0)

    # Inverse mapping: for every pixel in destination ROI, find source (x, y)
    rbf = RBFInterpolator(dst_pts, src_pts, kernel="thin_plate_spline")

    grid_y, grid_x = np.mgrid[min_y:max_y, min_x:max_x]
    eval_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mapped_src = rbf(eval_pts)

    map_x = mapped_src[:, 0].reshape(roi_h, roi_w).astype(np.float32)
    map_y = mapped_src[:, 1].reshape(roi_h, roi_w).astype(np.float32)

    # Remap color channels with INTER_CUBIC and alpha with INTER_LINEAR for crisp, smooth edges
    warped_roi = cv2.remap(
        nail_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return warped_roi, (min_x, min_y, roi_w, roi_h)
