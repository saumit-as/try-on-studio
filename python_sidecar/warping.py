"""
Press-On Nail Studio - Warping Engine
Implements:
  - 3-tier SciPy Thin-Plate Spline (TPS) transformation
  - Transverse cylindrical C-curve barrel distortion
  - Aspect-ratio preserving tip extension along the finger axis vector
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
    Applies transverse non-linear barrel compression across the nail's width
    to simulate the 3D curvature wrapping around the cylindrical nail plate.
    
    Parameters:
      image: Source image (BGR or BGRA).
      c_curve_intensity: Float in [0.0, 1.0], controls depth of curvature.
      roll_offset: Float in [-0.5, 0.5], shifts the apex ridge for lateral finger roll.
    """
    h, w = image.shape[:2]
    if w < 4 or h < 4 or c_curve_intensity <= 0.01:
        return image.copy()

    # Normalized x coordinates across width: -1.0 to 1.0
    x_lin = np.linspace(-1.0, 1.0, w) - roll_offset
    x_clamped = np.clip(x_lin, -0.999, 0.999)

    # Physical cylinder projection angle
    max_angle = (np.pi / 2.0) * min(0.95, max(0.05, c_curve_intensity))
    phi = np.arcsin(x_clamped * np.sin(max_angle))
    u_tex = phi / max_angle  # normalized texture coordinate in [-1, 1]

    # Map back to source pixel coordinates [0, w-1]
    src_x = (u_tex + 1.0) * 0.5 * (w - 1)

    map_x = np.tile(src_x, (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h)[:, None], (1, w)).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
    )
    return warped


def build_3tier_control_grid(
    src_w: float,
    src_h: float,
    cuticle_left: Tuple[float, float],
    cuticle_center: Tuple[float, float],
    cuticle_right: Tuple[float, float],
    distal_left: Tuple[float, float],
    distal_right: Tuple[float, float],
    length_scale: float = 1.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a 3-tier, 9-point TPS control grid:
      - Row 1 (Cuticle Fold): Left sidewall, Cuticle apex, Right sidewall.
      - Row 2 (Apex Arch at ~40%): Longitudinal arch pitch and sidewall width.
      - Row 3 (Extended Tip): Projected along the central finger axis scaled by length_scale.
      
    Returns:
      (src_points [9, 2], dst_points [9, 2])
    """
    # 1. Source grid on the flat, normalized nail texture
    # Row 1 (Cuticle at bottom, y = src_h): left, center, right
    # Row 2 (Apex at y = 0.4 * src_h): left, center, right
    # Row 3 (Tip at top, y = 0): left, center, right
    src_pts = np.array([
        # Row 1: Cuticle base
        [0.0, src_h],
        [src_w * 0.5, src_h],
        [src_w, src_h],
        # Row 2: Mid-body apex (~40% from cuticle = 0.6 * src_h)
        [0.0, src_h * 0.6],
        [src_w * 0.5, src_h * 0.6],
        [src_w, src_h * 0.6],
        # Row 3: Free edge / tip
        [0.0, 0.0],
        [src_w * 0.5, 0.0],
        [src_w, 0.0],
    ], dtype=np.float64)

    # 2. Destination landmarks on the finger
    c_l = np.array(cuticle_left, dtype=np.float64)
    c_c = np.array(cuticle_center, dtype=np.float64)
    c_r = np.array(cuticle_right, dtype=np.float64)

    d_l = np.array(distal_left, dtype=np.float64)
    d_r = np.array(distal_right, dtype=np.float64)

    # Finger centerline axis vector (from cuticle center to distal midpoint)
    distal_mid = (d_l + d_r) * 0.5
    axis_vector = distal_mid - c_c
    bed_length = np.linalg.norm(axis_vector)
    if bed_length > 0:
        axis_unit = axis_vector / bed_length
    else:
        axis_unit = np.array([0.0, -1.0])

    # Extension tip coordinates
    # Extended along the axis vector proportionally to length_scale
    extension_dist = bed_length * length_scale
    tip_center = c_c + axis_unit * extension_dist

    # Sidewall vectors for tip width
    # Orthogonal to axis vector
    perp_unit = np.array([-axis_unit[1], axis_unit[0]])
    tip_half_w = np.linalg.norm(d_r - d_l) * 0.45  # slightly tapered at tip
    tip_left = tip_center - perp_unit * tip_half_w
    tip_right = tip_center + perp_unit * tip_half_w

    # Apex arch row (~40% of the bed length from cuticle)
    apex_center = c_c + axis_unit * (bed_length * 0.42)
    apex_left = c_l + (d_l - c_l) * 0.42
    apex_right = c_r + (d_r - c_r) * 0.42

    dst_pts = np.array([
        # Row 1: Cuticle
        c_l,
        c_c,
        c_r,
        # Row 2: Apex
        apex_left,
        apex_center,
        apex_right,
        # Row 3: Tip
        tip_left,
        tip_center,
        tip_right,
    ], dtype=np.float64)

    return src_pts, dst_pts


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
      - Computes destination bounding box.
      - Evaluates inverse TPS only within the local target ROI.
      - Uses cv2.remap for hardware-accelerated sampling.
      
    Returns:
      (warped_rgba_roi, (x0, y0, roi_w, roi_h))
    """
    h_out, w_out = output_shape[:2]

    # Destination bounding box with 6px padding for anti-aliasing
    min_x = max(0, int(np.floor(np.min(dst_pts[:, 0]))) - 6)
    max_x = min(w_out, int(np.ceil(np.max(dst_pts[:, 0]))) + 6)
    min_y = max(0, int(np.floor(np.min(dst_pts[:, 1]))) - 6)
    max_y = min(h_out, int(np.ceil(np.max(dst_pts[:, 1]))) + 6)

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

    # Remap texture
    warped_roi = cv2.remap(
        nail_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return warped_roi, (min_x, min_y, roi_w, roi_h)
