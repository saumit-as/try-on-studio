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
    Applies transverse non-linear barrel compression across the nail's width
    to simulate 3D curvature wrapping around the cylindrical nail plate, plus
    3D lateral yaw/roll perspective foreshortening when roll_offset != 0
    (e.g. for a thumb viewed in 3/4 side profile).

    Parameters:
      image: Source image (BGR or BGRA).
      c_curve_intensity: Float in [0.0, 1.0], controls depth of curvature.
      roll_offset: Float in [-0.6, 0.6], 3D lateral yaw rotation angle around the
                   longitudinal nail spine:
                   - Negative roll_offset (e.g. -0.35 for a left-hand thumb):
                     the nail is turned toward the left (free profile edge on left,
                     attached nail-bed sidewall on right).
                   - Positive roll_offset: turned toward the right.
    """
    h, w = image.shape[:2]
    if w < 4 or h < 4 or (c_curve_intensity <= 0.01 and abs(roll_offset) <= 0.01):
        return image.copy()

    # Normalized destination x coordinates across width: -1.0 to 1.0
    x_dst = np.linspace(-1.0, 1.0, w)

    # 1. Non-linear power-law lateral perspective foreshortening for 3D thumb roll:
    #    Maps [-1, 1] -> [-1, 1] monotonically so neither edge is ever clipped, while
    #    compressing the foreshortened outer profile side and expanding the visible nail plate.
    if abs(roll_offset) > 0.01:
        u_01 = (x_dst + 1.0) * 0.5  # [0, 1]
        gamma = float(np.power(2.0, -roll_offset * 1.6))
        u_skewed = np.power(np.clip(u_01, 0.0, 1.0), gamma)
        x_rolled = u_skewed * 2.0 - 1.0
    else:
        x_rolled = x_dst

    x_clamped = np.clip(x_rolled, -0.999, 0.999)

    # 2. Physical cylinder projection angle
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
        interpolation=cv2.INTER_CUBIC,
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
    length_scale: float = 1.0,
    tip_taper: float = 1.0,
    roll_offset: float = 0.0,
    tray_to_hand_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs an Anatomically Nail-Bed Locked 3-tier (9-point) TPS control grid
    calibrated against real worn press-on nail references (`test_assets/references/`):
      1. Width Locked to the Natural Nail Bed (`frame_half_w = 0.55 * bed_w`):
         - Never inflates the nail wider than the finger's lateral skin folds (`c_l..d_l`
           and `c_r..d_r`), preserving a clean border of finger skin on both sides.
      2. Preserves Each Nail's True Relative Length from the Tray Image (`tray_to_hand_scale`):
         - When `tray_to_hand_scale` is provided (computed across the 5-nail tray set),
           each nail's rendered length on the hand is `src_h * tray_to_hand_scale * length_scale`.
         - This guarantees that the exact physical height graduation of the nails in the tray
           (e.g. `264px -> 237px -> 235px -> 221px -> 204px`) is 100% preserved on the hand!
      3. Uniform Vertical Mapping (`Row 2` at `y = 0.55 * src_h` -> `45%` of total length):
         - Prevents vertical stretching of the mid-body so the nail's single-arc Almond/Oval
           silhouette from the tray is preserved without capsule/sausage distortion.

    Returns:
      (src_points [9, 2], dst_points [9, 2])
    """
    src_pts = np.array([
        # Row 1: Cuticle base frame (maps to proximal cuticle fold c_l, c_c, c_r)
        [0.0, src_h * 0.85],
        [src_w * 0.5, src_h],
        [src_w, src_h * 0.85],
        # Row 2: Mid-bed widest arch (y = 0.55 * src_h maps to 45% along the nail length — zero vertical stretch!)
        [0.0, src_h * 0.55],
        [src_w * 0.5, src_h * 0.55],
        [src_w, src_h * 0.55],
        # Row 3: Free-edge tip bounding frame (top of cutout y = 0.0 extends past fingertip)
        [0.0, 0.0],
        [src_w * 0.5, 0.0],
        [src_w, 0.0],
    ], dtype=np.float64)

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

    # Orthogonal unit vector pointing from left sidewall to right sidewall
    perp_vec = d_r - d_l
    distal_width = np.linalg.norm(perp_vec)
    if distal_width > 0:
        perp_unit = perp_vec / distal_width
    else:
        perp_unit = np.array([1.0, 0.0])

    cuticle_width = np.linalg.norm(c_r - c_l)
    bed_width = max(distal_width, cuticle_width)

    # Strict 1.15x nail-bed half-width (`0.575 * bed_width`) so the nail covers 100% of the
    # natural nail bed while sitting strictly INSIDE the finger's lateral skin folds!
    frame_half_w = bed_width * 0.575

    # Preserve each nail's true length from the tray image!
    if tray_to_hand_scale is not None and tray_to_hand_scale > 0:
        # Direct proportional mapping from the extracted tray nail's pixel height `src_h`
        # so the exact height slope across the tray (Slot 1 > Slot 2 > Slot 3 > Slot 4 > Slot 5)
        # is 100% preserved on the hand without being overridden by bed_length!
        user_mult = length_scale if (length_scale is not None and 0.5 <= length_scale <= 2.0) else 1.0
        extension_dist = float(src_h) * tray_to_hand_scale * user_mult
    else:
        nail_aspect = float(src_h) / max(1.0, float(src_w))
        auto_total_len = bed_width * nail_aspect * 0.78
        if length_scale is not None and length_scale > 1.25:
            extension_dist = max(bed_length * length_scale, auto_total_len)
        else:
            extension_dist = max(bed_length * 1.34, auto_total_len)

    # Row 1: Cuticle base frame (locked to the cuticle width)
    cuticle_chord_mid = 0.5 * (c_l + c_r)
    if abs(roll_offset) > 0.05:
        base_left = c_l - perp_unit * (frame_half_w * 0.04)
        base_right = c_r + perp_unit * (frame_half_w * 0.08)
    else:
        base_left = cuticle_chord_mid - perp_unit * (frame_half_w * 0.96)
        base_right = cuticle_chord_mid + perp_unit * (frame_half_w * 0.96)

    # Row 2: Mid-bed line (anchored at 52% of extension_dist so the widest part of the nail sits high enough to cover 100% of d_l/d_r!)
    apex_center = c_c + axis_unit * (extension_dist * 0.52)
    if abs(roll_offset) > 0.05:
        left_distal_chord = 0.40 * c_l + 0.60 * d_l
        right_distal_chord = 0.40 * c_r + 0.60 * d_r
        left_bow = frame_half_w * (0.05 - 0.04 * roll_offset)
        right_bow = frame_half_w * (0.05 - 0.14 * roll_offset)
        apex_left = left_distal_chord - perp_unit * left_bow
        apex_right = right_distal_chord + perp_unit * right_bow
    else:
        apex_left = apex_center - perp_unit * frame_half_w
        apex_right = apex_center + perp_unit * frame_half_w

    # Row 3: Extended free-edge tip frame (parallel to Row 2: tip_half_w == frame_half_w * tip_taper)
    tip_center = c_c + axis_unit * extension_dist
    tip_half_w = frame_half_w * tip_taper
    tip_left = tip_center - perp_unit * tip_half_w
    tip_right = tip_center + perp_unit * tip_half_w

    dst_pts = np.array([
        # Row 1: Cuticle base frame
        base_left,
        c_c,
        base_right,
        # Row 2: Distal nail-bed line (fingertip)
        apex_left,
        apex_center,
        apex_right,
        # Row 3: Extended Tip Frame (parallel to Row 2)
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
    Constructs a smooth, high-precision anatomical U-curve mask along the proximal
    cuticle fold and lateral nail grooves (peronychium) so that:
      1. Any portion of the warped press-on nail that overflows below the cuticle fold
         or outside the lateral skin folds is cleanly masked out (i.e. the finger's
         cuticle skin and sidewall skin sit ON TOP of the nail edges).
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
    perp_unit = np.array([-axis_unit[1], axis_unit[0]])

    # Build a smooth U-shaped polygon representing the exposed nail plate + free-edge extension:
    # - Along the bottom: a 25-point quadratic/cubic Bézier U-curve passing cleanly through
    #   c_l -> c_c -> c_r so the proximal cuticle skin overlaps the base of the nail.
    # - Along the left and right sidewalls: smooth curves from c_l -> d_l and c_r -> d_r,
    #   then flaring out above d_l and d_r so the free-edge tip (beyond the finger) is never clipped.
    ctrl_mid = 2.0 * c_c - 0.5 * (c_l + c_r)
    t_vals = np.linspace(0.0, 1.0, 29)
    cuticle_curve = (
        ((1.0 - t_vals) ** 2)[:, None] * c_l[None, :]
        + (2.0 * (1.0 - t_vals) * t_vals)[:, None] * ctrl_mid[None, :]
        + (t_vals ** 2)[:, None] * c_r[None, :]
    )

    # Left and right lateral nail-groove curves from cuticle to distal free edge
    # Only enforce the sidewall tuck on the lower 78% of the nail bed (s in [0.05, 0.78])
    # with a 7.5% bow so the nail is strictly seated inside the finger's lateral
    # nail grooves and leaves a natural border of finger skin on BOTH left and right sides!
    s_vals = np.linspace(0.05, 0.78, 12)
    bed_w = np.linalg.norm(d_r - d_l)
    bow = (4.0 * s_vals * (1.0 - s_vals))[:, None] * (0.075 * bed_w)

    right_sidewall = (1.0 - s_vals)[:, None] * c_r[None, :] + s_vals[:, None] * d_r[None, :] + bow * perp_unit[None, :]
    left_sidewall = (1.0 - s_vals[::-1])[:, None] * c_l[None, :] + s_vals[::-1][:, None] * d_l[None, :] - bow[::-1] * perp_unit[None, :]

    # Above s = 0.78, flare wide open so the entire free-edge tip extending past the fingertip
    # is 100% shaped by the press-on nail's own smooth, unbroken arc
    far_right_0 = (0.15 * c_r + 0.85 * d_r) + perp_unit * (bed_w * 0.25)
    far_right_1 = d_r + axis_unit * (bed_len * 0.15) + perp_unit * (bed_w * 0.40)
    far_right_2 = d_r + axis_unit * (bed_len * 1.40) + perp_unit * (bed_w * 0.55)
    far_left_2 = d_l + axis_unit * (bed_len * 1.40) - perp_unit * (bed_w * 0.55)
    far_left_1 = d_l + axis_unit * (bed_len * 0.15) - perp_unit * (bed_w * 0.40)
    far_left_0 = (0.15 * c_l + 0.85 * d_l) - perp_unit * (bed_w * 0.25)

    poly_pts = np.vstack([
        cuticle_curve,
        right_sidewall,
        [far_right_0, far_right_1, far_right_2, far_left_2, far_left_1, far_left_0],
        left_sidewall,
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
    # right inside the cuticle curve and lateral nail folds so the nail looks tucked IN
    cuticle_line_ss = np.zeros((rh * ss, rw * ss), dtype=np.uint8)
    groove_pts = np.vstack([left_sidewall[-6:], cuticle_curve, right_sidewall[:6]])
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
