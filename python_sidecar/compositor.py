"""
Press-On Nail Studio - True-Texture PBR Studio Relighting & Compositing Engine
(docs/true-texture-composition-plan.md, Steps 3 & 4)

Transforms raw tray-extracted nails into photorealistic studio-lit nails matched to the hand
while preserving 100% of photographic texture (nail art, gold charms, cat-eye shimmer,
French smile lines, and sheer ombré gradients):

  Pass 1: Surgical Glare-Tick Inpainting & Perimeter Edge-Fringe Cleanup
  Pass 2: Anatomical Cuticle Pocket Tuck & Contact AO Generation
  Pass 3: 3D Cylindrical Surface Normal Mapping & Non-Destructive Diffuse Relighting
  Pass 4: Sheer Nail-Bed Subsurface Translucency (Texture-Preserving)
  Pass 5: Adaptive High-Gloss Gel Topcoat Reflection & Softbox Window Sheen
  Pass 6: Fingertip Cast Shadow, Sensor Micro-Grain & Sub-Pixel Anti-Aliased Composite
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import scipy.ndimage
from warping import (
    build_cuticle_pocket_mask,
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
)


def clean_tray_glare_preserving_texture(nail_rgba: np.ndarray) -> np.ndarray:
    """
    Surgically heals pinpoint overhead lightbox glare ticks from an extracted RGBA nail
    while preserving 100% of the true photographed texture (floral nail art, 3D gold charms,
    gold stripes, cat-eye magnetic shimmer, French smile lines, and sheer ombré gradients).

    Pipeline:
      1. Start with 100% original RGB pixels (out = nail_rgba.copy()).
      2. Compute white top-hat on CIE LAB luminance (L) scaled to nail width (~16% nail_w).
      3. Isolate overhead light ticks via 4 strict criteria:
         - Sharp local luminance spike: tophat_L > 22.0 and L > 200.0
         - Desaturated / achromatic reflection: HSV S < 24.0
         - Interior / width test: CC horizontal width < 0.22 * nail_w and not a French tip
         - Small connected-component area: blob area < 3.5% of valid nail pixels and not horizontal stripe
      4. Surgical fast-marching inpainting (cv2.inpaint with INPAINT_TELEA) + Gaussian feather blend.
      5. Perimeter edge-fringe cleanup: replace outer 2-3px ring with nearest interior pixels
         via distance transform to prevent white backing card halo leakage.

    Everything outside the glare mask and edge ring is byte-identical to the input.
    """
    out: np.ndarray = nail_rgba.copy()
    alpha: np.ndarray = out[:, :, 3]
    valid: np.ndarray = alpha > 128
    if not np.any(valid):
        return out

    ys, xs = np.where(valid)
    y_top: int = int(ys.min())
    y_bot: int = int(ys.max())
    nail_h: float = max(1.0, float(y_bot - y_top + 1))
    x_left: int = int(xs.min())
    x_right: int = int(xs.max())
    nail_w: float = max(1.0, float(x_right - x_left + 1))
    n_valid: int = int(np.sum(valid))

    # Kernel size scales with nail width (~16% of nail width, minimum 7) so rectangular
    # lightbox reflections (~10-15% nail_w) are captured cleanly across 60px to 900px nails.
    k_tophat: int = max(7, int(round(nail_w * 0.16)) | 1)
    kernel_tophat: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tophat, k_tophat))

    bgr: np.ndarray = out[:, :, :3]
    lab: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L_chan: np.ndarray = lab[:, :, 0]
    hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat: np.ndarray = hsv[:, :, 1]

    tophat_L: np.ndarray = cv2.morphologyEx(L_chan, cv2.MORPH_TOPHAT, kernel_tophat)

    # Relative luminance spike & saturation drop relative to interior nail tone
    # Catches both bright white ticks AND soft lightbox streaks on dark/colored nails
    med_L: float = float(np.median(L_chan[valid]))
    med_S: float = float(np.median(sat[valid]))
    # Glare must ALWAYS be significantly brighter than the nail's median luminance (med_L)!
    L_glare_thresh = max(165.0, min(242.0, med_L + 20.0))
    L_expand_thresh = max(150.0, min(230.0, med_L + 11.0))
    rel_L_spike = (tophat_L > 10.0) & (L_chan > L_glare_thresh)
    rel_S_drop = (sat < max(30.0, med_S * 0.65))
    glare_seed: np.ndarray = valid & rel_L_spike & rel_S_drop
    glare_expand: np.ndarray = valid & (tophat_L > 5.0) & (L_chan > L_expand_thresh) & (sat < max(42.0, med_S * 0.80))

    # Condition 3 & 4: Connected component geometric isolation
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glare_expand.astype(np.uint8))
    glare_mask: np.ndarray = np.zeros_like(valid, dtype=bool)

    for i in range(1, num_labels):
        if not np.any(glare_seed[labels == i]):
            continue
        bx, by, bw, bh, area = stats[i]
        # Condition 4: Max CC area < 8.5% of valid nail pixels to heal long vertical streaks
        if area > 0.085 * n_valid:
            continue
        # Vertically elongated or pinpoint (not wide horizontal stripes or charms)
        if bw > 2.5 * bh and bw > 0.15 * nail_w:
            continue
        # Condition 3: Not a White French Tip (which spans across the tip width)
        if (by - y_top) <= 0.14 * nail_h and bw >= 0.28 * nail_w:
            continue
        if bw >= 0.28 * nail_w:
            continue

        glare_mask |= (labels == i)

    # Surgical fast-marching inpainting with feather blending
    if np.any(glare_mask):
        k_dilate: int = max(3, int(round(nail_w * 0.015)) | 1)
        dil_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        dilated_glare: np.ndarray = cv2.dilate(glare_mask.astype(np.uint8), dil_kernel)

        inpaint_radius: float = max(2.0, round(nail_w * 0.015, 1))
        inpainted_bgr: np.ndarray = cv2.inpaint(
            out[:, :, :3], dilated_glare, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA
        )

        sigma: float = max(1.0, nail_w * 0.008)
        k_blur: int = max(3, int(round(sigma * 4)) | 1)
        feather: np.ndarray = cv2.GaussianBlur(dilated_glare.astype(np.float32), (k_blur, k_blur), sigma)[:, :, None]
        feather = np.clip(feather, 0.0, 1.0)
        out[:, :, :3] = np.clip(
            out[:, :, :3].astype(np.float32) * (1.0 - feather) + inpainted_bgr.astype(np.float32) * feather,
            0,
            255,
        ).astype(np.uint8)

    # 4.5% Perimeter Edge-Fringe Cleanup:
    # Overwrites pixels that have white backing-card bounce light with nearest clean interior pixels,
    # preserving dark nail art while stripping white card bevel reflection.
    fringe_px: int = max(3, min(9, int(round(nail_w * 0.045))))
    k_fringe: int = fringe_px * 2 + 1
    kernel_fringe: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fringe, k_fringe))
    clean_interior: np.ndarray = cv2.erode((alpha > 200).astype(np.uint8), kernel_fringe) > 0
    if not np.any(clean_interior):
        clean_interior = alpha > 128
    edge_ring: np.ndarray = (alpha > 25) & (~clean_interior)

    if np.any(clean_interior) and np.any(edge_ring):
        _, indices = scipy.ndimage.distance_transform_edt(~clean_interior, return_distances=True, return_indices=True)
        nearest_bgr = out[indices[0][edge_ring], indices[1][edge_ring], :3]
        nearest_lab = cv2.cvtColor(nearest_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        curr_bgr = out[edge_ring, :3]
        curr_lab = cv2.cvtColor(curr_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        curr_hsv = cv2.cvtColor(curr_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        nearest_hsv = cv2.cvtColor(nearest_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        # Overwrite only if edge pixel is contaminated by card reflection (brighter L by > 4.0 or desaturated by > 6.0)
        is_card_bleed = (curr_lab[:, 0].astype(np.float32) > nearest_lab[:, 0].astype(np.float32) + 4.0) | \
                        (curr_hsv[:, 1].astype(np.float32) < nearest_hsv[:, 1].astype(np.float32) - 6.0)
        if np.any(is_card_bleed):
            ring_ys = np.where(edge_ring)[0][is_card_bleed]
            ring_xs = np.where(edge_ring)[1][is_card_bleed]
            out[ring_ys, ring_xs, :3] = nearest_bgr[is_card_bleed]

    exterior_all: np.ndarray = alpha < 200
    if np.any(clean_interior) and np.any(exterior_all):
        _, (iy, ix) = scipy.ndimage.distance_transform_edt(~clean_interior, return_indices=True)
        out[:, :, :3][exterior_all] = out[:, :, :3][iy[exterior_all], ix[exterior_all]]

    return out


def recover_clean_nail_albedo(nail_rgba: np.ndarray) -> np.ndarray:
    """
    Backward-compatible alias for clean_tray_glare_preserving_texture.
    Preserves 100% of photographic texture while surgically removing tray glare ticks.
    """
    return clean_tray_glare_preserving_texture(nail_rgba)


def compute_3d_nail_geometry(
    alpha_mask: np.ndarray,
    roll_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-pixel normalized coordinates (u, v) and 3D unit surface normals (nx, ny, nz)
    for a warped nail mask:
      - u in [-1, +1]: transverse position from left lateral edge (-1) to right lateral edge (+1)
      - v in [0, 1]: longitudinal position from proximal cuticle base (0) to distal tip (1)
      - (nx, ny, nz): 3D curved dome surface normal vector
    """
    rh, rw = alpha_mask.shape[:2]
    u_map = np.zeros((rh, rw), dtype=np.float32)
    v_map = np.zeros((rh, rw), dtype=np.float32)

    active_rows = np.where(np.max(alpha_mask, axis=1) > 0.08)[0]
    if len(active_rows) < 2:
        nz = np.ones((rh, rw), dtype=np.float32)
        return u_map, v_map, u_map.copy(), v_map.copy(), nz

    y_top = int(active_rows[0])     # Distal tip (smallest y)
    y_bot = int(active_rows[-1])    # Cuticle base (largest y)
    nail_len = max(1.0, float(y_bot - y_top))

    centers_x = np.full(rh, rw * 0.5, dtype=np.float32)
    half_widths = np.full(rh, rw * 0.25, dtype=np.float32)

    for y in range(y_top, y_bot + 1):
        cols = np.where(alpha_mask[y] > 0.12)[0]
        if len(cols) >= 2:
            x_l = float(cols[0])
            x_r = float(cols[-1])
            centers_x[y] = 0.5 * (x_l + x_r)
            half_widths[y] = max(3.0, 0.5 * (x_r - x_l))

    centers_x[:y_top] = centers_x[y_top]
    centers_x[y_bot + 1:] = centers_x[y_bot]
    half_widths[:y_top] = half_widths[y_top]
    half_widths[y_bot + 1:] = half_widths[y_bot]

    centers_x = cv2.GaussianBlur(centers_x.reshape(-1, 1), (1, 29), 7.0).ravel()
    half_widths = cv2.GaussianBlur(half_widths.reshape(-1, 1), (1, 29), 7.0).ravel()

    x_grid = np.arange(rw, dtype=np.float32)[None, :]
    y_grid = np.arange(rh, dtype=np.float32)[:, None]

    u_map = np.clip((x_grid - centers_x[:, None]) / half_widths[:, None], -1.0, 1.0)
    v_map = np.clip((float(y_bot) - y_grid) / nail_len, 0.0, 1.0)

    max_angle_rad = np.deg2rad(58.0)
    theta_x = np.clip(u_map + roll_offset * 0.48, -1.15, 1.15) * max_angle_rad
    theta_y = (v_map - 0.36) * np.deg2rad(18.0)

    nx = np.sin(theta_x) * np.cos(theta_y)
    ny = -np.sin(theta_y)
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.04, 1.0))

    return u_map, v_map, nx, ny, nz


def composite_nail_pbr(
    composite_hand: np.ndarray,
    hand_orig: np.ndarray,
    warped_roi: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target: Dict[str, Any],
    lighting_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Natural Studio Retoucher Compositing Pipeline:
      - Zero black perimeter fringe (edge shading fades to 1.0 at the outer boundary so anti-aliased
        pixels never darken against the skin).
      - Warm subsurface shading (never muddy gray/black scalar multiplication).
      - Subtle, photography-preserving studio sheen (no metallic white pipe bar).
      - No exterior black drop-shadow halo along sidewalls or cuticles.
    """
    rx, ry, rw, rh = bbox
    if rw <= 0 or rh <= 0:
        return

    opt: Dict[str, Any] = lighting_options or {}
    gloss_mult: float = float(opt.get("glossIntensity", 1.0))
    shadow_mult: float = float(opt.get("shadowDepth", 1.0))
    light_angle_deg: float = float(opt.get("lightAngle", 0.0))
    preset: str = str(opt.get("preset", "studio"))

    warped_bgr: np.ndarray = warped_roi[:, :, :3].astype(np.float32)
    raw_alpha: np.ndarray = warped_roi[:, :, 3].astype(np.float32) / 255.0

    # Pass 1: Anatomical Cuticle & Sidewall Skin-Fold Tuck Mask
    pocket_allow, cuticle_ao = build_cuticle_pocket_mask(
        roi_bbox=bbox,
        cuticle_left=target["c_l"],
        cuticle_center=target["c_c"],
        cuticle_right=target["c_r"],
        distal_left=target["d_l"],
        distal_right=target["d_r"],
        feather_sigma=1.2,
    )

    tucked_alpha: np.ndarray = raw_alpha * pocket_allow
    alpha_smooth: np.ndarray = cv2.GaussianBlur(tucked_alpha, (3, 3), 0.65)[:, :, None]

    # Pass 2: 3D Cylindrical & Longitudinal Apex Normals
    roll_offset: float = float(target.get("roll_offset", 0.0))
    u_map, v_map, nx, ny, nz = compute_3d_nail_geometry(tucked_alpha, roll_offset=roll_offset)

    # Measure interior luminance & contrast
    nail_lum: np.ndarray = 0.114 * warped_bgr[:, :, 0] + 0.587 * warped_bgr[:, :, 1] + 0.299 * warped_bgr[:, :, 2]
    interior_bool: np.ndarray = tucked_alpha > 0.50
    med_nail_lum: float = float(np.median(nail_lum[interior_bool])) if np.any(interior_bool) else 160.0
    interior_lum_std: float = float(np.std(nail_lum[interior_bool])) if np.any(interior_bool) else 15.0

    hand_patch: np.ndarray = hand_orig[ry:ry + rh, rx:rx + rw].astype(np.float32)
    hand_bed_smooth: np.ndarray = cv2.GaussianBlur(hand_patch, (19, 19), 5.0)

    # Pass 3: Subtle Subsurface Warmth on Sheer/Light Nails (v_map < 0.65)
    v_bed_zone: np.ndarray = np.clip((0.65 - v_map) / 0.20, 0.0, 1.0)
    sheer_weight: np.ndarray = 0.08 * v_bed_zone
    base_bgr: np.ndarray = warped_bgr * (1.0 - sheer_weight[:, :, None]) + hand_bed_smooth * sheer_weight[:, :, None]

    # CRITICAL FIX FOR BLACK FRINGE:
    # 1. Keep cylindrical C-curve shading gentle (max 8% darkening, i.e. 0.92..1.02) so light pastel nails
    #    never get dark brown/black sidewalls.
    # 2. Multiply shading intensity by `interior_fade = np.clip((tucked_alpha - 0.60) / 0.35, 0.0, 1.0)`
    #    so the outer 1.5px anti-aliased boundary has ZERO artificial darkening (shading == 1.0 at the edge!).
    u_abs: np.ndarray = np.abs(u_map)
    t_side: np.ndarray = np.clip((u_abs - 0.45) / 0.55, 0.0, 1.0)
    w_side: np.ndarray = 0.5 * (1.0 - np.cos(np.pi * t_side))
    raw_shading: np.ndarray = 1.01 - 0.09 * w_side  # Gently rolls off from 1.01 at center to 0.92 at sides

    # Fade shading out at the very edge (tucked_alpha < 0.85) so the perimeter never turns into a dark line!
    interior_fade: np.ndarray = np.clip((tucked_alpha - 0.60) / 0.35, 0.0, 1.0)
    shading: np.ndarray = 1.0 - (1.0 - raw_shading) * interior_fade

    # Warm chromatic shading (darken Blue/Green slightly more than Red so shading is warm rose, never gray/black)
    shade_bgr: np.ndarray = np.dstack([
        shading ** 1.15,  # Blue
        shading ** 1.05,  # Green
        shading ** 0.85,  # Red (stays warmer)
    ])
    lit_bgr: np.ndarray = np.clip(base_bgr * shade_bgr, 0.0, 255.0)

    # Pass 4: Delicate Studio Softbox Sheen (Avoids the "3D Metallic Pipe" look!)
    spec_u_center: float = -0.20 - roll_offset * 0.30 + (light_angle_deg / 180.0) * 0.20
    u_delta: np.ndarray = u_map - spec_u_center

    v_sheen_env: np.ndarray = np.sin(np.clip((v_map - 0.08) / 0.82, 0.0, 1.0) * np.pi)
    broad_sheen: np.ndarray = np.exp(-0.5 * (u_delta / 0.22) ** 2) * v_sheen_env

    rib_half_w: np.ndarray = 0.075 * (0.65 + 0.35 * v_sheen_env)
    gloss_ribbon: np.ndarray = np.exp(-0.5 * (np.abs(u_delta) / np.maximum(0.015, rib_half_w)) ** 1.5) * v_sheen_env

    # Keep synthetic gloss subtle because real tray photos already contain natural surface sheen
    if preset == "ring":
        spec_total: np.ndarray = (0.08 * broad_sheen + 0.22 * gloss_ribbon) * gloss_mult
    elif preset == "flash":
        spec_total = (0.04 * broad_sheen + 0.32 * gloss_ribbon) * gloss_mult
    elif preset == "natural":
        spec_total = (0.08 * broad_sheen + 0.10 * gloss_ribbon) * gloss_mult
    else:
        # Studio default
        spec_total = (0.06 * broad_sheen + 0.16 * gloss_ribbon) * gloss_mult

    spec_total = cv2.GaussianBlur(spec_total, (5, 5), 1.0)

    # Clamp specular strictly inside the solid interior (zero highlight within 3px of edge)
    m_spec: np.ndarray = np.clip((tucked_alpha - 0.70) / 0.25, 0.0, 1.0)
    spec_alpha: np.ndarray = np.clip(spec_total * m_spec, 0.0, 0.22)[:, :, None]

    # Screen blend mode (never over-additive, naturally softer on light pink/white nails!)
    lit_bgr = lit_bgr + (250.0 - lit_bgr) * spec_alpha

    # Pass 5: Cuticle & Sidewall Skin Light-Wrap (Melts the outer 1.5px nail edge into the surrounding skin)
    # Never darken the skin sample! Use 100% skin brightness around the proximal cuticle & sidewalls.
    hand_blurred: np.ndarray = cv2.GaussianBlur(hand_patch, (11, 11), 3.0)
    rim_wrap: np.ndarray = (
        np.clip((0.92 - tucked_alpha) * 2.2, 0.0, 1.0)
        * (tucked_alpha > 0.05)
        * np.clip(1.0 - v_map * 1.25, 0.0, 1.0)
    )
    lit_bgr = lit_bgr * (1.0 - 0.28 * rim_wrap[:, :, None]) + hand_blurred * (0.28 * rim_wrap[:, :, None])

    # Pass 6: Sensor Micro-Grain ONLY inside the nail mask (fixes rectangular ROI box visibility!)
    rng: np.random.Generator = np.random.default_rng(seed=int(rx * 31 + ry * 17))
    sensor_grain: np.ndarray = rng.normal(0.0, 1.0, size=(rh, rw, 1)).astype(np.float32)
    lit_bgr = np.clip(lit_bgr + sensor_grain * alpha_smooth, 0.0, 255.0)

    # Gentle Warm Skin Crease Micro-Shadow (Max 6% warm darkening right in the cuticle tuck, NEVER black!)
    hand_curr: np.ndarray = composite_hand[ry:ry + rh, rx:rx + rw].copy()
    crease_weight: np.ndarray = (
        np.clip((1.0 - cuticle_ao) / 0.18, 0.0, 1.0)
        * (tucked_alpha < 0.50)
        * np.clip(1.0 - v_map * 2.2, 0.0, 1.0)
        * (0.06 * shadow_mult)
    )
    crease_mult_bgr: np.ndarray = np.dstack([
        1.0 - crease_weight * 1.25,  # Warm shadow (absorbs blue)
        1.0 - crease_weight * 1.10,  # Absorbs green
        1.0 - crease_weight * 0.70,  # Preserves red subsurface warmth
    ])
    hand_with_shadow: np.ndarray = hand_curr * crease_mult_bgr

    # Final sub-pixel anti-aliased composite
    composite_hand[ry:ry + rh, rx:rx + rw] = (
        lit_bgr * alpha_smooth + hand_with_shadow * (1.0 - alpha_smooth)
    )


def erase_exposed_natural_nails(
    composite_canvas: np.ndarray,
    landmarks: Dict[str, Any],
    mapped_nails_rgba: Optional[Dict[str, np.ndarray]] = None,
    downscale: float = 1.0,
) -> np.ndarray:
    """
    Surgical, texture-preserving natural-nail inpainting (hybrid-implementation-v2.md, Pillar 5).
    Evaluates per-side exposure (mL, mR) independently.
    If neither corner is exposed (e.g. square/wide press-ons), inpainting is 100% skipped.
    When exposed, strictly masks only the tiny lateral corner triangles at d_l and d_r,
    leaving fingertip pulp, central nail bed, and cuticles completely untouched.
    """
    h_canvas, w_canvas = composite_canvas.shape[:2]
    nail_corners_mask = np.zeros((h_canvas, w_canvas), dtype=np.uint8)
    has_any_inpaint = False

    for fname, lm in landmarks.items():
        if not isinstance(lm, dict) or "c_l" not in lm or "d_l" not in lm:
            continue
        c_l = np.array(lm["c_l"], dtype=np.float64) * downscale
        c_c = np.array(lm["c_c"], dtype=np.float64) * downscale
        c_r = np.array(lm["c_r"], dtype=np.float64) * downscale
        d_l = np.array(lm["d_l"], dtype=np.float64) * downscale
        d_r = np.array(lm["d_r"], dtype=np.float64) * downscale

        distal_mid = (d_l + d_r) * 0.5
        axis_vec = distal_mid - c_c
        bed_len = max(1.0, float(np.linalg.norm(axis_vec)))
        axis_u = axis_vec / bed_len
        perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)
        if np.dot(perp_u, d_r - d_l) < 0:
            perp_u = -perp_u

        nat_distal_w = float(np.linalg.norm(d_r - d_l))

        # Check press-on nail width and shape if available
        expose_left = False
        expose_right = False

        if mapped_nails_rgba and fname in mapped_nails_rgba:
            nail_img = mapped_nails_rgba[fname]
            if nail_img is not None and nail_img.size > 0:
                n_alpha = nail_img[:, :, 3]
                nh, nw = n_alpha.shape[:2]
                # Measure taper at distal 40% zone
                y_distal = int(nh * 0.42)
                row_xs = np.where(n_alpha[min(y_distal, nh - 1)] > 50)[0]
                if len(row_xs) > 0:
                    press_distal_w = float(row_xs[-1] - row_xs[0] + 1) * downscale
                    is_tapered = (nh / max(1.0, float(nw))) > 1.6
                    margin = (nat_distal_w - press_distal_w) * 0.5
                    if margin > 2.0 or is_tapered:
                        expose_left = True
                        expose_right = True
                else:
                    expose_left = True
                    expose_right = True
        else:
            expose_left = True
            expose_right = True

        # Build surgical triangular corner masks strictly at d_l and d_r
        corner_len = min(bed_len * 0.28, 14.0 * downscale)

        if expose_left:
            # Triangular wedge at d_l corner only
            tri_l = np.array([
                d_l,
                d_l + axis_u * (2.0 * downscale),
                d_l - perp_u * (2.0 * downscale) - axis_u * corner_len,
            ], dtype=np.int32)
            cv2.fillPoly(nail_corners_mask, [tri_l], 255)
            has_any_inpaint = True

        if expose_right:
            # Triangular wedge at d_r corner only
            tri_r = np.array([
                d_r,
                d_r + axis_u * (2.0 * downscale),
                d_r + perp_u * (2.0 * downscale) - axis_u * corner_len,
            ], dtype=np.int32)
            cv2.fillPoly(nail_corners_mask, [tri_r], 255)
            has_any_inpaint = True

    if has_any_inpaint and np.any(nail_corners_mask > 0):
        canvas_u8 = np.clip(composite_canvas, 0, 255).astype(np.uint8)
        inpainted_bgr = cv2.inpaint(
            canvas_u8,
            nail_corners_mask,
            inpaintRadius=max(2.0, 3.5 * downscale),
            flags=cv2.INPAINT_TELEA,
        )
        noise = np.random.default_rng(42).normal(0.0, 1.2, canvas_u8.shape).astype(np.float32)
        inpainted_textured = np.clip(inpainted_bgr.astype(np.float32) + noise, 0.0, 255.0)

        feather_mask = cv2.GaussianBlur(nail_corners_mask.astype(np.float32) / 255.0, (5, 5), 1.2)[:, :, None]
        composite_canvas = composite_canvas * (1.0 - feather_mask) + inpainted_textured * feather_mask

    return composite_canvas


def render_full_hand_composite(
    hand_bgr: np.ndarray,
    landmarks: Dict[str, Any],
    mapped_nails_rgba: Dict[str, np.ndarray],
    lighting_options: Optional[Dict[str, Any]] = None,
    downscale: float = 1.0,
) -> np.ndarray:
    """
    Renders the complete 5-finger composite with physical PBR lighting and TPS warping.
    Supports downscaled fast preview (<100ms) for responsive slider adjustments.
    """
    h_hand, w_hand = hand_bgr.shape[:2]
    opt = lighting_options or {}
    overall_len_scale = float(opt.get("nailLengthScale", 1.0))
    c_curve_user = float(opt.get("cCurveIntensity", 1.0))

    if downscale < 1.0:
        h_work = max(64, int(round(h_hand * downscale)))
        w_work = max(64, int(round(w_hand * downscale)))
        hand_work = cv2.resize(hand_bgr, (w_work, h_work), interpolation=cv2.INTER_AREA)
    else:
        h_work, w_work = h_hand, w_hand
        hand_work = hand_bgr.copy()

    composite_canvas = hand_work.copy().astype(np.float32)

    cleaned_nails: Dict[str, np.ndarray] = {}
    for fname, rgba in mapped_nails_rgba.items():
        if rgba is not None and rgba.size > 0:
            cleaned = clean_tray_glare_preserving_texture(rgba)
            if downscale < 1.0:
                cleaned = cv2.resize(cleaned, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
            cleaned_nails[fname] = cleaned

    if not cleaned_nails:
        return hand_work

    # Pre-composite: Surgical, texture-preserving inpainting of exposed natural nail corners
    composite_canvas = erase_exposed_natural_nails(composite_canvas, landmarks, cleaned_nails, downscale)

    finger_keys = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    hand_bed_widths = []
    tray_widths = []
    for fname in finger_keys:
        if fname in landmarks and fname in cleaned_nails:
            lm = landmarks[fname]
            cw = float(np.linalg.norm(np.array(lm["c_r"]) - np.array(lm["c_l"]))) * downscale
            dw = float(np.linalg.norm(np.array(lm["d_r"]) - np.array(lm["d_l"]))) * downscale
            hand_bed_widths.append(max(cw, dw))
            nd = cleaned_nails[fname]
            xs = np.where(nd[:, :, 3] > 25)[1]
            tray_widths.append(float(np.max(xs) - np.min(xs) + 1) if len(xs) > 0 else float(nd.shape[1]))

    tray_scale = 1.0
    if len(hand_bed_widths) > 0 and len(tray_widths) > 0 and sum(tray_widths) > 0:
        tray_scale = sum(hand_bed_widths) / max(1.0, sum(tray_widths))

    for fname in finger_keys:
        if fname not in landmarks or fname not in cleaned_nails:
            continue
        lm = landmarks[fname]
        clean_rgba = cleaned_nails[fname]

        # Extract per-finger adjustments (from lighting_options or landmarks)
        finger_adj = opt.get("finger_adjustments", {}).get(fname, {})

        c_curve_val = float(finger_adj.get("cCurve", finger_adj.get("c_curve", lm.get("c_curve", 0.28))))
        c_curve = c_curve_val * c_curve_user
        roll = float(finger_adj.get("roll_offset", lm.get("roll_offset", 0.0)))

        curved = apply_cylindrical_c_curve(clean_rgba, c_curve_intensity=c_curve, roll_offset=roll)
        src_h, src_w = curved.shape[:2]

        c_l = [lm["c_l"][0] * downscale, lm["c_l"][1] * downscale]
        c_c = [lm["c_c"][0] * downscale, lm["c_c"][1] * downscale]
        c_r = [lm["c_r"][0] * downscale, lm["c_r"][1] * downscale]
        d_l = [lm["d_l"][0] * downscale, lm["d_l"][1] * downscale]
        d_r = [lm["d_r"][0] * downscale, lm["d_r"][1] * downscale]

        len_scale = float(finger_adj.get("lengthScale", finger_adj.get("length_scale", lm.get("length_scale", 1.0)))) * overall_len_scale
        width_scale = float(finger_adj.get("widthScale", finger_adj.get("width_scale", lm.get("width_scale", 1.0))))
        axial_shift = float(finger_adj.get("axialShift", finger_adj.get("axial_shift", lm.get("axial_shift", 0.0)))) * downscale
        angle_offset = float(finger_adj.get("angleOffset", finger_adj.get("angle_offset", lm.get("angle_offset", 0.0))))
        tip_taper = float(finger_adj.get("tipTaper", finger_adj.get("tip_taper", lm.get("tip_taper", 1.0))))

        src_pts, dst_pts = build_3tier_control_grid(
            src_w=src_w,
            src_h=src_h,
            cuticle_left=c_l,
            cuticle_center=c_c,
            cuticle_right=c_r,
            distal_left=d_l,
            distal_right=d_r,
            length_scale=len_scale,
            tip_taper=tip_taper,
            roll_offset=roll,
            tray_to_hand_scale=tray_scale,
            width_scale=width_scale,
            axial_shift=axial_shift,
            angle_offset=angle_offset,
        )

        warped_roi, bbox = warp_nail_tps(curved, src_pts, dst_pts, output_shape=(h_work, w_work))

        # Update cuticle coordinates for anatomical tuck mask alignment
        tuck_c_c = c_c
        tuck_c_l = c_l
        tuck_c_r = c_r
        if abs(axial_shift) > 0.01:
            distal_mid = (np.array(d_l) + np.array(d_r)) * 0.5
            axis_vec = distal_mid - np.array(c_c)
            b_len = np.linalg.norm(axis_vec)
            if b_len > 0:
                ax_unit = axis_vec / b_len
                shift_vec = ax_unit * axial_shift
                tuck_c_c = (np.array(c_c) + shift_vec).tolist()
                tuck_c_l = (np.array(c_l) + shift_vec).tolist()
                tuck_c_r = (np.array(c_r) + shift_vec).tolist()

        target_dict = {
            "name": fname,
            "c_l": tuck_c_l,
            "c_c": tuck_c_c,
            "c_r": tuck_c_r,
            "d_l": d_l,
            "d_r": d_r,
            "c_curve": c_curve,
            "roll_offset": roll,
            "length_scale": len_scale,
            "tip_taper": tip_taper,
            "width_scale": width_scale,
            "axial_shift": axial_shift,
            "angle_offset": angle_offset,
        }

        composite_nail_pbr(composite_canvas, hand_work, warped_roi, bbox, target_dict, lighting_options=opt)

    return np.clip(composite_canvas, 0, 255).astype(np.uint8)
