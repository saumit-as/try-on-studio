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
    rel_L_spike = (tophat_L > 12.0) & (L_chan > min(185.0, med_L + 28.0))
    rel_S_drop = (sat < max(38.0, med_S * 0.72))
    glare_seed: np.ndarray = valid & rel_L_spike & rel_S_drop
    glare_expand: np.ndarray = valid & (tophat_L > 6.0) & (L_chan > min(160.0, med_L + 16.0)) & (sat < max(48.0, med_S * 0.85))

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
    Executes the 6-pass True-Texture PBR studio relighting and compositing pipeline on a warped nail ROI.

    Key improvements (Steps 3 & 4):
      - 3D Cylindrical Diffuse Ratio constrained to [0.84, 1.04]: center dome |u|<0.55 stays at
        0.97-1.03 (preserving 100% original color/pattern contrast), with smooth cosine roll-off
        to 0.84 on the shaded sidewall and 0.90 on the lit sidewall.
      - 0.88 gentle contact AO tuck at the proximal cuticle base.
      - Sheer nail-bed translucency restricted to max 10%, only where v_map < 0.62 AND local texture
        variance std(L) in 9x9 is low (< 4.0), preventing washout of floral art, charms, or French tips.
      - Adaptive clear-coat specular sheen: scales spec_alpha by 0.55 when interior luminance std is high.
      - Dynamic lighting_options support (glossIntensity, shadowDepth, lightAngle, preset).
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
        feather_sigma=1.5,
    )

    tucked_alpha: np.ndarray = raw_alpha * pocket_allow
    alpha_smooth: np.ndarray = cv2.GaussianBlur(tucked_alpha, (3, 3), 0.75)[:, :, None]

    # Pass 2: 3D Cylindrical & Longitudinal Apex Normals
    roll_offset: float = float(target.get("roll_offset", 0.0))
    u_map, v_map, nx, ny, nz = compute_3d_nail_geometry(tucked_alpha, roll_offset=roll_offset)

    # Pass 3: Studio Diffuse Relighting & Sheer Subsurface Translucency
    # Rotate key light around Y axis by light_angle_deg
    rad: float = float(np.radians(light_angle_deg))
    lx: float = 0.64 * np.cos(rad) + 0.755 * np.sin(rad)
    lz: float = -0.64 * np.sin(rad) + 0.755 * np.cos(rad)
    L_key: np.ndarray = np.array([lx, -0.14, lz], dtype=np.float32)
    L_key /= np.linalg.norm(L_key)
    ndotl_key: np.ndarray = np.clip(nx * L_key[0] + ny * L_key[1] + nz * L_key[2], 0.0, 1.0)

    # Measure interior luminance and local spatial texture variation
    nail_lum: np.ndarray = 0.114 * warped_bgr[:, :, 0] + 0.587 * warped_bgr[:, :, 1] + 0.299 * warped_bgr[:, :, 2]
    mean_L: np.ndarray = cv2.boxFilter(nail_lum, cv2.CV_32F, (9, 9))
    mean_L2: np.ndarray = cv2.boxFilter(nail_lum ** 2, cv2.CV_32F, (9, 9))
    local_std_L: np.ndarray = np.sqrt(np.maximum(0.0, mean_L2 - mean_L ** 2))

    interior_bool: np.ndarray = tucked_alpha > 0.35
    interior_lum_std: float = float(np.std(nail_lum[interior_bool])) if np.any(interior_bool) else 15.0

    # Sheer Translucency: Restricted strictly to the proximal nail bed zone (v_map < 0.62)
    v_bed_zone: np.ndarray = np.clip((0.62 - v_map) / 0.15, 0.0, 1.0)
    low_texture_var: np.ndarray = np.clip((4.0 - local_std_L) / 1.5, 0.0, 1.0)
    sheer_weight: np.ndarray = 0.10 * v_bed_zone * low_texture_var

    hand_patch: np.ndarray = hand_orig[ry:ry + rh, rx:rx + rw].astype(np.float32)
    hand_bed_smooth: np.ndarray = cv2.GaussianBlur(hand_patch, (19, 19), 5.0)
    base_with_subsurface: np.ndarray = (
        warped_bgr * (1.0 - sheer_weight[:, :, None]) + hand_bed_smooth * sheer_weight[:, :, None]
    )

    # 3D Cylindrical Diffuse Ratio: Deepen edge shading to [0.72, 1.04]
    center_diffuse: np.ndarray = 0.97 + 0.06 * ndotl_key
    u_abs: np.ndarray = np.abs(u_map)
    t_side: np.ndarray = np.clip((u_abs - 0.42) / 0.58, 0.0, 1.0)
    w_side: np.ndarray = 0.5 * (1.0 - np.cos(np.pi * t_side))
    target_sidewall: np.ndarray = np.where(u_map < 0.0, 0.72, 0.78)
    diffuse_ratio: np.ndarray = np.where(
        u_abs <= 0.42,
        center_diffuse,
        (1.0 - w_side) * 1.00 + w_side * target_sidewall,
    )
    diffuse_ratio = np.clip(diffuse_ratio, 0.72, 1.04)

    # Gentle 0.88 cuticle AO tuck at the proximal cuticle base
    soft_cuticle_ao: np.ndarray = 0.88 + 0.12 * cuticle_ao
    lit_bgr: np.ndarray = base_with_subsurface * (diffuse_ratio * soft_cuticle_ao)[:, :, None]

    # Pass 4: High-Gloss Clear Gel Topcoat Specular Reflections
    # Aligned to upper-left studio softbox matching hand-template.jpg
    spec_u_center: float = -0.22 - roll_offset * 0.35 + (light_angle_deg / 180.0) * 0.20
    u_delta: np.ndarray = u_map - spec_u_center

    # Broad studio dome sheen
    v_sheen_env: np.ndarray = np.sin(np.clip((v_map - 0.05) / 0.88, 0.0, 1.0) * np.pi)
    broad_sheen: np.ndarray = np.exp(-0.5 * (u_delta / 0.18) ** 2) * v_sheen_env

    # Softened vertical softbox window reflection ribbon (rib_half_w = 0.065, exp = 1.6)
    v_rib_in: np.ndarray = np.clip((v_map - 0.10) / 0.15, 0.0, 1.0)
    v_rib_out: np.ndarray = np.clip((0.90 - v_map) / 0.12, 0.0, 1.0)
    apex_hotspot: np.ndarray = 0.80 + 0.20 * np.exp(-0.5 * ((v_map - 0.48) / 0.22) ** 2)
    v_rib_env: np.ndarray = (v_rib_in ** 1.2) * (v_rib_out ** 1.3) * apex_hotspot

    rib_half_w: np.ndarray = 0.065 * (0.60 + 0.40 * np.sin(np.clip((v_map - 0.08) / 0.82, 0.0, 1.0) * np.pi))
    gloss_ribbon: np.ndarray = np.exp(-0.5 * (np.abs(u_delta) / np.maximum(0.012, rib_half_w)) ** 1.6) * v_rib_env

    # Secondary lateral fill sheen
    fill_u_delta: np.ndarray = np.abs(u_map - (0.42 - roll_offset * 0.30))
    fill_v_env: np.ndarray = np.maximum(0.0, np.sin(np.clip((v_map - 0.10) / 0.55, 0.0, 1.0) * np.pi)) ** 1.2
    fill_reflection: np.ndarray = np.exp(-0.5 * (fill_u_delta / 0.075) ** 2.0) * fill_v_env

    # Preset adaptations
    if preset == "ring":
        spec_total: np.ndarray = (0.25 * broad_sheen + 0.55 * gloss_ribbon + 0.20 * fill_reflection) * gloss_mult
    elif preset == "flash":
        spec_total = (0.05 * broad_sheen + 1.10 * gloss_ribbon + 0.05 * fill_reflection) * gloss_mult
    elif preset == "natural":
        spec_total = (0.45 * broad_sheen + 0.30 * gloss_ribbon + 0.15 * fill_reflection) * gloss_mult
    else:
        # Studio default: gentle natural softbox sheen
        spec_total = (0.10 * broad_sheen + 0.42 * gloss_ribbon + 0.08 * fill_reflection) * gloss_mult

    spec_total = cv2.GaussianBlur(spec_total, (3, 3), 0.5)

    high_contrast_weight: float = float(np.clip((interior_lum_std - 18.0) / 10.0, 0.0, 1.0))
    spec_scale: float = 1.0 - 0.45 * high_contrast_weight

    spec_color: np.ndarray = np.array([252.0, 254.0, 255.0], dtype=np.float32)[None, None, :]
    spec_alpha: np.ndarray = np.clip(spec_total[:, :, None] * spec_scale, 0.0, 0.85)
    lit_bgr = lit_bgr * (1.0 - spec_alpha * 0.75) + spec_color * spec_alpha

    # Pass 5: True Skin Light-Wrap & Warm Subsurface Crease Integration
    hand_blurred: np.ndarray = cv2.GaussianBlur(hand_patch, (9, 9), 2.5)
    rim_wrap: np.ndarray = (
        np.clip((1.0 - tucked_alpha) * 2.5, 0.0, 1.0)
        * (tucked_alpha > 0.05)
        * np.clip(1.0 - v_map * 1.4, 0.0, 1.0)
    )
    lit_bgr = lit_bgr * (1.0 - 0.22 * rim_wrap[:, :, None]) + hand_blurred * (0.22 * rim_wrap[:, :, None])

    # Pass 6: Camera sensor micro-grain matching hand-template.jpg & under-tip cast shadow
    rng: np.random.Generator = np.random.default_rng(seed=int(rx * 31 + ry * 17))
    sensor_grain: np.ndarray = rng.normal(0.0, 1.15, size=(rh, rw, 1)).astype(np.float32)
    lit_bgr = np.clip(lit_bgr + sensor_grain, 0.0, 255.0)

    # Warm Subsurface Contact Shadow along the cuticle crease
    valid_skin = (tucked_alpha < 0.15) & (hand_patch[:, :, 2] > 70.0)
    if np.any(valid_skin):
        skin_med_bgr = np.median(hand_patch[valid_skin], axis=0)
    else:
        skin_med_bgr = np.array([140.0, 155.0, 195.0], dtype=np.float32)
    warm_shadow_bgr: np.ndarray = skin_med_bgr * np.array([0.52, 0.58, 0.72], dtype=np.float32)

    crease_band: np.ndarray = np.clip((1.0 - cuticle_ao) / 0.18, 0.0, 1.0) * (tucked_alpha < 0.35) * np.clip(1.0 - v_map * 2.5, 0.0, 1.0)
    hand_curr: np.ndarray = composite_hand[ry:ry + rh, rx:rx + rw].copy()
    hand_curr = hand_curr * (1.0 - 0.40 * crease_band[:, :, None]) + warm_shadow_bgr[None, None, :] * (0.40 * crease_band[:, :, None])

    # Under-Tip Free-Edge Cast Shadow: RESTRICTED STRICTLY TO v_map > 0.65
    M_sh: np.ndarray = np.float32([[1.0, 0.0, 1.6 + light_angle_deg * 0.05], [0.0, 1.0, 4.2]])
    tip_zone_mask: np.ndarray = np.clip((v_map - 0.65) / 0.15, 0.0, 1.0)
    shifted_alpha: np.ndarray = cv2.warpAffine(tucked_alpha * tip_zone_mask, M_sh, (rw, rh), flags=cv2.INTER_LINEAR)
    shadow_blur: np.ndarray = cv2.GaussianBlur(shifted_alpha, (11, 11), 3.2)

    patch_hsv: np.ndarray = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
    skin_weight: np.ndarray = np.clip((patch_hsv[:, :, 1].astype(np.float32) - 18.0) / 25.0, 0.0, 1.0)
    under_shadow_mult: np.ndarray = 1.0 - (0.32 * shadow_mult) * shadow_blur * (1.0 - tucked_alpha) * skin_weight
    hand_with_shadow: np.ndarray = hand_curr * under_shadow_mult[:, :, None]

    # Sub-pixel anti-aliased composite
    composite_hand[ry:ry + rh, rx:rx + rw] = (
        lit_bgr * alpha_smooth + hand_with_shadow * (1.0 - alpha_smooth)
    )


def erase_exposed_natural_nails(
    composite_canvas: np.ndarray,
    landmarks: Dict[str, Any],
    downscale: float = 1.0,
) -> np.ndarray:
    """
    Inpaints and heals exposed bare natural nail corners (especially square free edges
    around d_l and d_r) with local skin tone and texture before press-on nails are composited.
    """
    h_canvas, w_canvas = composite_canvas.shape[:2]
    nail_corners_mask = np.zeros((h_canvas, w_canvas), dtype=np.uint8)

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

        # Expand polygon slightly distal (+6px) and lateral (+4px)
        p_d_ext = distal_mid + axis_u * (6.0 * downscale)
        p_dl_ext = d_l + axis_u * (6.0 * downscale) - perp_u * (4.0 * downscale)
        p_dr_ext = d_r + axis_u * (6.0 * downscale) + perp_u * (4.0 * downscale)
        p_cl_ext = c_l - perp_u * (3.0 * downscale)
        p_cr_ext = c_r + perp_u * (3.0 * downscale)

        # Polygon covering the distal half and lateral corners of the natural nail bed
        pts = np.array([
            p_cl_ext + axis_u * (bed_len * 0.35),
            p_dl_ext,
            p_d_ext,
            p_dr_ext,
            p_cr_ext + axis_u * (bed_len * 0.35),
        ], dtype=np.int32)
        cv2.fillPoly(nail_corners_mask, [pts], 255)

    if np.any(nail_corners_mask > 0):
        canvas_u8 = np.clip(composite_canvas, 0, 255).astype(np.uint8)
        inpainted_bgr = cv2.inpaint(
            canvas_u8,
            nail_corners_mask,
            inpaintRadius=max(3.0, 5.0 * downscale),
            flags=cv2.INPAINT_TELEA,
        )
        feather_mask = cv2.GaussianBlur(nail_corners_mask.astype(np.float32) / 255.0, (7, 7), 2.0)[:, :, None]
        composite_canvas = composite_canvas * (1.0 - feather_mask) + inpainted_bgr.astype(np.float32) * feather_mask

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

    # Pre-composite: Inpaint exposed natural nail corners on bare hand template
    composite_canvas = erase_exposed_natural_nails(composite_canvas, landmarks, downscale)

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
