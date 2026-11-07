"""
Press-On Nail Studio - Phase 4 Preview: 6-Pass PBR Studio Relighting & Compositing Engine
Transforms raw tray-extracted nails into photorealistic studio-lit nails matched to the hand:
  Pass 1: Intrinsic Albedo Recovery (supports both Monochrome Solid & French-Tip / Nail-Art sets)
  Pass 2: 3D Cylindrical & Longitudinal Apex Surface Normal Map Generation
  Pass 3: Studio Key + Fill Diffuse Shading & Sheer Nail-Bed Subsurface Translucency
  Pass 4: High-Gloss Gel Topcoat Anisotropic Softbox Reflection & Secondary Rim Sheen
  Pass 5: Cuticle & Sidewall Skin-Fold Tuck Mask + Warm Subsurface Contact AO
  Pass 6: Under-Tip Fingertip Cast Shadow, Sensor Micro-Grain & Anti-Aliased Composite
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np
from warping import build_cuticle_pocket_mask


def recover_clean_nail_albedo(nail_rgba: np.ndarray) -> np.ndarray:
    """
    Removes baked-in packaging tray lighting artifacts from an extracted RGBA nail while
    automatically distinguishing between:
      A. Monochrome Solid Nails: strips out both specular glare streaks and dark slot edge shadows.
      B. Multi-Tone / French-Tip / Nail-Art Nails: reconstructs a 100% bilaterally symmetric,
         razor-sharp French-tip smile line with full crescent wings and a warm, de-glared sheer
         blush-pink base.
    """
    out = nail_rgba.copy()
    alpha = out[:, :, 3].astype(np.float32) / 255.0
    h, w = alpha.shape[:2]

    valid = alpha > 0.5
    if not np.any(valid):
        return out

    lab = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    L_chan = lab[:, :, 0]
    A_chan = lab[:, :, 1]
    B_chan = lab[:, :, 2]

    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]

    # Check if the nail has a French Tip or multi-tone design by comparing centerline
    # luminance in the free-edge tip zone (y = 0.06..0.24*h) vs the main nail bed (y = 0.42..0.82*h)
    cx0, cx1 = int(w * 0.30), max(int(w * 0.30) + 1, int(w * 0.70))
    tip_slice = L_chan[int(h * 0.06):int(h * 0.24), cx0:cx1]
    body_slice = L_chan[int(h * 0.42):int(h * 0.82), cx0:cx1]

    tip_med_L = float(np.median(tip_slice)) if tip_slice.size > 0 else 100.0
    body_med_L = float(np.median(body_slice)) if body_slice.size > 0 else 100.0

    is_french_or_multitone = abs(body_med_L - tip_med_L) > 38.0

    if is_french_or_multitone:
        # --- PATH B: FRENCH TIP / MULTI-TONE NAIL ---
        # 1. Measure the clean base pigment in the lower-middle nail body (y = 0.44..0.86*h)
        body_mask = np.zeros((h, w), dtype=bool)
        body_mask[int(h * 0.44):int(h * 0.86), int(w * 0.22):int(w * 0.78)] = True
        body_mask = body_mask & (alpha > 0.85) & (L_chan > 110.0)

        if np.any(body_mask):
            p25 = float(np.percentile(L_chan[body_mask], 20))
            p75 = float(np.percentile(L_chan[body_mask], 75))
            clean_body = body_mask & (L_chan >= p25) & (L_chan <= p75)
            if not np.any(clean_body):
                clean_body = body_mask
            base_L = float(np.median(L_chan[clean_body]))
            base_A = float(np.median(A_chan[clean_body]))
            base_B = float(np.median(B_chan[clean_body]))
        else:
            base_L, base_A, base_B = 188.0, 135.0, 134.0

        # 2. Measure the French Tip smile-line center depth (y_center) directly from the nail's centerline
        active_ys = np.where(np.max(alpha, axis=1) > 0.2)[0]
        y_top_edge = int(active_ys[0]) if len(active_ys) > 0 else 0
        y_bot_edge = int(active_ys[-1]) if len(active_ys) > 0 else h - 1
        nail_h = max(1.0, float(y_bot_edge - y_top_edge))

        center_col_L = np.median(L_chan[:, int(w * 0.44):max(int(w * 0.44) + 1, int(w * 0.56))], axis=1)
        dark_rows = np.where((center_col_L < 115.0) & (np.arange(h) < y_top_edge + 0.45 * nail_h))[0]
        if len(dark_rows) > 0:
            y_smile_center = float(np.max(dark_rows))
        else:
            y_smile_center = y_top_edge + 0.21 * nail_h

        # Construct a 100% bilaterally symmetric, C2-smooth U-shaped French smile-line curve
        # using per-row normalized horizontal coordinate u_row(y, x) in [-1, 1] relative to the
        # nail's actual alpha width at each row y!
        row_centers = np.full(h, 0.5 * float(w - 1), dtype=np.float32)
        row_half_ws = np.full(h, max(5.0, 0.35 * float(w)), dtype=np.float32)
        for ry in range(y_top_edge, y_bot_edge + 1):
            r_xs = np.where(alpha[ry] > 0.15)[0]
            if len(r_xs) >= 2:
                row_centers[ry] = 0.5 * float(r_xs[0] + r_xs[-1])
                # Use 0.86 of the row half-width so the French tip wings reach full depth inside the sidewall tuck
                row_half_ws[ry] = max(3.0, 0.43 * float(r_xs[-1] - r_xs[0]))
        row_half_ws = cv2.GaussianBlur(row_half_ws.reshape(-1, 1), (1, 15), 3.5).ravel()

        x_grid = np.arange(w, dtype=np.float32)[None, :]
        y_grid = np.arange(h, dtype=np.float32)[:, None]
        u_norm = np.clip((x_grid - row_centers[:, None]) / row_half_ws[:, None], -1.25, 1.25)

        wing_drop = 0.23 * nail_h
        y_smile_curve = y_smile_center + wing_drop * (np.abs(u_norm) ** 2.0)

        # Sub-pixel anti-aliased transition across the smile line (1.2px feather)
        smile_analytic = np.clip((y_smile_curve - y_grid) / 1.25 + 0.5, 0.0, 1.0)
        tip_darkness = np.clip(smile_analytic, 0.0, 1.0)

        # 3. Reconstruct warm, rosy sheer-blush studio base for the Nude-Pink region
        # (Warming LAB A & B slightly so the pale card-lit pink matches a warm human nail bed)
        studio_pink_L = float(np.clip(base_L + 1.0, 180.0, 196.0))
        studio_pink_A = float(np.clip(base_A + 4.2, 137.5, 144.0))
        studio_pink_B = float(np.clip(base_B + 2.2, 134.5, 140.0))

        pink_glare = np.clip((L_chan - (base_L + 5.0)) / 9.0, 0.0, 1.0)
        pink_shadow = np.clip(((base_L - 5.0) - L_chan) / 11.0, 0.0, 1.0) * (1.0 - tip_darkness)
        pink_defect = cv2.GaussianBlur(np.clip(pink_glare + pink_shadow, 0.0, 1.0), (9, 9), 2.5)

        clean_pink_L = (1.0 - pink_defect) * L_chan + pink_defect * studio_pink_L
        clean_pink_L = 0.82 * studio_pink_L + 0.18 * cv2.GaussianBlur(clean_pink_L, (17, 17), 4.5)
        clean_pink_A = np.full_like(A_chan, studio_pink_A)
        clean_pink_B = np.full_like(B_chan, studio_pink_B)

        # 4. Rich, deep jet-black gel lacquer for the French Tip
        tip_L = np.full_like(L_chan, 12.0)
        tip_A = np.full_like(A_chan, 128.0)
        tip_B = np.full_like(B_chan, 127.0)

        final_L = (1.0 - tip_darkness) * clean_pink_L + tip_darkness * tip_L
        final_A = (1.0 - tip_darkness) * clean_pink_A + tip_darkness * tip_A
        final_B = (1.0 - tip_darkness) * clean_pink_B + tip_darkness * tip_B

        reconstructed_lab = np.stack([
            np.clip(final_L, 0, 255),
            np.clip(final_A, 0, 255),
            np.clip(final_B, 0, 255),
        ], axis=-1).astype(np.uint8)

        out[:, :, :3] = cv2.cvtColor(reconstructed_lab, cv2.COLOR_LAB2BGR)
        return out

    # --- PATH A: MONOCHROME SOLID NAIL ---
    k_core = max(5, int(min(h, w) * 0.14) | 1)
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core))
    interior_mask = cv2.erode((alpha > 0.85).astype(np.uint8), kernel_core) > 0
    if not np.any(interior_mask):
        interior_mask = valid

    L_int = L_chan[interior_mask]
    p25 = float(np.percentile(L_int, 25))
    p75 = float(np.percentile(L_int, 75))
    clean_pigment_mask = interior_mask & (L_chan >= p25) & (L_chan <= p75)

    if np.any(clean_pigment_mask):
        base_L = float(np.median(L_chan[clean_pigment_mask]))
        base_A = float(np.median(A_chan[clean_pigment_mask]))
        base_B = float(np.median(B_chan[clean_pigment_mask]))
    else:
        base_L = float(np.median(L_int))
        base_A = float(np.median(A_chan[interior_mask]))
        base_B = float(np.median(B_chan[interior_mask]))

    clean_sat = float(np.median(sat[clean_pigment_mask])) if np.any(clean_pigment_mask) else 60.0

    glare_score = np.clip((L_chan - (base_L + 9.0)) / 16.0, 0.0, 1.0)
    desat_score = np.clip(((clean_sat - 7.0) - sat) / max(12.0, clean_sat * 0.45), 0.0, 1.0)
    glare_weight = np.maximum(
        glare_score,
        desat_score * np.clip((L_chan - (base_L - 8.0)) / 14.0, 0.0, 1.0),
    )

    k_glare = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    glare_weight = cv2.dilate(glare_weight, k_glare)
    glare_weight = cv2.GaussianBlur(glare_weight, (15, 15), 4.5)

    shadow_weight = np.clip(((base_L - 3.0) - L_chan) / 13.0, 0.0, 1.0)
    shadow_weight = cv2.GaussianBlur(shadow_weight, (15, 15), 4.5)

    defect_weight = np.clip(glare_weight + shadow_weight, 0.0, 1.0)

    clean_L_field = np.where(defect_weight > 0.22, base_L, L_chan)
    clean_A_field = np.where(defect_weight > 0.22, base_A, A_chan)
    clean_B_field = np.where(defect_weight > 0.22, base_B, B_chan)

    clean_L_field = cv2.GaussianBlur(clean_L_field, (31, 31), 10.0)
    clean_A_field = cv2.GaussianBlur(clean_A_field, (31, 31), 10.0)
    clean_B_field = cv2.GaussianBlur(clean_B_field, (31, 31), 10.0)

    final_L = 0.76 * base_L + 0.24 * clean_L_field
    final_A = 0.82 * base_A + 0.18 * clean_A_field
    final_B = 0.82 * base_B + 0.18 * clean_B_field

    reconstructed_lab = np.stack([
        np.clip(final_L, 0, 255),
        np.clip(final_A, 0, 255),
        np.clip(final_B, 0, 255),
    ], axis=-1).astype(np.uint8)

    out[:, :, :3] = cv2.cvtColor(reconstructed_lab, cv2.COLOR_LAB2BGR)
    return out


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
) -> None:
    """
    Executes the 6-pass PBR studio relighting and compositing pipeline on a single warped nail ROI,
    matching the studio key/fill lighting and natural high-gloss gel reflections of hand-template.jpg.
    """
    rx, ry, rw, rh = bbox
    if rw <= 0 or rh <= 0:
        return

    warped_bgr = warped_roi[:, :, :3].astype(np.float32)
    raw_alpha = warped_roi[:, :, 3].astype(np.float32) / 255.0

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

    tucked_alpha = raw_alpha * pocket_allow
    alpha_smooth = cv2.GaussianBlur(tucked_alpha, (3, 3), 0.75)[:, :, None]

    # Pass 2: 3D Cylindrical & Longitudinal Apex Normals
    roll_offset = float(target.get("roll_offset", 0.0))
    u_map, v_map, nx, ny, nz = compute_3d_nail_geometry(tucked_alpha, roll_offset=roll_offset)

    # Pass 3: Studio Key + Fill Diffuse Shading & Sheer Nail-Bed Subsurface Translucency
    L_key = np.array([0.64, -0.14, 0.755], dtype=np.float32)
    L_key /= np.linalg.norm(L_key)

    L_fill = np.array([-0.50, -0.05, 0.864], dtype=np.float32)
    L_fill /= np.linalg.norm(L_fill)

    ndotl_key = np.clip(nx * L_key[0] + ny * L_key[1] + nz * L_key[2], 0.0, 1.0)
    ndotl_fill = np.clip(nx * L_fill[0] + ny * L_fill[1] + nz * L_fill[2], 0.0, 1.0)

    hand_patch = hand_orig[ry:ry + rh, rx:rx + rw].astype(np.float32)
    hand_curr = composite_hand[ry:ry + rh, rx:rx + rw]
    hand_gray = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hand_shading_blur = cv2.GaussianBlur(hand_gray, (25, 25), 8.0)

    mask_bool = tucked_alpha > 0.15
    mean_lum = float(np.mean(hand_shading_blur[mask_bool])) if np.any(mask_bool) else 172.0
    local_hand_ratio = np.clip(hand_shading_blur / max(1.0, mean_lum), 0.84, 1.12)

    right_sidewall_roll = 1.0 - 0.14 * np.clip((u_map - 0.72) / 0.28, 0.0, 1.0) ** 1.5
    cylinder_diffuse = (0.48 + 0.46 * ndotl_key + 0.13 * ndotl_fill) * right_sidewall_roll
    total_diffuse = cylinder_diffuse * (0.68 + 0.32 * local_hand_ratio)

    apex_dome_lift = 0.94 + 0.08 * np.sin(np.clip(v_map, 0.0, 1.0) * np.pi)
    total_diffuse *= apex_dome_lift

    # Detect light/sheer nude-pink regions vs dark pigment/French tips
    nail_lum = 0.114 * warped_bgr[:, :, 0] + 0.587 * warped_bgr[:, :, 1] + 0.299 * warped_bgr[:, :, 2]
    is_light_pink = np.clip((nail_lum - 125.0) / 45.0, 0.0, 1.0)

    # For sheer nude-pink regions, blend 24% with a smoothed version of the underlying natural
    # rosy nail-bed skin tone so the press-on has authentic sheer acrylic depth & warmth!
    hand_bed_smooth = cv2.GaussianBlur(hand_patch, (19, 19), 5.0)
    sheer_translucency = is_light_pink * np.clip((0.78 - v_map) / 0.20, 0.0, 1.0)
    base_with_subsurface = (
        warped_bgr * (1.0 - 0.24 * sheer_translucency[:, :, None])
        + hand_bed_smooth * (0.24 * sheer_translucency[:, :, None])
    )

    effective_diffuse = (1.0 - is_light_pink) * total_diffuse + is_light_pink * (0.26 + 0.74 * total_diffuse)
    soft_cuticle_ao = 1.0 - 0.72 * (1.0 - cuticle_ao)
    lit_bgr = base_with_subsurface * effective_diffuse[:, :, None] * soft_cuticle_ao[:, :, None]

    # Subtle Fresnel grazing rim reflection along the extreme left edge
    left_rim_mask = np.clip((-u_map - 0.74) / 0.24, 0.0, 1.0) ** 2.0 * np.sin(np.clip(v_map, 0.05, 0.92) * np.pi)
    env_color = np.array([205.0, 208.0, 216.0], dtype=np.float32)[None, None, :]
    lit_bgr = lit_bgr + left_rim_mask[:, :, None] * 0.10 * env_color

    # Pass 4: High-Gloss Clear Gel Topcoat Specular Reflections (Matched to hand-template.jpg)
    spec_u_center = 0.33 - roll_offset * 0.45
    u_delta = u_map - spec_u_center

    # 4a. Broad studio sheen across the upper-right dome (toned down on light pink so it never washes out chroma)
    v_sheen_env = np.sin(np.clip((v_map - 0.05) / 0.88, 0.0, 1.0) * np.pi)
    broad_sheen = np.exp(-0.5 * (u_delta / 0.21) ** 2) * v_sheen_env * (1.0 - 0.55 * is_light_pink)

    # 4b. Crisp, high-gloss softbox window reflection ribbon running up the dome and across the tip
    v_rib_in = np.clip((v_map - 0.13) / 0.16, 0.0, 1.0)
    v_rib_out = np.clip((0.86 - v_map) / 0.14, 0.0, 1.0)
    apex_hotspot = 0.74 + 0.26 * np.exp(-0.5 * ((v_map - 0.48) / 0.18) ** 2)
    v_rib_env = (v_rib_in ** 1.3) * (v_rib_out ** 1.4) * apex_hotspot

    rib_half_w = 0.036 * (0.55 + 0.45 * np.sin(np.clip((v_map - 0.10) / 0.78, 0.0, 1.0) * np.pi))
    gloss_ribbon = np.exp(-0.5 * (np.abs(u_delta) / np.maximum(0.012, rib_half_w)) ** 2.4) * v_rib_env

    # 4c. Soft secondary rim sheen near the right lateral bevel
    bevel_u_delta = u_map - (0.66 - roll_offset * 0.35)
    bevel_v_env = np.maximum(0.0, np.sin(np.clip((v_map - 0.20) / 0.50, 0.0, 1.0) * np.pi)) ** 1.4
    bevel_sheen = np.exp(-0.5 * (bevel_u_delta / 0.038) ** 2) * bevel_v_env * (1.0 - 0.45 * is_light_pink)

    # 4d. Soft lower-left studio reflector fill sheen
    fill_u_delta = np.abs(u_map - (-0.42 - roll_offset * 0.30))
    fill_v_env = np.maximum(0.0, np.sin(np.clip((v_map - 0.10) / 0.48, 0.0, 1.0) * np.pi)) ** 1.3
    fill_reflection = np.exp(-0.5 * (fill_u_delta / 0.065) ** 2.0) * fill_v_env * (1.0 - 0.55 * is_light_pink)

    spec_total = (
        0.13 * broad_sheen
        + 0.88 * gloss_ribbon
        + 0.11 * bevel_sheen
        + 0.11 * fill_reflection
    )
    spec_total = cv2.GaussianBlur(spec_total, (3, 3), 0.7)

    spec_color = np.array([248.0, 251.0, 255.0], dtype=np.float32)[None, None, :]
    spec_alpha = np.clip(spec_total[:, :, None], 0.0, 0.90)
    lit_bgr = lit_bgr * (1.0 - spec_alpha * 0.78) + spec_color * spec_alpha

    # Pass 5: Warm Skin Subsurface Scatter / Cuticle Rim Integration
    cuticle_edge_band = np.clip((1.0 - cuticle_ao) * 3.0, 0.0, 1.0) * np.clip(1.0 - v_map * 3.2, 0.0, 1.0)
    warm_skin_tint = hand_patch * 0.82
    lit_bgr = (
        lit_bgr * (1.0 - 0.18 * cuticle_edge_band[:, :, None])
        + warm_skin_tint * (0.18 * cuticle_edge_band[:, :, None])
    )

    # Pass 6: Match camera sensor micro-grain from hand-template.jpg
    rng = np.random.default_rng(seed=int(rx * 31 + ry * 17))
    sensor_grain = rng.normal(0.0, 1.15, size=(rh, rw, 1)).astype(np.float32)
    lit_bgr = np.clip(lit_bgr + sensor_grain, 0.0, 255.0)

    # Under-Tip Fingertip Cast Shadow
    M_sh = np.float32([[1.0, 0.0, -1.2], [0.0, 1.0, 3.2]])
    shifted_alpha = cv2.warpAffine(tucked_alpha, M_sh, (rw, rh), flags=cv2.INTER_LINEAR)
    shadow_blur = cv2.GaussianBlur(shifted_alpha, (9, 9), 2.5)

    patch_hsv = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
    skin_weight = np.clip((patch_hsv[:, :, 1].astype(np.float32) - 18.0) / 25.0, 0.0, 1.0)
    under_shadow_mult = 1.0 - 0.18 * shadow_blur * (1.0 - tucked_alpha) * skin_weight
    hand_with_shadow = hand_curr * under_shadow_mult[:, :, None]

    # Final sub-pixel anti-aliased composite
    composite_hand[ry:ry + rh, rx:rx + rw] = (
        lit_bgr * alpha_smooth + hand_with_shadow * (1.0 - alpha_smooth)
    )
