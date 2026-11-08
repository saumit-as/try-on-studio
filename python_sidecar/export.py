"""
Press-On Nail Studio - Phase 5 High-Res Production Export Engine
(docs/phases-roadmap.md, Phase 5)

Provides:
  1. 300 DPI Production PNG Export:
     - Embedded pHYs metadata for physical print compliance (300 DPI = 11811 px/m)
     - Presets: Original, 1:1 Square, 4:5 Social Portrait, 16:9 Banner
     - Solid Studio Backdrop or Transparent Cutout
  2. Multi-Layer Photoshop (.psd) Export:
     - Preserves 5 discrete layers with authentic blend modes and alpha channels:
       * Layer 4 (Top): Cuticle Pocket Occlusion Lip (Normal, RGBA)
       * Layer 3: Salon Gel Topcoat Specular Ribbons (Screen / Linear Dodge, RGBA)
       * Layer 2: Warped & Relit Press-On Nails (Normal, RGBA)
       * Layer 1: Contact AO & Drop Shadows (Multiply, RGBA, ~85% opacity)
       * Layer 0 (Bottom): Bare Hand Base (Normal, RGB / RGBA)
     - Includes automated NumPy 2.x compatibility monkey-patch for pytoshop RLE compression
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
from PIL import Image

# Setup pytoshop and monkey-patch for NumPy 2.x virtual channel compression bug
import pytoshop
import pytoshop.codecs
import pytoshop.enums
from pytoshop.user import nested_layers

try:
    import packbits
    pytoshop.codecs.packbits = packbits
except ImportError:
    pass

_orig_compress_image = pytoshop.codecs.compress_image

def _patched_compress_image(fd, image, compression, shape, num_channels, depth, version):
    """
    NumPy 2.x raises OverflowError on `np.dtype('uint8').type(-1)` when pytoshop
    compresses virtual channels (e.g. constant transparency value of -1).
    Mask negative scalar integers to valid unsigned bit ranges.
    """
    if isinstance(image, int):
        if depth == pytoshop.enums.ColorDepth.depth8:
            image = image & 0xff
        elif depth == pytoshop.enums.ColorDepth.depth16:
            image = image & 0xffff
        elif depth == pytoshop.enums.ColorDepth.depth32:
            image = image & 0xffffffff
    return _orig_compress_image(fd, image, compression, shape, num_channels, depth, version)

pytoshop.codecs.compress_image = _patched_compress_image


from compositor import (
    clean_tray_glare_preserving_texture,
    apply_cylindrical_c_curve,
    build_3tier_control_grid,
    warp_nail_tps,
    build_cuticle_pocket_mask,
    compute_3d_nail_geometry,
)


def apply_aspect_ratio_framing(
    img: np.ndarray,
    target_ratio: str = "original",
    fill_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Adjusts image canvas to the requested target aspect ratio without distorting or cropping the hand.
    Uses symmetrical canvas extension filled with sampled background tone or transparency.

    Supported ratios:
      - 'original': returns img unmodified
      - '1:1': Square (width expanded to height or vice versa)
      - '4:5': Vertical social/commerce portrait (0.80 aspect ratio)
      - '16:9': Horizontal presentation banner (1.777 aspect ratio)
    """
    if target_ratio == "original" or not target_ratio:
        return img

    h, w = img.shape[:2]
    is_rgba = img.shape[2] == 4

    # Determine desired target (target_w, target_h)
    if target_ratio in ("1:1", "square"):
        target_dim = max(h, w)
        target_w, target_h = target_dim, target_dim
    elif target_ratio in ("4:5", "portrait"):
        # 4:5 width to height ratio = 0.8
        req_w = int(round(h * 0.8))
        if req_w >= w:
            target_w, target_h = req_w, h
        else:
            target_w, target_h = w, int(round(w / 0.8))
    elif target_ratio in ("16:9", "banner"):
        # 16:9 width to height ratio = 1.777
        req_w = int(round(h * (16.0 / 9.0)))
        if req_w >= w:
            target_w, target_h = req_w, h
        else:
            target_w, target_h = w, int(round(w * (9.0 / 16.0)))
    else:
        return img

    # Compute background fill color
    if is_rgba:
        # Check if alpha has transparent pixels
        if np.mean(img[:, :, 3]) < 240:
            pad_val = (0, 0, 0, 0)
        else:
            corner_sample = np.median(img[:15, :15, :3], axis=(0, 1)).astype(int)
            pad_val = (int(corner_sample[0]), int(corner_sample[1]), int(corner_sample[2]), 255)
    else:
        if fill_color is not None:
            pad_val = fill_color
        else:
            corner_sample = np.median(img[:15, :15, :3], axis=(0, 1)).astype(int)
            pad_val = (int(corner_sample[0]), int(corner_sample[1]), int(corner_sample[2]))

    # Create new canvas and center original image
    new_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    new_img[:] = pad_val

    off_x = (target_w - w) // 2
    off_y = (target_h - h) // 2
    new_img[off_y:off_y + h, off_x:off_x + w] = img

    return new_img


def decompose_composite_layers(
    hand_bgr: np.ndarray,
    landmarks: Dict[str, Any],
    mapped_nails_rgba: Dict[str, np.ndarray],
    lighting_options: Optional[Dict[str, Any]] = None,
    transparent_background: bool = False,
    hand_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Decomposes the 6-pass PBR compositor into 5 discrete, isolated full-canvas layers
    for layered Photoshop (.psd) export.

    Returns dict with keys:
      - 'bare_hand': Base hand photo (RGB or RGBA)
      - 'contact_shadows': Contact AO & fingertip drop shadows (RGBA, Multiply blend)
      - 'press_on_nails': Warped, surface-shaded & lit press-on nails (RGBA, Normal blend)
      - 'gel_highlights': Salon gel topcoat specular ribbons & sheen (RGBA, Screen blend)
      - 'cuticle_lip': Proximal skin fold overlapping the nail plate (RGBA, Normal blend)
    """
    h_hand, w_hand = hand_bgr.shape[:2]
    opt = lighting_options or {}
    overall_len_scale = float(opt.get("nailLengthScale", 1.0))
    c_curve_user = float(opt.get("cCurveIntensity", 1.0))
    gloss_mult = float(opt.get("glossIntensity", 1.0))
    shadow_mult = float(opt.get("shadowDepth", 1.0))
    light_angle_deg = float(opt.get("lightAngle", 0.0))
    preset = str(opt.get("preset", "studio"))

    # 1. Base Hand Layer
    if transparent_background and hand_mask is not None:
        mask_u8 = hand_mask if hand_mask.dtype == np.uint8 else (hand_mask * 255).astype(np.uint8)
        bare_hand = np.dstack([hand_bgr, mask_u8])
    else:
        bare_hand = hand_bgr.copy()

    # Allocate full-canvas layer buffers (RGBA float32 for clean accumulation)
    shadow_canvas = np.zeros((h_hand, w_hand, 4), dtype=np.float32)
    nails_canvas = np.zeros((h_hand, w_hand, 4), dtype=np.float32)
    spec_canvas = np.zeros((h_hand, w_hand, 4), dtype=np.float32)
    cuticle_canvas = np.zeros((h_hand, w_hand, 4), dtype=np.float32)

    cleaned_nails: Dict[str, np.ndarray] = {}
    for fname, rgba in mapped_nails_rgba.items():
        if rgba is not None and rgba.size > 0:
            cleaned_nails[fname] = clean_tray_glare_preserving_texture(rgba)

    if not cleaned_nails:
        return {
            "bare_hand": bare_hand,
            "contact_shadows": shadow_canvas.astype(np.uint8),
            "press_on_nails": nails_canvas.astype(np.uint8),
            "gel_highlights": spec_canvas.astype(np.uint8),
            "cuticle_lip": cuticle_canvas.astype(np.uint8),
        }

    finger_keys = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    hand_bed_widths = []
    tray_widths = []
    for fname in finger_keys:
        if fname in landmarks and fname in cleaned_nails:
            lm = landmarks[fname]
            cw = float(np.linalg.norm(np.array(lm["c_r"]) - np.array(lm["c_l"])))
            dw = float(np.linalg.norm(np.array(lm["d_r"]) - np.array(lm["d_l"])))
            hand_bed_widths.append(max(cw, dw))
            nd = cleaned_nails[fname]
            xs = np.where(nd[:, :, 3] > 25)[1]
            tray_widths.append(float(np.max(xs) - np.min(xs) + 1) if len(xs) > 0 else float(nd.shape[1]))

    tray_scale = 1.0
    if len(hand_bed_widths) > 0 and len(tray_widths) > 0 and sum(tray_widths) > 0:
        tray_scale = (sum(hand_bed_widths) / max(1.0, sum(tray_widths))) * 1.05

    for fname in finger_keys:
        if fname not in landmarks or fname not in cleaned_nails:
            continue
        lm = landmarks[fname]
        clean_rgba = cleaned_nails[fname]

        finger_adj = opt.get("finger_adjustments", {}).get(fname, {})
        c_curve_val = float(finger_adj.get("cCurve", finger_adj.get("c_curve", lm.get("c_curve", 0.28))))
        c_curve = c_curve_val * c_curve_user
        roll = float(finger_adj.get("roll_offset", lm.get("roll_offset", 0.0)))

        curved = apply_cylindrical_c_curve(clean_rgba, c_curve_intensity=c_curve, roll_offset=roll)
        src_h, src_w = curved.shape[:2]

        c_l = [lm["c_l"][0], lm["c_l"][1]]
        c_c = [lm["c_c"][0], lm["c_c"][1]]
        c_r = [lm["c_r"][0], lm["c_r"][1]]
        d_l = [lm["d_l"][0], lm["d_l"][1]]
        d_r = [lm["d_r"][0], lm["d_r"][1]]

        len_scale = float(finger_adj.get("lengthScale", finger_adj.get("length_scale", lm.get("length_scale", 1.0)))) * overall_len_scale
        width_scale = float(finger_adj.get("widthScale", finger_adj.get("width_scale", lm.get("width_scale", 1.0))))
        axial_shift = float(finger_adj.get("axialShift", finger_adj.get("axial_shift", lm.get("axial_shift", 0.0))))
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

        warped_roi, bbox = warp_nail_tps(curved, src_pts, dst_pts, output_shape=(h_hand, w_hand))
        rx, ry, rw, rh = bbox
        if rw <= 0 or rh <= 0:
            continue

        warped_bgr = warped_roi[:, :, :3].astype(np.float32)
        raw_alpha = warped_roi[:, :, 3].astype(np.float32) / 255.0

        # Pass 1: Anatomical Cuticle Pocket Tuck Mask
        pocket_allow, cuticle_ao = build_cuticle_pocket_mask(
            roi_bbox=bbox,
            cuticle_left=c_l,
            cuticle_center=c_c,
            cuticle_right=c_r,
            distal_left=d_l,
            distal_right=d_r,
            feather_sigma=1.5,
        )

        tucked_alpha = raw_alpha * pocket_allow
        alpha_smooth = cv2.GaussianBlur(tucked_alpha, (3, 3), 0.75)

        # Cuticle skin fold overlap (skin lip covering the tucked nail base)
        lip_alpha = np.clip(raw_alpha * (1.0 - pocket_allow) * 2.0, 0.0, 1.0)
        lip_alpha = cv2.GaussianBlur(lip_alpha, (3, 3), 0.8)

        # Pass 2: 3D Surface Normals & Diffuse Relighting
        u_map, v_map, nx, ny, nz = compute_3d_nail_geometry(alpha_smooth, roll_offset=roll)
        hand_patch = hand_bgr[ry:ry + rh, rx:rx + rw].astype(np.float32)

        # Translucency in proximal nail bed
        lab_nail = cv2.cvtColor(warped_roi[:, :, :3], cv2.COLOR_BGR2LAB)
        L_nail = lab_nail[:, :, 0].astype(np.float32)
        L_mean = cv2.blur(L_nail, (9, 9))
        L_sq_mean = cv2.blur(L_nail ** 2, (9, 9))
        local_std = np.sqrt(np.maximum(0.0, L_sq_mean - L_mean ** 2))

        interior_valid = raw_alpha > 0.4
        interior_lum_std = float(np.std(L_nail[interior_valid])) if np.sum(interior_valid) > 25 else 0.0

        translucency_eligible = (
            np.clip((0.62 - v_map) / 0.18, 0.0, 1.0)
            * np.clip((1.0 - np.abs(u_map)) / 0.35, 0.0, 1.0)
            * np.clip((4.0 - local_std) / 2.5, 0.0, 1.0)
        )
        subsurface_weight = np.clip(0.10 * translucency_eligible, 0.0, 0.10)
        base_with_subsurface = (
            warped_bgr * (1.0 - subsurface_weight[:, :, None])
            + hand_patch * (0.85 * subsurface_weight[:, :, None])
        )

        # Constrained Diffuse Relighting
        u_abs = np.abs(u_map)
        center_diffuse = 0.97 + 0.03 * np.cos(np.clip(u_abs / 0.55, 0.0, 1.0) * (np.pi * 0.5))
        w_side = np.clip((u_abs - 0.55) / 0.45, 0.0, 1.0)
        target_sidewall = np.where(u_map > 0, 0.90, 0.84)
        diffuse_ratio = np.where(
            u_abs <= 0.55,
            center_diffuse,
            (1.0 - w_side) * 1.00 + w_side * target_sidewall,
        )
        diffuse_ratio = np.clip(diffuse_ratio, 0.84, 1.04)

        soft_cuticle_ao = 0.88 + 0.12 * cuticle_ao
        lit_bgr = base_with_subsurface * (diffuse_ratio * soft_cuticle_ao)[:, :, None]

        # Subtle rim reflection along left edge
        left_rim_mask = (
            np.clip((-u_map - 0.74) / 0.24, 0.0, 1.0) ** 2.0 * np.sin(np.clip(v_map, 0.05, 0.92) * np.pi)
        )
        env_color = np.array([205.0, 208.0, 216.0], dtype=np.float32)[None, None, :]
        lit_bgr = lit_bgr + left_rim_mask[:, :, None] * 0.06 * env_color

        # Pass 4: Topcoat Specular Reflections
        spec_u_center = 0.30 - roll * 0.40 + (light_angle_deg / 180.0) * 0.20
        u_delta = u_map - spec_u_center
        v_sheen_env = np.sin(np.clip((v_map - 0.05) / 0.88, 0.0, 1.0) * np.pi)
        broad_sheen = np.exp(-0.5 * (u_delta / 0.18) ** 2) * v_sheen_env

        v_rib_in = np.clip((v_map - 0.10) / 0.15, 0.0, 1.0)
        v_rib_out = np.clip((0.90 - v_map) / 0.12, 0.0, 1.0)
        apex_hotspot = 0.80 + 0.20 * np.exp(-0.5 * ((v_map - 0.48) / 0.22) ** 2)
        v_rib_env = (v_rib_in ** 1.2) * (v_rib_out ** 1.3) * apex_hotspot
        rib_half_w = 0.026 * (0.60 + 0.40 * np.sin(np.clip((v_map - 0.08) / 0.82, 0.0, 1.0) * np.pi))
        gloss_ribbon = np.exp(-0.5 * (np.abs(u_delta) / np.maximum(0.008, rib_half_w)) ** 2.2) * v_rib_env

        fill_u_delta = np.abs(u_map - (-0.46 - roll * 0.30))
        fill_v_env = np.maximum(0.0, np.sin(np.clip((v_map - 0.10) / 0.55, 0.0, 1.0) * np.pi)) ** 1.2
        fill_reflection = np.exp(-0.5 * (fill_u_delta / 0.055) ** 2.0) * fill_v_env

        if preset == "ring":
            spec_total = (0.25 * broad_sheen + 0.65 * gloss_ribbon + 0.25 * fill_reflection) * gloss_mult
        elif preset == "flash":
            spec_total = (0.05 * broad_sheen + 1.45 * gloss_ribbon + 0.05 * fill_reflection) * gloss_mult
        elif preset == "natural":
            spec_total = (0.45 * broad_sheen + 0.40 * gloss_ribbon + 0.15 * fill_reflection) * gloss_mult
        else:
            spec_total = (0.08 * broad_sheen + 1.05 * gloss_ribbon + 0.09 * fill_reflection) * gloss_mult

        spec_total = cv2.GaussianBlur(spec_total, (3, 3), 0.5)
        high_contrast_weight = float(np.clip((interior_lum_std - 18.0) / 10.0, 0.0, 1.0))
        spec_scale = 1.0 - 0.45 * high_contrast_weight
        spec_color = np.array([252.0, 254.0, 255.0], dtype=np.float32)
        spec_alpha_isolated = np.clip(spec_total * spec_scale * alpha_smooth, 0.0, 0.95)

        # Micro-grain on nail plate
        rng = np.random.default_rng(seed=int(rx * 31 + ry * 17))
        sensor_grain = rng.normal(0.0, 1.15, size=(rh, rw, 1)).astype(np.float32)
        lit_bgr = np.clip(lit_bgr + sensor_grain, 0.0, 255.0)

        # Pass 6: Fingertip Cast Shadow
        M_sh = np.float32([[1.0, 0.0, 1.6 + light_angle_deg * 0.05], [0.0, 1.0, 4.2]])
        shifted_alpha = cv2.warpAffine(tucked_alpha, M_sh, (rw, rh), flags=cv2.INTER_LINEAR)
        shadow_blur = cv2.GaussianBlur(shifted_alpha, (11, 11), 3.2)
        patch_hsv = cv2.cvtColor(hand_patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
        skin_weight = np.clip((patch_hsv[:, :, 1].astype(np.float32) - 18.0) / 25.0, 0.0, 1.0)
        shadow_strength = (0.28 * shadow_mult) * shadow_blur * (1.0 - tucked_alpha) * skin_weight

        # Accumulate onto full-canvas layers
        # 1. Contact Shadows Layer (Dark warm tone for Multiply blend)
        shadow_color = np.array([28.0, 24.0, 36.0], dtype=np.float32)  # BGR subtle skin shadow
        cur_sh_alpha = shadow_canvas[ry:ry + rh, rx:rx + rw, 3] / 255.0
        new_sh_alpha = np.maximum(cur_sh_alpha, shadow_strength)
        shadow_canvas[ry:ry + rh, rx:rx + rw, :3] = shadow_color[None, None, :]
        shadow_canvas[ry:ry + rh, rx:rx + rw, 3] = new_sh_alpha * 255.0

        # 2. Press-On Nails Layer
        nail_alpha_255 = alpha_smooth * 255.0
        nails_canvas[ry:ry + rh, rx:rx + rw, :3] = lit_bgr
        nails_canvas[ry:ry + rh, rx:rx + rw, 3] = nail_alpha_255

        # 3. Gel Highlights Layer (Screen / Linear Dodge blend)
        spec_canvas[ry:ry + rh, rx:rx + rw, :3] = spec_color[None, None, :]
        spec_canvas[ry:ry + rh, rx:rx + rw, 3] = spec_alpha_isolated * 255.0

        # 4. Cuticle Pocket Skin Lip Layer
        cuticle_canvas[ry:ry + rh, rx:rx + rw, :3] = hand_patch
        cuticle_canvas[ry:ry + rh, rx:rx + rw, 3] = lip_alpha * 255.0

    return {
        "bare_hand": bare_hand,
        "contact_shadows": np.clip(shadow_canvas, 0, 255).astype(np.uint8),
        "press_on_nails": np.clip(nails_canvas, 0, 255).astype(np.uint8),
        "gel_highlights": np.clip(spec_canvas, 0, 255).astype(np.uint8),
        "cuticle_lip": np.clip(cuticle_canvas, 0, 255).astype(np.uint8),
    }


def export_production_png(
    composite_bgr_or_rgba: np.ndarray,
    output_path: str,
    dpi: int = 300,
    aspect_ratio: str = "original",
) -> str:
    """
    Exports a high-resolution production PNG with embedded physical 300 DPI metadata.
    Pillow embeds the standard PNG 'pHYs' chunk with dots-per-meter:
      300 DPI = round(300 * 39.37007874) = 11811 pixels/meter.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    framed = apply_aspect_ratio_framing(composite_bgr_or_rgba, target_ratio=aspect_ratio)

    if framed.shape[2] == 4:
        # RGBA: convert BGR to RGB
        rgb = cv2.cvtColor(framed[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = framed[:, :, 3]
        rgba = np.dstack([rgb, alpha])
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        rgb = cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")

    pil_img.save(output_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return output_path


def export_layered_psd(
    layers_dict: Dict[str, np.ndarray],
    output_path: str,
    aspect_ratio: str = "original",
) -> str:
    """
    Exports an editable multi-layer Photoshop (.psd) file preserving all 5 isolated PBR passes:
      Layer 4: Cuticle Pocket Lip (Normal, RGBA)
      Layer 3: Gel Topcoat Highlights (Screen, RGBA)
      Layer 2: Press-On Nails (Normal, RGBA)
      Layer 1: Contact AO & Drop Shadows (Multiply, RGBA, 84% opacity)
      Layer 0: Bare Hand Base (Normal, RGB or RGBA)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Apply aspect ratio framing consistently across all layers
    framed_layers: Dict[str, np.ndarray] = {}
    for name, img in layers_dict.items():
        framed_layers[name] = apply_aspect_ratio_framing(img, target_ratio=aspect_ratio)

    sample_h, sample_w = framed_layers["bare_hand"].shape[:2]

    def _make_psd_layer(
        img_bgr_or_rgba: np.ndarray,
        layer_name: str,
        blend_mode: pytoshop.enums.BlendMode,
        opacity_val: int = 255,
    ) -> nested_layers.Image:
        h, w = img_bgr_or_rgba.shape[:2]
        is_4ch = img_bgr_or_rgba.shape[2] == 4

        # Convert BGR channels to RGB for Photoshop
        r_chan = img_bgr_or_rgba[:, :, 2]
        g_chan = img_bgr_or_rgba[:, :, 1]
        b_chan = img_bgr_or_rgba[:, :, 0]

        channels = {
            0: r_chan,
            1: g_chan,
            2: b_chan,
        }
        if is_4ch:
            channels[-1] = img_bgr_or_rgba[:, :, 3]

        return nested_layers.Image(
            name=layer_name,
            visible=True,
            opacity=opacity_val,
            blend_mode=blend_mode,
            top=0,
            left=0,
            bottom=h,
            right=w,
            channels=channels,
        )

    # Build layers from top to bottom (pytoshop inverts the list so layers[0] is topmost in Photoshop)
    psd_layer_list = [
        _make_psd_layer(
            framed_layers["cuticle_lip"],
            "Cuticle Pocket Lip Occlusion",
            pytoshop.enums.BlendMode.normal,
            opacity_val=255,
        ),
        _make_psd_layer(
            framed_layers["gel_highlights"],
            "Gel Topcoat Specular Highlights",
            pytoshop.enums.BlendMode.screen,
            opacity_val=255,
        ),
        _make_psd_layer(
            framed_layers["press_on_nails"],
            "Press-On Nails",
            pytoshop.enums.BlendMode.normal,
            opacity_val=255,
        ),
        _make_psd_layer(
            framed_layers["contact_shadows"],
            "Contact AO & Drop Shadows",
            pytoshop.enums.BlendMode.multiply,
            opacity_val=215,  # ~84% physical opacity
        ),
        _make_psd_layer(
            framed_layers["bare_hand"],
            "Bare Hand Base",
            pytoshop.enums.BlendMode.normal,
            opacity_val=255,
        ),
    ]

    psd = nested_layers.nested_layers_to_psd(
        psd_layer_list,
        color_mode=pytoshop.enums.ColorMode.rgb,
        depth=pytoshop.enums.ColorDepth.depth8,
        compression=pytoshop.enums.Compression.rle,
    )

    with open(output_path, "wb") as f:
        psd.write(f)

    return output_path


def export_composite_package(
    hand_bgr: np.ndarray,
    landmarks: Dict[str, Any],
    mapped_nails_rgba: Dict[str, np.ndarray],
    lighting_options: Optional[Dict[str, Any]] = None,
    export_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for Phase 5 production export requested via JSON-RPC.
    Generates 300 DPI PNG, Multi-Layer PSD, or both packages.
    """
    opt = export_options or {}
    export_format = str(opt.get("format", "both")).lower()  # "png", "psd", "both"
    aspect_ratio = str(opt.get("aspect_ratio", "original")).lower()  # "original", "1:1", "4:5", "16:9"
    dpi = int(opt.get("dpi", 300))
    transparent_bg = bool(opt.get("transparent_background", False))
    output_dir = str(opt.get("output_dir", "exports"))
    base_name = str(opt.get("filename", "try_on_studio_composite"))

    os.makedirs(output_dir, exist_ok=True)

    # 1. Decompose into the 5 discrete PBR layers
    layers = decompose_composite_layers(
        hand_bgr=hand_bgr,
        landmarks=landmarks,
        mapped_nails_rgba=mapped_nails_rgba,
        lighting_options=lighting_options,
        transparent_background=transparent_bg,
    )

    # 2. Render flattened composite for PNG export
    from compositor import render_full_hand_composite
    flattened_bgr = render_full_hand_composite(
        hand_bgr=hand_bgr,
        landmarks=landmarks,
        mapped_nails_rgba=mapped_nails_rgba,
        lighting_options=lighting_options,
        downscale=1.0,
    )

    if transparent_bg:
        # Attach hand cutout alpha
        hand_mask = layers["bare_hand"][:, :, 3] if layers["bare_hand"].shape[2] == 4 else None
        if hand_mask is not None:
            flattened_output = np.dstack([flattened_bgr, hand_mask])
        else:
            flattened_output = flattened_bgr
    else:
        flattened_output = flattened_bgr

    output_files: Dict[str, Any] = {}
    safe_ratio = aspect_ratio.replace(":", "-").replace("/", "-")
    safe_base = base_name.replace(":", "-").replace("/", "-")

    # Export PNG
    if export_format in ("png", "both"):
        png_filename = f"{safe_base}_{safe_ratio}_{dpi}dpi.png"
        png_path = os.path.join(output_dir, png_filename)
        export_production_png(
            flattened_output,
            png_path,
            dpi=dpi,
            aspect_ratio=aspect_ratio,
        )
        file_size = os.path.getsize(png_path)
        im_pil = Image.open(png_path)
        output_files["png"] = {
            "path": png_path,
            "filename": png_filename,
            "size_bytes": file_size,
            "dimensions": list(im_pil.size),
            "dpi": im_pil.info.get("dpi", (dpi, dpi)),
        }

    # Export PSD
    if export_format in ("psd", "both"):
        psd_filename = f"{safe_base}_{safe_ratio}_layered.psd"
        psd_path = os.path.join(output_dir, psd_filename)
        export_layered_psd(
            layers,
            psd_path,
            aspect_ratio=aspect_ratio,
        )
        file_size = os.path.getsize(psd_path)
        output_files["psd"] = {
            "path": psd_path,
            "filename": psd_filename,
            "size_bytes": file_size,
            "layers": [
                "Cuticle Pocket Lip Occlusion",
                "Gel Topcoat Specular Highlights",
                "Press-On Nails",
                "Contact AO & Drop Shadows",
                "Bare Hand Base",
            ],
        }

    return {
        "status": "ok",
        "output_dir": os.path.abspath(output_dir),
        "files": output_files,
    }
