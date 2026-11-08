# Technical Specification: Zero-Model Programmatic Retouching & True-Shape Fitting

**Target System:** Try-On Studio (`python_sidecar/segmentation.py`, `warping.py`, `compositor.py`, `template_generator.py`)  
**Objective:** Eliminate the "digital sticker" look, black perimeter fringes, flat-chopped tips, and metallic cylinder artifacts seen in `03_studio_relit_hand.jpg` and `image_e07d67.jpg` using 100% deterministic, classical image manipulation and computational geometry.

---

## 1. Root-Cause Audit of `37.jpg` + `segmentation_3.py`, `warping_2.py`, and `compositor_2.py`

By inspecting the latest code (`segmentation_3.py`, `warping_2.py`, `compositor_2.py`) against the soft sheer-pink tray image (`37.jpg`) and the resulting hand output (`image_e07d67.jpg`), here are the **6 exact bugs** in the current code:

### Bug 1: Why `37.jpg` (Soft Pink Almond/Oval Nails) Got Flat-Chopped Tips in `image_e07d67.jpg`
* **Exact Location:** `segmentation_3.py` lines 259–264 and lines 331–337 (`build_symmetric_smooth_nail_mask`).
* **What happens in the code:**
  1. Remember that in `37.jpg`, the nails are mounted **upside-down** (wide squoval cuticle at the top, rounded almond/oval tip at the bottom).
  2. Even when flipped right-side up (`y = 0` is tip), look at lines 260–264 of `segmentation_3.py`:
     ```python
     tip_zone_start = max(1, int(0.03 * nail_h))
     tip_zone_end = max(tip_zone_start + 1, int(0.15 * nail_h))
     w_tip_measured = float(np.median(w_pre[tip_zone_start:tip_zone_end]))
     flat_ratio = w_tip_measured / w_eq
     is_flat_tip = flat_ratio >= 0.55  # Square, Squoval, Coffin
     ```
     On the soft oval/almond nails in `37.jpg`, at `9%–15%` down from the tip, the width is already **`0.62 * w_eq`**! Because `0.62 >= 0.55`, `is_flat_tip` evaluates to **`True`**!
  3. Worse yet, if `should_invert` is ever `False` (when `exact_mask=True` or SAM is used, or when `score_extreme <= 1.04` on symmetric oval nails in `37.jpg`), the nail isn't flipped before `build_symmetric_smooth_nail_mask`, or the `else` branch (lines 290–338) runs and **inserts a flat horizontal line `[top_left, top_right]` across the top of the nail**! That is why every single nail in `image_e07d67.jpg` has a flat horizontal chopped tip!

### Bug 2: Why `warp_nail_tps` and `apply_cylindrical_c_curve` Create a Black Fringe Around Every Nail
* **Exact Location:** `warping_2.py` lines 97–104 (`apply_cylindrical_c_curve`) and lines 371–377 (`warp_nail_tps`).
* **What happens in the code:**
  In `segmentation_3.py`, your agent added `dematte_nail_edges` (which bleeds the pink nail color into the `alpha == 0` background of `patch_rgba`—a great start!).
  **HOWEVER**, look at what happens next in `warping_2.py`:
  1. First, `apply_cylindrical_c_curve` calls `cv2.remap(..., borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))` (line 103). Any pixel that maps outside `[0, w-1]` gets reset to **`(0, 0, 0, 0)` (PURE BLACK RGB)**!
  2. Second, in `warp_nail_tps` (line 346), the destination ROI box has `pad = 16`, which maps outside the source image bounds. `warp_nail_tps` calls:
     ```python
     warped_roi = cv2.remap(
         nail_rgba, map_x, map_y,
         interpolation=cv2.INTER_CUBIC,
         borderMode=cv2.BORDER_CONSTANT,
         borderValue=(0, 0, 0, 0)
     )
     ```
     Because `INTER_CUBIC` is a $4 \times 4$ bicubic kernel with negative lobes and `borderValue=(0, 0, 0, 0)` fills out-of-bounds with **black `[0,0,0]`**, `cv2.remap` blends the edge of the nail with `[0,0,0]` black AND produces bicubic ringing undershoots!
  3. **The 1-Line Fix in `warping_2.py`:** Both `cv2.remap` calls in `warping_2.py` **MUST** use `borderMode=cv2.BORDER_REPLICATE` for the RGB channels `(nail_rgba[:, :, :3])` and `borderMode=cv2.BORDER_CONSTANT, borderValue=0` **ONLY** for the Alpha channel (`nail_rgba[:, :, 3]`)! When RGB uses `BORDER_REPLICATE`, it is mathematically impossible for `[0, 0, 0]` black to ever enter the RGB channels during warping!

### Bug 3: Why `diffuse_ratio` (0.72) + `soft_cuticle_ao` (0.88) Creates a Dark Muddy Ring on Pastel Nails (`37.jpg`)
* **Exact Location:** `compositor_2.py` lines 248–260.
* **What happens in the code:**
  In `compositor_2.py` line 252, the agent changed `target_sidewall` to `0.72` (left) and `0.78` (right), and multiplied by `soft_cuticle_ao` (`0.88`).
  At the sidewall and cuticle edges of the soft baby-pink nail from `37.jpg` (where RGB is a delicate `[205, 185, 225]`), multiplying by `0.72 * 0.88 = 0.633` crushes the baby-pink RGB down to **`[130, 117, 142]` (a dark, muddy, bruise-colored gray-brown border)** right inside the outer `25%` of the nail!
  And because `u_map` reaches `-1.0` and `+1.0` right at the anti-aliased edge (`alpha_mask > 0.12`), the entire outer rim of the nail is painted `37%` darker black-gray!

### Bug 4: Why `crease_band` and `under_shadow_mult` Paint a Dark Halo Outside the Nail (`compositor_2.py` lines 318–337)
* **Exact Location:** `compositor_2.py` lines 323–337.
* **What happens in the code:**
  1. Look at line 321: `warm_shadow_bgr = skin_med_bgr * [0.52, 0.58, 0.72]`. That is `48%` darker than the finger skin! Then line 325 blends `0.40 * crease_band` where `tucked_alpha < 0.35`. Because `tucked_alpha` is feathery (`0.05..0.35`) along the entire cuticle and lower sidewalls, this paints a dark brown/black trench all around the outside of the cuticle!
  2. Look at line 336: `under_shadow_mult = 1.0 - (0.32 * shadow_mult) * shadow_blur * (1.0 - tucked_alpha) * skin_weight`. Even with `tip_zone_mask` (`v_map > 0.65`), `v_map` is computed across the **entire width of the ROI** (`0..rw`), and shifting `tucked_alpha` by `+1.6px X, +4.2px Y` pushes the shadow downward along the upper-left and upper-right sidewalls (`v_map` between `0.65` and `1.0` is the entire top 35% of the nail, which sits **on the finger skin**!). Multiplying the finger skin by `1.0 - 0.32` creates the dark black outline around the top third of every finger!

### Bug 5: Why the Nails Look Like "3D Metallic Tubes" (`compositor_2.py` lines 262–302)
* **Exact Location:** `compositor_2.py` lines 284–302.
* **What happens in the code:**
  Even though the agent lowered the studio weight to `0.42 * gloss_ribbon` on line 290, look at lines 301–302:
  ```python
  spec_color = np.array([252.0, 254.0, 255.0], dtype=np.float32)[None, None, :]
  spec_alpha = np.clip(spec_total[:, :, None] * spec_scale, 0.0, 0.85)
  lit_bgr = lit_bgr * (1.0 - spec_alpha * 0.75) + spec_color * spec_alpha
  ```
  Wait—look at the math on line 302: `lit_bgr * (1.0 - 0.75 * spec_alpha) + 255 * spec_alpha`!
  Because `1.0 - 0.75 * spec_alpha + spec_alpha = 1.0 + 0.25 * spec_alpha > 1.0`, this is an **over-additive blowout**! And combined with the dark `0.72` sidewalls on line 252, the contrast between the `0.72` dark edges and the `255` white center stripe makes the flat baby-pink nail from `37.jpg` look like a high-contrast 3D metallic cylinder!

### Bug 6: Why Dashed Rectangular Bounding Boxes Appear on the Hand (`image_e07d67.jpg`)
* Look closely at `image_e07d67.jpg` above each finger—there are faint colored dashed rectangles around every nail!
* **Why:** Either the UI canvas overlay is being captured in the screenshot, or `erase_exposed_natural_nails` (`compositor_2.py` line 430) is creating a rectangular boundary artifact where `inpainted_textured` (which adds random `normal(0.0, 1.2)` noise across the **entire `canvas_u8.shape`**!) or `warped_roi` (`bbox`) has a `1–2` level float32 rounding or noise discontinuity at `(rx, ry, rw, rh)`!

---

## 2. Exact Drop-In Code Fixes for the Coding Agent

Give the instructions below directly to your coding agent to update `segmentation.py`, `warping.py`, and `compositor.py`.

### Step 1: Fix `python_sidecar/warping.py` (`warping_2.py`) — Eliminate `[0, 0, 0]` Black Border Interpolation

In **`warping.py`**, replace the `cv2.remap` calls in **both** `apply_cylindrical_c_curve` (lines 97–104) and `warp_nail_tps` (lines 370–378) so RGB channels **never** use `BORDER_CONSTANT` with `(0, 0, 0)`:

```python
# In apply_cylindrical_c_curve (replace lines 97-104):
if has_alpha:
    warped_bgr = cv2.remap(
        image[:, :, :3],
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_a = cv2.remap(
        image[:, :, 3],
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = np.dstack([warped_bgr, warped_a])
else:
    warped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
```

And in **`warp_nail_tps`** (replace lines 370–378):
```python
# In warp_nail_tps (replace lines 370-378):
# CRITICAL: Remap BGR with BORDER_REPLICATE so out-of-bounds pixels NEVER pull [0,0,0] black into the edge!
# Use INTER_LINEAR (not INTER_CUBIC) to prevent bicubic negative-lobe dark ringing at high-contrast edges.
warped_bgr = cv2.remap(
    nail_rgba[:, :, :3],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REPLICATE,
)
warped_a = cv2.remap(
    nail_rgba[:, :, 3],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0,
)
warped_roi = np.dstack([warped_bgr, warped_a])
```

Also in **` deskew_nail_pca`** (`segmentation.py` lines 407–428):
* Whenever `cv2.warpAffine` is called on a 4-channel RGBA image, warp `image[:, :, :3]` with `borderMode=cv2.BORDER_REPLICATE` and `image[:, :, 3]` with `borderMode=cv2.BORDER_CONSTANT, borderValue=0`.

---

### Step 2: Fix `python_sidecar/segmentation.py` (`segmentation_3.py`) — Stop Slicing Tips Flat on Oval/Almond Nails (`37.jpg`)

1. **Fix `is_nail_inverted` for Symmetric Short/Squoval Nails (`37.jpg`):**
   In `37.jpg`, the nails are short squoval/oval nails mounted upside-down on the tray (cuticle at the top, rounded tip at the bottom). Because short squoval nails have a similar width at the top and bottom (`score_extreme ~ 1.01`), `is_nail_inverted` returns `False` when `exact_mask=True` (MobileSAM path)!
   * **The Fix in `segment_packaging_tray` (lines 586–592) and `extract_nail_rgba` (line 871):**
     Commercial press-on nail trays (`Nail BERRIE`) mount nails with the cuticle at the top (`inverted = True`). In `extract_nail_rgba`, default `should_invert = True if force_inverted is None else force_inverted` unless `score_extreme < 0.88` (which only happens if a nail is unambiguously point-up Stiletto/Almond).
2. **Fix `build_symmetric_smooth_nail_mask` (lines 259–265 of `segmentation_3.py`) so Oval/Almond Tips Are Never Sliced Flat:**
   * Replace lines 259–265 in `segmentation_3.py`:
     ```python
     # Measure tip width strictly in the top 1% to 5% zone (NOT 3% to 15%!)
     tip_zone_start = max(1, int(0.01 * nail_h))
     tip_zone_end = max(tip_zone_start + 1, int(0.05 * nail_h))
     w_tip_measured = float(np.median(w_pre[tip_zone_start:tip_zone_end]))
     flat_ratio = w_tip_measured / w_eq
     # Only classify as flat-tip if the very top 1-5% is at least 72% as wide as the equator!
     is_flat_tip = flat_ratio >= 0.72
     ```
   * Even when `is_flat_tip` is `True`, **NEVER** insert a sharp horizontal line `[top_left, top_right]` with un-rounded corners (lines 332–335); always round the top `6%` shoulders smoothly so there is never a polygonal corner step!

---

### Step 3: Fix `python_sidecar/compositor.py` (`compositor_2.py`) — Natural Soft Shading, Zero Black Rim, Zero "Metallic Tube"

Replace `composite_nail_pbr` in **`compositor.py`** (`compositor_2.py` lines 191–342) with this clean, natural-looking retoucher formulation:

```python
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
    # 2. Multiply shading intensity by `interior_fade = np.clip((tucked_alpha - 0.65) / 0.30, 0.0, 1.0)`
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
    spec_total: np.ndarray = (0.06 * broad_sheen + 0.16 * gloss_ribbon) * gloss_mult
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
```

---

### Step 4: Fix `clean_tray_glare_preserving_texture` on Light Pastel Nails (`37.jpg`)
In `compositor_2.py` lines 65–72:
```python
med_L: float = float(np.median(L_chan[valid]))
med_S: float = float(np.median(sat[valid]))
rel_L_spike = (tophat_L > 12.0) & (L_chan > min(185.0, med_L + 28.0))
rel_S_drop = (sat < max(38.0, med_S * 0.72))
```
Look at what happens on `37.jpg` (soft baby-pink nails):
* On `37.jpg`, the pink nail already has `med_L ~ 195`! Because `min(185.0, med_L + 28.0)` clamps the threshold down to **`185.0`**, `L_chan > 185.0` is true across the *entire nail*, and `glare_seed` only catches part of the white glare bar or leaves a half-inpainted smear!
* **The Fix:** Never clamp the glare threshold *below* `med_L + 18.0`! Replace lines 68–71 in `compositor_2.py` with:
  ```python
  # Glare must ALWAYS be significantly brighter than the nail's median luminance (med_L)!
  L_glare_thresh = max(165.0, min(242.0, med_L + 20.0))
  L_expand_thresh = max(150.0, min(230.0, med_L + 11.0))
  rel_L_spike = (tophat_L > 10.0) & (L_chan > L_glare_thresh)
  rel_S_drop = (sat < max(30.0, med_S * 0.65))
  glare_seed: np.ndarray = valid & rel_L_spike & rel_S_drop
  glare_expand: np.ndarray = valid & (tophat_L > 5.0) & (L_chan > L_expand_thresh) & (sat < max(42.0, med_S * 0.80))
  ```
* Also in `clean_tray_glare_preserving_texture` (lines 110–137):
  After the glare inpainting and card-bleed check, always fill `~clean_interior` (including `alpha == 0`) with the nearest `clean_interior` pixel so `clean_tray_glare_preserving_texture` guarantees non-black exterior RGB before warping:
  ```python
  exterior_all: np.ndarray = alpha < 200
  if np.any(clean_interior) and np.any(exterior_all):
      _, (iy, ix) = scipy.ndimage.distance_transform_edt(~clean_interior, return_indices=True)
      out[:, :, :3][exterior_all] = out[:, :, :3][iy[exterior_all], ix[exterior_all]]
  ```