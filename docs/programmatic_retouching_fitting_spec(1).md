# Technical Specification: Zero-Model Programmatic Retouching & True-Shape Fitting

**Target System:** Try-On Studio (`python_sidecar/segmentation.py`, `warping.py`, `compositor.py`, `template_generator.py`)  
**Objective:** Eliminate the "digital sticker" look seen in `03_studio_relit_hand.jpg` using 100% deterministic, classical image manipulation and computational geometry (zero neural networks required) so that **every** press-on nail set looks like an authentic studio photograph on the hand.

---

## 1. Root-Cause Audit: The 7 Code Bugs Causing the "Artificial" Look

After inspecting `segmentation_2.py`, `warping.py`, `compositor.py`, and `template_generator.py` against `03_studio_relit_hand.jpg`, here are the exact lines of code creating the artificial artifacts:

1. **Bug 1: Double Specular Reflections (Baked-in Tray Glare + Synthetic Ribbon):**
   * **Where:** `compositor.py` lines 65–95 (`clean_tray_glare_preserving_texture`).
   * **Why it fails:** `glare_seed` requires `L_chan > 190.0` and `sat < 24.0`. On dark burgundy or medium-tone nails (`nails-sample-tray.jpg`), the long vertical lightbox reflection strip only reaches $L \approx 135\text{--}175$ and retains slight color saturation ($S \approx 25\text{--}45$). Thus, `clean_tray_glare_preserving_texture` completely misses the tray's vertical white glare streak! Then, in Pass 4 (lines 259–299), `composite_nail_pbr` paints a **second** synthetic `gloss_ribbon` on top, resulting in ruler-straight, double white vertical stripes on every nail in `03_studio_relit_hand.jpg`.

2. **Bug 2: White Backing-Card Bevel Reflection (`left_rim_mask` + Shallow Fringe Cleanup):**
   * **Where:** `compositor.py` lines 118–128 (`fringe_px = nail_w * 0.009` = only 1–2px) and lines 252–256 (`left_rim_mask` deliberately adds `[205, 208, 216]` white-blue rim light to the left edge!).
   * **Why it fails:** In `nails-sample-tray.jpg`, white light from the backing card reflects onto the outer `4–6px` curved bevel of every nail. Because `fringe_px` only cleans `1–2px`, that chalky white card reflection stays on the nail—and `left_rim_mask` in `compositor.py` adds even more gray-white glow to the left border, creating a fake sticker rim.

3. **Bug 3: Bare Natural Nail Peeking Out Underneath Narrower Press-On Nails:**
   * **Where:** `compositor.py` (`render_full_hand_composite`, lines 331–444).
   * **Why it fails:** `hand-template.jpg` has square-ish, glossy natural nails with white free-edges (`d_l` to `d_r`). When placing a tapered Almond press-on nail (`nails-sample-tray.jpg`), the press-on tapers inward starting at the mid-body, leaving the upper-left and upper-right corners of the shiny natural nail in `hand-template.jpg` completely exposed (clearly visible on the Ring and Index fingers in `03_studio_relit_hand.jpg`).

4. **Bug 4: Black/Gray Outer Drop Shadow & Cuticle Halo:**
   * **Where:** `compositor.py` lines 313–322 (`under_shadow_mult` shifts `tucked_alpha` by `+1.6px X, +4.2px Y` and multiplies the hand by `1.0 - 0.28 * shadow_blur`) and `warping.py` lines 315–329 (`cuticle_ao` blurs a 10px line both inside and *outside* the nail).
   * **Why it fails:** Shifting the entire nail mask down by `+4.2px` (`M_sh`) casts a dark gray drop shadow **below the cuticle onto the knuckle skin** and along the lower sidewalls! Real nails only cast a shadow *under the free-edge tip* ($v > 0.72$), never below the cuticle base. Furthermore, multiplying RGB by a scalar (`0.72`) produces a muddy, achromatic gray-black shadow instead of a warm subsurface skin crease shadow.

5. **Bug 5: Egg/Jellybean Shape Distortion on Almond Nails:**
   * **Where:** `segmentation_2.py` lines 206–295 (`build_symmetric_smooth_nail_mask`) and `warping.py` lines 253–269 (`build_cuticle_pocket_mask`).
   * **Why it fails:**
     * In `segmentation_2.py`, when `is_flat_tip` is `False` (Almond/Oval), lines 224–226 force `p_cand` and `q_cand` toward `(1.82, 2.18)` and line 264 applies `ellip_tip` (`sqrt`), rounding the pointed Almond tip into a fat oval.
     * Even worse, in `warping.py` (`build_cuticle_pocket_mask`, line 276), `s_eq = 0.38 * bed_len` clips the bottom `38%` of the nail into a deep semi-circle (`u_cup_pts`), chopping off the natural squared-off cuticle corners of the press-on nail so both ends of the nail look identically egg-shaped!

6. **Bug 6: Sideways Thumb Rendered as a Squished Front-Facing Nail:**
   * **Where:** `warping.py` lines 77–88 (`apply_cylindrical_c_curve`) and lines 202–208 (`build_3tier_control_grid`).
   * **Why it fails:** When `roll_offset = -0.35` (Thumb), `apply_cylindrical_c_curve` maps `0..1` to `0..1` via `x_norm ** gamma`. This compresses the *entire* 100% width of the front-facing thumb nail (including both left and right sidewalls) onto the side-profile thumb. On a real thumb turned $60^\circ$ in profile (`hand-template.jpg`), the far 30% of the nail plate wraps behind the horizon of the thumb cylinder and is **invisible to the camera**.

7. **Bug 7: Over-Scaled Nail Width (`tray_scale * 1.05` and `frame_half_w = bed_width * 0.545`):**
   * **Where:** `compositor.py` line 378 (`tray_scale = ... * 1.05`) and `warping.py` line 186 (`frame_half_w = bed_width * 0.545`).
   * **Why it fails:** Setting `frame_half_w = 0.545 * bed_width` forces the nail to be **9% wider** than the physical landmarks (`d_l` to `d_r`), making the nails look oversized and overlapping the lateral finger skin.

---

## 2. File-by-File Implementation Instructions for the Coding Agent

### File 1: `python_sidecar/segmentation.py` (or `segmentation_2.py`)

#### Fix 1.1: Stop Turning Almond Nails into Ovals in `build_symmetric_smooth_nail_mask`
In `build_symmetric_smooth_nail_mask(mask, feather_px=1.5)`:
1. **Remove the Super-Ellipse Grid Search (`p_cand`, `q_cand`, lines 215–254) and Remove `ellip_tip` (lines 261–265):**
   * Do **not** replace `w_mono` with `0.94 * w_analytic + 0.06 * w_smooth`. That analytic super-ellipse is what turns tapered Almond tips and squared cuticles into uniform ovals.
2. **Use Shape-Faithful Savitzky-Golay Profile + True Tip Taper:**
   * Compute `w_pre` using `scipy.signal.savgol_filter(measured_half_w, window_length=max(5, int(nail_h * 0.09) | 1), polyorder=2)`.
   * Allow the anatomical equator `i_peak` to be found across `[0.45 * nail_h, 0.72 * nail_h]` (press-on nails on trays have wider cuticle bases than `0.58–0.66`).
   * Enforce monotonicity (`w_mono`) from `0 -> i_peak` and `nail_h - 1 -> i_peak`.
   * **Top Tip (`0` to `0.06 * nail_h`):** Only cap the very top `4%` (not `14%`) with a tangent curve matched to the local slope `dw/dy` at `0.05 * nail_h`, so Almond and Stiletto tips keep their true slender taper while Square/Coffin tips stay flat.
   * **Bottom Cuticle (`0.86 * nail_h` to `nail_h - 1`):** Keep the bottom `14%` `ellip_cut` rounding so the cuticle base forms a clean U-cup.

#### Fix 1.2: Deepen Perimeter De-Fringing in `extract_nail_rgba`
In `extract_nail_rgba` (lines 805–814):
* Change `erode_fringe` iterations from `iterations=2` to an adaptive pixel depth `fringe_px = max(3, int(round(w * 0.045)))` (covering the outer 4.5% width bevel where the white backing card reflects onto the curved plastic edge).

---

### File 2: `python_sidecar/warping.py`

#### Fix 2.1: True 3D Horizon Occlusion for Sideways Thumb (`apply_cylindrical_c_curve`)
In `apply_cylindrical_c_curve(image, c_curve_intensity=0.65, roll_offset=0.0)` (lines 77–88):
When `abs(roll_offset) > 0.10` (e.g., `Thumb` with `roll_offset = -0.35` on left hand or `+0.35` on right hand):
1. **Horizon Clipping (3D Cylinder Occlusion):**
   * A thumbnail viewed at a $\sim 55^\circ\text{--}65^\circ$ lateral roll angle has its far lateral side hidden behind the curvature of the nail barrel.
   * Instead of mapping `x_norm` from `[0.0, 1.0]`, crop/occlude the far lateral `24%` of the source nail texture (`u_start = 0.24 * abs(roll_offset) / 0.35` on the side turning away from the camera, i.e., the left side `x < u_start * w` when `roll_offset < 0`) and foreshorten the remaining visible `76%` of the nail plate using cylindrical projection:
     $$\theta(x) = \theta_{\text{horizon}} + (1 - x_{\text{norm}}) \cdot \Delta\theta, \quad x_{\text{proj}} = \frac{1 - \cos(\theta)}{1 - \cos(\theta_{\text{max}})}$$
   * Set the alpha channel at the far horizon edge (`x == 0`) to a crisp `1.0px` anti-aliased drop-off (since it is a 3D silhouette horizon against the grey background, **not** a skin seam).

#### Fix 2.2: Fix Oversized Width & Preserve Free-Edge Taper in `build_3tier_control_grid`
In `build_3tier_control_grid` (lines 182–218):
1. **Fix Nail Bed Width Multiplier (Line 186):**
   * Change `frame_half_w = bed_width * 0.545 * w_mult` $\rightarrow$ **`frame_half_w = bed_width * 0.495 * w_mult`**.
   * The landmarks `d_l` and `d_r` *already* mark the outer lateral nail folds (`bed_width`). Multiplying half-width by `0.545` made the nail `9%` wider than the nail bed! `0.495` seats the sidewalls right inside the paronychium groove.
2. **Place Row 2 Control Points at the Distal Nail Bed (`d_l, d_r`), Not `50%` of Total Length:**
   * Anchor Row 1 at the cuticle (`s = 0`), Row 2 near the distal nail-bed crease (`s = 0.42 * extension_dist`) matching `distal_width * 0.50`, and Row 3 at the tip (`s = 1.0 * extension_dist`) using `frame_half_w * tip_taper`.
   * For `abs(roll_offset) > 0.10` (Thumb), shift `base_left` and `apex_left` inward by `0.72 * frame_half_w` on the foreshortened horizon side and `0.95 * frame_half_w` on the near skin-tuck side so the thumb nail aligns with the visible half of the thumbnail bed in `hand-template.jpg`.

#### Fix 2.3: Fix Shallow U-Cup in `build_cuticle_pocket_mask`
In `build_cuticle_pocket_mask` (lines 274–285):
* Change `pocket_half_w = bed_w * 0.565` and `s_eq = 0.38 * bed_len` $\rightarrow$ **`pocket_half_w = cuticle_w * 0.505`** and **`s_eq = 0.18 * bed_len`**.
* Using `s_eq = 0.38 * bed_len` made the cuticle U-cup way too deep (38% of the nail bed), turning the bottom of every nail into a pointy parabola. Real cuticles (`hand-template.jpg`) have a flatter, squoval U-cup (`s_eq = 0.18 * bed_len`) that transitions quickly into straight parallel sidewalls.
* Change the curve exponent from a circular `sin(thetas)` to a squoval power `np.sin(thetas) ** 0.75` so the proximal cuticle corners (`c_l`, `c_r`) match the natural squoval cuticle contour of the finger.

---

### File 3: `python_sidecar/compositor.py`

#### Fix 3.1: Strip Broad Tray Studio Reflections & White Card Rim Bleed (`clean_tray_glare_preserving_texture`)
In `clean_tray_glare_preserving_texture(nail_rgba)` (lines 29–130):
1. **Detect Soft Studio Strip Reflections on Dark/Medium Nails:**
   * Currently, `glare_seed` requires `L_chan > 190.0` and `sat < 24.0`, which misses the vertical studio reflection on dark nails (`nails-sample-tray.jpg`) where the glare only raises $L$ from $\sim 75$ to $\sim 155$.
   * Compute the median interior luminance `med_L = float(np.median(L_chan[valid]))` and median saturation `med_S = float(np.median(sat[valid]))`.
   * Define relative glare seed:
     ```python
     # Catches both bright white ticks AND soft lightbox streaks on dark/colored nails
     rel_L_spike = (tophat_L > 12.0) & (L_chan > min(185.0, med_L + 28.0))
     rel_S_drop = (sat < max(38.0, med_S * 0.72))
     glare_seed = valid & rel_L_spike & rel_S_drop
     glare_expand = valid & (tophat_L > 6.0) & (L_chan > min(160.0, med_L + 16.0)) & (sat < max(48.0, med_S * 0.85))
     ```
   * Increase max connected-component area limit from `0.035 * n_valid` to **`0.085 * n_valid`** so full-length vertical ring-light streaks on glossy solid nails are completely healed before relighting.
2. **Increase Perimeter Edge-Fringe Choke (Lines 118–128):**
   * Change `fringe_px: int = max(1, min(4, int(round(nail_w * 0.009))))` $\rightarrow$ **`fringe_px: int = max(3, min(9, int(round(nail_w * 0.045))))`**.
   * In that outer `4.5%` ring (`edge_ring`), only overwrite pixels whose luminance `L_chan` is **brighter** than the nearest `clean_interior` pixel by more than `4.0` LAB units (or whose saturation is lower by `6.0`). This surgically strips the white backing-card bounce light from the bevel while preserving dark nail art at the edges.

#### Fix 3.2: Add Pre-Composite "Natural Nail Eraser" in `render_full_hand_composite`
In `render_full_hand_composite` (right before the `for fname in finger_keys:` loop at line 380):
* Add a helper function `erase_exposed_natural_nails(composite_canvas, landmarks, downscale)`:
  * For each finger (`c_l, c_c, c_r, d_l, d_r`), build a polygon covering the natural nail bed plus `+6px` distal and `+4px` lateral margin (especially around the upper corners `d_l` and `d_r` where natural square nails peek out behind Almond/Stiletto press-ons).
  * Sample the median skin color and texture from a `6px` band immediately outside `c_l..d_l` and `c_r..d_r` (the lateral finger pulp).
  * Use `cv2.inpaint` (or smoothly blend the sampled lateral skin tone over the outer `35%` margin of the natural nail bed) on `composite_canvas` **before** compositing the press-on nails.
  * Result: The shiny bare natural nail corners visible on the Ring and Index fingers in `03_studio_relit_hand.jpg` disappear 100%.

#### Fix 3.3: Fix Shading, Cuticle Light-Wrap, Specular Alignment, and Warm Contact Shadows in `composite_nail_pbr`
In `composite_nail_pbr` (lines 176–328):
1. **Delete `left_rim_mask` (Lines 252–256):**
   * Remove `lit_bgr = lit_bgr + left_rim_mask[:, :, None] * 0.06 * env_color` completely. Adding `[205, 208, 216]` gray-white paint to the left edge of the nail creates a fake sticker outline.
2. **Deepen Cylindrical C-Curve Edge Shading (Lines 237–246):**
   * Change `target_sidewall = np.where(u_map < 0.0, 0.84, 0.90)` $\rightarrow$ **`target_sidewall = np.where(u_map < 0.0, 0.72, 0.78)`** and start the cosine roll-off at `u_abs > 0.42` instead of `0.55`. Real glossy curved nails darken significantly right at the lateral curvature horizon.
3. **Transfer Natural Nail Lighting & Align Specular Highlight to Finger Axis (Pass 4, Lines 258–299):**
   * In `hand-template.jpg`, the studio softbox is on the **upper-left** (`u_center ~ -0.22`), whereas line 259 hardcodes `spec_u_center = +0.30` (putting the highlight on the *right* side of every nail, opposite to the hand's real lighting!).
   * Change `spec_u_center: float = -0.22 - roll_offset * 0.35 + (light_angle_deg / 180.0) * 0.20` so the primary highlight sits on the lit left side of the dome (matching `hand-template.jpg`).
   * Widen and soften `rib_half_w` from `0.026` (which looks like a razor-thin vector line) to **`0.065`** with a gentler Gaussian exponent (`1.6` instead of `2.2`) and lower default opacity (`0.42` instead of `1.05 * 0.84`), so the reflection looks like a natural curved studio softbox rather than a painted white stripe.
4. **Add True Skin Light-Wrap & Warm Subsurface Contact Shadow (Pass 5 & Pass 6, Lines 301–326):**
   * **Warm Subsurface Contact Shadow (Replaces Gray AO & Full-Nail Shift Shadow):**
     * Sample the local finger skin color `skin_med_bgr` from `hand_patch` around the cuticle. Compute a **warm subsurface shadow color** by multiplying `skin_med_bgr` by `[0.52, 0.58, 0.72]` (darkening Blue/Green more than Red so the crease shadow is warm skin-umber, never muddy gray/black).
     * Apply the contact shadow along `groove_blur` (from `build_cuticle_pocket_mask`) by blending `composite_hand` toward `warm_shadow_bgr` strictly within `1.5px` of the cuticle/sidewall crease.
   * **Restrict Under-Tip Cast Shadow (`under_shadow_mult`, Lines 313–322) to the Distal Free-Edge Only:**
     * Multiply `shadow_blur` by `tip_zone_mask = np.clip((v_map - 0.65) / 0.15, 0.0, 1.0)` (dilated downward by 4px) so the cast shadow is **only** cast by the overhanging free-edge tip (`v > 0.65`) onto the fingertip pulp, and **never** cast below the cuticle base (`v == 0`)!
   * **Cuticle & Sidewall Skin Light-Wrap:**
     * Blur `hand_patch` with `cv2.GaussianBlur(hand_patch, (9, 9), 2.5)`.
     * Create a `1.5px` inner border weight `rim_wrap = np.clip((1.0 - tucked_alpha) * 2.5, 0.0, 1.0) * (tucked_alpha > 0.05) * np.clip(1.0 - v_map * 1.4, 0.0, 1.0)`.
     * Blend `22%` of the blurred surrounding skin tone onto the outer `1.5px` rim of `lit_bgr` so the plastic border optically melts into the fleshy skin fold.

---

### File 4: `python_sidecar/template_generator.py`

#### Fix 4.1: Calibrate `DEFAULT_STUDIO_LANDMARKS` for `hand-template.jpg`
In `DEFAULT_STUDIO_LANDMARKS` (lines 24–76):
1. **Thumb Landmarks (Lines 25–35):**
   * In `hand-template.jpg` ($1536 \times 2752$), the visible thumbnail bed is in side profile on the left side of the thumb tip (`x: 205..262`).
   * Update `Thumb` to:
     ```python
     "Thumb": {
         "c_l": [204.0, 1418.0],
         "c_c": [232.0, 1432.0],
         "c_r": [264.0, 1412.0],
         "d_l": [196.0, 1278.0],
         "d_r": [258.0, 1282.0],
         "c_curve": 0.52,
         "roll_offset": -0.45,
         "length_scale": 0.95,
         "tip_taper": 0.92,
     }
     ```
2. **Tighten `d_l` and `d_r` on Index, Middle, Ring, and Pinky by `3px` Inward:**
   * Reduce the horizontal span `d_r[0] - d_l[0]` by `6px` (`+3px` on `d_l[0]`, `-3px` on `d_r[0]`) so the default anchors sit strictly inside the lateral nail grooves rather than on the outer skin bulge.

---

## 3. Summary Verification Checklist for the Coding Agent

After applying the changes above, run the test composite pipeline on `nails-sample-tray.jpg` + `hand-template.jpg` and verify:
- [ ] **Zero Double Glare:** The baked-in vertical lightbox stripe from `nails-sample-tray.jpg` is completely removed by `clean_tray_glare_preserving_texture`, and replaced by a single, natural, soft left-side studio sheen matching `hand-template.jpg`.
- [ ] **Zero Peeking Natural Nails:** The shiny upper-right natural nail corners on the Ring and Index fingers are completely healed with skin texture prior to nail placement.
- [ ] **Zero Gray Outer Halo:** No gray/black drop shadow appears below the cuticle base (`v == 0`); crease shadows are warm skin-umber, and cast shadows only appear under the overhanging free-edge tip (`v > 0.65`).
- [ ] **True Almond Silhouette:** The extracted nails keep a squoval cuticle base and tapered Almond tip (not an oval egg shape).
- [ ] **3D Side-Profile Thumb:** The thumb nail is foreshortened with far-side horizon occlusion so only the visible profile half wraps onto the thumb.