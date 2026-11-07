# True-Texture / Zero-Fabrication Nail Extraction & Composition Spec

> **Objective**: Replace all synthetic/fabricated shape templates and median-color/smile-line pixel replacements in `python_sidecar/segmentation.py` and `python_sidecar/compositor.py` with a **100% True-Texture Retrieval Pipeline** that preserves every detail from the photographed nail (floral nail art, 3D gold charms, gold chrome lines, cat-eye magnetic gradients, white/black French tips, and sheer ombré translucency) across all sets in `test_assets/test_nails/` and tilted/angled tray photos.

---

## 1. Root-Cause Diagnosis of Current Artifacts

1. **Fabricated Interior Colors (`compositor.py:18-202`, `recover_clean_nail_albedo`)**:
   - **Current flaw**: Overwrites the nail's interior pixels with a solid median LAB color (`PATH A`) or a synthetically drawn black parabolic smile line (`PATH B`, `smile_analytic`).
   - **Why it breaks**: Destroys hand-painted floral art, gold chrome stripes, 3D charms, cat-eye velvet gradients (`29.jpg`), white French tips (`33.jpg`), and sheer ombré gradients (`47.jpg`).
2. **Forced Analytical Almond Silhouette (`segmentation.py:18-96`, `build_symmetric_smooth_nail_mask`)**:
   - **Current flaw**: Blends `20%` measured width (`smooth_w`) with `80%` hardcoded super-ellipse (`analytic_w` with `t_peak = 0.60, power = 0.56`).
   - **Why it breaks**: Forces Square/Squoval nails (`29.jpg`, `39.jpg`) and Short Oval nails (`43.jpg`) into pointy Almond shapes, clipping the lateral corners of French tips and floral artwork.
3. **Color-Only Foreground Thresholding (`segmentation.py:331-334` & `495-500`)**:
   - **Current flaw**: Uses `(((A >= 131) | (sat >= 18)) & (gray < bg_gray - 5)) | (gray < 88)`.
   - **Why it breaks**: Misses **White French Tips** (`33.jpg`) where the white tip (`gray ~ 215, sat < 10, A ~ 128`) has nearly the same brightness and neutrality as the white backing card (`gray ~ 225`).
4. **No Perspective Rectification for Tilted/Angled Camera Poses (`segmentation.py:296-322`)**:
   - **Current flaw**: Assumes the white backing card is axis-aligned (`cv2.boundingRect(card_c)`).
   - **Why it breaks**: When a user shoots the nail box from an oblique/side angle (e.g. the floral/gold-charm set), the card is a rotated perspective trapezoid, causing row merging, skewed aspect ratios, and distorted nail lengths.

---

## 2. Exact File-by-File Changes Required

### Step 1: Add Backing-Card Perspective Rectification (`python_sidecar/segmentation.py`)

**Target**: Add `rectify_backing_card(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` and call it at the start of `segment_packaging_tray()`.

**What to implement**:
1. Threshold the white backing card inside the black presentation box (`(gray > 165) & (sat < 38)`), close internal nail holes with a large morphological closing (`(35, 35)`), and take the largest external contour `card_c`.
2. Fit a 4-corner polygon to `card_c` using `cv2.approxPolyDP` (or `cv2.minAreaRect` / convex hull corner extremes if the card corners are slightly rounded).
3. Order the 4 corners as `[top_left, top_right, bottom_right, bottom_left]`.
4. If the card is tilted (`abs(angle) > 1.5°`) or foreshortened in perspective (top width vs. bottom width differs by `> 3%`, or left height vs. right height differs by `> 3%`):
   - Compute target dimensions `dst_w = int(max(width_top, width_bottom))`, `dst_h = int(max(height_left, height_right))` (matching the standard square/near-square Nail BERRIE card ratio).
   - Compute homography `H = cv2.getPerspectiveTransform(src_quad, dst_quad)` and warp `rectified_img = cv2.warpPerspective(image, H, (dst_w, dst_h), flags=cv2.INTER_CUBIC)`.
5. Store or return `rectified_img` and `H_inv` so segmentation and RGBA nail extraction operate on the perspective-rectified fronto-parallel plane.

---

### Step 2: Upgrade Multi-Cue Segmentation for White French Tips & Sheer Cuticles (`python_sidecar/segmentation.py`)

**Target**: [`segment_packaging_tray()`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L296-L448) and [`extract_nail_rgba()`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L451-L583).

**What to change**:
1. **Include Physical Rim Edge Gradients & Rim Shadow for White-on-White Tips (`33.jpg`)**:
   - In addition to chrominance (`lab_A >= 131` or `sat >= 16` or `abs(lab_B - bg_B) >= 6`) and dark pigment (`gray < bg_gray - 12`), compute a Scharr/Sobel gradient magnitude on `lab_L` and `lab_A` plus a local dark rim-shadow detector (`gray < local_bg_max - 8`).
   - Even on a white French tip (`33.jpg`), the physical 3D plastic free edge casts a `2px–4px` shadow loop around the bottom tip against the card. Closing the union of `(pink_body_mask | tip_rim_shadow_edge)` bridges across the white crescent so the white French tip is 100% included inside the nail contour.
   - Distinguish the true nail tip rim from the clear rectangular plastic mounting tab by checking **width Continuity**: the clear plastic mounting tab protrudes *horizontally* between nails (creating a low vertical height `< 35%` of row height), whereas the white French tip sits directly below the pink nail body within the nail's `[x_left, x_right]` column span.
2. **Replace Forced Almond Template with Data-Driven Bilateral Profile Smoothing in [`build_symmetric_smooth_nail_mask()`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L18-L96)**:
   - Remove `analytic_w` (`0.80 * analytic_w + 0.20 * smooth_w`) at lines `62–79`.
   - Instead, measure `w_left(y)` and `w_right(y)` at every scanline `y` from `y_min` to `y_max`:
     - Take the outlier-resistant bilateral consensus: `raw_half_w[y] = 0.65 * max(w_left, w_right) + 0.35 * min(w_left, w_right)`.
     - Enforce anatomical monotonicity above and below the widest equator band (`y_peak` in `[0.35 * nail_h, 0.70 * nail_h]`): width must increase smoothly from `y_min` (distal tip) to `y_peak`, and decrease smoothly from `y_peak` to `y_max` (cuticle base), using isotonic/running-envelope clamping so small edge nicks are filled without changing the tip shape.
     - Smooth `raw_half_w` with a 1D Savitzky-Golay filter (window `15%` of `nail_h`, polyorder `3`) or narrow Gaussian (`sigma = 0.03 * nail_h`), preserving **95%+ of the measured width profile** while rounding only the extreme top `3%` tip cap and bottom `4%` cuticle cap.

---

### Step 3: Replace `recover_clean_nail_albedo()` with Surgical Glare-Tick Inpainting (`python_sidecar/compositor.py`)

**Target**: [`recover_clean_nail_albedo(nail_rgba: np.ndarray)`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/compositor.py#L18-L202).

**What to delete**:
- Delete both `PATH B` (lines `54–141`, which fabricates the black French smile line `smile_analytic` and overwrites the pink body) and `PATH A` (lines `143–201`, which overwrites the nail with median LAB colors).

**What to implement instead (`clean_tray_glare_preserving_texture`)**:
1. **Start with 100% of the original RGB pixels (`out = nail_rgba.copy()`)**:
   - Every flower petal, gold charm, gold stripe, cat-eye shimmer band, French smile line, and sheer gradient is preserved verbatim by default.
2. **Isolate Only Pinpoint Overhead Light-Box Glare Ticks**:
   - Convert `out[:, :, :3]` to CIE LAB (`L, A, B`) and HSV (`H, S, V`).
   - Compute a morphological **White Top-Hat** on `L` using an elliptical structuring element of size `(11, 11)`:
     - `tophat_L = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel_11x11)`
   - A pixel is part of a **tray overhead light reflection tick** if and only if ALL 4 conditions hold:
     1. **Sharp local luminance spike**: `tophat_L > 22.0` and `L > 210.0` (the pixel is significantly brighter than its immediate `11px` neighborhood).
     2. **Achromatic / desaturated white light**: `S < 22.0` (protects gold chrome stripes, gold 3D charms, red cat-eye shimmer, and colored floral art).
     3. **Interior location (not a White French Tip or rim highlight)**: Distance from the distal free-edge boundary `y_top` is `> 0.14 * nail_h` OR the connected component's horizontal width is `< 0.22 * nail_w` (white French tips span `100%` of the nail width across the top edge, whereas tray studio light ticks are narrow vertical streaks `< 15%` of the nail width!).
     4. **Small connected-component area**: Connected blob area `< 3.5%` of total valid nail pixels and aspect ratio is vertically elongated or pinpoint.
3. **Surgical Fast-Marching Inpainting (`cv2.inpaint`)**:
   - Dilate the detected `glare_tick_mask` by `2px` (`3x3` ellipse) to capture the 1px halo around the light tick.
   - Call `inpainted_bgr = cv2.inpaint(out[:, :, :3], glare_tick_mask, inpaintRadius=3.0, flags=cv2.INPAINT_TELEA)`.
   - Blend `inpainted_bgr` softly (`2px` Gaussian feather on `glare_tick_mask`) into `out[:, :, :3]`.
4. **2px Perimeter Edge-Fringe Cleanup**:
   - Erode the valid alpha mask `(alpha > 200)` by `3px` to define `clean_interior`.
   - For pixels in the outer `3px` boundary ring, replace their RGB color with the nearest pixel inside `clean_interior` (using `cv2.inpaint` on the outer rim or distance-transform nearest-neighbor lookup) so zero white backing-card pixels leak into the anti-aliased edge.

---

### Step 4: Non-Destructive Intrinsic Ratio Relighting (`python_sidecar/compositor.py`)

**Target**: [`composite_nail_pbr()`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/compositor.py#L264-L413).

**What to change**:
1. **Replace Aggressive Diffuse Overwrite with Gentle 3D Cylindrical Ratio Shading**:
   - Keep `compute_3d_nail_geometry()` (`u_map, v_map, nx, ny, nz`).
   - Constrain the multiplicative diffuse shading ratio `effective_diffuse` to a conservative, color-faithful range `[0.84, 1.04]`:
     - Center dome (`|u| < 0.55`): `0.97` to `1.03` so the original nail color, floral pattern, cat-eye shimmer, or French tip contrast is **unchanged**.
     - Lateral sidewalls (`|u| -> 1.0`): smooth cosine roll-off down to `0.84` on the shaded sidewall and `0.90` on the lit sidewall, giving the flat tray photo convincing 3D cylindrical curvature around the finger.
     - Proximal cuticle base (`v -> 0.0`): gentle `0.88` contact AO tuck (`soft_cuticle_ao`).
2. **Disable Artificial Subsurface Skin Blur on Patterned / Opaque Regions**:
   - In lines `327–337`, `sheer_translucency` currently blends `24%` of `hand_bed_smooth` into any pixel with `nail_lum > 125`. That would wash out white floral petals (`media_1794042448908_b0e89e26.jpg`) or white French tips (`33.jpg`)!
   - Restrict `sheer_translucency` (`max 10%` blend) strictly to the **proximal nail bed zone** (`v_map < 0.62`, away from the free-edge tip) AND only where local spatial texture variance (`std(L)` in a `9x9` window) is low (`< 4.0`, i.e., smooth sheer gel, never over floral petals, gold charms, or French smile lines).
3. **Adaptive Clear-Coat Specular Sheen**:
   - If the nail already contains strong intrinsic specular/metallic contrast (such as 3D gold charms or cat-eye magnetic shimmer where interior luminance std-dev is high), scale down the synthetic clear-coat specular ribbon (`spec_alpha *= 0.55`) so it enhances rather than competes with the real nail art.

---

### Step 5: Batch Benchmark & Verification Script (`python_sidecar/test_pipeline.py`)

**Target**: Add a batch runner in `python_sidecar/test_pipeline.py` (or `python_sidecar/test_all_nails.py`) that iterates over all sets in `test_assets/test_nails/` (`29.jpg`, `31.jpg`, `33.jpg`, `37.jpg`, `39.jpg`, `43.jpg`, `47.jpg`), `test_assets/nails-french-tray.jpg`, and the angled floral/gold-charm image, outputting side-by-side diagnostic sheets in `test_output/`:
- `<set>_segmented.jpg` (showing card homography + 15/15 slot contours + shape labels)
- `<set>_extracted_strip.png` (showing the 5 extracted RGBA nails before & after surgical glare-tick removal, verifying 100% pattern/shape preservation)
- `<set>_composited_hand.jpg` (final studio-relit hand composite)

---

## 3. What the Final Results Must Look Like (Acceptance Criteria)

| Test Asset | Nail Style & Challenge | Required Visual Result |
| :--- | :--- | :--- |
| **Floral & Gold Charm Set** *(angled photo)* | Tilted perspective card, hand-painted white/red hibiscus flowers, thin gold horizontal stripes, raised 3D gold charms | 1. Card is perspective-rectified to a flat top-down view before segmentation.<br>2. **100% of the flower petals, gold stripes, and 3D gold charms** appear sharp and intact on the fingers—zero blurring, zero color replacement.<br>3. Squoval shape and per-finger lengths are preserved. |
| **`29.jpg`** *(Cat-Eye Velvet)* | Dark crimson/black squoval with glowing vertical magnetic red shimmer band + tiny white studio light tick on left | 1. The tiny white studio light tick on the left is surgically healed.<br>2. The **glowing red cat-eye magnetic gradient** in the center is 100% preserved.<br>3. The **Squoval** flat-tip silhouette with rounded corners is preserved (not turned into an Almond tip). |
| **`33.jpg`** *(White French on Sheer Blush)* | Pale sheer blush body with bright **White French Tips** on a white backing card | 1. Segmentation captures the full **white French tip** down to its physical bottom edge (no clipped tips).<br>2. The **original white French tip & smile line** from the photo are preserved (not redrawn in black or washed out). |
| **`nails-french-tray.jpg`** *(Black French Tip)* | Sheer nude-pink oval nails with jet-black French tips | 1. The **original black crescent tip** from each nail is preserved without clipping its left/right wings.<br>2. The vertical white strip-light glare tick in the pink bed is removed via local inpainting without flattening the natural sheer pink gradient. |
| **`31.jpg` & `37.jpg`** *(Peach & Baby Pink)* | Warm peach (`31.jpg`) and sheer baby-pink (`37.jpg`) oval/squoval nails | 1. True peach vs. cool baby-pink hue distinction is preserved (`31.jpg` looks warm apricot-peach; `37.jpg` looks cool pastel pink).<br>2. Pinpoint tray light reflections are healed cleanly; smooth studio softbox ribbon sits naturally on top. |
| **`39.jpg` & `43.jpg`** *(Sage Grey & Deep Burgundy)* | High-gloss opaque sage-taupe squoval (`39.jpg`) and short oval burgundy (`43.jpg`) | 1. `39.jpg` retains its longer **Squoval** silhouette; `43.jpg` retains its **Short Oval** length (proportional to its real tray height relative to card width).<br>2. Zero white card halo at the cuticle or sidewalls. |
| **`47.jpg`** *(Sheer Terracotta Jelly)* | Translucent jelly rose-terracotta almond nails with lighter cuticle base and richer tip | 1. The natural longitudinal **jelly ombré gradient** (sheerer near cuticle, deeper terracotta toward tip) is preserved verbatim from the tray photo. |
