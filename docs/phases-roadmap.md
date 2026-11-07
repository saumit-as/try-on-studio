# Try-On Studio: Phased Development Roadmap

This roadmap breaks down the full Try-On Studio architecture into **5 sequential, testable phases**. Each phase ends in an explicit verification checkpoint where you test the output and provide feedback before proceeding to the next.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Feedback-Driven Execution Loop                    │
└────────────────────────────────────────────────────────────────────────┘
  Phase 1: Python Engine Core (Segmentation, PCA Deskew, Shape Classifier, TPS Warp)
      │
      ▼ [User Verification: Python Test Runner / Isolated Image Output]
  Phase 2: Tray Extractor UI & Interactive Slot Mapping (Live Contours & Shape Badge)
      │
      ▼ [User Verification: Real Tray Photo Upload, Auto-Detection & Slot Assignment]
  Phase 3: Hand Rigging Canvas & Bundled Master Hand Template
      │
      ▼ [User Verification: 5-Point Anchor Placement & Bundled Studio Hand Validation]
  Phase 4: Real-Time PBR Compositor & Dual-Tier Lighting Engine
      │
      ▼ [User Verification: 6-Pass PBR Rendering, Length Presets & Realistic C-Curve]
  Phase 5: High-Res Production Export (300 DPI PNG & Layered PSD)
      │
      ▼ [User Verification: Full-DPI Quality & Multi-Layer Photoshop File Output]
```

---

## Phase 1: Core Processing Engine (Backend Algorithms) — [COMPLETED ✅]

> **Objective:** Build and verify the computer vision and mathematical foundation in the Python sidecar before connecting to the UI.

### Tasks
- [x] Implement `python_sidecar/segmentation.py`:
  - Adaptive thresholding & core chrominance segmentation for packaging trays.
  - **$G^2$-Continuous Super-Ellipse Contour Fitting**: Two-piece super-ellipse ($y_{\text{eq}} \approx 0.64$, $p_{\text{up}} \approx 1.82$, $q_{\text{up}} \approx 2.18$) with sub-pixel cosine pole sampling, removing polygonal jaggedness.
  - **Card Homography Rectification**: Detect packaging card quad and suppress edge shadow fringes.
  - **PCA Deskewing**: Extract the principal major eigenvector on contour points to find the true nail spine ($v_{\text{spine}}$) and rotate by $-\theta_{\text{tray}}$ to deskew every nail to true $0^\circ$ vertical.
  - **Automatic 180° Orientation**: Cuticle-down, tapered tip-up detection based on width taper and area moments.
  - **Geometric Shape Classifier**:
    - Tip flattening ratio (top 20% width vs. mid-body width): $> 0.68 \rightarrow$ **Square**, $0.38 - 0.68 \rightarrow$ **Coffin**, $< 0.35 \rightarrow$ Tapered.
    - Tip vertex angle & curvature approximation: $< 50^\circ \rightarrow$ **Stiletto**, $50^\circ - 80^\circ \rightarrow$ **Almond**, smooth radius $\rightarrow$ **Oval**.
- [x] Implement `python_sidecar/warping.py`:
  - 2D Thin-Plate Spline (TPS) transformation via SciPy's `RBFInterpolator` with basis function \(r^2 \ln r\).
  - **3-Tier Control Grid**: Row 1 (Cuticle anchor), Row 2 (Apex arch at ~40%), Row 3 (Tip extension) to capture natural longitudinal downward pitch.
  - **Transverse C-Curve Barrel Distortion**: Normalized per-scanline horizontal compression simulating the 3D cylindrical wrap around the finger plate without outer capsule bloating.
  - **Anatomical Cuticle Pocket Tuck**: $G^1$-tangent semi-elliptical cuticle lip mask (`build_cuticle_pocket_mask`) ensuring natural finger skin sits cleanly over the press-on base.
- [x] Implement JSON-RPC handlers in `python_sidecar/main.py`:
  - `segment_tray(image_path_or_base64)` $\rightarrow$ returns detected bounding boxes, contours, PCA angles, and classified shapes.
  - `warp_nail(nail_image, source_contour, target_landmarks, length_scale, c_curve_depth)`.
  - `composite_hand(tray_path, hand_path, finger_mapping)`.
- [x] Automated test pipeline & quality benchmark harness:
  - `python_sidecar/test_pipeline.py`: Comprehensive test runner covering single and multi-tray runs with automated metric tracking.
  - `python_sidecar/score_realism_agent.py`: 5-dimension objective realism scoring agent (Shape smoothness, Specular gloss ribbon, Cuticle pocket seating, Under-tip contact shadow, Photographic color fidelity).
  - `python_sidecar/test_all_nails.py`: Multi-tray batch testing harness across 11+ packaging tray formats.
  - `python_sidecar/render_target_hand.py`: Production end-to-end rendering pipeline for user photo try-ons.

### 🧪 Testable Deliverable (Verified)
- Benchmarked on real user packaging tray (`target_tray.jpg`) and bare-hand photo (`target_hand.jpg`).
- Achieved **83.2 / 100 Realism Score** on `score_realism_agent.py`.
- Side-by-side verification deliverables saved to `test_output/` and artifacts directory.

---

## Phase 2: Tray Extractor UI & Interactive Slot Mapping

> **Objective:** Connect the React frontend to the sidecar's segmentation engine so users can upload photos, inspect detected nails, and assign them to fingers.

### Tasks
- [ ] Implement image file picker & drag-and-drop upload in [`CanvasViewport.tsx`](file:///c:/Users/Timuas/Projects/try-on-studio/src/components/canvas/CanvasViewport.tsx).
- [ ] Connect [`TrayControls.tsx`](file:///c:/Users/Timuas/Projects/try-on-studio/src/components/inspector/TrayControls.tsx) to sidecar `segment_tray`:
  - Display classified shape badge (e.g., `[Detected Shape: Coffin]`) with manual override dropdown.
  - Render interactive SVG/Canvas overlay over the tray photo with numbered slots (1–5 / 10).
  - Draggable contour vertices and bounding boxes for manual micro-nudging.
- [ ] Wire orientation controls:
  - 180° flip (swaps cuticle and tip).
  - Fine-rotation slider (-45° to +45°).
- [ ] Wire Slot-to-Finger chip assignment (Thumb, Index, Middle, Ring, Pinky).

### 🧪 Testable Deliverable
Launch `pnpm tauri dev`, open the **Tray Extractor** tab, upload a photo of a press-on nail packaging tray:
1. Slots are automatically detected with colored contour overlays and classified shapes.
2. Clicking slots 1–5 lets you reassign them to fingers.
3. Flipping or rotating the nail updates the preview in real time.

> [!IMPORTANT]
> **User Feedback Checkpoint 2**: Test various packaging types (blister packs, clear trays, glossy cards) and evaluate slot detection accuracy and manual adjustment ergonomics.

---

## Phase 3: Hand Template Rigging & Bundled Master Hand

> **Objective:** Enable users to set up hand templates by placing finger anchor landmarks and automatically generating procedural occlusion and shadow maps, plus ship a pre-rigged master hand template.

### Tasks
- [ ] Ship with a built-in **Default Studio Hand Template** (`templates/default_studio_hand/`):
  - High-res unpolished bare hand image with all 5 fingers facing forward on a single focal plane.
  - Pre-measured 5-point landmark coordinates per finger.
  - Pre-baked cuticle alpha mask, contact AO shadow map, and specular gloss map.
- [ ] Implement Bare Hand canvas in [`CanvasViewport.tsx`](file:///c:/Users/Timuas/Projects/try-on-studio/src/components/canvas/CanvasViewport.tsx):
  - Zoom and pan controls for sub-pixel landmark placement.
  - Interactive 5-point landmark placement per finger:
    1. Cuticle Apex (top center of proximal fold)
    2. Left Lateral Sidewall
    3. Right Lateral Sidewall
    4. Distal Edge Left
    5. Distal Edge Right
- [ ] Implement procedural map generation in `python_sidecar/template_generator.py`:
  - Eponychium skin fold occlusion mask ($0.5\text{px} - 0.7\text{px}$ Gaussian feathered edge).
  - Contact ambient occlusion (AO) shadow map along the cuticle seam.
  - Specular gloss reflection vector map.
- [ ] Implement `.nailtemplate` serialization (JSON metadata + generated PNG masks) in OS AppData directory.

### 🧪 Testable Deliverable
Open the **Hand Rigging** tab:
1. Select the pre-baked "Default Studio Hand" or load a custom bare hand image.
2. Click and adjust the 5 anchor points on each finger.
3. Click "Generate Rig Maps" to inspect the resulting cuticle overlay and shadow mask passes.
4. Save and load `.nailtemplate` bundles.

> [!IMPORTANT]
> **User Feedback Checkpoint 3**: Verify whether 5-point placement is fast and intuitive, and verify that the generated cuticle skin fold looks natural.

---

## Phase 4: Real-Time PBR Compositing & Dual-Tier Lighting Engine

> **Objective:** Combine extracted nails from Phase 2 with the rigged hand template from Phase 3 into a photorealistic, lighting-adjustable mockup.

### Tasks
- [x] Implement `python_sidecar/compositor.py` (6-layer physical blending engine):
  - [x] **Pass 1: Glare Inpainting & Edge Cleanup**: Fast-marching inpainting (`clean_tray_glare_preserving_texture`) to heal lightbox reflections and remove white card edge fringes.
  - [x] **Pass 2: Cuticle Pocket Tuck & Contact AO**: Seamless anatomical tuck with gentle contact shadow ($\text{soft\_cuticle\_ao} = 0.88 + 0.12 \times \text{ao}$).
  - [x] **Pass 3: 3D Surface Normals & Diffuse Relighting**: 3D cylindrical normal calculation with diffuse ratio constrained to $[0.84, 1.04]$ to protect 100% of photographic texture and contrast.
  - [x] **Pass 4: Sheer Subsurface Translucency**: Translucency restricted strictly to proximal nail-bed zones with low local variance.
  - [x] **Pass 5: Salon Gel Topcoat Specular Ribbon**: Crisp vertical softbox reflection ($\sigma \approx 0.026$) aligned with studio key light.
  - [x] **Pass 6: Tip Under-Shadow & Sensor Micro-Grain**: Key-light aligned directional contact drop shadow ($+1.6\text{px}$ X, $+4.2\text{px}$ Y) cast exclusively on fingertip skin pixels.
- [ ] Real-time caching & fast downscaled preview rendering for responsive slider movement (<150ms).
- [ ] Connect [`LightingControls.tsx`](src/components/inspector/LightingControls.tsx):
  - Presets: Studio Softbox, Ring Light, Direct Flash, Natural Daylight.
  - Anatomical Length Presets: Quick buttons for **Short** ($k \approx 0.2$), **Medium** ($k \approx 0.5$), **Long** ($k \approx 1.0$), **Extra-Long** ($k \approx 1.5$) alongside the continuous slider.
  - Sliders: Gloss Intensity, Shadow Depth, Transverse C-Curve, Light Angle.

### 🧪 Testable Deliverable
Open the **Compositor & Lighting** tab:
1. The 5 extracted nails are automatically warped and placed onto the rigged hand.
2. Clicking length presets (**Short / Med / Long / XL**) or dragging the slider lengthens the nails while preserving natural curvature.
3. Switching lighting presets updates reflections and contact shadows immediately.

> [!IMPORTANT]
> **User Feedback Checkpoint 4**: Evaluate visual realism (Does the cuticle tuck look real? Are shadows believable? Does the C-curve look 3D?).

---

## Phase 5: Production Export & Packaging

> **Objective:** Export print-ready 300 DPI composites and layered PSD files for professional e-commerce use.

### Tasks
- [ ] Implement full-resolution composite export in `python_sidecar/export.py`:
  - 300 DPI rendering in standard ratios (1:1 Square, 4:5 Portrait, 16:9 Banner).
  - Transparent Hand PNG (cutout without background).
- [ ] Multi-layer PSD generation via `pytoshop`:
  - Preserves 6 distinct layers with matching blend modes (Normal, Multiply, Screen).
- [ ] Wire [`ExportControls.tsx`](file:///c:/Users/Timuas/Projects/try-on-studio/src/components/inspector/ExportControls.tsx) with native OS Save Dialog (`@tauri-apps/plugin-dialog`).

### 🧪 Testable Deliverable
In the **Export** tab:
1. Export a Flattened 300 DPI PNG and inspect its crispness in an image viewer.
2. Export a Multi-Layer PSD and open it in Photoshop/GIMP to verify individual layer blend modes and transparency.

> [!IMPORTANT]
> **User Feedback Checkpoint 5**: Final review of output assets, file sizes, and e-commerce readiness.
