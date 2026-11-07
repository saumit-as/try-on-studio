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

## Phase 1: Core Processing Engine (Backend Algorithms)

> **Objective:** Build and verify the computer vision and mathematical foundation in the Python sidecar before connecting to the UI.

### Tasks
- [ ] Implement `python_sidecar/segmentation.py`:
  - Adaptive thresholding & Otsu contour detection for nail packaging slots.
  - **PCA Deskewing**: Extract the principal major eigenvector on contour points to find the true nail spine ($v_{\text{spine}}$) and rotate by $-\theta_{\text{tray}}$ to deskew every nail to true $0^\circ$ vertical.
  - **Geometric Shape Classifier**:
    - Tip flattening ratio (top 20% width vs. mid-body width): $> 0.68 \rightarrow$ **Square**, $0.38 - 0.68 \rightarrow$ **Coffin**, $< 0.35 \rightarrow$ Tapered.
    - Tip vertex angle & curvature approximation: $< 50^\circ \rightarrow$ **Stiletto**, $50^\circ - 80^\circ \rightarrow$ **Almond**, smooth radius $\rightarrow$ **Oval**.
  - GrabCut foreground extraction fallback for clear or low-contrast acrylic nails.
- [ ] Implement `python_sidecar/warping.py`:
  - 2D Thin-Plate Spline (TPS) transformation via SciPy's `RBFInterpolator` with basis function \(r^2 \ln r\).
  - **3-Tier Control Grid**: Row 1 (Cuticle anchor), Row 2 (Apex arch at ~40%), Row 3 (Tip extension) to capture natural longitudinal downward pitch.
  - **Transverse C-Curve Barrel Distortion**: Non-linear horizontal compression simulating the 3D cylindrical wrap around the finger plate.
- [ ] Implement JSON-RPC handlers in `python_sidecar/main.py`:
  - `segment_tray(image_path_or_base64)` $\rightarrow$ returns detected bounding boxes, contours, PCA angles, and classified shapes.
  - `refine_contour_grabcut(image_path, bbox, points)` $\rightarrow$ returns refined mask.
  - `warp_nail(nail_image, source_contour, target_landmarks, length_scale, c_curve_depth)`.
- [ ] Create a standalone test script `python_sidecar/test_pipeline.py` with synthetic/sample nail test assets.

### 🧪 Testable Deliverable
Run `python python_sidecar/test_pipeline.py`:
1. Processes a sample tray photo: isolates nail contours, deskews them to vertical, and outputs the detected shape (e.g., `Classified: Coffin`).
2. Warps the nail onto target finger landmarks with C-curve distortion.
3. Saves side-by-side verification images: `Original` $\rightarrow$ `Deskewed Mask` $\rightarrow$ `Warped Texture`.

> [!IMPORTANT]
> **User Feedback Checkpoint 1**: Inspect the edge smoothness, alpha anti-aliasing, shape classification accuracy, and 3D curvature realism of the warped output before connecting to React.

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
- [ ] Implement `python_sidecar/compositor.py` (6-layer physical blending engine):
  - **Layer 1: Base Hand**: Clean photograph of the hand.
  - **Layer 2: Tip Under-Shadow**: Subtle contact drop shadow on fingertip skin directly under the extended tip.
  - **Layer 3: Warped Nail**: TPS + transverse C-curve warped nail texture.
  - **Layer 4: Ambient Occlusion**: Cuticle contact shadow via Multiply blend ($\text{Nail} \times \text{ShadowMap}$).
  - **Layer 5: Specular Gloss**: Studio specular reflection via Screen blend ($1.0 - (1.0 - \text{Base}) \times (1.0 - \text{Specular} \times \text{opacity})$).
  - **Layer 6: Cuticle Skin Overlay**: Eponychium skin fold composited over the nail base to physically tuck the artificial nail under the skin.
- [ ] Real-time caching & fast downscaled preview rendering for responsive slider movement (<150ms).
- [ ] Connect [`LightingControls.tsx`](file:///c:/Users/Timuas/Projects/try-on-studio/src/components/inspector/LightingControls.tsx):
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
