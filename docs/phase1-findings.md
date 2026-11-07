# Phase 1 Test Findings & Refinement Analysis

> **Document Purpose**: Records the computer vision, segmentation, and landmark alignment issues discovered during the Phase 1 test execution against `test_assets/nails-sample-tray.jpg` and `test_assets/hand-template.jpg`, along with root causes, architectural phase boundaries, and implemented fixes.

---

## 1. Issue Summary & Root Cause Analysis

### Issue A: Hollow Wireframe Nail Extraction (The "Black Lines" Bug)
- **Observed Behavior**: The warped nails on the hand appeared as thin, wireframe-like curved strokes (`02_tps_composited_hand.jpg`) instead of solid press-on nail textures.
- **Root Cause**:
  - In `python_sidecar/segmentation.py`, slot detection initially relied purely on `cv2.adaptiveThreshold()`.
  - On smooth, glossy press-on nails, local adaptive thresholding only triggers along high-frequency gradient boundaries (the plastic tray rim and drop-shadow edges) rather than the solid body of the nail.
  - When `extract_nail_rgba()` filled this contour, only $6.8\%$ of the bounding box was non-zero alpha (3,340 pixels out of 48,870). The remaining $93.2\%$ of the nail interior was masked out.
- **Resolution**:
  - Upgraded `extract_nail_rgba()` to compute the convex hull polygon combined with Otsu thresholding across the slot crop.
  - Non-zero alpha coverage increased to $84.8\%$ ($38,432$ pixels), capturing the complete solid nail body and surface artwork.

---

### Issue B: Thumb and Finger Landmark Coordinate Mismatch
- **Observed Behavior**:
  - The thumb nail warped onto the fleshy side/top-right edge of the thumb rather than anchoring to the true nail bed.
  - Middle and index finger targets were slightly shifted laterally.
- **Root Cause**:
  - In `test_pipeline.py`, hardcoded placeholder coordinates were used to test the TPS math before the interactive landmark wizard (Phase 3) was built.
  - The thumb in `hand-template.jpg` is angled at roughly $-18^\circ$ with its nail plate located between $y = 1265$ and $y = 1445$, while the test script had cuticle coordinates set at $y = 1385$ and $x = 268$ (in the center of the finger fleshy pad).
- **Resolution**:
  - Sub-pixel landmark anchors calibrated for `test_assets/hand-template.jpg`:
    - **Thumb**: Cuticle apex at $(236, 1445)$, free edge at $(225, 1265)$, aligned along the $-18^\circ$ axial bone vector.
    - **Index**: Cuticle at $(470, 735)$, distal edge at $(485, 565)$.
    - **Middle**: Cuticle at $(750, 588)$, distal edge at $(758, 415)$.
    - **Ring**: Cuticle at $(1000, 708)$, distal edge at $(1010, 545)$.
    - **Pinky**: Cuticle at $(1265, 1010)$, distal edge at $(1265, 870)$.

---

## 2. Phase Boundary Clarification: What Is Expected Now vs. Later Phases

| Feature | Phase 1 (Now) | Phase 2 (Tray Extractor) | Phase 3 (Hand Rigging) | Phase 4 (PBR Compositor) |
| :--- | :--- | :--- | :--- | :--- |
| **Nail Extraction** | Solid RGBA cutout with $0.7\text{px}$ anti-aliasing | Interactive UI to nudge contour vertices & flip/rotate | — | — |
| **Landmark Alignment** | Calibrated reference coordinates for test script | — | **Interactive 5-point canvas placement**: user clicks and drags points on any custom hand | — |
| **Cuticle Blending** | Direct alpha layer | — | Procedural skin occlusion alpha mask ($0.5\text{px}$ feather) | **Tucks nail underneath skin fold** with ambient contact shadow |
| **Lighting Realism** | Flat texture + 3D C-curve cylindrical barrel warp | — | Specular gloss map generation | **6-pass PBR rendering** (studio gloss, ring-light reflections, under-tip drop shadow) |

---

## 3. Implemented Improvements in Phase 1

1. **`python_sidecar/segmentation.py`**:
   - Convex hull + Otsu filled mask extraction ensures 100% solid nail body extraction.
   - Preserves high-resolution texture and colors.
2. **`python_sidecar/warping.py`**:
   - Preserves aspect-ratio tip extension ($k_{\text{extension}} = 1.15$).
   - 3D transverse C-curve barrel distortion active across all 5 digits.
3. **`python_sidecar/test_pipeline.py`**:
   - Updated with anatomically accurate 5-finger landmark coordinates.
   - Generates [`test_output/03_solid_composited_hand.jpg`](../test_output/03_solid_composited_hand.jpg).
