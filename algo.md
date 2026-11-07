# Computer Vision & Detection Algorithms in Try-On Studio

**Try-On Studio** uses a hybrid computer vision and computational geometry pipeline implemented across the Python sidecar daemon (`python_sidecar/segmentation.py`, `template_generator.py`, `warping.py`, and `compositor.py`). 

Instead of relying on heavy, non-deterministic deep learning models that require multi-gigabyte GPU weights, Try-On Studio employs **deterministic, sub-millisecond, sub-pixel accurate classical computer vision (OpenCV, SciPy, NumPy)** tailored specifically for press-on nail packaging trays and hand anatomical geometry.

---

## Architecture Overview of Detection Pipelines

```mermaid
flowchart TD
    subgraph Packaging Tray Detection Pipeline
        A["Raw Tray Photograph"] --> B["1. Backing Card Detection & Homography Rectification"]
        B --> C["2. Fronto-Parallel Coordinate Transformation"]
        C --> D["3. Multi-Cue Color Space Segmentation (HSV, Lab, Warmth)"]
        D --> E["4. 1D Projection Peak/Valley Cutting (Rows & Columns)"]
        E --> F["5. Directional Flash Shadow Suppression"]
        F --> G["6. Central Moments / PCA Deskewing"]
        G --> H["7. Cuticle vs. Tip Inversion Analysis"]
        H --> I["8. Geometric Shape Classification (Almond, Coffin, etc.)"]
        I --> J["9. G2 Super-Elliptical Profile Fitting & Extraction"]
    end

    subgraph Hand & Template Detection Pipeline
        K["Bare Hand Photograph"] --> L["10. Skin Masking (YCrCb + HSV + Background Distance)"]
        L --> M["11. Distance Transform Palm Center Localization"]
        M --> N["12. 1D Radial Distance Peak Detection (Fingertips)"]
        N --> O["13. Handedness & Thumb Asymmetry Classification"]
        O --> P["14. 5-Point Anchor Estimation (c_l, c_c, c_r, d_l, d_r)"]
    end

    subgraph Defect & Artifact Detection
        Q["Extracted RGBA Nail"] --> R["15. White Top-Hat Morphological Glare Spike Detection"]
        R --> S["16. Telea Fast-Marching Glare Healing & Perimeter Edge Cleanup"]
    end
```

---

## 1. Backing Card & Perspective Rectification

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L53-L172)*

Photos of press-on nail packaging are often taken at an angle, with perspective tilt or foreshortening. Before detecting nail slots, the system detects the rectangular presentation backing card and transforms the image into a fronto-parallel plane.

### Algorithm Steps:
1. **Scale Normalization**: Downsamples the image to a maximum dimension of 1000px for noise-immune card boundary detection.
2. **Card Binary Segmentation**:
   $$\text{CardMask} = (\text{Gray} > 165) \land (\text{HSV}_{\text{Saturation}} < 40)$$
   Followed by morphological closing with a rectangular structuring element ($k \approx 0.8\%$ of image dimension).
3. **Upper Lid Reflection Rejection**: Identifies candidate card contours ($> 15\%$ total image area). It penalizes mirror reflections in the upper box lid by damping contours whose centroid $c_y < 0.35 \times H_{\text{img}}$.
4. **Quadrilateral Corner Fitting**:
   - Runs Douglas-Peucker polygon approximation (`cv2.approxPolyDP`) with an escalating tolerance:
     $$\epsilon \in [0.015 \times \text{perimeter}, 0.065 \times \text{perimeter}]$$
     until a convex 4-vertex polygon is identified.
   - Fallback: Uses `cv2.minAreaRect` if polygon fitting fails.
5. **Canonical Polar Point Ordering (`order_quad_points`)**: Sorts the 4 corners in clockwise order $[TL, TR, BR, BL]$ by polar angle from the centroid, then identifies the top-left vertex by minimizing $(x + y)$ subject to the top-edge angle $\theta \in [-45^\circ, +45^\circ]$.
6. **Rectification Decision Thresholds**:
   - Top edge tilt $> 1.5^\circ$
   - Minimum area bounding box tilt $> 1.5^\circ$
   - Trapezoidal width or height divergence (foreshortening) $> 3\%$
7. **Homography Projection**: Computes the $3 \times 3$ perspective transformation matrix $H$:
   $$\mathbf{x}_{\text{rect}} \sim H \mathbf{x}_{\text{orig}}$$
   using `cv2.getPerspectiveTransform` and performs bicubic resampling (`cv2.warpPerspective`). All coordinates detected downstream are projected back to original image space via $H^{-1}$.

---

## 2. Multi-Cue Press-On Nail Foreground Segmentation

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L579-L695)*

Commercial press-on nails present extreme optical challenges: sheer/jelly nude bases, pure white French tips against white backing cards, dark cat-eye pigments, and directional phone flash shadows. A single color threshold always fails; Try-On Studio uses a **triple-cue logical OR** detector inside the backing card ROI:

### 1. Chrominance & Warmth Cue (Sheer, Nude, Pink, Terracotta, Gold)
- Extracts three color metrics:
  - CIE LAB $A^*$ channel (green-to-red axis)
  - HSV Saturation channel $S$
  - Red-channel warmth metric:
    $$W = (R - B) + (R - G)$$
- Establishes adaptive background baseline percentiles ($p_{50}$ for background, $p_{82}$ for luminance).
- Condition:
  $$\text{Cue}_{\text{chroma}} = (\text{Gray} \ge 35) \land \left( (A^* \ge A_{\text{thresh}}) \lor (S \ge S_{\text{thresh}}) \lor (W \ge W_{\text{thresh}}) \right) \land (\text{Gray} < \text{BG}_{\text{gray}} - 5)$$
  *(Note: $\text{Gray} \ge 35$ filters out deep black cast shadows where HSV saturation numerically spikes due to zero division).*

### 2. Dark Pigment Cue (Jet-Black Tips, Deep Crimson, Cat-Eye)
- Differentiates dark lacquer from achromatic cast shadows by requiring either color saturation or $A^*$ chrominance separation:
  $$\text{Cue}_{\text{dark}} = (\text{Gray} < \text{BG}_{\text{gray}} - 32) \land \left( (S \ge \text{BG}_S + 12) \lor (A^* \ge \text{BG}_A + 2.5) \lor (W \ge \text{BG}_W + 14) \right)$$
- Spatially masks out the printed header logo zone ("Nail BERRIE") to avoid false detections.

### 3. White French Tip & Rim Shadow Cue
- For white tips on white cards (luminance delta $\approx 0$), color thresholding is blind.
- Computes Scharr gradient magnitude on CIE LAB Luminance $L^*$:
  $$\nabla L = \sqrt{\left(\frac{\partial L}{\partial x}\right)^2 + \left(\frac{\partial L}{\partial y}\right)^2}$$
- Detects the tiny contact shadow cast by the tip's physical thickness via box-filtered local background subtraction:
  $$\text{RimShadow} = \text{Gray} < (\text{BoxFilter}(\text{Gray}) - 6)$$
- Requires the gradient spike to be vertically connected to a detected pink/chromatic nail body via a directional vertical morphological dilation kernel $(1 \times 0.08 \cdot H_{\text{roi}})$.

---

## 3. 1D Projection Peak/Valley Cutting (Row & Column Disambiguation)

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L695-L751)*

Nails packed closely in trays frequently touch or overlap in binary masks. Try-On Studio decouples them using projection profiles:

1. **Horizontal Projection Profile (Y-axis)**:
   $$P_y(y) = \sum_{x} \text{Mask}(x, y)$$
   Smoothed with a 1D Gaussian kernel ($k \approx 2\%$ of card height).
2. **Row Valley Detection**:
   - Uses `scipy.signal.find_peaks` on $-P_y$ with a minimum row distance constraint ($\Delta y \ge 20\% \cdot H_{\text{card}}$) and prominence thresholding.
   - Slices the foreground binary mask across identified row valley coordinates to prevent vertically adjacent nails from merging.
3. **Vertical Projection Profile (X-axis)**:
   - For each detected row, computes column-wise projection $P_x(x) = \sum_{y} \text{RowMask}(x, y)$ smoothed with a 1D Gaussian kernel.
   - Identifies inter-nail valleys with `scipy.signal.find_peaks` on $-P_x$ with minimum spacing $\Delta x \ge 8\% \cdot W_{\text{row}}$.
   - Slices the columns, producing individual connected components for each nail slot.

---

## 4. PCA Orientation & Longitudinal Deskewing

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L374-L457)*

Nails placed in packaging trays are rarely perfectly upright (often tilted $\pm 1^\circ$ to $\pm 6^\circ$).

### Dense 2D Central Area Moments
Rather than performing PCA on sparse convex-hull boundary vertices (which suffers from severe vertex-clustering bias around high-curvature tips), Try-On Studio rasterizes the nail into a solid binary patch and computes **continuous 2D image moments**:
$$m_{pq} = \sum_{x} \sum_{y} x^p y^q I(x, y)$$
Centroid:
$$(\bar{x}, \bar{y}) = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$$
Central moments:
$$\mu_{20} = \frac{m_{20}}{m_{00}} - \bar{x}^2, \quad \mu_{02} = \frac{m_{02}}{m_{00}} - \bar{y}^2, \quad \mu_{11} = \frac{m_{11}}{m_{00}} - \bar{x}\bar{y}$$
The principal inertial axis tilt angle $\theta$ is:
$$\theta = \frac{1}{2} \operatorname{atan2}\left(2\mu_{11}, \mu_{02} - \mu_{20}\right)$$
The deskew rotation matrix is constructed with `cv2.getRotationMatrix2D` around $(\bar{x}, \bar{y})$ and applied via bilinear affine transformation.

---

## 5. Cuticle vs. Tip Inversion Detection

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L459-L503)*

Commercial tray packaging almost universally arranges press-on nails **upside-down** (cuticle base at the top, free-edge tip pointing down toward the customer). The software detects this automatically:

1. **Scanline Band Width Ratios**:
   - Extreme bands: top $2\%-20\%$ width vs. bottom $80\%-98\%$ width.
   - Quarter bands: upper quarter ($12\%-35\%$) vs. lower quarter ($65\%-88\%$).
   - If $\frac{W_{\text{top}}}{W_{\text{bottom}}} > 1.04$, the nail is flagged as inverted (cuticle at top).
2. **Luminance Gradient Verification (French / Ombré Nails)**:
   - For nails where shapes are nearly symmetric, it samples the luminance of the top core zone ($10\%-45\%$) vs. bottom core zone ($72\%-95\%$).
   - If the bottom zone is significantly brighter ($L_{\text{bot}} - L_{\text{top}} > 35$), the French smile line is at the bottom, confirming inversion.
3. **Row-Level Heuristic Consensus**:
   - Compares the standard deviation of slot top coordinates vs. bottom coordinates across the entire row. Because tips vary in length while cuticles are aligned on the card tape, the side with lower variance corresponds to the cuticle base.

---

## 6. Geometric Nail Shape Classification

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L505-L576)*

The algorithm categorizes each nail into standard salon silhouettes (**Stiletto, Almond, Square, Squoval, Coffin, Oval**):

### Metric Computation:
1. **Tip Flattening Ratio**:
   $$\text{Ratio}_{\text{flat}} = \frac{W_{\text{tip}}}{W_{\text{max}}}$$
   measured across the distal $3\%-15\%$ zone.
2. **Aspect Ratio**:
   $$\text{Aspect} = \frac{H_{\text{nail}}}{W_{\text{nail}}}$$
3. **Vertex Count**: Number of vertices after Douglas-Peucker approximation ($\epsilon = 0.02 \times \text{perimeter}$).

### Decision Tree:
| Shape | Flattening Ratio | Aspect Ratio | Characteristics |
| :--- | :--- | :--- | :--- |
| **Stiletto** | $< 0.36$ | Any | Sharp needle point |
| **Almond** | $0.36 \le r < 0.54$ | Any | Tapered convex curve |
| **Oval** | $0.50 \le r < 0.60$ | Any | Smooth rounded arch |
| **Square** | $> 0.80$ | Any | Wide, sharp perpendicular edge |
| **Squoval** | $0.58 \le r \le 0.80$ | $< 1.70$ | Flat center with rounded shoulders |
| **Coffin / Ballerina** | $0.60 \le r \le 0.78$ | $\ge 1.70$ | Long tapered sidewalls ending in flat tip (vertices $\le 7$) |

---

## 7. Data-Driven G2 Super-Elliptical Profile Fitting

*Source: [`python_sidecar/segmentation.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/segmentation.py#L174-L372)*

Raw contour edges from noisy packaging trays contain jagged steps. To create a photorealistic composite, the boundary is mathematically modeled using a **2-piece $G^2$-continuous super-ellipse**:

1. **Equator Detection**:
   - Half-widths $w(y)$ are measured along every scanline from the vertical centerline.
   - Filtered with a Savitzky-Golay polynomial filter (`scipy.signal.savgol_filter`).
   - Anatomical maximum width (equator $y_{\text{eq}}$) is constrained to $58\%-66\%$ from the tip toward the cuticle base.
   - Enforces anatomical monotonicity (width strictly increases towards the equator from both ends).
2. **Super-Ellipse Optimization**:
   Fits Lamé curve exponents $p$ and $q$ separately for the tip arch and cuticle cup:
   $$\frac{w(u)}{w_{\text{eq}}} = \left(1 - u^p\right)^{1/q}$$
   - **Upper Tip Arch**: Optimized over $p \in [1.80, 3.40]$, $q \in [2.12, 2.65]$ with prior regularization around $(p=1.82, q=2.18)$.
   - **Lower Cuticle Cup**: Optimized over $p \in [2.00, 2.45]$, $q \in [2.30, 2.65]$ with prior around $(p=2.05, q=2.40)$.
3. **Sub-Pixel Cosine-Spaced Rasterization**:
   Evaluates dense points using cosine parameterization near the poles:
   $$u = \cos(\theta), \quad \theta \in [0, \pi/2]$$
   guaranteeing a true horizontal tangent at both the tip and cuticle without scanline staircase artifacts.
4. **4x Supersampled Anti-Aliasing**: Rasterized at $4\times$ supersampling and downscaled with area averaging (`cv2.INTER_AREA`), followed by a $1.5\text{px}$ Gaussian feather.

---

## 8. Hand Landmark & Fingertip Detection

*Source: [`python_sidecar/template_generator.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/template_generator.py#L484-L650)* and [`test_hand_detect.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/test_hand_detect.py#L10-L200)*

When authoring a hand template, the system automatically detects the 5 fingers and their anatomical rigging landmarks:

### 1. Dual Color-Space Skin Segmentation
- **YCrCb Range**: $Cr \in [130, 185]$, $Cb \in [75, 135]$
- **HSV Skin Range**: $H \in [0, 32]$, $S \in [20, 255]$
- Combined with background color distance rejection sampled from image border corners.
- Morphological closing & opening isolate the primary hand contour.

### 2. Distance Transform Palm Center Localization
- Computes Euclidean distance transform on the binary hand silhouette:
  $$D(x, y) = \min_{(x', y') \in \text{BG}} \sqrt{(x - x')^2 + (y - y')^2}$$
- Excludes the bottom $22\%$ of the image (wrist area).
- The global maximum of $D(x, y)$ yields the **palm center** $(X_p, Y_p)$ and **inscribed palm radius** $R_p$.

### 3. Fingertip Peak Detection via Radial Contour Profiling
- For all contour points above the palm center ($y < Y_p - 0.22 \cdot R_p$), groups points by column and extracts the highest point (minimum $y$).
- Smooths the upper envelope with a 1D Gaussian kernel.
- Runs local minima detection on $y(x)$ (corresponding to upward fingertip peaks) enforcing minimum spacing:
  $$\Delta x_{\text{fingers}} \ge 0.26 \cdot R_p$$
- Identifies the 4 upright fingers (Index, Middle, Ring, Pinky).

### 4. Handedness & Lateral Thumb Detection
- Samples lateral contour points to the left and right of the palm center ($Y_p \pm 0.5 \cdot R_p$).
- Computes maximum lateral distance:
  $$\text{RightExtent} = \max(X_{\text{right}} - X_p), \quad \text{LeftExtent} = \max(X_p - X_{\text{left}})$$
- **Classification**:
  - If $\text{RightExtent} \ge \text{LeftExtent}$: **Left hand** (dorsal view, thumb pointing right).
  - Otherwise: **Right hand** (dorsal view, thumb pointing left).

### 5. 5-Point Anchor Estimation
For each finger, the axis vector $\vec{v}_{\text{axis}} = P_{\text{tip}} - P_{\text{palm}}$ and perpendicular unit $\vec{v}_{\text{perp}}$ are calculated. By raymarching across the hand mask at $25\%$ below the tip, the finger width $2 \cdot W_{\text{half}}$ is measured, estimating the 5 primary landmarks:
- **Cuticle Apex** ($c_c$): $P_{\text{distal\_mid}} - 2.2 \cdot W_{\text{half}} \cdot \vec{v}_{\text{axis}}$
- **Cuticle Corners** ($c_l, c_r$): $c_c \pm 0.95 \cdot W_{\text{half}} \cdot \vec{v}_{\text{perp}}$
- **Distal Sidewalls** ($d_l, d_r$): $P_{\text{distal\_mid}} \pm 0.88 \cdot W_{\text{half}} \cdot \vec{v}_{\text{perp}}$

---

## 9. Overhead Lightbox Glare Detection

*Source: [`python_sidecar/compositor.py`](file:///c:/Users/Timuas/Projects/try-on-studio/python_sidecar/compositor.py#L24-L100)*

Tray photos often contain harsh white specular glare spots from overhead ring lights that destroy the illusion when composited onto a hand.

### Multi-Criteria White Top-Hat Detection:
1. **White Top-Hat Transform**:
   $$\text{TopHat}(L) = L - (L \circ K)$$
   where $L$ is CIE LAB Luminance, and $K$ is an elliptical structuring element with width $\approx 16\% \cdot W_{\text{nail}}$.
2. **Glare Seed Isolation**:
   - Luminance spike: $\text{TopHat}(L) > 20.0$ and $L > 190.0$
   - Desaturation: $\text{HSV}_S < 24.0$ (pinpoint glare is completely achromatic)
3. **Morphological Expansion & Connected Component Filtering**:
   - Expands to halo ($L > 160.0$, $S < 32.0$).
   - Connected component width must be $< 22\% \cdot W_{\text{nail}}$ (rejects French tip smile bands).
   - Blob area must be $< 3.5\%$ of nail area (rejects large white decorative nail art).
4. **Surgical Telea Inpainting**:
   - Applies Fast-Marching Method (`cv2.inpaint` with `INPAINT_TELEA`, radius $2\text{px}$) strictly within the identified glare masks.
   - All surrounding nail art, glitter, French tips, and gold charms remain 100% byte-identical to the source photograph.

---

## Summary of Libraries and Mathematical Foundations

| Capability | Core Algorithms / Mathematics | Implementation |
| :--- | :--- | :--- |
| **Perspective Rectification** | Douglas-Peucker approximation, Homography projection ($H$), Fronto-parallel warping | `cv2.approxPolyDP`, `cv2.getPerspectiveTransform`, `cv2.warpPerspective` |
| **Nail Extraction** | Multi-cue color spaces (HSV, CIE LAB, Warmth), Scharr gradient magnitude, morphological opening/closing | `cv2.Scharr`, `cv2.cvtColor`, `cv2.morphologyEx` |
| **Slot Disambiguation** | 1D projection profiles ($P_x, P_y$), peak & valley detection | `scipy.signal.find_peaks`, `cv2.GaussianBlur` |
| **Nail Alignment** | 2D Continuous Central Area Moments ($\mu_{20}, \mu_{02}, \mu_{11}$), Inertial PCA tilt angle | `cv2.moments`, `cv2.getRotationMatrix2D`, `cv2.warpAffine` |
| **Orientation / Inversion** | Multi-band scanline width ratios, vertical luminance gradient | Custom NumPy vectorized scanline analysis |
| **Shape Classification** | Free-edge flattening ratio, aspect ratio, polygon vertex count | Rule-based geometric decision tree |
| **Contour Smoothing** | 2-piece $G^2$ super-ellipse ($1 - u^p = v^q$), Savitzky-Golay filter, sub-pixel cosine sampling | `scipy.signal.savgol_filter`, supersampled `cv2.fillPoly` |
| **Hand Rigging Detection** | Dual-space skin color segmentation, $L_2$ Distance Transform, radial envelope peak finding | `cv2.distanceTransform`, `cv2.inRange`, `scipy.signal.find_peaks` |
| **Glare Detection & Healing**| White Top-Hat transform, connected component morphological filter, Telea Fast-Marching Method | `cv2.morphologyEx(MORPH_TOPHAT)`, `cv2.inpaint(INPAINT_TELEA)` |
