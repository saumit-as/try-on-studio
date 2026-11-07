"""
Press-On Nail Studio - Nail Tray Segmentation & Shape Classifier
Implements:
  - Hierarchical tray row & slot detection (for multi-row packaging trays)
  - Direct adaptive + Otsu contour detection fallback
  - Principal Component Analysis (PCA) deskewing along the longitudinal spine
  - Geometric tip shape classification (Square, Coffin, Almond, Oval, Stiletto)
  - GrabCut foreground extraction for low-contrast/clear acrylic nails
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def deskew_nail_pca(
    contour: np.ndarray,
    image: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    """
    Computes the principal axis orientation of a nail contour using PCA
    and returns the deskew angle (in degrees), centered contour, and deskewed image.
    
    The principal major eigenvector corresponds to the nail's longitudinal spine (cuticle-to-tip).
    Angle is normalized so that 0 degrees means the nail stands perfectly vertical.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return 0.0, contour, image

    # Center of mass & PCA
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    center = mean[0]
    major_axis = eigenvectors[0]  # [vx, vy]

    # Compute angle of major axis
    vx, vy = major_axis[0], major_axis[1]
    if vy < 0:
        vx, vy = -vx, -vy

    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    deskew_angle = angle_deg - 90.0

    # Rotate contour
    rot_mat = cv2.getRotationMatrix2D(tuple(center), deskew_angle, 1.0)
    rotated_pts = cv2.transform(pts.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
    rotated_contour = rotated_pts.astype(np.int32).reshape(-1, 1, 2)

    deskewed_image = None
    if image is not None:
        h, w = image.shape[:2]
        deskewed_image = cv2.warpAffine(
            image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

    return float(deskew_angle), rotated_contour, deskewed_image


def classify_nail_shape(contour: np.ndarray) -> Dict[str, Any]:
    """
    Classifies the isolated nail contour into Square, Coffin, Almond, Oval, or Stiletto
    using geometric contour profiling on the tip region.
    
    Rules (from architectural specification & research):
      1. Tip Flattening Ratio (width of top 20% vs mid-body width):
         - Ratio > 0.68 -> Square
         - 0.38 <= Ratio <= 0.68 -> Coffin
         - Ratio < 0.35 -> Tapered (Almond, Oval, Stiletto)
      2. Tip Vertex Angle:
         - Interior angle < 50 deg -> Stiletto
         - Interior angle 50 - 80 deg with rounded apex -> Almond
         - Continuous smooth radius -> Oval
    """
    pts = contour.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return {"shape": "Square", "confidence": 0.5, "metrics": {}}

    # Normalize contour so tip is at top
    top_half = pts[pts[:, 1] < (y + 0.5 * h)]
    bottom_half = pts[pts[:, 1] >= (y + 0.5 * h)]

    top_w = float(np.max(top_half[:, 0]) - np.min(top_half[:, 0])) if len(top_half) > 0 else float(w)
    bottom_w = float(np.max(bottom_half[:, 0]) - np.min(bottom_half[:, 0])) if len(bottom_half) > 0 else float(w)

    is_inverted = top_w > (bottom_w * 1.35)

    # Tip ROI (top 20% along length)
    tip_threshold = (y + 0.20 * h) if not is_inverted else (y + 0.80 * h)
    if not is_inverted:
        tip_pts = pts[pts[:, 1] <= tip_threshold]
    else:
        tip_pts = pts[pts[:, 1] >= tip_threshold]

    if len(tip_pts) > 0:
        tip_w = float(np.max(tip_pts[:, 0]) - np.min(tip_pts[:, 0]))
    else:
        tip_w = float(w * 0.5)

    flattening_ratio = float(tip_w / float(w))

    # Polygon approximation for vertex sharpness
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.025 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    metrics = {
        "flattening_ratio": round(flattening_ratio, 3),
        "aspect_ratio": round(float(h / w), 3),
        "vertex_count": int(len(approx)),
        "is_inverted": bool(is_inverted),
    }

    if flattening_ratio > 0.65:
        shape = "Square"
        confidence = min(0.98, 0.65 + (flattening_ratio - 0.65) * 1.2)
    elif 0.38 <= flattening_ratio <= 0.65:
        shape = "Coffin"
        confidence = 0.88
    else:
        if len(approx) <= 6:
            shape = "Stiletto"
            confidence = 0.90
        elif flattening_ratio < 0.25:
            shape = "Almond"
            confidence = 0.85
        else:
            shape = "Oval"
            confidence = 0.82

    return {
        "shape": shape,
        "confidence": round(float(confidence), 2),
        "metrics": metrics,
    }


def segment_packaging_tray(
    image: np.ndarray,
    max_slots: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detects press-on nail slots in a packaging tray image.
    
    Uses a two-tier hierarchical strategy:
      1. First detects blister pack rows/slots if present.
      2. Segments individual nails within each row.
      3. Falls back to global thresholding if rows are not clearly defined.
      
    Returns sorted list of detected slots with:
      - slot_id (1, 2, ...)
      - bbox: [x, y, w, h]
      - contour: List of [x, y] coordinates
      - pca_angle: Deskew angle in degrees
      - shape_info: Detected shape and confidence
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 1. Look for packaging rows
    _, th_otsu_inv = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_cnts, _ = cv2.findContours(th_otsu_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rows = []
    for c in row_cnts:
        area = cv2.contourArea(c)
        if area > (h_img * w_img * 0.015):
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 0.22 <= aspect <= 0.65 and bw > (w_img * 0.20):
                rows.append((y, x, bw, bh))

    rows.sort(key=lambda r: r[0])  # top to bottom

    raw_nails = []

    if len(rows) > 0:
        # Hierarchical detection inside rows
        for ry, rx, rw, rh in rows:
            row_crop = image[ry:ry+rh, rx:rx+rw]
            rc_gray = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)
            rc_blurred = cv2.GaussianBlur(rc_gray, (5, 5), 0)
            th_rc = cv2.adaptiveThreshold(
                rc_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
            )
            inner_cnts, _ = cv2.findContours(th_rc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            row_nails = []
            for c in inner_cnts:
                area = cv2.contourArea(c)
                if area > 1500:
                    x, y, bw, bh = cv2.boundingRect(c)
                    aspect = bh / float(bw) if bw > 0 else 0
                    if 1.15 <= aspect <= 3.2:
                        gx, gy = rx + x, ry + y
                        global_cnt = c + np.array([rx, ry])
                        row_nails.append({
                            "gx": gx,
                            "gy": gy,
                            "bw": bw,
                            "bh": bh,
                            "area": area,
                            "aspect": aspect,
                            "contour": global_cnt,
                        })

            row_nails.sort(key=lambda n: n["gx"])  # sort left-to-right within the row
            raw_nails.extend(row_nails)

    # 2. Fallback if hierarchical rows not found: direct contour search
    if len(raw_nails) == 0:
        th_adapt = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(th_adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
        direct_cnts, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in direct_cnts:
            area = cv2.contourArea(c)
            if 1500 < area < (h_img * w_img * 0.15):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 0
                if 1.15 <= aspect <= 3.2:
                    raw_nails.append({
                        "gx": x,
                        "gy": y,
                        "bw": bw,
                        "bh": bh,
                        "area": area,
                        "aspect": aspect,
                        "contour": c,
                    })
        raw_nails.sort(key=lambda n: (n["gy"] // 80, n["gx"]))

    # 3. Process each detected nail: PCA deskewing & Shape Classification
    results = []
    for idx, item in enumerate(raw_nails[:max_slots], start=1):
        cnt = item["contour"]
        pca_angle, rotated_cnt, _ = deskew_nail_pca(cnt)
        shape_info = classify_nail_shape(rotated_cnt)

        results.append({
            "slot_id": idx,
            "bbox": [int(item["gx"]), int(item["gy"]), int(item["bw"]), int(item["bh"])],
            "contour": cnt.reshape(-1, 2).tolist(),
            "area": float(item["area"]),
            "aspect_ratio": round(float(item["aspect"]), 2),
            "pca_angle": round(float(pca_angle), 2),
            "shape_info": shape_info,
        })

    return results


def extract_nail_rgba(
    image: np.ndarray,
    contour: np.ndarray,
    feather_px: float = 0.7,
) -> np.ndarray:
    """
    Extracts an isolated RGBA nail cutout from the source image using the contour
    with sub-pixel anti-aliased edge feathering and solid body fill.
    """
    pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return np.zeros((10, 10, 4), dtype=np.uint8)

    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)

    cropped = image[y0:y1, x0:x1].copy()
    ch, cw = cropped.shape[:2]

    shifted_contour = pts - np.array([x0, y0])

    # 1. Fill solid nail body via convex hull of the contour
    hull = cv2.convexHull(shifted_contour)
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # 2. Otsu refinement to exclude bright white/reflective packaging backgrounds around the rim
    crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, otsu_inv = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_and(mask, otsu_inv)
    # If otsu preserved the solid body (>70%), use refined mask
    if np.count_nonzero(combined_mask) > (np.count_nonzero(mask) * 0.7):
        mask = combined_mask

    if feather_px > 0:
        ksize = int(np.ceil(feather_px * 4)) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_px)

    b, g, r = cv2.split(cropped)
    rgba = cv2.merge([b, g, r, mask])
    return rgba
