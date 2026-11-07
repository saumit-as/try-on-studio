#!/usr/bin/env python3
"""
Test and refine automatic hand & fingernail landmark detection.
"""

import cv2
import numpy as np
import os

def detect_hand_landmarks_cv(img: np.ndarray):
    h, w = img.shape[:2]
    
    # 1. Skin & Hand Segmentation
    # Combine YCrCb and HSV skin models
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Skin bounds
    skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 185, 135]))
    skin_hsv = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([35, 255, 255]))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    
    # Background distance filter from corners
    corner_samples = np.vstack([
        img[:25, :25].reshape(-1, 3),
        img[:25, -25:].reshape(-1, 3),
        img[-25:, :25].reshape(-1, 3),
        img[-25:, -25:].reshape(-1, 3),
    ])
    bg_color = np.median(corner_samples, axis=0)
    color_dist = np.linalg.norm(img.astype(np.float32) - bg_color[None, None, :], axis=2)
    fg_mask = (color_dist > 25).astype(np.uint8) * 255
    
    combined = cv2.bitwise_and(skin_mask, fg_mask)
    if cv2.countNonZero(combined) < 1000:
        # Fallback to foreground if lighting/color is unusual
        combined = fg_mask
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest contour (Hand)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No hand detected in image")
    
    hand_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand_cnt) < 5000:
        raise ValueError("Detected contour too small to be a hand")
        
    # Hand mask
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hand_mask, [hand_cnt], -1, 255, -1)
    
    # 2. Palm Center via Distance Transform
    dist_map = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    # Exclude bottom 20% to avoid wrist
    valid_dist = dist_map.copy()
    valid_dist[int(h * 0.75):, :] = 0
    _, max_val, _, max_loc = cv2.minMaxLoc(valid_dist)
    palm_x, palm_y = max_loc
    palm_r = max(20.0, float(max_val))
    
    # 3. Fingertip Candidates
    pts = hand_cnt.squeeze()
    if len(pts.shape) != 2 or pts.shape[0] < 50:
        raise ValueError("Invalid contour shape")
        
    # Compute Euclidean distance from palm center to every contour point
    dx = pts[:, 0] - palm_x
    dy = pts[:, 1] - palm_y
    dists = np.sqrt(dx**2 + dy**2)
    
    # Smooth distances with 1D Gaussian
    k_size = max(15, (len(dists) // 40) | 1)
    smooth_dists = cv2.GaussianBlur(dists[:, None], (1, k_size), k_size / 3).ravel()
    
    # Find local peaks
    peaks = []
    window = max(10, len(dists) // 50)
    n = len(smooth_dists)
    for i in range(n):
        val = smooth_dists[i]
        if val < palm_r * 1.35:
            continue
        is_peak = True
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            idx = (i + offset) % n
            if smooth_dists[idx] > val:
                is_peak = False
                break
        if is_peak:
            peaks.append((val, i, pts[i]))
            
    # Suppress peaks that are too close to each other
    peaks.sort(key=lambda p: p[0], reverse=True)
    selected_peaks = []
    min_dist_between_tips = palm_r * 0.45
    for p in peaks:
        pt = p[2]
        too_close = False
        for sp in selected_peaks:
            spt = sp[2]
            if np.linalg.norm(pt - spt) < min_dist_between_tips:
                too_close = True
                break
        if not too_close:
            selected_peaks.append(p)
        if len(selected_peaks) >= 5:
            break
            
    if len(selected_peaks) < 5:
        # Fallback to top peaks
        selected_peaks = peaks[:5]
        
    print(f"Detected {len(selected_peaks)} fingertip peaks around palm ({palm_x}, {palm_y}) r={palm_r:.1f}")
    
    # Sort the detected peaks by x-coordinate (from left to right)
    selected_peaks.sort(key=lambda p: p[2][0])
    
    # Determine handedness (Thumb on right = Left hand; Thumb on left = Right hand)
    # Check distances from palm center: Thumb is usually lower (closer to palm) and angled outward
    leftmost = selected_peaks[0]
    rightmost = selected_peaks[-1]
    
    # Compare y-levels: Thumb has larger y (lower down) than index/pinky
    # Or compare gap: gap between thumb and index is larger than pinky and ring
    gap_left = np.linalg.norm(selected_peaks[1][2] - selected_peaks[0][2])
    gap_right = np.linalg.norm(selected_peaks[-1][2] - selected_peaks[-2][2])
    
    if gap_right > gap_left * 1.2 or rightmost[2][1] > leftmost[2][1] + palm_r * 0.3:
        # Thumb is on the right -> Left hand (dorsal view)
        finger_order = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
        is_left_hand = True
    else:
        # Thumb is on the left -> Right hand (dorsal view)
        finger_order = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        is_left_hand = False
        
    print(f"Hand orientation identified: {'Left Hand (Thumb Right)' if is_left_hand else 'Right Hand (Thumb Left)'}")
    
    landmarks_dict = {}
    
    for i, fname in enumerate(finger_order):
        if i >= len(selected_peaks):
            break
        tip_pt = selected_peaks[i][2].astype(np.float64)
        
        # Finger axis vector from palm center to tip
        v_axis = tip_pt - np.array([palm_x, palm_y], dtype=np.float64)
        v_axis_len = np.linalg.norm(v_axis)
        v_unit = v_axis / max(1e-4, v_axis_len)
        
        # Perpendicular unit vector (pointing towards right lateral sidewall)
        v_perp = np.array([-v_unit[1], v_unit[0]], dtype=np.float64)
        if v_perp[0] < 0:
            v_perp = -v_perp
            
        # Estimate finger width by scanning across v_perp at 15% down from tip
        check_pt = tip_pt - v_unit * (palm_r * 0.25)
        # Scan left and right
        w_left = 0
        w_right = 0
        for step in range(1, int(palm_r * 0.8)):
            p_test = check_pt - v_perp * step
            ix, iy = int(round(p_test[0])), int(round(p_test[1]))
            if 0 <= ix < w and 0 <= iy < h and hand_mask[iy, ix] > 0:
                w_left = step
            else:
                break
        for step in range(1, int(palm_r * 0.8)):
            p_test = check_pt + v_perp * step
            ix, iy = int(round(p_test[0])), int(round(p_test[1]))
            if 0 <= ix < w and 0 <= iy < h and hand_mask[iy, ix] > 0:
                w_right = step
            else:
                break
                
        half_w = max(10.0, float(w_left + w_right) * 0.5)
        nail_len = half_w * 2.2 # standard nail bed aspect ratio
        
        # Distal midpoint slightly below fingertip
        distal_mid = tip_pt - v_unit * (half_w * 0.35)
        d_l = distal_mid - v_perp * (half_w * 0.88)
        d_r = distal_mid + v_perp * (half_w * 0.88)
        
        # Cuticle apex
        c_c = distal_mid - v_unit * nail_len
        c_l = c_c - v_perp * (half_w * 0.95)
        c_r = c_c + v_perp * (half_w * 0.95)
        
        # Refine sidewalls against hand contour
        c_curve = 0.42 if fname == "Thumb" else 0.28
        roll = 0.0
        if fname == "Thumb":
            roll = -0.35 if is_left_hand else 0.35
            
        landmarks_dict[fname] = {
            "c_l": [round(float(c_l[0]), 1), round(float(c_l[1]), 1)],
            "c_c": [round(float(c_c[0]), 1), round(float(c_c[1]), 1)],
            "c_r": [round(float(c_r[0]), 1), round(float(c_r[1]), 1)],
            "d_l": [round(float(d_l[0]), 1), round(float(d_l[1]), 1)],
            "d_r": [round(float(d_r[0]), 1), round(float(d_r[1]), 1)],
            "c_curve": c_curve,
            "roll_offset": roll,
            "length_scale": 1.0,
            "tip_taper": 1.0,
        }
        
    return landmarks_dict, hand_cnt, (palm_x, palm_y, palm_r)

if __name__ == "__main__":
    for img_path in ["test_output/user_hand_exact.jpg", "test_assets/target_hand.jpg"]:
        if os.path.exists(img_path):
            print(f"\nProcessing {img_path}...")
            img = cv2.imread(img_path)
            lm, cnt, palm = detect_hand_landmarks_cv(img)
            print("Extracted landmarks successfully:")
            for k, v in lm.items():
                print(f"  {k}: apex={v['c_c']}, d_l={v['d_l']}, d_r={v['d_r']}")
                
            # Draw on image to verify
            vis = img.copy()
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(vis, (palm[0], palm[1]), int(palm[2]), (255, 0, 0), 2)
            for fname, pts in lm.items():
                cc = (int(pts['c_c'][0]), int(pts['c_c'][1]))
                dl = (int(pts['d_l'][0]), int(pts['d_l'][1]))
                dr = (int(pts['d_r'][0]), int(pts['d_r'][1]))
                cv2.circle(vis, cc, 5, (0, 0, 255), -1)
                cv2.circle(vis, dl, 4, (0, 255, 255), -1)
                cv2.circle(vis, dr, 4, (0, 255, 255), -1)
                cv2.putText(vis, fname, (cc[0] - 20, cc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            out_vis = img_path.replace(".jpg", "_detected_vis.jpg")
            cv2.imwrite(out_vis, vis)
            print(f"Saved visual verification to: {out_vis}")
