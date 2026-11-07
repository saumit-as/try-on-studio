import cv2
import numpy as np

def fit_finger_in_roi(roi, bx, by, finger_name, handedness="left"):
    bh, bw = roi.shape[:2]
    
    # Segment finger within ROI using YCrCb skin chrominance
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([40, 133, 77]), np.array([255, 185, 135]))
    
    corner_bgr = np.median(np.vstack([roi[:5, :5], roi[:5, -5:]]), axis=(0,1))
    diff = np.linalg.norm(roi.astype(np.float32) - corner_bgr[None, None, :], axis=2)
    fg = (diff > 25).astype(np.uint8) * 255
    combined = cv2.bitwise_and(skin, fg)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=1)
    
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        tx, ty = bx + bw // 2, by + bh // 3
        spine_unit = np.array([0.0, -1.0])
        fw = float(bw * 0.70)
    else:
        best_cnt = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(clean)
        cv2.drawContours(filled, [best_cnt], -1, 255, -1)
        ys, xs = np.where(filled > 0)
        
        is_thumb = (finger_name == "Thumb")
        if not is_thumb:
            min_y = np.min(ys)
            top_xs = xs[ys < min_y + 6]
            tx = bx + int(np.median(top_xs))
            ty = by + int(min_y)
            
            # Find cross sections to determine local spine angle
            mids = []
            widths = []
            for dy in range(6, min(35, bh - min_y), 3):
                curr_y = min_y + dy
                row_xs = xs[ys == curr_y]
                if len(row_xs) > 4:
                    xl = np.min(row_xs)
                    xr = np.max(row_xs)
                    mids.append([(xl + xr) * 0.5 + bx, curr_y + by])
                    widths.append(float(xr - xl))
            
            if len(mids) >= 3:
                mids_arr = np.array(mids)
                v = mids_arr[0] - mids_arr[-1] # points upwards to tip
                v_len = np.linalg.norm(v)
                # Keep spine close to vertical (clamp tilt to +/- 25 degrees)
                spine_raw = v / max(1e-4, v_len)
                if spine_raw[1] > -0.70: # excessive horizontal tilt
                    spine_unit = np.array([0.0, -1.0])
                else:
                    spine_unit = spine_raw
                fw = float(np.median(widths))
            else:
                spine_unit = np.array([0.0, -1.0])
                fw = float(bw * 0.70)
        else:
            thumb_sign = 1.0 if handedness == "left" else -1.0
            scores = xs * thumb_sign * 1.0 - ys * 1.3
            best_i = np.argmax(scores)
            tx = bx + int(xs[best_i])
            ty = by + int(ys[best_i])
            
            v_thumb = np.array([thumb_sign * 0.65, -0.76])
            spine_unit = v_thumb / np.linalg.norm(v_thumb)
            fw = float(bw * 0.75)

    perp_unit = np.array([-spine_unit[1], spine_unit[0]], dtype=np.float64)
    if perp_unit[0] < 0:
        perp_unit = -perp_unit

    is_thumb = (finger_name == "Thumb")
    nail_w = max(16.0, fw * 0.74)
    nail_len = nail_w * (0.95 if is_thumb else 1.05)

    distal_mid = np.array([tx, ty], dtype=np.float64) - spine_unit * 3.0
    d_l = distal_mid - perp_unit * (nail_w * 0.50)
    d_r = distal_mid + perp_unit * (nail_w * 0.50)

    c_c = distal_mid - spine_unit * nail_len
    c_l = c_c - perp_unit * (nail_w * 0.48)
    c_r = c_c + perp_unit * (nail_w * 0.48)

    c_curve = 0.42 if is_thumb else 0.28
    roll_offset = 0.0
    if is_thumb:
        roll_offset = -0.35 if handedness == "left" else 0.35

    return {
        "finger_name": finger_name,
        "landmarks": {
            "c_l": [round(float(c_l[0]), 1), round(float(c_l[1]), 1)],
            "c_c": [round(float(c_c[0]), 1), round(float(c_c[1]), 1)],
            "c_r": [round(float(c_r[0]), 1), round(float(c_r[1]), 1)],
            "d_l": [round(float(d_l[0]), 1), round(float(d_l[1]), 1)],
            "d_r": [round(float(d_r[0]), 1), round(float(d_r[1]), 1)],
            "c_curve": c_curve,
            "roll_offset": roll_offset,
            "length_scale": 1.0,
            "tip_taper": 1.0,
        }
    }

if __name__ == "__main__":
    img = cv2.imread("test_output/user_full_canvas_crop.jpg")
    bboxes = {
        "Pinky": [80, 235, 75, 110],
        "Ring": [165, 155, 65, 110],
        "Middle": [230, 115, 65, 110],
        "Index": [315, 145, 75, 110],
        "Thumb": [405, 315, 85, 110]
    }
    
    colors = {
        "Thumb": (94, 63, 244),   # Rose/Red (#f43f5e)
        "Index": (11, 158, 245),  # Amber (#f59e0b)
        "Middle": (129, 185, 16), # Green (#10b981)
        "Ring": (246, 130, 59),   # Blue (#3b82f6)
        "Pinky": (247, 85, 168),  # Purple (#a855f7)
    }

    vis = img.copy()
    for fname, bbox in bboxes.items():
        bx, by, bw, bh = bbox
        roi = img[by:by+bh, bx:bx+bw]
        res = fit_finger_in_roi(roi, bx, by, fname, handedness="left")
        lm = res["landmarks"]
        
        c_l = tuple(map(int, lm["c_l"]))
        c_c = tuple(map(int, lm["c_c"]))
        c_r = tuple(map(int, lm["c_r"]))
        d_l = tuple(map(int, lm["d_l"]))
        d_r = tuple(map(int, lm["d_r"]))
        col = colors[fname]
        
        pts = np.array([c_l, c_c, c_r, d_r, d_l], dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.40, vis, 0.60, 0, vis)
        cv2.polylines(vis, [pts], True, col, 2)
        for p in [c_l, c_c, c_r, d_l, d_r]:
            cv2.circle(vis, p, 3, (255, 255, 255), -1)
            cv2.circle(vis, p, 2, col, -1)
        
        # Bounding box dashed/thin
        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (200, 200, 200), 1)
        # Finger label
        cv2.putText(vis, fname, (c_c[0]-20, c_c[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        cv2.putText(vis, fname, (c_c[0]-20, c_c[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    cv2.imwrite("test_output/subagent_review_target.jpg", vis)
    print("Saved test_output/subagent_review_target.jpg")
