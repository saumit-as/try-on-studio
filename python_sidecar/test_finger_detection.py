import cv2
import numpy as np

def detect_5_fingers(img):
    h, w = img.shape[:2]
    
    # 1. Background subtraction from corners
    corner_samples = np.vstack([
        img[:25, :25].reshape(-1, 3),
        img[:25, -25:].reshape(-1, 3),
        img[-25:, :25].reshape(-1, 3),
        img[-25:, -25:].reshape(-1, 3),
    ])
    bg_color = np.median(corner_samples, axis=0)
    color_dist = np.linalg.norm(img.astype(np.float32) - bg_color[None, None, :], axis=2)
    fg_mask = (color_dist > 25).astype(np.uint8) * 255

    # 2. Skin mask (YCrCb + HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 185, 135]))
    skin_hsv = cv2.inRange(hsv, np.array([0, 15, 30]), np.array([35, 255, 255]))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)

    combined = cv2.bitwise_and(skin_mask, fg_mask)
    if cv2.countNonZero(combined) < (h * w * 0.03):
        combined = fg_mask

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k_close, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hand_cnt = max(contours, key=cv2.contourArea)

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [hand_cnt], -1, 255, -1)

    # Palm center via distance transform
    dist_map = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist_search = dist_map.copy()
    dist_search[int(h * 0.78):, :] = 0
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_search)
    palm_x, palm_y = max_loc
    palm_r = max(18.0, float(max_val))

    # Approximate polygon to find vertices
    peri = cv2.arcLength(hand_cnt, True)
    approx = cv2.approxPolyDP(hand_cnt, 0.006 * peri, True).squeeze()

    candidates = []
    for pt in approx:
        px, py = float(pt[0]), float(pt[1])
        if py > palm_y + palm_r * 0.65:
            continue
        dist_palm = np.sqrt((px - palm_x)**2 + (py - palm_y)**2)
        if dist_palm < palm_r * 1.15:
            continue
        angle = np.arctan2(py - palm_y, px - palm_x) * 180.0 / np.pi
        candidates.append((int(px), int(py), dist_palm, angle))

    # Cluster within 30px
    clustered = []
    for c in sorted(candidates, key=lambda x: -x[2]):
        if not any(np.hypot(c[0] - ex[0], c[1] - ex[1]) < 30 for ex in clustered):
            clustered.append(c)

    # Sort clustered tips by angle from left (-180) to right (0)
    clustered.sort(key=lambda x: x[3])
    
    # Filter to the 5 primary tips with highest prominence in their angular clusters
    # There are 5 fingers: cluster into 5 angular bins
    final_5 = []
    # If more than 5, group tips within 22 degrees of each other and pick the one with max distance to palm
    merged = []
    for c in clustered:
        if not merged:
            merged.append([c])
        else:
            if abs(c[3] - merged[-1][-1][3]) < 20:
                merged[-1].append(c)
            else:
                merged.append([c])
    
    for group in merged:
        best = max(group, key=lambda x: x[2])
        final_5.append(best)

    # If we have more than 5, keep the 5 with largest distance to palm
    if len(final_5) > 5:
        final_5 = sorted(sorted(final_5, key=lambda x: -x[2])[:5], key=lambda x: x[3])

    print(f"Final {len(final_5)} tips:")
    for i, f in enumerate(final_5):
        print(f"  Digit {i}: ({f[0]}, {f[1]}), angle={f[3]:.1f} deg, dist={f[2]:.1f} px")

    # Determine handedness
    if len(final_5) >= 2:
        leftmost = final_5[0]
        rightmost = final_5[-1]
        if rightmost[1] > leftmost[1] + 25:
            handedness = "left" # thumb on right
        elif leftmost[1] > rightmost[1] + 25:
            handedness = "right" # thumb on left
        else:
            handedness = "left"
    else:
        handedness = "left"

    print("Handedness:", handedness)

    if len(final_5) == 5:
        if handedness == "left":
            mapping = {
                "Pinky": final_5[0],
                "Ring": final_5[1],
                "Middle": final_5[2],
                "Index": final_5[3],
                "Thumb": final_5[4],
            }
        else:
            mapping = {
                "Thumb": final_5[0],
                "Index": final_5[1],
                "Middle": final_5[2],
                "Ring": final_5[3],
                "Pinky": final_5[4],
            }
        for name, tip in mapping.items():
            print(f"  -> {name}: ({tip[0]}, {tip[1]})")

    return final_5

if __name__ == "__main__":
    img = cv2.imread("test_output/user_full_canvas_crop.jpg")
    detect_5_fingers(img)
