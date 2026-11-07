import cv2
import numpy as np

def test_robust_fingertip_finder(img_path, name="test"):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return
    h, w = img.shape[:2]

    # Background subtraction
    corner_samples = np.vstack([
        img[:25, :25].reshape(-1, 3),
        img[:25, -25:].reshape(-1, 3),
        img[-25:, :25].reshape(-1, 3),
        img[-25:, -25:].reshape(-1, 3),
    ])
    bg_color = np.median(corner_samples, axis=0)
    color_dist = np.linalg.norm(img.astype(np.float32) - bg_color[None, None, :], axis=2)
    fg_mask = (color_dist > 25).astype(np.uint8) * 255

    # Skin mask (YCrCb + HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 185, 135]))
    skin_hsv = cv2.inRange(hsv, np.array([0, 15, 30]), np.array([35, 255, 255]))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)

    combined = cv2.bitwise_and(skin_mask, fg_mask)
    if cv2.countNonZero(combined) < (h * w * 0.03):
        combined = fg_mask

    # Mild morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found")
        return
    hand_cnt = max(contours, key=cv2.contourArea)

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [hand_cnt], -1, 255, -1)

    # Palm center
    dist_map = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist_search = dist_map.copy()
    dist_search[int(h * 0.78):, :] = 0
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_search)
    palm_x, palm_y = max_loc
    palm_r = max(18.0, float(max_val))

    pts = hand_cnt.squeeze()
    # Filter valid points above lower wrist
    valid = pts[pts[:, 1] < palm_y + palm_r * 0.70]
    dx = valid[:, 0] - palm_x
    dy = valid[:, 1] - palm_y
    radii = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx) * 180.0 / np.pi

    # Filter points further than palm_r * 1.05
    mask_outer = radii > (palm_r * 1.05)
    v_outer = valid[mask_outer]
    r_outer = radii[mask_outer]
    a_outer = angles[mask_outer]

    # Approximate polygon on contour
    peri = cv2.arcLength(hand_cnt, True)
    approx = cv2.approxPolyDP(hand_cnt, 0.005 * peri, True).squeeze()
    
    # Candidate tips: polygon vertices that are far from palm
    candidates = []
    for p in approx:
        px, py = float(p[0]), float(p[1])
        if py >= palm_y + palm_r * 0.70:
            continue
        d = np.hypot(px - palm_x, py - palm_y)
        if d < palm_r * 1.10:
            continue
        ang = np.arctan2(py - palm_y, px - palm_x) * 180.0 / np.pi
        candidates.append((int(px), int(py), d, ang))

    # Cluster candidates by angle within 12 degrees
    candidates.sort(key=lambda c: c[3])
    ang_clusters = []
    for c in candidates:
        if not ang_clusters:
            ang_clusters.append([c])
        else:
            if abs(c[3] - ang_clusters[-1][-1][3]) < 14.0:
                ang_clusters[-1].append(c)
            else:
                ang_clusters.append([c])

    # Pick max radius in each angular cluster
    cluster_tips = [max(g, key=lambda x: x[2]) for g in ang_clusters]

    print(f"[{name}] Found {len(cluster_tips)} angular tip clusters:")
    for ct in cluster_tips:
        print(f"  Point: ({ct[0]}, {ct[1]}), angle: {ct[3]:.1f} deg, dist: {ct[2]:.1f} px")

    # If we have 5 clusters:
    if len(cluster_tips) == 5:
        # Handedness check
        leftmost = cluster_tips[0]
        rightmost = cluster_tips[-1]
        if rightmost[1] > leftmost[1] + 25:
            handedness = "left"
        elif leftmost[1] > rightmost[1] + 25:
            handedness = "right"
        else:
            handedness = "left"

        if handedness == "left":
            fingers = {
                "Pinky": cluster_tips[0],
                "Ring": cluster_tips[1],
                "Middle": cluster_tips[2],
                "Index": cluster_tips[3],
                "Thumb": cluster_tips[4],
            }
        else:
            fingers = {
                "Thumb": cluster_tips[0],
                "Index": cluster_tips[1],
                "Middle": cluster_tips[2],
                "Ring": cluster_tips[3],
                "Pinky": cluster_tips[4],
            }
        print(f"[{name}] PERFECT 5/5 DETECTED! Handedness: {handedness}")
        for fname, tip in fingers.items():
            print(f"  {fname}: ({tip[0]}, {tip[1]})")

test_robust_fingertip_finder("test_output/user_full_canvas_crop.jpg", "User Hand Crop")
test_robust_fingertip_finder("test_assets/hand-template.jpg", "Master Studio Template")
