import cv2
import json
import numpy as np
from template_generator import auto_detect_hand_landmarks

img = cv2.imread('test_output/user_hand_clean_v2.jpg')
print("Image shape:", img.shape if img is not None else None)
if img is not None:
    res = auto_detect_hand_landmarks(img)
    print("Handedness:", res["handedness"])
    print("Palm center:", res["palm_center"])
    print("Palm radius:", res["palm_radius"])
    for k, v in res["landmarks"].items():
        print(k, ": c_c =", v["c_c"], "d_l =", v["d_l"], "d_r =", v["d_r"])
