#!/usr/bin/env python3
"""
Test MediaPipe 3D Skeletal Hand Landmarker integration for hand template rigging.
"""

import os
import sys
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from template_generator import (
    detect_hand_landmarks_mediapipe,
    auto_detect_hand_landmarks,
    refit_finger_landmarks,
    get_mediapipe_hand_detector,
)
from main import handle_auto_detect_hand

def test_mediapipe_detection():
    print("Testing MediaPipe detector initialization...")
    detector = get_mediapipe_hand_detector()
    assert detector is not None, "MediaPipe HandLandmarker detector failed to initialize"
    print("  -> MediaPipe detector initialized successfully!")

    target_hand_path = os.path.join(current_dir, "..", "test_assets", "target_hand.jpg")
    studio_hand_path = os.path.join(current_dir, "..", "test_assets", "hand-template.jpg")

    for path in [target_hand_path, studio_hand_path]:
        print(f"\nTesting detection on {os.path.basename(path)}...")
        img = cv2.imread(path)
        assert img is not None, f"Failed to load {path}"

        res = auto_detect_hand_landmarks(img)
        assert res is not None, "Detection returned None"
        assert res.get("detection_method") == "mediapipe", f"Expected 'mediapipe', got {res.get('detection_method')}"
        assert res["handedness"] == "right", f"Expected right hand, got {res['handedness']}"
        assert len(res["landmarks"]) == 5, f"Expected 5 fingers, got {len(res['landmarks'])}"

        # Validate that all 5 landmark points exist and are valid numbers
        for fname in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
            lm = res["landmarks"][fname]
            for key in ["c_l", "c_c", "c_r", "d_l", "d_r"]:
                assert key in lm, f"Missing {key} in {fname}"
                assert len(lm[key]) == 2
                assert lm[key][0] > 0 and lm[key][1] > 0

            assert 0.0 < lm["c_curve"] <= 1.0
            assert "roll_offset" in lm
            assert lm["length_scale"] > 0
            assert lm["tip_taper"] > 0

        print(f"  -> Successfully detected 5 fingers on {os.path.basename(path)} (method: {res['detection_method']}, handedness: {res['handedness']})")

    print("\nTesting RPC handle_auto_detect_hand...")
    rpc_res = handle_auto_detect_hand({"image_path": target_hand_path})
    assert rpc_res["status"] == "ok"
    assert rpc_res["detection_method"] == "mediapipe"
    assert rpc_res["handedness"] == "right"
    print("  -> RPC auto_detect_hand returned valid response with detection_method='mediapipe'")

    print("\nALL MEDIAPIPE HAND RIGGING TESTS PASSED!")

if __name__ == "__main__":
    test_mediapipe_detection()
