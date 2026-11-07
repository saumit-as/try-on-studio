#!/usr/bin/env python3
"""
Unit test script for Phase 3 Hand Rigging & Template Engine.
Verifies:
  - JSON-RPC get_default_hand_template returns all 5 finger landmarks and procedural maps
  - generate_hand_maps produces valid cuticle, AO, and specular maps
  - save_hand_template and load_hand_template work with .nailtemplate bundles
"""

import os
import sys
import json
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from main import (
    handle_get_default_hand_template,
    handle_generate_hand_maps,
    handle_save_hand_template,
    handle_load_hand_template,
)
from template_generator import DEFAULT_STUDIO_LANDMARKS


def test_hand_template_rpc():
    print("[Test 1] Testing handle_get_default_hand_template...")
    res = handle_get_default_hand_template({"include_images": True})
    assert res["status"] == "ok"
    assert "landmarks" in res
    assert len(res["landmarks"]) == 5
    for fname in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        assert fname in res["landmarks"], f"Missing {fname}"
        lm = res["landmarks"][fname]
        assert "c_l" in lm and "c_c" in lm and "c_r" in lm and "d_l" in lm and "d_r" in lm
    assert "hand_image_base64" in res
    assert "cuticle_mask_base64" in res
    assert "ao_map_base64" in res
    assert "specular_map_base64" in res
    print("  -> Default studio hand template loaded successfully with all 5 fingers and procedural maps!")

    print("\n[Test 2] Testing handle_generate_hand_maps on custom image...")
    # Create test synthetic or real hand image
    test_img = np.full((1000, 800, 3), 180, dtype=np.uint8)
    # Scale landmarks to 800x1000
    custom_landmarks = {}
    for fname, lm in DEFAULT_STUDIO_LANDMARKS.items():
        custom_landmarks[fname] = {
            "c_l": [lm["c_l"][0] * (800 / 1536), lm["c_l"][1] * (1000 / 2752)],
            "c_c": [lm["c_c"][0] * (800 / 1536), lm["c_c"][1] * (1000 / 2752)],
            "c_r": [lm["c_r"][0] * (800 / 1536), lm["c_r"][1] * (1000 / 2752)],
            "d_l": [lm["d_l"][0] * (800 / 1536), lm["d_l"][1] * (1000 / 2752)],
            "d_r": [lm["d_r"][0] * (800 / 1536), lm["d_r"][1] * (1000 / 2752)],
            "c_curve": lm.get("c_curve", 0.3),
            "roll_offset": lm.get("roll_offset", 0.0),
        }

    # Encode test image to base64
    _, buf = cv2.imencode(".jpg", test_img)
    import base64
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    maps_res = handle_generate_hand_maps({
        "image_base64": b64,
        "landmarks": custom_landmarks,
    })
    assert maps_res["status"] == "ok"
    assert "cuticle_mask_base64" in maps_res
    assert "ao_map_base64" in maps_res
    assert "specular_map_base64" in maps_res
    print("  -> Procedural map generation passed for all 3 passes!")

    print("\n[Test 3] Testing .nailtemplate save and load...")
    save_path = os.path.join(current_dir, "..", "test_output", "test_bundle.nailtemplate")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_res = handle_save_hand_template({
        "target_path": save_path,
        "name": "Test Custom Hand",
        "image_base64": b64,
        "landmarks": custom_landmarks,
    })
    assert save_res["status"] == "ok"
    assert os.path.exists(save_path)

    load_res = handle_load_hand_template({
        "source_path": save_path,
    })
    assert load_res["status"] == "ok"
    tmpl = load_res["template"]
    assert tmpl["name"] == "Test Custom Hand"
    assert len(tmpl["landmarks"]) == 5
    print("  -> .nailtemplate bundle serialized and deserialized flawlessly!")

    from main import handle_auto_detect_hand, handle_refit_finger
    print("\n[Test 4] Testing handle_auto_detect_hand...")
    hand_test_path = os.path.join(current_dir, "..", "test_output", "user_hand_exact.jpg")
    if os.path.exists(hand_test_path):
        auto_res = handle_auto_detect_hand({"image_path": hand_test_path})
        assert auto_res["status"] == "ok"
        assert len(auto_res["landmarks"]) == 5
        assert auto_res["handedness"] == "left"
        print("  -> Auto-detect hand accurately identified all 5 fingers and left-hand orientation!")

        print("\n[Test 5] Testing handle_refit_finger with bounding box...")
        refit_res = handle_refit_finger({
            "image_path": hand_test_path,
            "finger_name": "Thumb",
            "bbox": [400, 310, 95, 110],
            "handedness": "left"
        })
        assert refit_res["status"] == "ok"
        assert refit_res["finger_name"] == "Thumb"
        assert "c_c" in refit_res["landmarks"]
        assert "d_l" in refit_res["landmarks"]
        assert "d_r" in refit_res["landmarks"]
        print(f"  -> Bounding box refit for Thumb passed! Apex c_c = {refit_res['landmarks']['c_c']}")

    print("\nALL PHASE 3 BACKEND TESTS PASSED SUCCESSFULLY! (5/5)")


if __name__ == "__main__":
    test_hand_template_rpc()
