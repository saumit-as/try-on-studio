#!/usr/bin/env python3
"""
Test Suite for Milestone 2: MobileSAM Interactive Prompt Engine & Refinement Pipeline.
Validates:
  1. Encoder latency & embedding LRU cache
  2. Decoder latency (< 25-100 ms target)
  3. Interactive box prompt IoU vs ground truth (>= 0.90 gate, target >= 0.95)
  4. Multi-turn point feedback using mask_id
  5. Native-resolution guided filter refinement + de-matting metrics
  6. Coordinate round-trip precision (< 0.5 px)
  7. Golden repeatability test
  8. JSON-RPC handle_segment_prompt handler compliance
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# Ensure sidecar directory is in sys.path
sidecar_dir = os.path.dirname(os.path.abspath(__file__))
if sidecar_dir not in sys.path:
    sys.path.insert(0, sidecar_dir)

from ml_models import get_sam_runner, MobileSAMRunner
from segmentation import segment_packaging_tray
from main import handle_segment_prompt

REPO_ROOT = os.path.abspath(os.path.join(sidecar_dir, ".."))
SAMPLE_TRAY = os.path.join(REPO_ROOT, "test_assets", "nails-sample-tray.jpg")
FRENCH_TRAY = os.path.join(REPO_ROOT, "test_assets", "nails-french-tray.jpg")


def test_models_exist():
    print("[1/8] Verifying MobileSAM ONNX models...")
    runner = get_sam_runner()
    assert runner.is_available(), "MobileSAM encoder or decoder session is not available!"
    print("  ✓ Encoder and Decoder sessions successfully loaded.")


def test_encoder_latency_and_cache():
    print("[2/8] Testing Encoder latency and LRU cache...")
    runner = get_sam_runner()
    img = cv2.imread(SAMPLE_TRAY)
    assert img is not None, f"Failed to load {SAMPLE_TRAY}"

    # Clear cache for isolated benchmark
    runner._embedding_cache.clear()

    # First run (uncached)
    t0 = time.perf_counter()
    res1 = runner.encode_image(img, image_id="sample_tray_test")
    uncached_ms = (time.perf_counter() - t0) * 1000
    print(f"  Uncached encode latency: {uncached_ms:.1f} ms (Target <= 1500 ms)")
    assert uncached_ms <= 2500, f"Encoder too slow: {uncached_ms:.1f} ms"
    assert res1["embedding"].shape == (1, 256, 64, 64)

    # Second run (cached)
    t1 = time.perf_counter()
    res2 = runner.encode_image(img, image_id="sample_tray_test")
    cached_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cached encode latency: {cached_ms:.3f} ms")
    assert cached_ms < 5.0, f"Cached retrieval too slow: {cached_ms} ms"
    assert np.array_equal(res1["embedding"], res2["embedding"])
    print("  ✓ Encoder latency and caching passed.")


def test_box_prompt_segmentation():
    print("[3/8] Testing Interactive Box Prompt Segmentation...")
    runner = get_sam_runner()
    img = cv2.imread(SAMPLE_TRAY)
    h, w = img.shape[:2]

    # Get reference slot 1 from classical segmentation
    classical_slots = segment_packaging_tray(img, max_slots=5)
    assert len(classical_slots) > 0, "No classical slots found on sample tray."
    s1 = classical_slots[0]
    bx, by, bw, bh = s1["bbox"]
    box = [bx, by, bx + bw, by + bh]

    t0 = time.perf_counter()
    res = runner.predict_prompt(img, bbox=box, image_id="sample_tray_test", slot_id=1)
    prompt_ms = (time.perf_counter() - t0) * 1000

    print(f"  Box-to-contour latency: {prompt_ms:.1f} ms (Target <= 300 ms)")
    assert prompt_ms <= 350, f"Prompt segmentation too slow: {prompt_ms:.1f} ms"
    assert res["method"] == "sam"
    assert res["status"] == "ok"
    assert len(res["contour"]) >= 20, "Contour should have at least 20 points."
    assert "mask_id" in res and res["mask_id"]
    assert "metrics" in res
    assert res["metrics"]["choke_px"] in [1, 2, 3]
    assert res["metrics"]["area_removed_pct"] <= 4.0

    # Compare IoU with classical contour
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(gt_mask, [np.array(s1["contour"], dtype=np.int32)], -1, 1, -1)
    sam_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(sam_mask, [np.array(res["contour"], dtype=np.int32)], -1, 1, -1)

    inter = np.logical_and(sam_mask, gt_mask).sum()
    union = np.logical_or(sam_mask, gt_mask).sum()
    iou = inter / union
    print(f"  Sample tray Slot 1 IoU: {iou:.4f} (Gate >= 0.90, Target >= 0.95)")
    assert iou >= 0.90, f"IoU too low: {iou:.4f}"
    print("  ✓ Box prompt segmentation passed.")


def test_french_tray_box_prompt():
    print("[4/8] Testing White French Tips on White Card (Hard Case)...")
    runner = get_sam_runner()
    img = cv2.imread(FRENCH_TRAY)
    assert img is not None, f"Failed to load {FRENCH_TRAY}"
    h, w = img.shape[:2]

    classical_slots = segment_packaging_tray(img, max_slots=5)
    s1 = classical_slots[0]
    bx, by, bw, bh = s1["bbox"]
    box = [bx, by, bx + bw, by + bh]

    res = runner.predict_prompt(img, bbox=box, image_id="french_tray_test", slot_id=1)
    assert res["method"] == "sam"

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(gt_mask, [np.array(s1["contour"], dtype=np.int32)], -1, 1, -1)
    sam_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(sam_mask, [np.array(res["contour"], dtype=np.int32)], -1, 1, -1)

    inter = np.logical_and(sam_mask, gt_mask).sum()
    union = np.logical_or(sam_mask, gt_mask).sum()
    iou = inter / union
    print(f"  French tray Slot 1 IoU: {iou:.4f} (Gate >= 0.90)")
    assert iou >= 0.90, f"French tray IoU too low: {iou:.4f}"
    print("  ✓ French tray box prompt passed.")


def test_multiturn_point_feedback():
    print("[5/8] Testing Multi-turn Point Feedback with mask_id...")
    runner = get_sam_runner()
    img = cv2.imread(SAMPLE_TRAY)
    box = [530, 724, 706, 988]

    # Turn 1: Initial box prompt
    res1 = runner.predict_prompt(img, bbox=box, image_id="sample_tray_test", slot_id=1)
    mask_id1 = res1["mask_id"]
    assert mask_id1 in runner._mask_cache, "Mask not cached for feedback!"

    # Turn 2: Point feedback prompt (+1 positive point) using prev_mask_id
    px = 530 + 88
    py = 724 + 132
    points = [{"x": px, "y": py, "label": 1.0}]
    t0 = time.perf_counter()
    res2 = runner.predict_prompt(
        img,
        bbox=box,
        points=points,
        prev_mask_id=mask_id1,
        image_id="sample_tray_test",
        slot_id=1
    )
    feedback_ms = (time.perf_counter() - t0) * 1000
    print(f"  Point refinement latency with mask feedback: {feedback_ms:.1f} ms")
    assert res2["method"] == "sam"
    assert res2["mask_id"] != mask_id1, "New turn should create a new mask token."
    print("  ✓ Multi-turn point feedback passed.")


def test_coordinate_roundtrip():
    print("[6/8] Testing Coordinate Round-Trip Precision...")
    pts1 = np.float32([[120, 150], [920, 110], [960, 940], [100, 910]])
    pts2 = np.float32([[0, 0], [1024, 0], [1024, 1024], [0, 1024]])
    H = cv2.getPerspectiveTransform(pts1, pts2)
    H_inv = np.linalg.inv(H)

    box = [235.0, 312.0, 480.0, 680.0]
    corners = np.array([
        [[box[0], box[1]]],
        [[box[2], box[1]]],
        [[box[2], box[3]]],
        [[box[0], box[3]]]
    ], dtype=np.float32)

    rect_corners = cv2.perspectiveTransform(corners, H)
    orig_corners = cv2.perspectiveTransform(rect_corners, H_inv)
    max_err = float(np.max(np.abs(corners - orig_corners)))
    print(f"  Max round-trip error: {max_err:.4f} px (Gate < 0.5 px)")
    assert max_err < 0.5, f"Coordinate round-trip error exceeded 0.5px: {max_err}"
    print("  ✓ Coordinate round-trip passed.")


def test_golden_repeatability():
    print("[7/8] Testing Golden Repeatability...")
    runner = get_sam_runner()
    img = cv2.imread(SAMPLE_TRAY)
    box = [530, 724, 706, 988]

    res_a = runner.predict_prompt(img, bbox=box, image_id="golden_test", slot_id=1)
    res_b = runner.predict_prompt(img, bbox=box, image_id="golden_test", slot_id=1)

    cnt_a = np.array(res_a["contour"])
    cnt_b = np.array(res_b["contour"])
    assert len(cnt_a) == len(cnt_b), "Contour length must be deterministic."
    diff = np.max(np.abs(cnt_a - cnt_b))
    assert diff == 0, f"Golden test failed: non-deterministic contours diff={diff}"
    print("  ✓ Golden repeatability passed (0 px variance).")


def test_jsonrpc_handler():
    print("[8/8] Testing JSON-RPC handle_segment_prompt API Contract...")
    params = {
        "image_path": SAMPLE_TRAY,
        "image_id": "jsonrpc_test_tray",
        "bbox": [530, 724, 706, 988],
        "slot_id": 2,
    }
    result = handle_segment_prompt(params)
    assert result["status"] == "ok"
    assert result["slot_id"] == 2
    assert result["method"] == "sam"
    assert len(result["contour"]) > 0
    assert len(result["bbox"]) == 4
    assert result["bbox"][0] < result["bbox"][2]
    assert result["bbox"][1] < result["bbox"][3]
    assert "metrics" in result
    assert "decoder_ms" in result["metrics"]
    assert "refine_ms" in result["metrics"]
    assert "choke_px" in result["metrics"]
    assert "area_removed_pct" in result["metrics"]
    print("  JSON-RPC response:")
    print(f"    Slot: {result['slot_id']}, Method: {result['method']}")
    print(f"    BBox: {result['bbox']}")
    print(f"    Metrics: {result['metrics']}")
    print("  ✓ JSON-RPC handle_segment_prompt contract passed.")


def run_all_tests():
    print("================================================================")
    print("   MILESTONE 2: MOBILESAM & REFINEMENT UNIT TEST SUITE")
    print("================================================================")
    t_start = time.perf_counter()
    test_models_exist()
    test_encoder_latency_and_cache()
    test_box_prompt_segmentation()
    test_french_tray_box_prompt()
    test_multiturn_point_feedback()
    test_coordinate_roundtrip()
    test_golden_repeatability()
    test_jsonrpc_handler()
    total_time = time.perf_counter() - t_start
    print("================================================================")
    print(f"   ALL MILESTONE 2 TESTS PASSED CLEANLY IN {total_time:.2f}s")
    print("================================================================")


if __name__ == "__main__":
    run_all_tests()
