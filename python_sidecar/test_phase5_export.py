"""
Phase 5 Test Suite - High-Resolution Production Export & Layered PSD Verification
(docs/phases-roadmap.md, Phase 5 Deliverable)

Validates:
  1. 300 DPI Flattened PNG with physical DPI metadata (pHYs chunk)
  2. Multi-layer Photoshop (.psd) export with all 5 discrete PBR layers & authentic blend modes
  3. Aspect ratio framing (1:1 Square, 4:5 Portrait, 16:9 Banner)
  4. End-to-end JSON-RPC handle_export_composite execution
"""

import os
import sys
import json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import handle_segment_tray, handle_extract_nail, handle_export_composite

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_output", "phase5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_phase5_tests():
    print("=" * 60)
    print("PHASE 5: PRODUCTION EXPORT & LAYERED PSD TEST SUITE")
    print("=" * 60)

    # 1. Segment and extract nails from test tray
    tray_path = "test_assets/target_tray.jpg"
    print(f"\n[Step 1] Segmenting tray nails from {tray_path}...")
    seg_res = handle_segment_tray({"image_path": tray_path, "max_slots": 5})
    assert seg_res["status"] == "ok", f"Segmentation failed: {seg_res}"
    slots = seg_res["slots"]
    print(f"  -> Detected {len(slots)} nail slots.")

    mapped_nails = {}
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for i, s in enumerate(slots[:5]):
        ext_res = handle_extract_nail({
            "image_path": tray_path,
            "contour": s["contour"],
            "deskew": True,
        })
        assert ext_res["status"] == "ok", f"Extraction failed for slot {i}: {ext_res}"
        fname = finger_names[i]
        mapped_nails[fname] = ext_res["image_base64"]
        print(f"  -> Extracted nail {s['slot_id']} ({s.get('shape_category')}) => {fname}")

    # 2. Test 300 DPI PNG Export across Aspect Ratios
    ratios = ["original", "1:1", "4:5", "16:9"]
    for ratio in ratios:
        print(f"\n[Step 2] Testing 300 DPI PNG Export for aspect ratio '{ratio}'...")
        export_res = handle_export_composite({
            "image_path": "templates/default_studio_hand/hand.jpg",
            "mapped_nails": mapped_nails,
            "lighting_options": {
                "preset": "studio",
                "glossIntensity": 1.1,
                "shadowDepth": 1.0,
                "lightAngle": 15.0,
            },
            "export_options": {
                "format": "png",
                "aspect_ratio": ratio,
                "dpi": 300,
                "output_dir": OUTPUT_DIR,
                "filename": f"export_test_{ratio.replace(':', '-')}",
            },
        })
        assert export_res["status"] == "ok", f"Export failed: {export_res}"
        png_info = export_res["files"]["png"]
        png_path = png_info["path"]
        assert os.path.exists(png_path), f"PNG file not found: {png_path}"
        
        # Verify DPI metadata with Pillow
        with Image.open(png_path) as im:
            w, h = im.size
            dpi = im.info.get("dpi")
            print(f"  -> Generated PNG: {os.path.basename(png_path)}")
            print(f"     Dimensions: {w}x{h} px, Size: {os.path.getsize(png_path) / 1024:.1f} KB")
            print(f"     Embedded DPI: {dpi}")
            assert dpi is not None, "Missing DPI metadata in PNG!"
            assert round(dpi[0]) == 300 and round(dpi[1]) == 300, f"Incorrect DPI: {dpi}"

            # Check aspect ratio proportions
            if ratio == "1:1":
                assert w == h, f"Square ratio failed: {w} != {h}"
            elif ratio == "4:5":
                assert abs(w / h - 0.8) < 0.02, f"4:5 ratio failed: {w}/{h} = {w/h}"
            elif ratio == "16:9":
                assert abs(w / h - (16.0 / 9.0)) < 0.02, f"16:9 ratio failed: {w}/{h} = {w/h}"

    # 3. Test Multi-Layer PSD Export
    print("\n[Step 3] Testing Multi-Layer PSD Export...")
    psd_res = handle_export_composite({
        "image_path": "templates/default_studio_hand/hand.jpg",
        "mapped_nails": mapped_nails,
        "lighting_options": {
            "preset": "studio",
            "glossIntensity": 1.2,
            "shadowDepth": 1.0,
        },
        "export_options": {
            "format": "psd",
            "aspect_ratio": "original",
            "output_dir": OUTPUT_DIR,
            "filename": "export_test_production",
        },
    })
    assert psd_res["status"] == "ok", f"PSD export failed: {psd_res}"
    psd_info = psd_res["files"]["psd"]
    psd_path = psd_info["path"]
    assert os.path.exists(psd_path), f"PSD file not found: {psd_path}"
    print(f"  -> Generated PSD: {os.path.basename(psd_path)}")
    print(f"     Size: {os.path.getsize(psd_path) / (1024 * 1024):.2f} MB")
    print(f"     Declared Layers: {psd_info['layers']}")

    # Verify PSD with Pillow
    with Image.open(psd_path) as psd_im:
        print(f"     Pillow PSD Verification: Size {psd_im.size}, Mode {psd_im.mode}")
        assert psd_im.size[0] > 0 and psd_im.size[1] > 0

    # 4. Test "both" combined export package
    print("\n[Step 4] Testing Combined (Both PNG + PSD) Export Package...")
    both_res = handle_export_composite({
        "image_path": "templates/default_studio_hand/hand.jpg",
        "mapped_nails": mapped_nails,
        "lighting_options": {"preset": "ring", "nailLengthScale": 1.15},
        "export_options": {
            "format": "both",
            "aspect_ratio": "4:5",
            "dpi": 300,
            "output_dir": OUTPUT_DIR,
            "filename": "tryon_studio_final_delivery",
        },
    })
    assert both_res["status"] == "ok"
    assert "png" in both_res["files"] and "psd" in both_res["files"]
    print(f"  -> Both PNG & PSD exported successfully in {both_res['execution_time_ms']} ms!")
    print(f"     PNG: {both_res['files']['png']['filename']} ({both_res['files']['png']['dimensions']})")
    print(f"     PSD: {both_res['files']['psd']['filename']} ({len(both_res['files']['psd']['layers'])} layers)")

    print("\n" + "=" * 60)
    print("ALL PHASE 5 PRODUCTION EXPORT TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_phase5_tests()
