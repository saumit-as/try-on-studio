import os
import sys
import time
import base64
import json
import cv2
import numpy as np

sys.path.append("python_sidecar")
from main import handle_segment_tray, handle_extract_nail, handle_render_composite

def test_phase4_suite():
    os.makedirs("test_output/phase4", exist_ok=True)
    tray_path = "test_assets/target_tray.jpg"
    print(f"[1/4] Segmenting studio tray: {tray_path}...")
    seg_res = handle_segment_tray({"image_path": tray_path, "max_slots": 5})
    assert seg_res["status"] == "ok"
    slots = seg_res["slots"]
    print(f"  -> Detected {len(slots)} nail slots.")

    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    mapped_nails = {}

    print(f"[2/4] Extracting transparent nail RGBA cutouts...")
    for i, slot in enumerate(slots[:5]):
        ext_res = handle_extract_nail({
            "image_path": tray_path,
            "contour": slot["contour"],
            "deskew": True
        })
        assert ext_res["status"] == "ok"
        mapped_nails[fingers[i]] = ext_res["image_base64"]
        print(f"  -> Extracted nail for {fingers[i]}")

    print(f"[3/4] Testing composite rendering across presets and resolutions...")
    test_cases = [
        {"name": "studio_preview", "preset": "studio", "scale": 1.05, "downscale": 0.5, "gloss": 1.0, "shadow": 1.0},
        {"name": "studio_full_300dpi", "preset": "studio", "scale": 1.05, "downscale": 1.0, "gloss": 1.0, "shadow": 1.0},
        {"name": "ring_light", "preset": "ring", "scale": 1.05, "downscale": 0.5, "gloss": 1.4, "shadow": 0.8},
        {"name": "direct_flash", "preset": "flash", "scale": 1.05, "downscale": 0.5, "gloss": 1.6, "shadow": 1.3},
        {"name": "natural_daylight", "preset": "natural", "scale": 1.05, "downscale": 0.5, "gloss": 0.9, "shadow": 0.9},
        {"name": "short_nails", "preset": "studio", "scale": 0.85, "downscale": 0.5, "gloss": 1.0, "shadow": 1.0},
        {"name": "extra_long_nails", "preset": "studio", "scale": 1.65, "downscale": 0.5, "gloss": 1.2, "shadow": 1.1},
    ]

    results = []
    for tc in test_cases:
        t0 = time.time()
        res = handle_render_composite({
            "mapped_nails": mapped_nails,
            "lighting_options": {
                "preset": tc["preset"],
                "glossIntensity": tc["gloss"],
                "shadowDepth": tc["shadow"],
                "nailLengthScale": tc["scale"],
                "cCurveIntensity": 1.0,
                "lightAngle": 45.0,
            },
            "downscale": tc["downscale"],
        })
        elapsed_ms = (time.time() - t0) * 1000.0
        assert res["status"] == "ok", f"Render failed for {tc['name']}"

        # Save image to disk
        data_bytes = base64.b64decode(res["composite_base64"].split(",", 1)[1])
        out_path = f"test_output/phase4/{tc['name']}.jpg"
        with open(out_path, "wb") as f:
            f.write(data_bytes)

        print(f"  -> [{tc['name']}] Size: {res['image_size']}, Latency: {res['render_time_ms']}ms (wall: {elapsed_ms:.1f}ms) -> saved to {out_path}")
        results.append({
            "name": tc["name"],
            "image_size": res["image_size"],
            "latency_ms": res["render_time_ms"],
            "output_path": out_path,
        })

    print("[4/4] All Phase 4 test cases completed successfully!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_phase4_suite()
