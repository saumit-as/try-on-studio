import os
import sys
import base64
import json
import time

sys.path.append("python_sidecar")
from main import handle_segment_tray, handle_extract_nail, handle_render_composite

def test_per_finger_render():
    os.makedirs("test_output/per_finger", exist_ok=True)
    tray_path = "test_assets/target_tray.jpg"
    seg_res = handle_segment_tray({"image_path": tray_path, "max_slots": 5})
    slots = seg_res["slots"]
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    mapped_nails = {}

    for i, s in enumerate(slots[:5]):
        ext_res = handle_extract_nail({
            "image_path": tray_path,
            "contour": s["contour"],
            "deskew": True
        })
        mapped_nails[fingers[i]] = ext_res["image_base64"]

    # Test distinct customization on each finger:
    # Thumb: 3/4 side profile, standard length
    # Index: Long stiletto (1.45x length, tapered 0.8x)
    # Middle: Extra-long statement (1.60x length, wide 1.1x)
    # Ring: Medium classic (1.05x length, deep cuticle tuck -8px)
    # Pinky: Petite natural (0.85x length, slight tilt +4 deg)
    test_options = {
        "preset": "studio",
        "glossIntensity": 1.2,
        "shadowDepth": 1.0,
        "nailLengthScale": 1.0,
        "finger_adjustments": {
            "Thumb": {
                "lengthScale": 1.05,
                "widthScale": 1.0,
                "axialShift": 0,
                "angleOffset": 0,
                "cCurve": 50,
                "tipTaper": 1.0
            },
            "Index": {
                "lengthScale": 1.45,
                "widthScale": 0.95,
                "axialShift": 2,
                "angleOffset": 0,
                "cCurve": 60,
                "tipTaper": 0.82
            },
            "Middle": {
                "lengthScale": 1.60,
                "widthScale": 1.12,
                "axialShift": -4,
                "angleOffset": 0,
                "cCurve": 55,
                "tipTaper": 0.95
            },
            "Ring": {
                "lengthScale": 1.05,
                "widthScale": 1.0,
                "axialShift": -8,
                "angleOffset": 0,
                "cCurve": 50,
                "tipTaper": 1.0
            },
            "Pinky": {
                "lengthScale": 0.85,
                "widthScale": 0.92,
                "axialShift": 0,
                "angleOffset": 4.0,
                "cCurve": 50,
                "tipTaper": 0.95
            }
        }
    }

    t0 = time.time()
    res = handle_render_composite({
        "mapped_nails": mapped_nails,
        "lighting_options": test_options,
        "downscale": 0.5
    })
    elapsed = (time.time() - t0) * 1000.0
    assert res["status"] == "ok"

    data_bytes = base64.b64decode(res["composite_base64"].split(",", 1)[1])
    out_path = "test_output/per_finger/customized_fingers_preview.jpg"
    with open(out_path, "wb") as f:
        f.write(data_bytes)

    print(f"Per-finger composite rendered in {res['render_time_ms']}ms (wall: {elapsed:.1f}ms)")
    print(f"Saved output to {out_path}")

if __name__ == "__main__":
    test_per_finger_render()
