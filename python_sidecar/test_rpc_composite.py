import os
import sys
sys.path.append("python_sidecar")

from main import handle_render_composite

def test_rpc_render():
    res = handle_render_composite({
        "lighting_options": {
            "preset": "studio",
            "glossIntensity": 1.2,
            "shadowDepth": 1.0,
            "nailLengthScale": 1.0,
            "cCurveIntensity": 1.0,
            "lightAngle": 0.0,
        },
        "downscale": 0.5,
    })
    assert res["status"] == "ok"
    assert "composite_base64" in res
    print("handle_render_composite executed successfully!")
    print(f"Render time: {res['render_time_ms']}ms, size: {res['image_size']}")

if __name__ == "__main__":
    test_rpc_render()
