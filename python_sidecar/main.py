#!/usr/bin/env python3
"""
Press-On Nail Mockup Generator - Python Sidecar
Persistent JSON-RPC stdio daemon handling:
  - Nail contour segmentation & GrabCut snapping
  - Thin-Plate Spline (TPS) warping & C-curve cylindrical distortion
  - Multi-pass PBR lighting compositing
  - High-DPI and multi-layer PSD exports
"""

import sys
import json
import traceback

def handle_ping(params):
    return {"status": "pong", "version": "0.1.0"}

def handle_echo(params):
    return params

HANDLERS = {
    "ping": handle_ping,
    "echo": handle_echo,
}

def register_handler(name, func):
    HANDLERS[name] = func

def main():
    # Ensure stdout is unbuffered for immediate line-delimited JSON-RPC response delivery
    sys.stdout.reconfigure(line_buffering=True)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method not in HANDLERS:
                response = {
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }
            else:
                try:
                    result = HANDLERS[method](params)
                    response = {"id": req_id, "result": result}
                except Exception as ex:
                    response = {
                        "id": req_id,
                        "error": {
                            "code": -32000,
                            "message": str(ex),
                            "traceback": traceback.format_exc()
                        }
                    }
        except Exception as e:
            response = {
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
