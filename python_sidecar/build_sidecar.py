#!/usr/bin/env python3
"""
Build script to package python_sidecar into a standalone binary
formatted with the Tauri v2 externalBin naming convention:
  nail-sidecar-<target_triplet>[.exe]
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def get_default_target_triplet():
    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        return "x86_64-apple-darwin"
    elif system == "windows":
        return "x86_64-pc-windows-msvc"
    elif system == "linux":
        if machine in ("arm64", "aarch64"):
            return "aarch64-unknown-linux-gnu"
        return "x86_64-unknown-linux-gnu"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    sidecar_dir = repo_root / "python_sidecar"
    binaries_dir = repo_root / "src-tauri" / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)

    triplet = sys.argv[1] if len(sys.argv) > 1 else get_default_target_triplet()
    binary_name = f"nail-sidecar-{triplet}"
    if "windows" in triplet:
        binary_name += ".exe"

    output_path = binaries_dir / binary_name
    print(f"Building sidecar for target: {triplet}")
    print(f"Destination: {output_path}")

    dist_dir = sidecar_dir / "dist"
    build_dir = sidecar_dir / "build"
    shutil.rmtree(dist_dir, ignore_errors=True)
    shutil.rmtree(build_dir, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconsole",
        "--onefile",
        "--name", f"nail-sidecar-{triplet}",
        "--distpath", str(binaries_dir),
        str(sidecar_dir / "main.py")
    ]
    
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=sidecar_dir)
    if res.returncode == 0:
        print(f"Successfully generated: {output_path}")
    else:
        print(f"PyInstaller failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
