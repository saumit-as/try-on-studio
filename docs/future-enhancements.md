# Future Enhancements & Performance Backlog

This document tracks identified performance optimizations, architectural improvements, and non-blocking features parked for future milestones.

---

## 1. Sidecar Cold-Start & Startup Latency Optimization

### Problem Statement
On cold launch, the desktop app takes several seconds before the daemon connection status turns green (`Connected`). While the frontend renders quickly, `App.tsx` calls `pingSidecar()` on mount, which synchronously blocks in Tauri Rust until Python has finished executing all top-level imports.

### Benchmarked Breakdown
| Component | Measured Overhead | Root Cause |
| :--- | :--- | :--- |
| `scipy.signal` | **~988 ms** | Eagerly imported at top of `segmentation.py`; loads `scipy.stats`, `scipy.optimize`, and full distribution tables. |
| `scipy.ndimage` | **~534 ms** | Eagerly imported at module root. |
| `cv2` (OpenCV headless) | **~264 ms** | Loaded at module root. |
| `pytoshop` + `PIL` | **~100 ms** | Eagerly imported in `export.py` at launch. |
| ONNX Session Init (SAM) | **~853 ms** | Loading `mobile_sam.encoder.onnx` (28 MB) and `decoder.onnx` (16 MB) upfront if initialized at boot. |
| **Total Cold-Start Delay** | **~2.45 – 3.3 s** | Python cannot read first line of `stdin` until all modules load. |

### Planned Solutions
1. **Lazy / Deferred Module Imports in `main.py`:**
   - Move module imports (`segmentation`, `warping`, `template_generator`, `compositor`, `export`) inside their respective JSON-RPC handler functions instead of the global file header.
   - **Expected Gain:** Daemon enters `main()` and responds to `ping` in **< 40 ms** (instant connection).
2. **In-Function Scoping of `scipy.signal`:**
   - Move `import scipy.signal` inside `build_symmetric_smooth_nail_mask()` and `_detect_card_rows_and_cols()` where `savgol_filter` and `find_peaks` are actually executed.
   - **Expected Gain:** Eliminates **~1,000 ms** from segmentation load time.
3. **On-Demand ONNX Model Initialization:**
   - Defer initializing `ort.InferenceSession` until the user draws their first bounding box or uploads a tray image, rather than inside `MobileSAMRunner.__init__`.
   - **Expected Gain:** Eliminates **~850 ms** of CPU work during app launch.
4. **Optimistic Non-Blocking Frontend Health Check:**
   - Decouple the React mount lifecycle from the sidecar ping so the UI is immediately interactive with optimistic state while ping validates in the background.

---

## 2. Release Sidecar Binary Compilation (PyInstaller / Nuitka)

### Problem Statement
In development mode, Tauri spawns the virtualenv interpreter (`python.exe ../python_sidecar/main.py`), requiring Windows NTFS to dynamically resolve dozens of `.pyd` C-extension libraries.

### Planned Solutions
- Build a standalone, frozen executable using `python_sidecar/build_sidecar.py` into `src-tauri/binaries/nail-sidecar-x86_64-pc-windows-msvc.exe`.
- Bypasses runtime Python interpreter initialization and `.venv` filesystem checks.

---

## 3. Windows Dev-Build Linker Acceleration

### Problem Statement
During `pnpm tauri dev`, the default MSVC linker (`link.exe`) takes 4–8 seconds to link Tauri v2, WebView2, and Wry crates even when no Rust code has changed.

### Planned Solutions
- Configure the LLVM linker (`lld`) or `mold` via `.cargo/config.toml` on Windows:
  ```toml
  [target.x86_64-pc-windows-msvc]
  linker = "rust-lld.exe"
  ```
- Reduces incremental linking time from ~5s down to < 1s.

---

## 4. Local Learning Flywheel (Privacy-Preserving Dataset Capture)

- Opt-in local storage of accepted masks and user-corrected Bezier contours.
- Provides a regression benchmark and training data for future model calibration without any cloud dependencies.
