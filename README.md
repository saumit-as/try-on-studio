# Try-On Studio

> A local-first desktop press-on nail mockup generator built with **Tauri v2** (Rust + React 19) and a persistent **Python/OpenCV sidecar**.

Automates the extraction of press-on nail textures from packaging tray photographs and photorealistically composites them onto a 5-finger bare-hand template using Thin-Plate Spline (TPS) warping, 3D cylindrical C-curve distortion, and multi-pass PBR contact shadows and specular reflections.

---

## 🛠️ Tech Stack & Architecture

- **Desktop Shell**: [Tauri v2](https://tauri.app/) (Rust 2024 edition)
- **Frontend**: React 19 + TypeScript + Vite 8 + Tailwind CSS v4 + Lucide Icons
- **Sidecar Daemon**: Python 3.10+ (OpenCV, NumPy, SciPy, Pillow, Pytoshop)
- **IPC Bridge**: High-throughput line-delimited JSON-RPC over `stdio`
- **Platforms**: macOS (Apple Silicon `aarch64` & Intel `x86_64`) and Windows (10/11 `x86_64`)

---

## 📋 Implementation Roadmap & Task Status

### Phase 0: Architecture & Repository Setup
- [x] Comprehensive architectural specification and design plan
- [x] Cross-platform architecture definition (macOS & Windows parity)
- [x] GitHub repository created and connected: [saumit-as/try-on-studio](https://github.com/saumit-as/try-on-studio)

### Phase 1: Core Tauri v2 Shell & Python CV / Warping Engine
- [x] Initialize Tauri v2 project structure (`src-tauri` Rust core)
- [x] Configure React 19 + TypeScript + Vite + Tailwind CSS v4 frontend
- [x] Implement studio shell UI with 4-tab workflow (Tray, Hand Rigging, Compositor, Export)
- [x] Build Python sidecar daemon (`python_sidecar/main.py`) with line-delimited JSON-RPC over stdio
- [x] Implement Rust sidecar IPC client (`src-tauri/src/sidecar.rs`) with dev script fallback and production binary lookup
- [x] Implement TypeScript IPC bridge (`src/services/sidecar.ts`) and ping health check
- [x] Create PyInstaller cross-platform packaging script (`python_sidecar/build_sidecar.py`)
- [x] Implement OpenCV contour detection & adaptive chrominance thresholding (`python_sidecar/segmentation.py`)
- [x] Implement $G^2$-continuous 2-piece super-ellipse contour smoothing & sub-pixel cosine pole sampling
- [x] Implement packaging card homography rectification and shadow-fringe suppression
- [x] Implement PCA deskewing & automatic 180° orientation (cuticle-down / tip-up)
- [x] Implement geometric nail shape classifier (Almond, Coffin, Square, Squoval, Stiletto, Oval)
- [x] Implement SciPy Thin-Plate Spline (TPS) coordinate warping & 3-tier control grid (`python_sidecar/warping.py`)
- [x] Implement normalized transverse cylindrical C-curve barrel distortion
- [x] Implement anatomical cuticle pocket tuck mask & contact AO (`build_cuticle_pocket_mask`)
- [x] Automated test pipeline (`python_sidecar/test_pipeline.py`)
- [x] 5-dimension realism scoring agent (`python_sidecar/score_realism_agent.py`) & batch harness (`python_sidecar/test_all_nails.py`)

### Phase 2: Tray Extractor UI & Interactive Slot Mapping
- [ ] Connect [`CanvasViewport.tsx`](src/components/canvas/CanvasViewport.tsx) to sidecar `segment_tray`
- [ ] Interactive SVG/Canvas contour overlays with numbered slot badges
- [ ] Manual contour vertex editor & drag handles for micro-nudging
- [ ] Orientation controls (180° flip toggle and fine-rotation wheel)
- [ ] Drag-and-drop slot-to-finger assignment chips (Thumb, Index, Middle, Ring, Pinky)

### Phase 3: Hand Template Authoring Wizard & Master Hand
- [x] Bundled master studio hand photo (`test_assets/hand-template.jpg`) at 1536×2752
- [x] Calibrated biometric 5-finger landmark targets (cuticle arch, distal points, 3D roll offsets)
- [ ] Interactive 5-point landmark placement canvas (Apex, Left/Right Corners, Sidewalls, Directional Vector)
- [ ] Canvas zoom, pan, and interactive touch-up brush/eraser tools for mask refinement
- [ ] `.nailtemplate` bundle serializer (JSON rig metadata + 16-bit/8-bit PNG passes)
- [ ] AppData template library manager (macOS `Application Support` / Windows `%APPDATA%`)

### Phase 4: PBR Compositing & Dual-Tier Lighting Engine
- [x] 6-pass PBR compositing pipeline in Python sidecar (`python_sidecar/compositor.py`):
  - [x] Pass 1: Surgical glare-tick inpainting (`clean_tray_glare_preserving_texture`) & edge-fringe cleanup
  - [x] Pass 2: Anatomical cuticle pocket tuck & contact ambient occlusion
  - [x] Pass 3: 3D surface normal mapping & non-destructive diffuse relighting ([0.84, 1.04] range)
  - [x] Pass 4: Texture-preserving sheer nail-bed translucency
  - [x] Pass 5: High-gloss clear salon gel topcoat specular reflection ribbon ($\sigma \approx 0.026$)
  - [x] Pass 6: Key-light aligned directional under-tip contact drop shadows on skin pad with sensor micro-grain
- [ ] Frontend lighting inspector controls connected to JSON-RPC compositor parameters
- [ ] Dual-tier lighting controls (Basic Studio Presets + Advanced Sliders)
- [ ] Real-time caching & thumbnail generation for sub-second UI responsiveness

### Phase 5: Production Export & CI/CD Packaging
- [ ] Flattened 300 DPI high-resolution export in standard aspect ratios (1:1, 4:5, 16:9)
- [ ] Transparent cutout hand PNG export
- [ ] Multi-layer Photoshop PSD export with isolated blend modes (`pytoshop`)
- [ ] GitHub Actions CI workflow for macOS `.dmg`/`.app` and Windows `.exe` (NSIS)/`.msi` installers

---

## 🚀 Development Quickstart

### Prerequisites
- Node.js 20+ & `pnpm`
- Rust 1.77+
- Python 3.10+

### Setup & Run
```bash
# Install frontend dependencies
pnpm install

# Test sidecar communication
python3 python_sidecar/main.py

# Launch Tauri development mode
pnpm tauri dev
```
