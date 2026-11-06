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

### Phase 1: Core Tauri v2 Shell & Python Sidecar Setup
- [x] Initialize Tauri v2 project structure (`src-tauri` Rust core)
- [x] Configure React 19 + TypeScript + Vite + Tailwind CSS v4 frontend
- [x] Implement studio shell UI with 4-tab workflow (Tray, Hand Rigging, Compositor, Export)
- [x] Build Python sidecar daemon (`python_sidecar/main.py`) with line-delimited JSON-RPC over stdio
- [x] Implement Rust sidecar IPC client (`src-tauri/src/sidecar.rs`) with dev script fallback and production binary lookup
- [x] Implement TypeScript IPC bridge (`src/services/sidecar.ts`) and ping health check
- [x] Create PyInstaller cross-platform packaging script (`python_sidecar/build_sidecar.py`)
- [ ] Implement OpenCV contour detection & adaptive thresholding (`python_sidecar/segmentation.py`)
- [ ] Implement GrabCut foreground refinement for low-contrast/clear press-on nails
- [ ] Implement contour moment-based axis determination and cuticle vs. tip orientation
- [ ] Implement SciPy RBF Thin-Plate Spline (TPS) coordinate warping (`python_sidecar/warping.py`)
- [ ] Implement transverse C-curve cylindrical barrel distortion

### Phase 2: Hand Template Authoring Wizard
- [ ] Interactive 5-point landmark placement canvas (Apex, Left/Right Corners, Sidewalls, Directional Vector)
- [ ] Procedural cuticle occlusion alpha mask generator (eponychium skin fold overlay)
- [ ] Procedural ambient occlusion (AO) contact shadow generator
- [ ] Procedural specular gloss reflection map generator aligned to finger axis
- [ ] Interactive touch-up brush/eraser tools on canvas for mask refinement
- [ ] `.nailtemplate` bundle serializer (JSON rig metadata + 16-bit/8-bit PNG passes)
- [ ] AppData template library manager (macOS `Application Support` / Windows `%APPDATA%`)

### Phase 3: Tray Extraction & Interactive Mapping
- [ ] Tray layout template manager (2×5, 2×12 presets and custom polygon drawing)
- [ ] Interactive contour vertex editor with drag handles
- [ ] Manual brush/eraser boundary correction for tricky packaging glare
- [ ] Auto-orientation with 180° flip and fine-rotation wheel controls
- [ ] Visual drag-and-drop finger chip assignment (Thumb, Index, Middle, Ring, Pinky)

### Phase 4: PBR Compositing & Dual-Tier Lighting Engine
- [ ] 6-pass compositing pipeline in Python sidecar (Base Hand, Cast Shadow, Warped Nail, AO, Gloss, Cuticle Fold)
- [ ] Real-time caching & thumbnail generation for sub-second UI responsiveness
- [ ] Dual-tier lighting controls (Basic Studio Presets + Advanced Sliders)

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
