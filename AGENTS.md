# AGENTS.md — Try-On Studio Engineering Guide

> **Agent Context**: This document guides AI coding assistants and developers collaborating on **Try-On Studio**. It defines the system architecture, current development state, multi-phase execution protocol, and team handoff rules.

---

## 1. Project Overview & Architecture

**Try-On Studio** is a local-first desktop application that automates extracting press-on nail textures from product tray photos and photorealistically composites them onto a 5-finger bare-hand template.

### Technical Stack
- **Desktop Harness**: [Tauri v2](https://tauri.app/) (Rust 2024 edition)
- **Frontend**: React 19 + TypeScript + Vite 8 + Tailwind CSS v4 + Lucide Icons
- **Sidecar Daemon**: Python 3.10+ (`opencv-python-headless`, `numpy`, `scipy`, `pillow`, `pytoshop`)
- **IPC Protocol**: Line-delimited JSON-RPC over `stdio` (zero local open ports, firewall-free)
- **Target Platforms**: macOS (Apple Silicon `aarch64` & Intel `x86_64`) and Windows (10/11 `x86_64`)

### Deep References
- **Full Architecture Specification**: [`docs/arch-spec.md`](docs/arch-spec.md)
- **Testable Phases & Feedback Gates**: [`docs/phases-roadmap.md`](docs/phases-roadmap.md)

---

## 2. Directory Layout & Key Modules

```
try-on-studio/
├── src-tauri/                 # Tauri v2 Rust backend
│   ├── Cargo.toml             # Rust dependencies (Tauri 2, serde, serde_json)
│   ├── tauri.conf.json        # Window & externalBin sidecar bundle declarations
│   ├── binaries/              # Packaged sidecar binaries (prefixed by target triple)
│   └── src/
│       ├── lib.rs             # Tauri command handlers (`call_sidecar`)
│       ├── main.rs            # Desktop entrypoint
│       └── sidecar.rs         # Cross-platform sidecar lifecycle & venv/binary discovery
├── python_sidecar/            # Python image processing engine
│   ├── main.py                # Persistent stdio JSON-RPC server (`ping`, `echo`)
│   ├── segmentation.py        # [Phase 1] PCA deskewing, contour detection & shape classifier
│   ├── warping.py             # [Phase 1] 3-tier SciPy TPS warping & C-curve cylindrical barrel distortion
│   ├── compositor.py          # [Phase 4] 6-pass PBR lighting & blending engine
│   ├── export.py              # [Phase 5] 300 DPI PNG & layered PSD output
│   ├── build_sidecar.py       # PyInstaller standalone binary compilation script
│   └── requirements.txt       # Python dependencies
├── src/                       # React 19 Frontend
│   ├── types/index.ts         # Central TypeScript interfaces (ActiveTab, LightingState, TrayState)
│   ├── services/sidecar.ts    # JSON-RPC IPC client bridge (`pingSidecar`, `callSidecar`)
│   ├── components/
│   │   ├── layout/Header.tsx  # Workflow tabs & sidecar daemon connection pill
│   │   ├── canvas/            # CanvasViewport.tsx (Tray, Rigging, Composite, Export stages)
│   │   └── inspector/         # Tooling sidebars (TrayControls, LightingControls, HandControls, ExportControls)
│   └── App.tsx                # Lightweight state coordinator
└── docs/                      # Architectural specs & phase roadmaps
```

---

## 3. Current Implementation Status

| Component | Status | Verification |
| :--- | :--- | :--- |
| **Tauri v2 Core** | ✅ Complete | Runs with `pnpm tauri dev` |
| **Cross-Platform Sidecar Discovery** | ✅ Complete | Multi-directory search (`resource_dir`, `current_exe().parent()`) |
| **Frontend Deconstruction** | ✅ Complete | Modular architecture in `src/components/`, builds cleanly with `pnpm run build` |
| **Stdio JSON-RPC IPC** | ✅ Complete | Tested via `pingSidecar()` $\rightarrow$ `pong` |
| **Phase 1: Python CV & Warping Core** | ✅ Complete | G2 super-ellipse, PCA deskew, 3-tier TPS, C-curve, Cuticle tuck |
| **Realism Scoring & Quality Benchmark** | ✅ Complete | `score_realism_agent.py` & `test_all_nails.py` (83.2+ score) |
| **Phase 4 Backend: 6-Pass PBR Compositor** | ✅ Complete | Glare healing, studio diffuse relighting, gel topcoat ribbon, contact drop shadows |
| **Phase 2: Tray Extractor UI & Pen Tool** | ✅ Complete | Photoshop Bezier pen tool, pan/zoom, inspection modal, exact-mask design preservation |
| **Phase 3: Hand Rigging Canvas & Bundled Master Template** | ✅ Complete | 5-point landmark rigging, pan/zoom canvas, procedural cuticle/AO/specular maps, `.nailtemplate` bundle |
| **Phase 4: Real-Time PBR Compositor & Live Viewport** | ✅ Complete | Real-time viewport, dual-tier lighting controls, 98/100 visual audit score |
| **Phase 5: Production Export & Layered PSD** | ✅ Complete | 300 DPI PNG (pHYs metadata) & 5-layer PSD (pytoshop RLE) |

---

## 4. Multi-Agent & Multi-User Collaboration Rules

When picking up a task or continuing development:

### 1. Follow the 5-Phase Sequential Gate Protocol
- Work on **one phase at a time**.
- Every phase concludes with a standalone, checkable test deliverable (see [`docs/phases-roadmap.md`](docs/phases-roadmap.md)).
- **Halt and present the test results to the user for feedback before proceeding to the next phase.**

### 2. Code Integrity & Anti-Elision
- **Zero code placeholders**: Never write `// ... existing code ...` or omit imports/declarations.
- Provide complete files or total block replacements to avoid accidental deletion of functionality.

### 3. Build & Type Safety Validation
- **Rust changes**: Always verify with `cargo check --manifest-path src-tauri/Cargo.toml`.
- **Frontend changes**: Always verify with `pnpm run build` (`tsc && vite build`).
- **Python sidecar changes**: Always verify with a standalone test script (e.g. `python_sidecar/test_pipeline.py`).

### 4. Cross-Platform Parity
- **Windows**: Use `#[cfg(target_os = "windows")]` for Win32-specific APIs (e.g. `CREATE_NO_WINDOW`, Job Objects).
- **macOS**: Target triples are `aarch64-apple-darwin` (Apple Silicon) and `x86_64-apple-darwin` (Intel). Binaries are resolved via `Contents/MacOS/` or `Contents/Resources/`.
- Paths must be normalized with `PathBuf` / `pathlib.Path` across JSON-RPC boundaries.

### 5. Knowledge Graph & Architecture Queries (Graphify Skill)
- **Consult Graphify First**: Whenever investigating codebase architecture, cross-module relationships, call flows, symbol definitions, or data flows, always check the local knowledge graph in `graphify-out/` before brute-force file grepping:
  - **CLI Queries**: Use `graphify query "<question>"`, `graphify path "<source>" "<target>"`, or `graphify explain "<concept>"`.
  - **MCP Navigation**: If the `graphify` MCP server is active, use `query_graph`, `shortest_path`, or `get_node`.
  - **Visual & Structural Reference**: Consult [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) or open [`graphify-out/graph.html`](graphify-out/graph.html).
- **Keep Graph in Sync**: After modifying or adding files during a session, run:
  ```powershell
  graphify update .
  ```
  to keep `graphify-out/graph.json` synchronized without incurring LLM token costs.
