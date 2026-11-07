# Graph Report - try-on-studio  (2026-11-08)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 355 nodes · 809 edges · 15 communities (10 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fce82375`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 13

## God Nodes (most connected - your core abstractions)
1. `segment_packaging_tray()` - 20 edges
2. `extract_nail_rgba()` - 17 edges
3. `run_set()` - 17 edges
4. `run_tray_set()` - 16 edges
5. `compilerOptions` - 16 edges
6. `SidecarClient` - 15 edges
7. `apply_cylindrical_c_curve()` - 13 edges
8. `build_cuticle_pocket_mask()` - 13 edges
9. `warp_nail_tps()` - 13 edges
10. `composite_nail_pbr()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `get_cleaner()` --indirect_call--> `clean_tray_glare_preserving_texture()`  [INFERRED]
  python_sidecar/test_all_nails.py → python_sidecar/compositor.py
- `extract_supports_homography()` --indirect_call--> `extract_nail_rgba()`  [INFERRED]
  python_sidecar/test_all_nails.py → python_sidecar/segmentation.py
- `HandControlsProps` --references--> `HandState`  [EXTRACTED]
  src/components/inspector/HandControls.tsx → src/types/index.ts
- `LightingControlsProps` --references--> `LightingState`  [EXTRACTED]
  src/components/inspector/LightingControls.tsx → src/types/index.ts
- `TrayControlsProps` --references--> `TrayState`  [EXTRACTED]
  src/components/inspector/TrayControls.tsx → src/types/index.ts

## Import Cycles
- None detected.

## Communities (15 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (24): clean_tray_glare_preserving_texture(), composite_nail_pbr(), compute_3d_nail_geometry(), recover_clean_nail_albedo(), handle_warp_nail(), main(), evaluate_nail_realism(), main() (+16 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (49): lucide-react@^1.69.0, lucide-react, react, App(), testConnection(), CanvasViewport(), CanvasViewportProps, constructPathD() (+41 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (20): _encode_jpg_base64(), _encode_png_base64(), handle_extract_nail(), handle_generate_hand_maps(), handle_get_default_hand_template(), handle_load_hand_template(), handle_refit_slot_contour(), handle_save_hand_template() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (38): dependencies, clsx@^2.1.1, react@^19.1.0, react-dom@^19.1.0, tailwind-merge@^3.7.0, @tauri-apps/api@^2, @tauri-apps/plugin-opener@^2, devDependencies (+30 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (5): call_sidecar(), JsonRpcRequest, JsonRpcResponse, SidecarClient, SidecarState

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (12): composite_on_hand(), discover_sets(), draw_label(), extract_supports_homography(), get_cleaner(), main(), render_extracted_strip(), render_segmentation_sheet() (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (18): app, security, windows, build, beforeBuildCommand, beforeDevCommand, devUrl, frontendDist (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (18): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+10 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (7): compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (5): description, identifier, permissions, $schema, windows

## Knowledge Gaps
- **1 isolated node(s):** `nail-mockup-generator`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 148 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `react` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `nail-mockup-generator` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07942097026604068 - nodes in this community are weakly interconnected._
- **Why does `lucide-react` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08361581920903954 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.10083256244218317 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.06504065040650407 - nodes in this community are weakly interconnected._