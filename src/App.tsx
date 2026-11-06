import { useState, useEffect } from "react";
import {
  Sparkles,
  Layers,
  Hand,
  Sliders,
  Download,
  RotateCw,
  RefreshCw,
  Sun,
  ChevronRight,
} from "lucide-react";
import { pingSidecar } from "./services/sidecar";
import "./App.css";

type ActiveTab = "tray" | "hand" | "composite" | "export";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("composite");
  const [sidecarStatus, setSidecarStatus] = useState<"checking" | "connected" | "disconnected">("checking");
  const [sidecarVersion, setSidecarVersion] = useState<string>("");

  // Composite & Lighting State
  const [lightingPreset, setLightingPreset] = useState("studio");
  const [glossIntensity, setGlossIntensity] = useState(75);
  const [shadowDepth, setShadowDepth] = useState(60);
  const [lightAngle, setLightAngle] = useState(45);
  const [cCurveIntensity, setCCurveIntensity] = useState(65);
  const [nailLengthScale, setNailLengthScale] = useState(1.15);
  const [showAdvancedLighting, setShowAdvancedLighting] = useState(false);

  // Tray State
  const [selectedSlot, setSelectedSlot] = useState<number>(1);
  const [slotFinger, setSlotFinger] = useState<Record<number, string>>({
    1: "Thumb",
    2: "Index",
    3: "Middle",
    4: "Ring",
    5: "Pinky",
  });
  const [nailRotation, setNailRotation] = useState(0);

  // Check Python sidecar connection
  useEffect(() => {
    async function testConnection() {
      try {
        const res = await pingSidecar();
        if (res && res.status === "pong") {
          setSidecarStatus("connected");
          setSidecarVersion(res.version || "0.1.0");
        } else {
          setSidecarStatus("disconnected");
        }
      } catch (err) {
        // In browser dev mode without Tauri runner, mark graceful fallback
        setSidecarStatus("disconnected");
      }
    }
    testConnection();
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-neutral-950 text-neutral-100 font-sans select-none overflow-hidden">
      {/* Top Application Bar */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-neutral-800 bg-neutral-900/70 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-rose-500 to-amber-400 flex items-center justify-center shadow-lg shadow-rose-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-sm tracking-wide text-neutral-100 flex items-center gap-2">
              Press-On Nail Studio
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                Tauri v2
              </span>
            </h1>
            <p className="text-[11px] text-neutral-400">PBR Warping & Mockup Engine</p>
          </div>
        </div>

        {/* Workflow Navigation Tabs */}
        <nav className="flex items-center bg-neutral-900 p-1 rounded-xl border border-neutral-800">
          <button
            onClick={() => setActiveTab("tray")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "tray"
                ? "bg-neutral-800 text-neutral-100 shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            1. Tray Extractor
          </button>
          <button
            onClick={() => setActiveTab("hand")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "hand"
                ? "bg-neutral-800 text-neutral-100 shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Hand className="w-3.5 h-3.5 text-rose-400" />
            2. Hand Rigging
          </button>
          <button
            onClick={() => setActiveTab("composite")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "composite"
                ? "bg-neutral-800 text-neutral-100 shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Sliders className="w-3.5 h-3.5 text-emerald-400" />
            3. Compositor & Lighting
          </button>
          <button
            onClick={() => setActiveTab("export")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "export"
                ? "bg-neutral-800 text-neutral-100 shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Download className="w-3.5 h-3.5 text-sky-400" />
            4. Export
          </button>
        </nav>

        {/* Sidecar Connection Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-neutral-900 border border-neutral-800 text-[11px]">
            {sidecarStatus === "connected" ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-emerald-400 font-mono">Python Sidecar v{sidecarVersion}</span>
              </>
            ) : sidecarStatus === "checking" ? (
              <>
                <RefreshCw className="w-3 h-3 text-amber-400 animate-spin" />
                <span className="text-neutral-400">Connecting...</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-neutral-500" />
                <span className="text-neutral-400 font-mono">Sidecar Idle (Local Mode)</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Interactive Canvas Area */}
        <div className="flex-1 flex flex-col bg-neutral-950 relative overflow-hidden border-r border-neutral-800/80">
          <div className="flex-1 flex items-center justify-center p-6 relative">
            {/* Viewport Canvas Frame */}
            <div className="relative w-full max-w-2xl aspect-[4/5] bg-neutral-900/60 rounded-2xl border border-neutral-800 flex items-center justify-center shadow-2xl overflow-hidden group">
              {activeTab === "tray" && (
                <div className="flex flex-col items-center justify-center text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-4">
                    <Layers className="w-8 h-8 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-neutral-200 mb-1">Tray Segmentation Canvas</h3>
                  <p className="text-xs text-neutral-400 max-w-sm mb-4">
                    Drop a packaging tray photo or select a predefined 2x5 grid template to snap contours.
                  </p>
                  <button className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors">
                    Upload Tray Photo (.PNG / .JPG)
                  </button>
                </div>
              )}

              {activeTab === "hand" && (
                <div className="flex flex-col items-center justify-center text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-4">
                    <Hand className="w-8 h-8 text-rose-400" />
                  </div>
                  <h3 className="font-semibold text-neutral-200 mb-1">Bare Hand Template Rigging</h3>
                  <p className="text-xs text-neutral-400 max-w-sm mb-4">
                    Click 5 anchor points per finger to auto-generate cuticle occlusion, ambient shadows, and gloss maps.
                  </p>
                  <button className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors">
                    Load Bare Hand Photo
                  </button>
                </div>
              )}

              {activeTab === "composite" && (
                <div className="relative w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-neutral-900/40 to-neutral-950">
                  {/* Mock Hand Render Canvas Simulation */}
                  <div className="relative flex flex-col items-center">
                    <div className="w-48 h-64 rounded-t-full bg-neutral-800/40 border border-dashed border-neutral-700 flex flex-col items-center justify-center p-4 text-center">
                      <Sparkles className="w-8 h-8 text-emerald-400 mb-2" />
                      <span className="text-xs font-medium text-neutral-300">TPS Composite Canvas</span>
                      <span className="text-[10px] text-neutral-500 mt-1">Aspect-ratio preserved + C-curve active</span>
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-[11px] text-neutral-400 bg-neutral-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-neutral-800">
                    <span>Active Template: Default Studio Hand</span>
                    <span>5 / 5 Nails Rigged</span>
                  </div>
                </div>
              )}

              {activeTab === "export" && (
                <div className="flex flex-col items-center justify-center text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mb-4">
                    <Download className="w-8 h-8 text-sky-400" />
                  </div>
                  <h3 className="font-semibold text-neutral-200 mb-1">Production Export Studio</h3>
                  <p className="text-xs text-neutral-400 max-w-sm mb-4">
                    Ready to export 300 DPI high-res composites, transparent cutouts, or layered PSDs.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Parameter & Tooling Inspector */}
        <aside className="w-96 bg-neutral-900/40 backdrop-blur-md flex flex-col overflow-y-auto border-l border-neutral-800">
          {activeTab === "composite" && (
            <div className="p-5 space-y-6">
              <div>
                <h2 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <Sun className="w-4 h-4 text-amber-400" />
                  PBR Lighting Environment
                </h2>
                <p className="text-xs text-neutral-400 mt-0.5">Dual-tier studio lighting and gloss controls</p>
              </div>

              {/* Basic Lighting Presets */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-300">Environment Preset</label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: "studio", label: "Studio Softbox" },
                    { id: "ring", label: "Ring Light" },
                    { id: "flash", label: "Direct Flash" },
                    { id: "natural", label: "Natural Daylight" },
                  ].map((preset) => (
                    <button
                      key={preset.id}
                      onClick={() => setLightingPreset(preset.id)}
                      className={`px-3 py-2 text-xs rounded-lg border transition-all text-left ${
                        lightingPreset === preset.id
                          ? "bg-rose-500/10 border-rose-500/40 text-rose-300 font-medium"
                          : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Master Sliders */}
              <div className="space-y-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-neutral-300">Specular Gloss Intensity</span>
                    <span className="font-mono text-neutral-400">{glossIntensity}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={glossIntensity}
                    onChange={(e) => setGlossIntensity(Number(e.target.value))}
                    className="w-full accent-rose-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-neutral-300">Contact AO & Drop Shadow</span>
                    <span className="font-mono text-neutral-400">{shadowDepth}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={shadowDepth}
                    onChange={(e) => setShadowDepth(Number(e.target.value))}
                    className="w-full accent-rose-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                  />
                </div>
              </div>

              {/* 3D Curvature & Length Mechanics */}
              <div className="space-y-4 pt-2 border-t border-neutral-800">
                <h3 className="text-xs font-semibold text-neutral-300">TPS Curvature & Extension</h3>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-neutral-300">Transverse C-Curve Depth</span>
                    <span className="font-mono text-neutral-400">{cCurveIntensity}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={cCurveIntensity}
                    onChange={(e) => setCCurveIntensity(Number(e.target.value))}
                    className="w-full accent-emerald-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-neutral-300">Tip Length Scale</span>
                    <span className="font-mono text-neutral-400">{nailLengthScale.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.8"
                    max="1.8"
                    step="0.05"
                    value={nailLengthScale}
                    onChange={(e) => setNailLengthScale(Number(e.target.value))}
                    className="w-full accent-emerald-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                  />
                </div>
              </div>

              {/* Collapsible Advanced Lighting Menu */}
              <div className="pt-2 border-t border-neutral-800">
                <button
                  onClick={() => setShowAdvancedLighting(!showAdvancedLighting)}
                  className="flex items-center justify-between w-full text-xs font-medium text-neutral-400 hover:text-neutral-200 transition-colors"
                >
                  <span>Advanced PBR Parameters</span>
                  <ChevronRight
                    className={`w-3.5 h-3.5 transition-transform ${showAdvancedLighting ? "rotate-90" : ""}`}
                  />
                </button>

                {showAdvancedLighting && (
                  <div className="mt-3 space-y-3 pl-1">
                    <div>
                      <div className="flex justify-between text-[11px] mb-1 text-neutral-400">
                        <span>Light Elevation Angle</span>
                        <span className="font-mono">{lightAngle}°</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="360"
                        value={lightAngle}
                        onChange={(e) => setLightAngle(Number(e.target.value))}
                        className="w-full accent-amber-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                      />
                    </div>
                    <div className="p-2.5 rounded bg-neutral-900 border border-neutral-800 text-[11px] text-neutral-400 space-y-1">
                      <div className="flex justify-between">
                        <span>Tray Glare Suppression</span>
                        <span className="text-emerald-400 font-medium">Active</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Cuticle Eponychium Alpha</span>
                        <span className="text-neutral-300">Auto (Procedural)</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "tray" && (
            <div className="p-5 space-y-6">
              <div>
                <h2 className="text-sm font-semibold text-neutral-200">Slot-to-Finger Mapping</h2>
                <p className="text-xs text-neutral-400 mt-0.5">Assign detected nails to hand positions</p>
              </div>

              {/* Slot Selector */}
              <div className="space-y-2">
                <span className="text-xs font-medium text-neutral-300">Active Tray Slot</span>
                <div className="grid grid-cols-5 gap-1.5">
                  {[1, 2, 3, 4, 5].map((slot) => (
                    <button
                      key={slot}
                      onClick={() => setSelectedSlot(slot)}
                      className={`py-2 text-center text-xs font-medium rounded-lg border transition-colors ${
                        selectedSlot === slot
                          ? "bg-amber-500/20 border-amber-500/50 text-amber-300"
                          : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:text-neutral-200"
                      }`}
                    >
                      Slot {slot}
                    </button>
                  ))}
                </div>
              </div>

              {/* Finger Assignment Chips */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-neutral-300">Assign Slot {selectedSlot} to:</span>
                  <span className="text-amber-400 font-mono">{slotFinger[selectedSlot] || "Unassigned"}</span>
                </div>
                <div className="grid grid-cols-5 gap-1.5">
                  {(["Thumb", "Index", "Middle", "Ring", "Pinky"] as const).map((finger) => (
                    <button
                      key={finger}
                      onClick={() => setSlotFinger((prev) => ({ ...prev, [selectedSlot]: finger }))}
                      className={`py-2 text-center text-xs font-medium rounded-lg border transition-colors ${
                        slotFinger[selectedSlot] === finger
                          ? "bg-amber-500/20 border-amber-500/60 text-amber-300"
                          : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-amber-500/30"
                      }`}
                    >
                      {finger[0]}
                    </button>
                  ))}
                </div>
              </div>

              {/* Rotation & Orientation */}
              <div className="space-y-3 pt-2 border-t border-neutral-800">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-neutral-300">Orientation Correction</span>
                  <button
                    onClick={() => setNailRotation((prev) => (prev + 180) % 360)}
                    className="px-2 py-1 bg-neutral-800 hover:bg-neutral-700 text-[11px] rounded flex items-center gap-1 border border-neutral-700"
                  >
                    <RotateCw className="w-3 h-3 text-amber-400" />
                    Flip 180°
                  </button>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1 text-neutral-400">
                    <span>Fine Rotation</span>
                    <span className="font-mono">{nailRotation}°</span>
                  </div>
                  <input
                    type="range"
                    min="-45"
                    max="45"
                    value={nailRotation}
                    onChange={(e) => setNailRotation(Number(e.target.value))}
                    className="w-full accent-amber-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === "export" && (
            <div className="p-5 space-y-6">
              <div>
                <h2 className="text-sm font-semibold text-neutral-200">Export Settings</h2>
                <p className="text-xs text-neutral-400 mt-0.5">Production-ready outputs</p>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-medium text-neutral-300">Aspect Ratio Preset</span>
                <div className="grid grid-cols-3 gap-2">
                  {["1:1 Square", "4:5 Portrait", "16:9 Banner"].map((ratio) => (
                    <button
                      key={ratio}
                      className="px-2 py-2 text-xs rounded bg-neutral-900 border border-neutral-800 hover:border-sky-500/50 text-neutral-300 text-center"
                    >
                      {ratio}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-neutral-800">
                <span className="text-xs font-medium text-neutral-300">Output Packages</span>
                <button className="w-full py-2.5 px-3 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 text-neutral-200 text-left flex items-center justify-between">
                  <span>Flattened 300 DPI PNG</span>
                  <Download className="w-3.5 h-3.5 text-neutral-400" />
                </button>
                <button className="w-full py-2.5 px-3 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 text-neutral-200 text-left flex items-center justify-between">
                  <span>Transparent Cutout PNG</span>
                  <Download className="w-3.5 h-3.5 text-neutral-400" />
                </button>
                <button className="w-full py-2.5 px-3 bg-rose-500/10 hover:bg-rose-500/20 text-xs font-medium rounded-lg border border-rose-500/30 text-rose-300 text-left flex items-center justify-between">
                  <span>Multi-Layer Photoshop PSD</span>
                  <Download className="w-3.5 h-3.5 text-rose-400" />
                </button>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
