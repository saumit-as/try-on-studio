import React from "react";
import {
  Sun,
  ChevronRight,
  Sliders,
  RotateCcw,
  Sparkles,
  Copy,
  Layers,
  MoveVertical,
  MoveHorizontal,
  Compass,
} from "lucide-react";
import { FingerName, FingerNailAdjustment, LightingState } from "../../types";
import {
  DEFAULT_FINGER_ADJUSTMENTS,
  FINGER_COLORS,
} from "../../constants/handDefaults";

interface LightingControlsProps {
  lightingState: LightingState;
  setLightingState: React.Dispatch<React.SetStateAction<LightingState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

const PRESETS = [
  { id: "studio", label: "Studio Softbox", desc: "Diffused key light + soft topcoat ribbon" },
  { id: "ring", label: "Ring Light", desc: "Uniform dome ambient with dual-edge sheen" },
  { id: "flash", label: "Direct Flash", desc: "Crisp on-axis specular with punchy highlight" },
  { id: "natural", label: "Natural Daylight", desc: "Soft ambient sunlight with skin bounce" },
];

const LENGTH_PRESETS = [
  { id: "short", label: "Short", scale: 0.85 },
  { id: "medium", label: "Medium", scale: 1.05 },
  { id: "long", label: "Long", scale: 1.35 },
  { id: "extra_long", label: "XL", scale: 1.65 },
];

const WIDTH_PRESETS = [
  { label: "Slim", scale: 0.9 },
  { label: "Standard", scale: 1.0 },
  { label: "Wide", scale: 1.15 },
];

const TAPER_PRESETS = [
  { label: "Almond", taper: 0.85 },
  { label: "Square", taper: 1.0 },
  { label: "Coffin", taper: 0.75 },
];

const ANGLE_PRESETS = [
  { label: "Front (0°)", angle: 0 },
  { label: "Left (45°)", angle: 45 },
  { label: "Overhead (90°)", angle: 90 },
  { label: "Right (135°)", angle: 135 },
];

export const LightingControls: React.FC<LightingControlsProps> = ({
  lightingState,
  setLightingState,
}) => {
  const {
    preset,
    glossIntensity,
    shadowDepth,
    lightAngle,
    cCurveIntensity,
    nailLengthScale,
    showAdvanced,
    selectedFinger = "All",
    fingerAdjustments = DEFAULT_FINGER_ADJUSTMENTS,
  } = lightingState;

  const updateState = (partial: Partial<LightingState>) => {
    setLightingState((prev) => ({ ...prev, ...partial }));
  };

  const currentAdj: FingerNailAdjustment =
    selectedFinger === "All"
      ? {
          lengthScale: nailLengthScale,
          widthScale: 1.0,
          axialShift: 0,
          angleOffset: 0,
          cCurve: cCurveIntensity,
          tipTaper: 1.0,
        }
      : fingerAdjustments[selectedFinger] ||
        DEFAULT_FINGER_ADJUSTMENTS[selectedFinger];

  const updateFingerAdj = (partial: Partial<FingerNailAdjustment>) => {
    if (selectedFinger === "All") {
      if (partial.lengthScale !== undefined) {
        updateState({ nailLengthScale: partial.lengthScale });
      }
      if (partial.cCurve !== undefined) {
        updateState({ cCurveIntensity: partial.cCurve });
      }
      const next: Record<FingerName, FingerNailAdjustment> = { ...fingerAdjustments };
      for (const f of FINGERS) {
        next[f] = { ...(next[f] || DEFAULT_FINGER_ADJUSTMENTS[f]), ...partial };
      }
      updateState({ fingerAdjustments: next });
    } else {
      const next: Record<FingerName, FingerNailAdjustment> = {
        ...fingerAdjustments,
        [selectedFinger]: {
          ...(fingerAdjustments[selectedFinger] ||
            DEFAULT_FINGER_ADJUSTMENTS[selectedFinger]),
          ...partial,
        },
      };
      updateState({ fingerAdjustments: next });
    }
  };

  const handleResetSelected = () => {
    if (selectedFinger === "All") {
      updateState({
        nailLengthScale: 1.05,
        cCurveIntensity: 65,
        fingerAdjustments: JSON.parse(JSON.stringify(DEFAULT_FINGER_ADJUSTMENTS)),
      });
    } else {
      const next = {
        ...fingerAdjustments,
        [selectedFinger]: { ...DEFAULT_FINGER_ADJUSTMENTS[selectedFinger] },
      };
      updateState({ fingerAdjustments: next });
    }
  };

  const handleApplyToAll = () => {
    if (selectedFinger === "All") return;
    const src = fingerAdjustments[selectedFinger] || DEFAULT_FINGER_ADJUSTMENTS[selectedFinger];
    const next: Record<FingerName, FingerNailAdjustment> = { ...fingerAdjustments };
    for (const f of FINGERS) {
      next[f] = { ...src };
    }
    updateState({
      nailLengthScale: src.lengthScale,
      cCurveIntensity: src.cCurve,
      fingerAdjustments: next,
    });
  };

  return (
    <div className="p-5 space-y-6">
      {/* Header & Reset */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-rose-400" />
            Nail Fitting & PBR Studio
          </h2>
          <p className="text-xs text-neutral-400 mt-0.5">
            Per-finger anatomical sizing and studio relighting
          </p>
        </div>
        <button
          onClick={handleResetSelected}
          className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 transition-colors cursor-pointer"
          title={`Reset ${selectedFinger === "All" ? "all nails" : selectedFinger + " nail"} to defaults`}
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Finger Selector Bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-neutral-300">
            Select Nail to Edit
          </label>
          <span className="text-[11px] text-neutral-400 font-mono">
            {selectedFinger === "All" ? "Global Mode" : `${selectedFinger} Nail`}
          </span>
        </div>
        <div className="grid grid-cols-6 gap-1 bg-neutral-900/80 p-1 rounded-xl border border-neutral-800">
          <button
            onClick={() => updateState({ selectedFinger: "All" })}
            className={`py-1.5 text-[11px] font-medium rounded-lg transition-all cursor-pointer text-center ${
              selectedFinger === "All"
                ? "bg-rose-500 text-white shadow-sm font-semibold"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
            }`}
          >
            All
          </button>
          {FINGERS.map((f) => {
            const isSel = selectedFinger === f;
            const color = FINGER_COLORS[f];
            return (
              <button
                key={f}
                onClick={() => updateState({ selectedFinger: f })}
                className={`py-1.5 text-[11px] font-medium rounded-lg transition-all cursor-pointer flex flex-col items-center justify-center gap-1 ${
                  isSel
                    ? "bg-neutral-800 text-neutral-100 border border-neutral-700 shadow-sm"
                    : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/40"
                }`}
                title={`Edit ${f} Nail`}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: color.hex }}
                />
                <span className="text-[10px] truncate max-w-full px-0.5">
                  {f.slice(0, 3)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Per-Finger Geometry & Fitting Controls */}
      <div className="p-3.5 rounded-2xl bg-neutral-900/50 border border-neutral-800 space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800/80 pb-2">
          <div className="flex items-center gap-2">
            {selectedFinger !== "All" && (
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: FINGER_COLORS[selectedFinger].hex }}
              />
            )}
            <span className="text-xs font-semibold text-neutral-200">
              {selectedFinger === "All"
                ? "All Nails Fitting"
                : `${selectedFinger} Finger Fitting`}
            </span>
          </div>
          {selectedFinger !== "All" && (
            <button
              onClick={handleApplyToAll}
              className="px-2 py-0.5 rounded text-[10px] bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors cursor-pointer flex items-center gap-1 border border-neutral-700"
              title="Copy this nail's sizing to all other fingers"
            >
              <Copy className="w-3 h-3" />
              Apply to All
            </button>
          )}
        </div>

        {/* 1. Tip Length Scale */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-neutral-300 flex items-center gap-1.5">
              <MoveVertical className="w-3.5 h-3.5 text-emerald-400" />
              Tip Length Scale
            </span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.lengthScale.toFixed(2)}x
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1">
            {LENGTH_PRESETS.map((lp) => {
              const active = Math.abs(currentAdj.lengthScale - lp.scale) < 0.04;
              return (
                <button
                  key={lp.id}
                  onClick={() => updateFingerAdj({ lengthScale: lp.scale })}
                  className={`py-1 text-[10px] rounded border transition-colors cursor-pointer text-center ${
                    active
                      ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 font-semibold"
                      : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  {lp.label} ({lp.scale}x)
                </button>
              );
            })}
          </div>
          <input
            type="range"
            min="0.5"
            max="2.5"
            step="0.02"
            value={currentAdj.lengthScale}
            onChange={(e) =>
              updateFingerAdj({ lengthScale: Number(e.target.value) })
            }
            className="w-full accent-emerald-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>

        {/* 2. Width / Bed Coverage Scale */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-neutral-300 flex items-center gap-1.5">
              <MoveHorizontal className="w-3.5 h-3.5 text-cyan-400" />
              Nail Width / Bed Fit
            </span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.widthScale.toFixed(2)}x
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {WIDTH_PRESETS.map((wp) => {
              const active = Math.abs(currentAdj.widthScale - wp.scale) < 0.04;
              return (
                <button
                  key={wp.label}
                  onClick={() => updateFingerAdj({ widthScale: wp.scale })}
                  className={`py-1 text-[10px] rounded border transition-colors cursor-pointer text-center ${
                    active
                      ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-300 font-semibold"
                      : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  {wp.label} ({wp.scale}x)
                </button>
              );
            })}
          </div>
          <input
            type="range"
            min="0.7"
            max="1.4"
            step="0.02"
            value={currentAdj.widthScale}
            onChange={(e) =>
              updateFingerAdj({ widthScale: Number(e.target.value) })
            }
            className="w-full accent-cyan-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>

        {/* 3. Cuticle Pocket Tuck / Axial Shift */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-rose-400" />
              Cuticle Pocket Tuck
            </span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.axialShift > 0
                ? `+${currentAdj.axialShift}px (Out)`
                : currentAdj.axialShift < 0
                ? `${currentAdj.axialShift}px (Tuck)`
                : "Centered"}
            </span>
          </div>
          <input
            type="range"
            min="-30"
            max="30"
            step="1"
            value={currentAdj.axialShift}
            onChange={(e) =>
              updateFingerAdj({ axialShift: Number(e.target.value) })
            }
            className="w-full accent-rose-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-neutral-500 px-0.5 mt-0.5">
            <span>Deeper Tuck</span>
            <span>0</span>
            <span>Tip Extension</span>
          </div>
        </div>

        {/* 4. Angle / Rotation Offset */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-300 flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-amber-400" />
              Nail Tilt Angle
            </span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.angleOffset > 0 ? `+` : ""}
              {currentAdj.angleOffset}°
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="15"
            step="0.5"
            value={currentAdj.angleOffset}
            onChange={(e) =>
              updateFingerAdj({ angleOffset: Number(e.target.value) })
            }
            className="w-full accent-amber-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>

        {/* 5. Transverse C-Curve Depth */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-300">C-Curve Curvature</span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.cCurve}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={currentAdj.cCurve}
            onChange={(e) => updateFingerAdj({ cCurve: Number(e.target.value) })}
            className="w-full accent-indigo-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>

        {/* 6. Tip Taper / Silhouette */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-neutral-300">Tip Taper</span>
            <span className="font-mono text-neutral-400 text-xs">
              {currentAdj.tipTaper.toFixed(2)}x
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {TAPER_PRESETS.map((tp) => {
              const active = Math.abs(currentAdj.tipTaper - tp.taper) < 0.04;
              return (
                <button
                  key={tp.label}
                  onClick={() => updateFingerAdj({ tipTaper: tp.taper })}
                  className={`py-1 text-[10px] rounded border transition-colors cursor-pointer text-center ${
                    active
                      ? "bg-indigo-500/20 border-indigo-500/50 text-indigo-300 font-semibold"
                      : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  {tp.label} ({tp.taper}x)
                </button>
              );
            })}
          </div>
          <input
            type="range"
            min="0.6"
            max="1.3"
            step="0.02"
            value={currentAdj.tipTaper}
            onChange={(e) =>
              updateFingerAdj({ tipTaper: Number(e.target.value) })
            }
            className="w-full accent-indigo-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* PBR Lighting Environment Section */}
      <div className="space-y-4 pt-2 border-t border-neutral-800">
        <div>
          <h3 className="text-xs font-semibold text-neutral-300 flex items-center gap-1.5">
            <Sun className="w-3.5 h-3.5 text-amber-400" />
            Studio Lighting Environment
          </h3>
          <p className="text-[11px] text-neutral-400 mt-0.5">
            Global photorealistic lighting and clear-coat gloss
          </p>
        </div>

        {/* Environment Presets */}
        <div className="grid grid-cols-2 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => updateState({ preset: p.id })}
              className={`px-3 py-2 text-xs rounded-lg border transition-all text-left cursor-pointer flex flex-col justify-between ${
                preset === p.id
                  ? "bg-rose-500/10 border-rose-500/40 text-rose-300 font-medium"
                  : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700"
              }`}
            >
              <span>{p.label}</span>
              <span className="text-[10px] text-neutral-500 line-clamp-1 mt-0.5">
                {p.desc}
              </span>
            </button>
          ))}
        </div>

        {/* Sliders */}
        <div className="space-y-3.5 pt-1">
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
              onChange={(e) =>
                updateState({ glossIntensity: Number(e.target.value) })
              }
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
              onChange={(e) =>
                updateState({ shadowDepth: Number(e.target.value) })
              }
              className="w-full accent-rose-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Advanced PBR Relighting Accordion */}
      <div className="pt-2 border-t border-neutral-800">
        <button
          onClick={() => updateState({ showAdvanced: !showAdvanced })}
          className="flex items-center justify-between w-full text-xs font-medium text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5" />
            Advanced Light Parameters
          </span>
          <ChevronRight
            className={`w-3.5 h-3.5 transition-transform ${
              showAdvanced ? "rotate-90" : ""
            }`}
          />
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-3.5 pl-1">
            <div>
              <div className="flex justify-between text-[11px] mb-1.5 text-neutral-400">
                <span>Light Elevation Angle</span>
                <span className="font-mono text-amber-400">{lightAngle}°</span>
              </div>
              <div className="grid grid-cols-4 gap-1 mb-2">
                {ANGLE_PRESETS.map((ap) => (
                  <button
                    key={ap.angle}
                    onClick={() => updateState({ lightAngle: ap.angle })}
                    className={`py-1 text-[10px] rounded border transition-colors cursor-pointer text-center ${
                      lightAngle === ap.angle
                        ? "bg-amber-500/20 border-amber-500/40 text-amber-300 font-medium"
                        : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
                    }`}
                  >
                    {ap.label}
                  </button>
                ))}
              </div>
              <input
                type="range"
                min="0"
                max="360"
                value={lightAngle}
                onChange={(e) =>
                  updateState({ lightAngle: Number(e.target.value) })
                }
                className="w-full accent-amber-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
              />
            </div>

            <div className="p-2.5 rounded-lg bg-neutral-900 border border-neutral-800 text-[11px] text-neutral-400 space-y-1.5">
              <div className="flex justify-between items-center">
                <span>Tray Glare Suppression</span>
                <span className="text-emerald-400 font-medium flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  Active
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span>Cuticle Eponychium Alpha</span>
                <span className="text-neutral-300">Dynamic Pocket Tuck</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Gel Topcoat Ribbon</span>
                <span className="text-neutral-300">G2 Curvature Follower</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
