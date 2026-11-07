import React from "react";
import { Sun, ChevronRight } from "lucide-react";
import { LightingState } from "../../types";

interface LightingControlsProps {
  lightingState: LightingState;
  setLightingState: React.Dispatch<React.SetStateAction<LightingState>>;
}

const PRESETS = [
  { id: "studio", label: "Studio Softbox" },
  { id: "ring", label: "Ring Light" },
  { id: "flash", label: "Direct Flash" },
  { id: "natural", label: "Natural Daylight" },
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
  } = lightingState;

  const updateState = (partial: Partial<LightingState>) => {
    setLightingState((prev) => ({ ...prev, ...partial }));
  };

  return (
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
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => updateState({ preset: p.id })}
              className={`px-3 py-2 text-xs rounded-lg border transition-all text-left cursor-pointer ${
                preset === p.id
                  ? "bg-rose-500/10 border-rose-500/40 text-rose-300 font-medium"
                  : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {p.label}
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
            onChange={(e) => updateState({ glossIntensity: Number(e.target.value) })}
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
            onChange={(e) => updateState({ shadowDepth: Number(e.target.value) })}
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
            onChange={(e) => updateState({ cCurveIntensity: Number(e.target.value) })}
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
            onChange={(e) => updateState({ nailLengthScale: Number(e.target.value) })}
            className="w-full accent-emerald-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* Collapsible Advanced Lighting Menu */}
      <div className="pt-2 border-t border-neutral-800">
        <button
          onClick={() => updateState({ showAdvanced: !showAdvanced })}
          className="flex items-center justify-between w-full text-xs font-medium text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer"
        >
          <span>Advanced PBR Parameters</span>
          <ChevronRight
            className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
          />
        </button>

        {showAdvanced && (
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
                onChange={(e) => updateState({ lightAngle: Number(e.target.value) })}
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
  );
};
