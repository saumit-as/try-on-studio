import React from "react";
import { Sparkles, Layers, Hand, Download } from "lucide-react";
import { ActiveTab } from "../../types";

interface CanvasViewportProps {
  activeTab: ActiveTab;
}

export const CanvasViewport: React.FC<CanvasViewportProps> = ({ activeTab }) => {
  return (
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
              <button className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors cursor-pointer">
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
              <button className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors cursor-pointer">
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
  );
};
