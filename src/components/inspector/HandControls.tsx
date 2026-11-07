import React from "react";
import { Hand, CheckCircle2 } from "lucide-react";

export const HandControls: React.FC = () => {
  return (
    <div className="p-5 space-y-6">
      <div>
        <h2 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
          <Hand className="w-4 h-4 text-rose-400" />
          Hand Template Setup
        </h2>
        <p className="text-xs text-neutral-400 mt-0.5">Rig bare hand anchors & landmark geometry</p>
      </div>

      <div className="space-y-3">
        <span className="text-xs font-medium text-neutral-300">Rigging Steps</span>
        <div className="space-y-2 text-xs">
          <div className="p-2.5 rounded-lg bg-neutral-900 border border-neutral-800 flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-neutral-200 font-medium block">1. 5-Point Cuticle Bounds</span>
              <span className="text-neutral-400 text-[11px]">Left/right sidewalls, cuticle apex, and distal groove.</span>
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-neutral-900 border border-neutral-800 flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-neutral-200 font-medium block">2. Procedural Occlusion Map</span>
              <span className="text-neutral-400 text-[11px]">Auto-generates eponychium skin fold overlay mask.</span>
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-neutral-900 border border-neutral-800 flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-neutral-200 font-medium block">3. Ambient Contact Shadow</span>
              <span className="text-neutral-400 text-[11px]">Computes soft AO pocket along the cuticle curve.</span>
            </div>
          </div>
        </div>
      </div>

      <div className="pt-2 border-t border-neutral-800">
        <button className="w-full py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors text-neutral-200 cursor-pointer">
          Reset All Finger Anchors
        </button>
      </div>
    </div>
  );
};
