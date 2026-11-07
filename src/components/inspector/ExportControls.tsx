import React from "react";
import { Download } from "lucide-react";

export const ExportControls: React.FC = () => {
  return (
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
              className="px-2 py-2 text-xs rounded bg-neutral-900 border border-neutral-800 hover:border-sky-500/50 text-neutral-300 text-center cursor-pointer transition-colors"
            >
              {ratio}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2 pt-2 border-t border-neutral-800">
        <span className="text-xs font-medium text-neutral-300">Output Packages</span>
        <button className="w-full py-2.5 px-3 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 text-neutral-200 text-left flex items-center justify-between cursor-pointer transition-colors">
          <span>Flattened 300 DPI PNG</span>
          <Download className="w-3.5 h-3.5 text-neutral-400" />
        </button>
        <button className="w-full py-2.5 px-3 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 text-neutral-200 text-left flex items-center justify-between cursor-pointer transition-colors">
          <span>Transparent Cutout PNG</span>
          <Download className="w-3.5 h-3.5 text-neutral-400" />
        </button>
        <button className="w-full py-2.5 px-3 bg-rose-500/10 hover:bg-rose-500/20 text-xs font-medium rounded-lg border border-rose-500/30 text-rose-300 text-left flex items-center justify-between cursor-pointer transition-colors">
          <span>Multi-Layer Photoshop PSD</span>
          <Download className="w-3.5 h-3.5 text-rose-400" />
        </button>
      </div>
    </div>
  );
};
