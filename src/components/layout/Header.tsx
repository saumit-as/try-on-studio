import React from "react";
import {
  Sparkles,
  Layers,
  Hand,
  Sliders,
  Download,
  RefreshCw,
} from "lucide-react";
import { ActiveTab, SidecarStatus } from "../../types";

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  sidecarStatus: SidecarStatus;
  sidecarVersion: string;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  sidecarStatus,
  sidecarVersion,
}) => {
  return (
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
  );
};
