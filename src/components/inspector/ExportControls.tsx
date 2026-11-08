import React, { useState } from "react";
import {
  Download,
  FileImage,
  Layers,
  Sparkles,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Ratio,
  Printer,
} from "lucide-react";
import { HandState, LightingState, TrayState } from "../../types";
import { callSidecar } from "../../services/sidecar";
import { ExportAspectRatio } from "../canvas/ExportCanvas";

interface ExportControlsProps {
  trayState: TrayState;
  handState: HandState;
  lightingState: LightingState;
  aspectRatio: ExportAspectRatio;
  setAspectRatio: (r: ExportAspectRatio) => void;
  onExportStart?: () => void;
  onExportEnd?: () => void;
}

interface ExportResult {
  status: string;
  output_dir: string;
  files: {
    png?: {
      path: string;
      filename: string;
      size_bytes: number;
      dimensions: [number, number];
      dpi: [number, number];
    };
    psd?: {
      path: string;
      filename: string;
      size_bytes: number;
      layers: string[];
    };
  };
  execution_time_ms?: number;
}

export const ExportControls: React.FC<ExportControlsProps> = ({
  trayState,
  handState,
  lightingState,
  aspectRatio,
  setAspectRatio,
  onExportStart,
  onExportEnd,
}) => {
  const [format, setFormat] = useState<"both" | "png" | "psd">("both");
  const [transparentCutout, setTransparentCutout] = useState<boolean>(false);
  const [outputDir, setOutputDir] = useState<string>("exports");
  const [filename, setFilename] = useState<string>("press_on_studio_mockup");
  const [dpi, setDpi] = useState<number>(300);

  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ExportResult | null>(null);

  // Compile mapped nails dictionary
  const getMappedNailsDict = () => {
    const dict: Record<string, string> = {};
    for (const [slotIdStr, finger] of Object.entries(trayState.slotFinger)) {
      const sid = Number(slotIdStr);
      if (trayState.extractedPreviews[sid]) {
        dict[finger] = trayState.extractedPreviews[sid];
      }
    }
    return dict;
  };

  const handleExecuteExport = async () => {
    setIsExporting(true);
    setExportError(null);
    if (onExportStart) onExportStart();

    try {
      const mappedNails = getMappedNailsDict();
      const res = await callSidecar<ExportResult>("export_composite", {
        hand_image_base64: handState.currentTemplate?.imageSrc,
        landmarks: handState.currentTemplate?.landmarks,
        mapped_nails: mappedNails,
        lighting_options: {
          preset: lightingState.preset,
          glossIntensity: lightingState.glossIntensity / 50.0,
          shadowDepth: lightingState.shadowDepth / 50.0,
          lightAngle: lightingState.lightAngle,
          cCurveIntensity: lightingState.cCurveIntensity / 50.0,
          nailLengthScale: lightingState.nailLengthScale,
          finger_adjustments: lightingState.fingerAdjustments,
        },
        export_options: {
          format,
          aspect_ratio: aspectRatio,
          dpi,
          transparent_background: transparentCutout,
          output_dir: outputDir,
          filename,
        },
      });

      if (res && res.status === "ok") {
        setLastResult(res);
      } else {
        throw new Error("Export failed to return valid files.");
      }
    } catch (err: any) {
      console.error("Export error:", err);
      setExportError(err?.message || "Failed to generate export packages.");
    } finally {
      setIsExporting(false);
      if (onExportEnd) onExportEnd();
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  return (
    <div className="p-5 space-y-6">
      {/* Title & Description */}
      <div>
        <div className="flex items-center gap-2">
          <Printer className="w-4 h-4 text-amber-400" />
          <h2 className="text-sm font-semibold text-neutral-200">
            Phase 5: Production Export
          </h2>
        </div>
        <p className="text-xs text-neutral-400 mt-0.5 leading-relaxed">
          Generate print-ready 300 DPI composite PNGs and multi-layer Photoshop (.psd) files with preserved layer hierarchy.
        </p>
      </div>

      {/* Aspect Ratio Presets */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300 flex items-center gap-1.5">
            <Ratio className="w-3.5 h-3.5 text-neutral-400" />
            Aspect Ratio Preset
          </span>
          <span className="text-[11px] font-mono text-amber-400">
            {aspectRatio === "original"
              ? "1536 × 2752 px"
              : aspectRatio === "1:1"
              ? "2752 × 2752 px"
              : aspectRatio === "4:5"
              ? "2202 × 2752 px"
              : "4892 × 2752 px"}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {(
            [
              { id: "original", label: "Original (Native)", sub: "1536×2752 (9:16)" },
              { id: "1:1", label: "1:1 Square", sub: "2752×2752 (Feed / Grid)" },
              { id: "4:5", label: "4:5 Portrait", sub: "2202×2752 (Instagram / Shop)" },
              { id: "16:9", label: "16:9 Banner", sub: "4892×2752 (Hero Banner)" },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              onClick={() => setAspectRatio(item.id)}
              className={`p-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                aspectRatio === item.id
                  ? "bg-amber-500/15 border-amber-500/60 text-neutral-100 shadow-md"
                  : "bg-neutral-900 border-neutral-800 hover:border-neutral-700 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              <div className="text-xs font-semibold text-neutral-200">
                {item.label}
              </div>
              <div className="text-[10px] text-neutral-500 mt-0.5">
                {item.sub}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Output Package Formats */}
      <div className="space-y-2 pt-2 border-t border-neutral-800">
        <span className="text-xs font-medium text-neutral-300">
          Output Packages
        </span>
        <div className="grid grid-cols-3 gap-2">
          {(
            [
              { id: "both", label: "Complete Bundle", icon: Sparkles, desc: "PNG + PSD" },
              { id: "png", label: "300 DPI PNG", icon: FileImage, desc: "Print Flat" },
              { id: "psd", label: "Photoshop PSD", icon: Layers, desc: "5 Layers" },
            ] as const
          ).map((pkg) => {
            const Icon = pkg.icon;
            const isSelected = format === pkg.id;
            return (
              <button
                key={pkg.id}
                onClick={() => setFormat(pkg.id)}
                className={`p-2.5 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? "bg-amber-500/15 border-amber-500/60 text-neutral-100 shadow-md"
                    : "bg-neutral-900 border-neutral-800 hover:border-neutral-700 text-neutral-400 hover:text-neutral-200"
                }`}
              >
                <Icon
                  className={`w-4 h-4 mb-2 ${
                    isSelected ? "text-amber-400" : "text-neutral-500"
                  }`}
                />
                <div>
                  <div className="text-xs font-medium text-neutral-200">
                    {pkg.label}
                  </div>
                  <div className="text-[10px] text-neutral-500 mt-0.5">
                    {pkg.desc}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Photoshop Layer Hierarchy Inspection Card */}
      {(format === "both" || format === "psd") && (
        <div className="space-y-2.5 pt-2 border-t border-neutral-800">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-neutral-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-sky-400" />
              Photoshop Layers Preserved (5 Passes)
            </span>
            <span className="text-[10px] text-neutral-500">Top to Bottom</span>
          </div>

          <div className="space-y-1.5 bg-neutral-900/60 p-3 rounded-xl border border-neutral-800 text-[11px]">
            <div className="flex items-center justify-between text-neutral-200">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                <span>Layer 4: Cuticle Pocket Lip</span>
              </div>
              <span className="text-[10px] font-mono text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                Normal (RGBA)
              </span>
            </div>

            <div className="flex items-center justify-between text-neutral-200">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                <span>Layer 3: Gel Topcoat Highlights</span>
              </div>
              <span className="text-[10px] font-mono text-sky-400 bg-neutral-800 px-1.5 py-0.5 rounded">
                Screen (RGBA)
              </span>
            </div>

            <div className="flex items-center justify-between text-neutral-200">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Layer 2: Press-On Nails Plate</span>
              </div>
              <span className="text-[10px] font-mono text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                Normal (RGBA)
              </span>
            </div>

            <div className="flex items-center justify-between text-neutral-200">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <span>Layer 1: Contact AO & Shadows</span>
              </div>
              <span className="text-[10px] font-mono text-amber-400 bg-neutral-800 px-1.5 py-0.5 rounded">
                Multiply (84%)
              </span>
            </div>

            <div className="flex items-center justify-between text-neutral-200">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-neutral-400" />
                <span>Layer 0: Bare Hand Base</span>
              </div>
              <span className="text-[10px] font-mono text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                Normal (RGB)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Background Options */}
      <div className="space-y-2 pt-2 border-t border-neutral-800">
        <span className="text-xs font-medium text-neutral-300">
          Background Transparency
        </span>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setTransparentCutout(false)}
            className={`p-2.5 rounded-xl border text-xs font-medium transition-all cursor-pointer ${
              !transparentCutout
                ? "bg-neutral-800 border-amber-500/60 text-neutral-100"
                : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
            }`}
          >
            Solid Studio Backdrop
          </button>
          <button
            onClick={() => setTransparentCutout(true)}
            className={`p-2.5 rounded-xl border text-xs font-medium transition-all cursor-pointer ${
              transparentCutout
                ? "bg-neutral-800 border-amber-500/60 text-neutral-100"
                : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
            }`}
          >
            Transparent Hand Cutout
          </button>
        </div>
      </div>

      {/* Physical Print DPI */}
      <div className="space-y-2 pt-2 border-t border-neutral-800">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">
            Physical Print Resolution (DPI)
          </span>
          <span className="text-[11px] font-mono text-amber-400">
            {dpi === 300 ? "300 DPI (Standard Print)" : `${dpi} DPI`}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[150, 300, 600].map((d) => (
            <button
              key={d}
              onClick={() => setDpi(d)}
              className={`py-2 px-2.5 rounded-xl border text-xs font-medium transition-all cursor-pointer ${
                dpi === d
                  ? "bg-amber-500/15 border-amber-500/60 text-neutral-100 shadow-md"
                  : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {d === 150 ? "150 DPI (Web)" : d === 300 ? "300 DPI (Print)" : "600 DPI (Ultra)"}
            </button>
          ))}
        </div>
      </div>

      {/* File Destination & Name */}
      <div className="space-y-2 pt-2 border-t border-neutral-800">
        <label className="text-xs font-medium text-neutral-300 block">
          Base Filename
        </label>
        <input
          type="text"
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          className="w-full bg-neutral-900 border border-neutral-800 focus:border-amber-500/60 text-neutral-200 text-xs px-3 py-2 rounded-xl outline-none font-mono"
        />

        <label className="text-xs font-medium text-neutral-300 block pt-1">
          Output Directory
        </label>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="w-full bg-neutral-900 border border-neutral-800 focus:border-amber-500/60 text-neutral-200 text-xs px-3 py-2 rounded-xl outline-none font-mono"
          />
        </div>
      </div>

      {/* Export Action Button */}
      <div className="pt-2">
        <button
          onClick={handleExecuteExport}
          disabled={isExporting}
          className="w-full py-3 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-neutral-950 font-semibold text-xs rounded-xl shadow-lg transition-all cursor-pointer flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {isExporting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Rendering Full 300 DPI Export Packages...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Export Production Assets Now
            </>
          )}
        </button>
      </div>

      {/* Error Message */}
      {exportError && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{exportError}</span>
        </div>
      )}

      {/* Successful Export Result Card */}
      {lastResult && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-4 h-4" />
            <span>Export Complete ({lastResult.execution_time_ms} ms)</span>
          </div>

          <div className="text-[11px] text-neutral-300 space-y-2">
            {lastResult.files.png && (
              <div className="bg-neutral-900/80 p-2.5 rounded-xl border border-neutral-800">
                <div className="flex items-center justify-between text-neutral-200 font-medium">
                  <span className="flex items-center gap-1.5">
                    <FileImage className="w-3.5 h-3.5 text-amber-400" />
                    {lastResult.files.png.filename}
                  </span>
                  <span className="text-neutral-400 text-[10px]">
                    {formatFileSize(lastResult.files.png.size_bytes)}
                  </span>
                </div>
                <div className="text-[10px] text-neutral-500 mt-1">
                  {lastResult.files.png.dimensions[0]} × {lastResult.files.png.dimensions[1]} px • 300 DPI
                </div>
              </div>
            )}

            {lastResult.files.psd && (
              <div className="bg-neutral-900/80 p-2.5 rounded-xl border border-neutral-800">
                <div className="flex items-center justify-between text-neutral-200 font-medium">
                  <span className="flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-sky-400" />
                    {lastResult.files.psd.filename}
                  </span>
                  <span className="text-neutral-400 text-[10px]">
                    {formatFileSize(lastResult.files.psd.size_bytes)}
                  </span>
                </div>
                <div className="text-[10px] text-neutral-500 mt-1">
                  {lastResult.files.psd.layers.length} Layers (Normal, Screen, Multiply)
                </div>
              </div>
            )}

            <div className="text-[10px] text-neutral-400 truncate pt-1">
              Destination: <span className="font-mono text-neutral-300">{lastResult.output_dir}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
