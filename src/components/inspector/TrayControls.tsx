import React, { useEffect, useState } from "react";
import {
  RotateCw,
  Wand2,
  ChevronDown,
  RotateCcw,
  Loader2,
  Move,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Sliders,
  PenTool,
  Maximize2,
  X,
  ZoomIn,
  ZoomOut,
  Bug,
  Sparkles,
  Download,
  CheckCircle2,
} from "lucide-react";
import { FingerName, NailShape, TrayState, DetectedSlot, ExtractionDebugInfo } from "../../types";
import { callSidecar } from "../../services/sidecar";

interface TrayControlsProps {
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const SHAPES: NailShape[] = [
  "Almond",
  "Coffin",
  "Square",
  "Squoval",
  "Stiletto",
  "Oval",
];

export const TrayControls: React.FC<TrayControlsProps> = ({
  trayState,
  setTrayState,
}) => {
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [isRefitting, setIsRefitting] = useState(false);
  const [nudgeStep, setNudgeStep] = useState<number>(5);
  const [isPreviewModalOpen, setIsPreviewModalOpen] = useState(false);
  const [modalZoom, setModalZoom] = useState<number>(1);
  const [forceExactMask, setForceExactMask] = useState<boolean>(true);
  const [isRefiningSam, setIsRefiningSam] = useState<boolean>(false);
  const [isDumpingDebug, setIsDumpingDebug] = useState<boolean>(false);
  const [debugMessage, setDebugMessage] = useState<string | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(true);

  const {
    selectedSlot,
    slotFinger,
    nailRotation,
    slotInverted,
    shapeOverrides,
    detectedSlots,
    extractedPreviews,
    isManualEditMode,
    originalSlots,
  } = trayState;

  const activeSlotData: DetectedSlot | undefined = detectedSlots.find(
    (s) => s.slot_id === selectedSlot
  );

  const currentShape =
    shapeOverrides[selectedSlot] ||
    activeSlotData?.shape_info.shape ||
    "Almond";

  const currentRotation = nailRotation[selectedSlot] || 0;
  const isInverted = slotInverted[selectedSlot] || false;
  const activePreview = extractedPreviews[selectedSlot];

  // Auto-fetch upright cutout preview whenever selected slot changes or photo is loaded
  useEffect(() => {
    if (!trayState.trayImageSrc || !activeSlotData) return;
    if (extractedPreviews[selectedSlot]) return;

    let isMounted = true;
    setIsLoadingPreview(true);

    const isCustomContour =
      !!(activeSlotData.pen_anchors && activeSlotData.pen_anchors.length > 0) ||
      !!(trayState.slotPenAnchors?.[selectedSlot] && trayState.slotPenAnchors[selectedSlot].length > 0);

    callSidecar<{
      status: string;
      width: number;
      height: number;
      image_base64: string;
      debug_info?: ExtractionDebugInfo;
    }>("extract_nail", {
      image_base64: trayState.trayImageSrc,
      contour: activeSlotData.contour,
      slot_id: selectedSlot,
      deskew: true,
      auto_orient_cuticle_down: true,
      exact_mask: forceExactMask || isCustomContour,
    })
      .then((ext) => {
        if (isMounted && ext && ext.status === "ok" && ext.image_base64) {
          const previewUrl = ext.image_base64.startsWith("data:")
            ? ext.image_base64
            : `data:image/png;base64,${ext.image_base64}`;
          setTrayState((prev) => ({
            ...prev,
            extractedPreviews: {
              ...prev.extractedPreviews,
              [selectedSlot]: previewUrl,
            },
            extractedDebug: ext.debug_info
              ? {
                  ...prev.extractedDebug,
                  [selectedSlot]: ext.debug_info,
                }
              : prev.extractedDebug,
          }));
        }
      })
      .catch((err) => {
        console.error(`[TrayControls] Auto-extraction error for slot #${selectedSlot}:`, err);
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingPreview(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [trayState.trayImageSrc, selectedSlot, activeSlotData, extractedPreviews, forceExactMask, setTrayState]);

  const handleSlotSelect = (slotId: number) => {
    setTrayState((prev) => ({ ...prev, selectedSlot: slotId }));
  };

  const handleFingerAssign = (finger: FingerName) => {
    setTrayState((prev) => {
      const nextMapping = { ...prev.slotFinger };
      Object.entries(nextMapping).forEach(([sId, fName]) => {
        if (fName === finger) {
          delete nextMapping[Number(sId)];
        }
      });
      nextMapping[selectedSlot] = finger;
      return { ...prev, slotFinger: nextMapping };
    });
  };

  const handleUnassign = () => {
    setTrayState((prev) => {
      const nextMapping = { ...prev.slotFinger };
      delete nextMapping[selectedSlot];
      return { ...prev, slotFinger: nextMapping };
    });
  };

  const handleAutoAssignRow1 = () => {
    setTrayState((prev) => {
      const nextMapping: Record<number, FingerName> = {};
      const slots = prev.detectedSlots.slice(0, 5);
      slots.forEach((s, idx) => {
        nextMapping[s.slot_id] = FINGERS[idx];
      });
      return { ...prev, slotFinger: nextMapping };
    });
  };

  const handleShapeOverride = (shape: string) => {
    setTrayState((prev) => ({
      ...prev,
      shapeOverrides: {
        ...prev.shapeOverrides,
        [selectedSlot]: shape,
      },
    }));
  };

  const handleRotationFlip = () => {
    setTrayState((prev) => ({
      ...prev,
      slotInverted: {
        ...prev.slotInverted,
        [selectedSlot]: !prev.slotInverted[selectedSlot],
      },
    }));
  };

  const handleRotationChange = (val: number) => {
    setTrayState((prev) => ({
      ...prev,
      nailRotation: {
        ...prev.nailRotation,
        [selectedSlot]: val,
      },
    }));
  };

  const handleResetRotation = () => {
    setTrayState((prev) => ({
      ...prev,
      nailRotation: {
        ...prev.nailRotation,
        [selectedSlot]: 0,
      },
    }));
  };

  // Manual Nudge Handler (dx, dy in image pixels)
  const handleNudge = (dx: number, dy: number) => {
    if (!activeSlotData) return;
    const [bx, by, bw, bh] = activeSlotData.bbox;
    const newBbox: [number, number, number, number] = [bx + dx, by + dy, bw, bh];
    const newContour = activeSlotData.contour.map(
      ([x, y]) => [x + dx, y + dy] as [number, number]
    );

    setTrayState((prev) => ({
      ...prev,
      detectedSlots: prev.detectedSlots.map((s) =>
        s.slot_id === selectedSlot ? { ...s, bbox: newBbox, contour: newContour } : s
      ),
      extractedPreviews: {
        ...prev.extractedPreviews,
        [selectedSlot]: "",
      },
    }));
  };

  // Manual Scale Handler (deltaW, deltaH)
  const handleScale = (deltaW: number, deltaH: number) => {
    if (!activeSlotData) return;
    const [bx, by, bw, bh] = activeSlotData.bbox;
    const newBw = Math.max(16, bw + deltaW);
    const newBh = Math.max(16, bh + deltaH);
    const newBx = Math.round(bx - (newBw - bw) / 2);
    const newBy = Math.round(by - (newBh - bh) / 2);
    const newBbox: [number, number, number, number] = [newBx, newBy, newBw, newBh];

    const newContour = activeSlotData.contour.map(
      ([x, y]) =>
        [
          Math.round(newBx + ((x - bx) / Math.max(1, bw)) * newBw),
          Math.round(newBy + ((y - by) / Math.max(1, bh)) * newBh),
        ] as [number, number]
    );

    setTrayState((prev) => ({
      ...prev,
      detectedSlots: prev.detectedSlots.map((s) =>
        s.slot_id === selectedSlot ? { ...s, bbox: newBbox, contour: newContour } : s
      ),
      extractedPreviews: {
        ...prev.extractedPreviews,
        [selectedSlot]: "",
      },
    }));
  };

  // Reset current slot to original auto-detected contour
  const handleResetToAuto = () => {
    const orig = originalSlots?.[selectedSlot];
    if (!orig) return;
    setTrayState((prev) => ({
      ...prev,
      detectedSlots: prev.detectedSlots.map((s) =>
        s.slot_id === selectedSlot ? JSON.parse(JSON.stringify(orig)) : s
      ),
      extractedPreviews: {
        ...prev.extractedPreviews,
        [selectedSlot]: "",
      },
    }));
  };

  // Re-run CV snapping inside user's manually adjusted bounding box
  const handleAutoRefit = async () => {
    if (!trayState.trayImageSrc || !activeSlotData) return;
    setIsRefitting(true);
    try {
      const res = await callSidecar<{
        status: string;
        bbox: [number, number, number, number];
        contour: [number, number][];
        pca_angle: number;
        shape_info: { shape: string; confidence: number; metrics: any };
      }>("refit_slot_contour", {
        image_base64: trayState.trayImageSrc,
        bbox: activeSlotData.bbox,
      });

      if (res && res.status === "ok") {
        setTrayState((prev) => ({
          ...prev,
          detectedSlots: prev.detectedSlots.map((s) =>
            s.slot_id === selectedSlot
              ? {
                  ...s,
                  bbox: res.bbox,
                  contour: res.contour,
                  pca_angle: res.pca_angle,
                  shape_info: res.shape_info,
                }
              : s
          ),
          extractedPreviews: {
            ...prev.extractedPreviews,
            [selectedSlot]: "",
          },
        }));
      }
    } catch (e) {
      console.error("Auto-refit failed:", e);
    } finally {
      setIsRefitting(false);
    }
  };

  // Re-extract fresh cutout from current contour with exact_mask or adaptive mask
  const handleReExtract = async (overrideExactMask?: boolean, debugDump?: boolean) => {
    if (!trayState.trayImageSrc || !activeSlotData) return;
    setIsLoadingPreview(true);
    const useExact = overrideExactMask !== undefined ? overrideExactMask : forceExactMask;
    try {
      const isCustomContour =
        !!(activeSlotData.pen_anchors && activeSlotData.pen_anchors.length > 0) ||
        !!(trayState.slotPenAnchors?.[selectedSlot] && trayState.slotPenAnchors[selectedSlot].length > 0);

      const ext = await callSidecar<{
        status: string;
        width: number;
        height: number;
        image_base64: string;
        debug_info?: ExtractionDebugInfo;
      }>("extract_nail", {
        image_base64: trayState.trayImageSrc,
        contour: activeSlotData.contour,
        slot_id: selectedSlot,
        deskew: true,
        auto_orient_cuticle_down: !isInverted,
        exact_mask: isCustomContour || useExact,
        debug_dump: !!debugDump,
      });

      if (ext && ext.status === "ok" && ext.image_base64) {
        const previewUrl = ext.image_base64.startsWith("data:")
          ? ext.image_base64
          : `data:image/png;base64,${ext.image_base64}`;
        setTrayState((prev) => ({
          ...prev,
          extractedPreviews: {
            ...prev.extractedPreviews,
            [selectedSlot]: previewUrl,
          },
          extractedDebug: ext.debug_info
            ? {
                ...prev.extractedDebug,
                [selectedSlot]: ext.debug_info,
              }
            : prev.extractedDebug,
        }));

        if (debugDump && ext.debug_info?.debug_dump_path) {
          setDebugMessage(`Debug files dumped to: ${ext.debug_info.debug_dump_path}`);
          setTimeout(() => setDebugMessage(null), 8000);
        }
      }
    } catch (err) {
      console.error("Re-extract failed:", err);
    } finally {
      setIsLoadingPreview(false);
    }
  };

  // 1-Click zero-shot contour refinement using MobileSAM
  const handleRefineWithSam = async () => {
    if (!trayState.trayImageSrc || !activeSlotData) return;
    setIsRefiningSam(true);
    try {
      const bx = activeSlotData.bbox[0];
      const by = activeSlotData.bbox[1];
      const bw = activeSlotData.bbox[2];
      const bh = activeSlotData.bbox[3];
      const samRes = await callSidecar<{
        status: string;
        contour: [number, number][];
        bbox: [number, number, number, number];
        metrics?: Record<string, any>;
      }>("segment_prompt", {
        image_base64: trayState.trayImageSrc,
        bbox: [bx, by, bx + bw, by + bh],
        slot_id: selectedSlot,
      });

      if (samRes && samRes.status === "ok" && samRes.contour && samRes.contour.length > 5) {
        setTrayState((prev) => {
          const nextSlots = prev.detectedSlots.map((s) => {
            if (s.slot_id === selectedSlot) {
              return {
                ...s,
                contour: samRes.contour,
                bbox: samRes.bbox || s.bbox,
                method: "sam",
                method_note: "MobileSAM zero-shot contour refinement",
              };
            }
            return s;
          });
          return { ...prev, detectedSlots: nextSlots };
        });

        // Trigger immediate re-extraction with new contour
        setTimeout(() => {
          handleReExtract(true);
        }, 50);
        setDebugMessage("✓ Nail contour successfully refined with MobileSAM AI!");
        setTimeout(() => setDebugMessage(null), 5000);
      }
    } catch (err) {
      console.error("MobileSAM refinement failed:", err);
    } finally {
      setIsRefiningSam(false);
    }
  };

  const handleDumpDebug = async () => {
    if (!trayState.trayImageSrc || !activeSlotData) return;
    setIsDumpingDebug(true);
    try {
      await handleReExtract(forceExactMask, true);
    } catch (err) {
      console.error("Debug dump failed:", err);
    } finally {
      setIsDumpingDebug(false);
    }
  };

  return (
    <div className="p-5 space-y-6 select-none">
      <div>
        <h2 className="text-sm font-semibold text-neutral-200">
          Tray Extractor & Slot Mapping
        </h2>
        <p className="text-xs text-neutral-400 mt-0.5">
          Assign detected nail packaging slots to hand fingers
        </p>
      </div>

      {/* Detected Slots Grid Selector */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">
            Detected Slots ({detectedSlots.length > 0 ? detectedSlots.length : "None"})
          </span>
          {detectedSlots.length > 0 && (
            <button
              onClick={handleAutoAssignRow1}
              className="text-[11px] text-amber-400 hover:text-amber-300 flex items-center gap-1 cursor-pointer font-medium"
            >
              <Wand2 className="w-3 h-3" />
              <span>Auto-Map Row 1</span>
            </button>
          )}
        </div>

        {detectedSlots.length === 0 ? (
          <div className="p-3 bg-neutral-900/60 rounded-xl border border-neutral-800 text-xs text-neutral-500 text-center">
            Upload a tray photo on the left canvas to populate detected nail slots.
          </div>
        ) : (
          <div className="grid grid-cols-5 gap-1.5 max-h-36 overflow-y-auto pr-1">
            {detectedSlots.map((slot) => {
              const isSelected = selectedSlot === slot.slot_id;
              const assigned = slotFinger[slot.slot_id];
              return (
                <button
                  key={slot.slot_id}
                  onClick={() => handleSlotSelect(slot.slot_id)}
                  className={`py-2 px-1 text-center rounded-lg border transition-all cursor-pointer flex flex-col items-center justify-center ${
                    isSelected
                      ? "bg-amber-500/25 border-amber-500 text-amber-200 shadow-sm"
                      : "bg-neutral-800/80 border-neutral-700/80 text-neutral-400 hover:text-neutral-200 hover:border-neutral-600"
                  }`}
                >
                  <span className="text-[11px] font-semibold">#{slot.slot_id}</span>
                  <span
                    className={`text-[9px] font-medium mt-0.5 px-1 rounded ${
                      assigned
                        ? "bg-amber-500/30 text-amber-300 font-bold"
                        : "text-neutral-500"
                    }`}
                  >
                    {assigned ? assigned.slice(0, 3) : "—"}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected Slot Information Card */}
      {activeSlotData && (
        <div className="p-3.5 bg-neutral-900/80 rounded-xl border border-neutral-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-md bg-amber-500/20 text-amber-400 font-bold text-xs flex items-center justify-center">
                #{activeSlotData.slot_id}
              </span>
              <span className="text-xs font-semibold text-neutral-200">
                Slot {activeSlotData.slot_id} Details
              </span>
            </div>

            <span className="text-[10px] text-neutral-400 font-mono">
              {activeSlotData.bbox[2]}×{activeSlotData.bbox[3]}px
            </span>
          </div>

          {/* Upright Deskewed Cutout Thumbnail */}
          <div className="flex items-center gap-3 pt-1">
            <div
              onClick={() => {
                if (activePreview) {
                  setIsPreviewModalOpen(true);
                  setModalZoom(1);
                }
              }}
              className={`w-16 h-20 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-center overflow-hidden relative shrink-0 transition-all ${
                activePreview
                  ? "cursor-pointer hover:border-amber-500/80 hover:shadow-lg hover:shadow-amber-500/10 group"
                  : ""
              }`}
              title={activePreview ? "Click to inspect enlarged cutout & trim edges" : "Cutout preview"}
            >
              {activePreview ? (
                <>
                  <img
                    src={activePreview}
                    alt={`Nail #${selectedSlot}`}
                    className="max-w-full max-h-full object-contain group-hover:scale-105 transition-transform"
                    style={{
                      transform: `rotate(${currentRotation}deg) ${
                        isInverted ? "rotate(180deg)" : ""
                      }`,
                    }}
                  />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <Maximize2 className="w-4 h-4 text-amber-400 drop-shadow" />
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center gap-1 text-[9px] text-neutral-500 text-center px-1">
                  {isLoadingPreview ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin text-amber-500" />
                      <span>Extracting...</span>
                    </>
                  ) : (
                    <span>Cutout preview</span>
                  )}
                </div>
              )}
            </div>

            {/* Shape Classification & Manual Override */}
            <div className="flex-1 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-neutral-400 text-[11px]">Classified Shape:</span>
                <span className="text-amber-400 font-medium text-[11px]">
                  {activeSlotData.shape_info.shape} (
                  {Math.round(activeSlotData.shape_info.confidence * 100)}%)
                </span>
              </div>

              <div className="relative">
                <select
                  value={currentShape}
                  onChange={(e) => handleShapeOverride(e.target.value)}
                  className="w-full bg-neutral-800 border border-neutral-700 hover:border-neutral-600 text-neutral-200 text-xs rounded-lg px-2.5 py-1.5 appearance-none cursor-pointer focus:outline-none focus:border-amber-500/80"
                >
                  {SHAPES.map((shape) => (
                    <option key={shape} value={shape}>
                      Shape: {shape}
                    </option>
                  ))}
                </select>
                <ChevronDown className="w-3.5 h-3.5 text-neutral-400 absolute right-2.5 top-2.5 pointer-events-none" />
              </div>

              <div className="text-[10px] text-neutral-500">
                PCA Deskew Angle: {activeSlotData.pca_angle.toFixed(1)}°
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Extraction Diagnostics & Debug Tools Section */}
      {activeSlotData && (
        <div className="p-3.5 bg-neutral-900/90 rounded-xl border border-neutral-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Bug className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs font-semibold text-neutral-200">
                Extraction Diagnostics & Debug
              </span>
            </div>
            <button
              onClick={() => setShowDiagnostics((prev) => !prev)}
              className="text-[11px] text-neutral-400 hover:text-neutral-200 cursor-pointer flex items-center gap-1"
            >
              <span>{showDiagnostics ? "Hide" : "Show"}</span>
              <ChevronDown className={`w-3 h-3 transition-transform ${showDiagnostics ? "rotate-180" : ""}`} />
            </button>
          </div>

          {debugMessage && (
            <div className="p-2 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-[11px] text-emerald-300 flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-emerald-400" />
              <span className="truncate">{debugMessage}</span>
            </div>
          )}

          {showDiagnostics && (
            <div className="space-y-3 pt-1">
              {/* Method & Quality Pill Header */}
              <div className="flex items-center justify-between text-[11px] bg-neutral-950/70 p-2 rounded-lg border border-neutral-800/80">
                <span className="text-neutral-400">Segmentation Method:</span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide ${
                    activeSlotData.method === "sam"
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      : activeSlotData.method === "yolo"
                      ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                      : activeSlotData.pen_anchors?.length
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                      : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                  }`}
                >
                  {activeSlotData.method === "sam"
                    ? "MobileSAM AI"
                    : activeSlotData.method === "yolo"
                    ? "YOLO11 ML"
                    : activeSlotData.pen_anchors?.length
                    ? "Pen Tool"
                    : "Classical CV"}
                </span>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-2 bg-neutral-950/50 rounded-lg border border-neutral-800/60">
                  <span className="text-neutral-500 block text-[10px]">Contour Vertices</span>
                  <span className="text-neutral-200 font-mono font-medium">
                    {activeSlotData.contour.length} points
                  </span>
                </div>
                <div className="p-2 bg-neutral-950/50 rounded-lg border border-neutral-800/60">
                  <span className="text-neutral-500 block text-[10px]">De-Matting Choke</span>
                  <span className="text-neutral-200 font-mono font-medium">
                    {trayState.extractedDebug?.[selectedSlot]?.dematting_metrics?.choke_px ?? 2} px (
                    {trayState.extractedDebug?.[selectedSlot]?.dematting_metrics?.area_removed_pct ?? 0}%)
                  </span>
                </div>
                <div className="p-2 bg-neutral-950/50 rounded-lg border border-neutral-800/60">
                  <span className="text-neutral-500 block text-[10px]">Extraction Mode</span>
                  <span className="text-amber-300 font-medium text-[10px]">
                    {forceExactMask ? "Exact Shape Contour" : "Adaptive Super-Ellipse"}
                  </span>
                </div>
                <div className="p-2 bg-neutral-950/50 rounded-lg border border-neutral-800/60">
                  <span className="text-neutral-500 block text-[10px]">Orientation</span>
                  <span className="text-neutral-200 font-medium text-[10px]">
                    {isInverted ? "Cuticle Tucked (180°)" : "Standard Upright"}
                  </span>
                </div>
              </div>

              {/* Mask Strategy Switcher */}
              <div className="space-y-1">
                <span className="text-[10px] text-neutral-400 font-medium">Contour Mask Mode:</span>
                <div className="grid grid-cols-2 gap-1.5 bg-neutral-950 p-1 rounded-lg border border-neutral-800">
                  <button
                    onClick={() => {
                      setForceExactMask(true);
                      handleReExtract(true);
                    }}
                    className={`py-1 px-2 rounded text-[10px] font-medium transition-all cursor-pointer ${
                      forceExactMask
                        ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                        : "text-neutral-400 hover:text-neutral-200"
                    }`}
                  >
                    Exact Contour
                  </button>
                  <button
                    onClick={() => {
                      setForceExactMask(false);
                      handleReExtract(false);
                    }}
                    className={`py-1 px-2 rounded text-[10px] font-medium transition-all cursor-pointer ${
                      !forceExactMask
                        ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                        : "text-neutral-400 hover:text-neutral-200"
                    }`}
                  >
                    Super-Ellipse
                  </button>
                </div>
              </div>

              {/* AI & Debug Trigger Buttons */}
              <div className="space-y-1.5 pt-1">
                <button
                  onClick={handleRefineWithSam}
                  disabled={isRefiningSam || isLoadingPreview}
                  className="w-full py-2 px-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-xs rounded-lg flex items-center justify-center gap-2 cursor-pointer shadow-md transition-all disabled:opacity-50"
                  title="Run zero-shot MobileSAM on this slot to snap tightly around the true nail silhouette"
                >
                  {isRefiningSam ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="w-3.5 h-3.5 text-amber-200" />
                  )}
                  <span>{isRefiningSam ? "Refining with MobileSAM..." : "Refine Slot with MobileSAM AI"}</span>
                </button>

                <button
                  onClick={handleDumpDebug}
                  disabled={isDumpingDebug || isLoadingPreview}
                  className="w-full py-1.5 px-3 bg-neutral-800 hover:bg-neutral-750 text-neutral-300 border border-neutral-700/80 rounded-lg text-xs font-medium flex items-center justify-center gap-2 cursor-pointer transition-all disabled:opacity-50"
                  title="Export intermediate crop, deskewed mask, and JSON diagnostic metadata to test_output/debug_extractions/"
                >
                  {isDumpingDebug ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
                  ) : (
                    <Download className="w-3.5 h-3.5 text-neutral-400" />
                  )}
                  <span>Export Debug Images & JSON</span>
                </button>
              </div>

              {trayState.extractedDebug?.[selectedSlot]?.debug_dump_path && (
                <div className="p-2 bg-neutral-950 rounded border border-neutral-800/80 text-[9px] font-mono text-neutral-400 break-all select-all">
                  Dumped to: {trayState.extractedDebug[selectedSlot].debug_dump_path}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Manual Selection Modification Section */}
      {activeSlotData && (
        <div className="p-3.5 bg-neutral-900/90 rounded-xl border border-neutral-800 space-y-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs font-semibold text-neutral-200">
                Manual Selection Help
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              {/* Toggle Pen Tool */}
              <button
                onClick={() =>
                  setTrayState((prev) => ({
                    ...prev,
                    activeTool: prev.activeTool === "pen" ? "select" : "pen",
                    isManualEditMode: prev.activeTool === "pen" ? prev.isManualEditMode : false,
                  }))
                }
                className={`px-2 py-0.5 text-[10px] font-medium rounded-md border transition-all cursor-pointer flex items-center gap-1 ${
                  trayState.activeTool === "pen"
                    ? "bg-amber-500/25 border-amber-500 text-amber-300 font-semibold shadow-xs"
                    : "bg-neutral-800 hover:bg-neutral-750 border-neutral-700 text-neutral-300"
                }`}
                title="Draw custom curved nail contour with Pen Tool"
              >
                <PenTool className="w-2.5 h-2.5" />
                <span>{trayState.activeTool === "pen" ? "Pen: Active" : "Pen Tool"}</span>
              </button>

              {/* Toggle Interactive Canvas Handles */}
              <button
                onClick={() =>
                  setTrayState((prev) => ({
                    ...prev,
                    isManualEditMode: !prev.isManualEditMode,
                    activeTool: !prev.isManualEditMode ? "select" : prev.activeTool,
                  }))
                }
                className={`px-2 py-0.5 text-[10px] font-medium rounded-md border transition-all cursor-pointer flex items-center gap-1 ${
                  isManualEditMode
                    ? "bg-amber-500/25 border-amber-500 text-amber-300 font-semibold shadow-xs"
                    : "bg-neutral-800 hover:bg-neutral-750 border-neutral-700 text-neutral-300"
                }`}
                title="Toggle draggable bounding box corner handles on canvas"
              >
                <Move className="w-2.5 h-2.5" />
                <span>{isManualEditMode ? "Handles: ON" : "Handles"}</span>
              </button>
            </div>
          </div>

          <p className="text-[11px] text-neutral-400 leading-relaxed">
            {trayState.activeTool === "pen"
              ? "Pen Tool active: click and drag to curve around Slot #" +
                selectedSlot +
                ". Other boundaries are hidden to prevent clutter. Press Enter or click start point to finish."
              : isManualEditMode
              ? "Drag handles directly on the canvas to stretch, resize, or reposition the selection box."
              : "Use the micro-nudge D-pad, dimension adjusters, or the Pen Tool to curve custom contours."}
          </p>

          {/* D-Pad Micro-Nudge & Step Toggle */}
          <div className="space-y-2 pt-1 border-t border-neutral-800/80">
            <div className="flex items-center justify-between text-xs text-neutral-400">
              <span className="text-[11px] font-medium text-neutral-300">Position Nudge</span>
              <div className="flex items-center gap-1 bg-neutral-950 p-0.5 rounded border border-neutral-800">
                <button
                  onClick={() => setNudgeStep(1)}
                  className={`px-1.5 py-0.5 text-[10px] rounded cursor-pointer ${
                    nudgeStep === 1
                      ? "bg-amber-500 text-neutral-950 font-bold"
                      : "text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  1px
                </button>
                <button
                  onClick={() => setNudgeStep(5)}
                  className={`px-1.5 py-0.5 text-[10px] rounded cursor-pointer ${
                    nudgeStep === 5
                      ? "bg-amber-500 text-neutral-950 font-bold"
                      : "text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  5px
                </button>
              </div>
            </div>

            {/* D-Pad Buttons */}
            <div className="flex items-center justify-center">
              <div className="grid grid-cols-3 gap-1 w-32">
                <div />
                <button
                  onClick={() => handleNudge(0, -nudgeStep)}
                  className="p-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 active:bg-amber-500/30 border border-neutral-700 text-neutral-200 flex items-center justify-center cursor-pointer transition-all"
                  title={`Nudge Up (-${nudgeStep}px)`}
                >
                  <ArrowUp className="w-3.5 h-3.5" />
                </button>
                <div />
                <button
                  onClick={() => handleNudge(-nudgeStep, 0)}
                  className="p-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 active:bg-amber-500/30 border border-neutral-700 text-neutral-200 flex items-center justify-center cursor-pointer transition-all"
                  title={`Nudge Left (-${nudgeStep}px)`}
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                </button>
                <div className="flex items-center justify-center text-[10px] text-neutral-400 font-mono font-bold">
                  #{selectedSlot}
                </div>
                <button
                  onClick={() => handleNudge(nudgeStep, 0)}
                  className="p-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 active:bg-amber-500/30 border border-neutral-700 text-neutral-200 flex items-center justify-center cursor-pointer transition-all"
                  title={`Nudge Right (+${nudgeStep}px)`}
                >
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
                <div />
                <button
                  onClick={() => handleNudge(0, nudgeStep)}
                  className="p-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 active:bg-amber-500/30 border border-neutral-700 text-neutral-200 flex items-center justify-center cursor-pointer transition-all"
                  title={`Nudge Down (+${nudgeStep}px)`}
                >
                  <ArrowDown className="w-3.5 h-3.5" />
                </button>
                <div />
              </div>
            </div>
          </div>

          {/* Dimension Adjusters */}
          <div className="space-y-2 pt-1 border-t border-neutral-800/80">
            <span className="text-[11px] font-medium text-neutral-300">
              Dimensions & Scale
            </span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-neutral-400">Width:</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleScale(-4, 0)}
                    className="flex-1 py-1 px-2 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-neutral-300 font-mono text-[11px] cursor-pointer"
                  >
                    -4px
                  </button>
                  <button
                    onClick={() => handleScale(4, 0)}
                    className="flex-1 py-1 px-2 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-neutral-300 font-mono text-[11px] cursor-pointer"
                  >
                    +4px
                  </button>
                </div>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-neutral-400">Length:</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleScale(0, -6)}
                    className="flex-1 py-1 px-2 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-neutral-300 font-mono text-[11px] cursor-pointer"
                  >
                    -6px
                  </button>
                  <button
                    onClick={() => handleScale(0, 6)}
                    className="flex-1 py-1 px-2 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-neutral-300 font-mono text-[11px] cursor-pointer"
                  >
                    +6px
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Auto-Refit and Reset Action Buttons */}
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-neutral-800/80">
            <button
              onClick={handleAutoRefit}
              disabled={isRefitting}
              className="py-1.5 px-2.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-300 text-[11px] font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer disabled:opacity-50"
            >
              {isRefitting ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Snapping...</span>
                </>
              ) : (
                <>
                  <Wand2 className="w-3 h-3 text-amber-400" />
                  <span>Re-snap Contour</span>
                </>
              )}
            </button>

            <button
              onClick={handleResetToAuto}
              className="py-1.5 px-2.5 rounded-lg bg-neutral-800 hover:bg-neutral-750 border border-neutral-700 text-neutral-300 hover:text-neutral-100 text-[11px] font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer"
            >
              <RotateCcw className="w-3 h-3 text-neutral-400" />
              <span>Reset to Auto</span>
            </button>
          </div>
        </div>
      )}

      {/* Finger Assignment Chips */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-medium text-neutral-300">
            Assign Slot #{selectedSlot} to Finger:
          </span>
          <span className="text-amber-400 font-medium">
            {slotFinger[selectedSlot] || "Unassigned"}
          </span>
        </div>

        <div className="grid grid-cols-5 gap-1.5">
          {FINGERS.map((finger) => {
            const isAssigned = slotFinger[selectedSlot] === finger;
            return (
              <button
                key={finger}
                onClick={() => handleFingerAssign(finger)}
                className={`py-2 text-center text-xs font-medium rounded-lg border transition-all cursor-pointer ${
                  isAssigned
                    ? "bg-amber-500/25 border-amber-500 text-amber-300 font-semibold shadow-sm"
                    : "bg-neutral-800/80 border-neutral-700/80 text-neutral-400 hover:text-neutral-200 hover:border-neutral-600"
                }`}
              >
                {finger}
              </button>
            );
          })}
        </div>

        {slotFinger[selectedSlot] && (
          <button
            onClick={handleUnassign}
            className="w-full text-center text-[11px] text-neutral-500 hover:text-rose-400 transition-colors cursor-pointer py-1"
          >
            Unassign Slot #{selectedSlot}
          </button>
        )}
      </div>

      {/* Rotation & Orientation Fine-Tuning */}
      <div className="space-y-3 pt-2 border-t border-neutral-800">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">
            Orientation Correction
          </span>
          <button
            onClick={handleRotationFlip}
            className={`px-2.5 py-1 text-[11px] rounded-lg flex items-center gap-1.5 border transition-all cursor-pointer ${
              isInverted
                ? "bg-amber-500/20 border-amber-500/60 text-amber-300"
                : "bg-neutral-800 hover:bg-neutral-700 border-neutral-700 text-neutral-300"
            }`}
          >
            <RotateCw className="w-3 h-3" />
            <span>180° Flip ({isInverted ? "Inverted" : "Upright"})</span>
          </button>
        </div>

        {/* Fine Rotation Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs text-neutral-400">
            <span>Fine Angle Adjustment</span>
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-neutral-200">
                {currentRotation > 0 ? `+${currentRotation}` : currentRotation}°
              </span>
              {currentRotation !== 0 && (
                <button
                  onClick={handleResetRotation}
                  className="p-0.5 text-neutral-500 hover:text-neutral-300"
                  title="Reset to 0°"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
          <input
            type="range"
            min="-45"
            max="45"
            value={currentRotation}
            onChange={(e) => handleRotationChange(Number(e.target.value))}
            className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
        </div>
      </div>

      {/* Enlarged Cutout Inspection Modal */}
      {isPreviewModalOpen && activePreview && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 select-none"
          onClick={() => setIsPreviewModalOpen(false)}
        >
          <div
            className="bg-neutral-900 border border-neutral-700/80 rounded-2xl shadow-2xl max-w-md w-full overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="px-5 py-3.5 border-b border-neutral-800 flex items-center justify-between bg-neutral-950/60">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-amber-500/20 text-amber-400 font-bold text-xs flex items-center justify-center">
                  #{selectedSlot}
                </span>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100">
                    Slot #{selectedSlot} Cutout Inspection
                  </h3>
                  <span className="text-[11px] text-neutral-400">
                    {activeSlotData?.shape_info.shape} • {activeSlotData?.bbox[2]}×{activeSlotData?.bbox[3]}px • {activeSlotData?.pca_angle.toFixed(1)}° Deskew
                  </span>
                </div>
              </div>

              <button
                onClick={() => setIsPreviewModalOpen(false)}
                className="p-1.5 text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800 rounded-lg transition-colors cursor-pointer"
                title="Close (Esc)"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Checkerboard Transparency Canvas */}
            <div className="relative p-6 flex items-center justify-center bg-neutral-950 min-h-64 max-h-80 overflow-hidden">
              {/* Checkerboard Pattern */}
              <div
                className="absolute inset-0 opacity-20"
                style={{
                  backgroundImage: `linear-gradient(45deg, #333 25%, transparent 25%), linear-gradient(-45deg, #333 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #333 75%), linear-gradient(-45deg, transparent 75%, #333 75%)`,
                  backgroundSize: "16px 16px",
                  backgroundPosition: "0 0, 0 8px, 8px -8px, -8px 0px",
                }}
              />

              <img
                src={activePreview}
                alt={`Cutout #${selectedSlot}`}
                className="relative max-h-64 object-contain transition-transform duration-150 drop-shadow-2xl"
                style={{
                  transform: `scale(${modalZoom}) rotate(${currentRotation}deg) ${
                    isInverted ? "rotate(180deg)" : ""
                  }`,
                }}
              />

              {/* Floating Zoom Controls */}
              <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-neutral-900/90 backdrop-blur-md px-2 py-1 rounded-lg border border-neutral-800 shadow-md">
                <button
                  onClick={() => setModalZoom((prev) => Math.max(0.5, Number((prev - 0.25).toFixed(2))))}
                  className="p-1 hover:text-amber-400 text-neutral-400 cursor-pointer"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <span className="text-[10px] font-mono font-bold text-neutral-300 w-10 text-center">
                  {Math.round(modalZoom * 100)}%
                </span>
                <button
                  onClick={() => setModalZoom((prev) => Math.min(3, Number((prev + 0.25).toFixed(2))))}
                  className="p-1 hover:text-amber-400 text-neutral-400 cursor-pointer"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Quality Diagnosis Notice */}
            <div className="p-4 bg-neutral-900/90 border-t border-neutral-800 space-y-3">
              <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-200/90 leading-relaxed">
                <span className="font-semibold text-amber-300">Extraction Check:</span> If tray packaging, shadows, or neighboring nails are visible in the cutout, use the Pen Tool or Bounding Box handles to trim them out.
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between gap-2 pt-1">
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={handleRotationFlip}
                    className="px-2.5 py-1.5 bg-neutral-800 hover:bg-neutral-750 text-neutral-200 rounded-lg text-xs font-medium flex items-center gap-1.5 cursor-pointer border border-neutral-700 transition-all"
                    title="Flip orientation 180°"
                  >
                    <RotateCw className="w-3.5 h-3.5" />
                    <span>180° Flip</span>
                  </button>

                  <button
                    onClick={() => handleReExtract()}
                    disabled={isLoadingPreview}
                    className="px-2.5 py-1.5 bg-neutral-800 hover:bg-neutral-750 text-amber-300 rounded-lg text-xs font-medium flex items-center gap-1.5 cursor-pointer border border-neutral-700 transition-all disabled:opacity-50"
                    title="Re-extract fresh cutout from current contour (preserves all designs & black tips)"
                  >
                    {isLoadingPreview ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
                    ) : (
                      <Wand2 className="w-3.5 h-3.5 text-amber-400" />
                    )}
                    <span>Re-extract</span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setIsPreviewModalOpen(false);
                      setTrayState((prev) => ({
                        ...prev,
                        isManualEditMode: true,
                        activeTool: "select",
                      }));
                    }}
                    className="px-3 py-1.5 bg-neutral-800 hover:bg-neutral-750 text-neutral-200 rounded-lg text-xs font-medium flex items-center gap-1.5 cursor-pointer border border-neutral-700 transition-all"
                  >
                    <Move className="w-3.5 h-3.5" />
                    <span>Adjust Handles</span>
                  </button>

                  <button
                    onClick={() => {
                      setIsPreviewModalOpen(false);
                      setTrayState((prev) => ({
                        ...prev,
                        activeTool: "pen",
                        isManualEditMode: false,
                      }));
                    }}
                    className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-neutral-950 rounded-lg text-xs font-bold flex items-center gap-1.5 cursor-pointer transition-all shadow-md"
                  >
                    <PenTool className="w-3.5 h-3.5" />
                    <span>Edit with Pen Tool</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};