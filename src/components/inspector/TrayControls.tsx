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
} from "lucide-react";
import { FingerName, NailShape, TrayState, DetectedSlot } from "../../types";
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

    callSidecar<{
      status: string;
      width: number;
      height: number;
      image_base64: string;
    }>("extract_nail", {
      image_base64: trayState.trayImageSrc,
      contour: activeSlotData.contour,
      deskew: true,
      auto_orient_cuticle_down: true,
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
          }));
        }
      })
      .catch(() => {})
      .finally(() => {
        if (isMounted) {
          setIsLoadingPreview(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [trayState.trayImageSrc, selectedSlot, activeSlotData, extractedPreviews, setTrayState]);

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
            <div className="w-16 h-20 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-center overflow-hidden relative shrink-0">
              {activePreview ? (
                <img
                  src={activePreview}
                  alt={`Nail #${selectedSlot}`}
                  className="max-w-full max-h-full object-contain"
                  style={{
                    transform: `rotate(${currentRotation}deg) ${
                      isInverted ? "rotate(180deg)" : ""
                    }`,
                  }}
                />
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
    </div>
  );
};