import React from "react";
import {
  RotateCw,
  Wand2,
  ChevronDown,
  RotateCcw,
} from "lucide-react";
import { FingerName, NailShape, TrayState, DetectedSlot } from "../../types";

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
  const {
    selectedSlot,
    slotFinger,
    nailRotation,
    slotInverted,
    shapeOverrides,
    detectedSlots,
    extractedPreviews,
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

  const handleSlotSelect = (slotId: number) => {
    setTrayState((prev) => ({ ...prev, selectedSlot: slotId }));
  };

  const handleFingerAssign = (finger: FingerName) => {
    setTrayState((prev) => {
      const nextMapping = { ...prev.slotFinger };
      // If finger is already assigned elsewhere, unassign it to prevent duplicates
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

  const activePreview = extractedPreviews[selectedSlot];

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
                <div className="text-[9px] text-neutral-600 text-center px-1">
                  Cutout preview
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
