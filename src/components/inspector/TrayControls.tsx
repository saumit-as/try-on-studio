import React from "react";
import { RotateCw } from "lucide-react";
import { FingerName, TrayState } from "../../types";

interface TrayControlsProps {
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

export const TrayControls: React.FC<TrayControlsProps> = ({
  trayState,
  setTrayState,
}) => {
  const { selectedSlot, slotFinger, nailRotation } = trayState;

  const handleSlotSelect = (slot: number) => {
    setTrayState((prev) => ({ ...prev, selectedSlot: slot }));
  };

  const handleFingerAssign = (finger: FingerName) => {
    setTrayState((prev) => ({
      ...prev,
      slotFinger: {
        ...prev.slotFinger,
        [selectedSlot]: finger,
      },
    }));
  };

  const handleRotationFlip = () => {
    setTrayState((prev) => ({
      ...prev,
      nailRotation: (prev.nailRotation + 180) % 360,
    }));
  };

  const handleRotationChange = (val: number) => {
    setTrayState((prev) => ({ ...prev, nailRotation: val }));
  };

  return (
    <div className="p-5 space-y-6">
      <div>
        <h2 className="text-sm font-semibold text-neutral-200">Slot-to-Finger Mapping</h2>
        <p className="text-xs text-neutral-400 mt-0.5">Assign detected nails to hand positions</p>
      </div>

      {/* Slot Selector */}
      <div className="space-y-2">
        <span className="text-xs font-medium text-neutral-300">Active Tray Slot</span>
        <div className="grid grid-cols-5 gap-1.5">
          {[1, 2, 3, 4, 5].map((slot) => (
            <button
              key={slot}
              onClick={() => handleSlotSelect(slot)}
              className={`py-2 text-center text-xs font-medium rounded-lg border transition-colors cursor-pointer ${
                selectedSlot === slot
                  ? "bg-amber-500/20 border-amber-500/50 text-amber-300"
                  : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Slot {slot}
            </button>
          ))}
        </div>
      </div>

      {/* Finger Assignment Chips */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-medium text-neutral-300">Assign Slot {selectedSlot} to:</span>
          <span className="text-amber-400 font-mono">{slotFinger[selectedSlot] || "Unassigned"}</span>
        </div>
        <div className="grid grid-cols-5 gap-1.5">
          {FINGERS.map((finger) => (
            <button
              key={finger}
              onClick={() => handleFingerAssign(finger)}
              className={`py-2 text-center text-xs font-medium rounded-lg border transition-colors cursor-pointer ${
                slotFinger[selectedSlot] === finger
                  ? "bg-amber-500/20 border-amber-500/60 text-amber-300"
                  : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-amber-500/30"
              }`}
            >
              {finger[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Rotation & Orientation */}
      <div className="space-y-3 pt-2 border-t border-neutral-800">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">Orientation Correction</span>
          <button
            onClick={handleRotationFlip}
            className="px-2 py-1 bg-neutral-800 hover:bg-neutral-700 text-[11px] rounded flex items-center gap-1 border border-neutral-700 cursor-pointer"
          >
            <RotateCw className="w-3 h-3 text-amber-400" />
            Flip 180°
          </button>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1 text-neutral-400">
            <span>Fine Rotation</span>
            <span className="font-mono">{nailRotation}°</span>
          </div>
          <input
            type="range"
            min="-45"
            max="45"
            value={nailRotation}
            onChange={(e) => handleRotationChange(Number(e.target.value))}
            className="w-full accent-amber-500 bg-neutral-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
