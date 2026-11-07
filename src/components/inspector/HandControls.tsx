import React, { useRef, useState } from "react";
import {
  Hand,
  CheckCircle2,
  Upload,
  RotateCcw,
  Wand2,
  Loader2,
  Eye,
  EyeOff,
  Sliders,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Download,
  FolderOpen,
  Sparkles,
  Crop,
} from "lucide-react";
import { FingerName, HandState, LandmarkKey } from "../../types";
import {
  DEFAULT_STUDIO_LANDMARKS,
  FINGER_COLORS,
  INITIAL_DEFAULT_HAND_TEMPLATE,
  LANDMARK_LABELS,
} from "../../constants/handDefaults";
import { callSidecar } from "../../services/sidecar";

interface HandControlsProps {
  handState: HandState;
  setHandState: React.Dispatch<React.SetStateAction<HandState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const LANDMARK_KEYS: LandmarkKey[] = ["c_c", "c_l", "c_r", "d_l", "d_r"];

export const HandControls: React.FC<HandControlsProps> = ({
  handState,
  setHandState,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const [nudgeStep, setNudgeStep] = useState<number>(5);
  const [isAutoDetecting, setIsAutoDetecting] = useState<boolean>(false);

  const {
    currentTemplate,
    activeFinger,
    selectedLandmark,
    isGeneratingMaps,
    generateError,
    overlayToggles,
  } = handState;

  const activeFingerName: FingerName =
    activeFinger === "All" ? "Middle" : activeFinger;

  const currentLandmarks =
    currentTemplate?.landmarks?.[activeFingerName] ||
    DEFAULT_STUDIO_LANDMARKS[activeFingerName];

  const currentSelectedKey: LandmarkKey =
    selectedLandmark?.finger === activeFingerName
      ? selectedLandmark.key
      : "c_c";

  const currentCoords = currentLandmarks[currentSelectedKey] || [0, 0];

  // Auto-detect hand landmarks using Computer Vision
  const handleAutoDetectFingers = async (
    imageSrcOverride?: string,
    sizeOverride?: [number, number],
    handednessOverride?: "left" | "right"
  ) => {
    const src = imageSrcOverride || currentTemplate?.imageSrc;
    if (!src) return;

    setIsAutoDetecting(true);
    setHandState((prev) => ({ ...prev, generateError: null }));

    try {
      const targetHandedness =
        handednessOverride || currentTemplate?.handedness || "auto";
      const res = await callSidecar<{
        status: string;
        landmarks: Record<FingerName, any>;
        handedness: "left" | "right";
        palm_center: [number, number];
        palm_radius: number;
      }>("auto_detect_hand", {
        image_base64: src,
        handedness: targetHandedness,
      });

      if (res && res.status === "ok" && res.landmarks) {
        setHandState((prev) => {
          if (!prev.currentTemplate) return prev;
          return {
            ...prev,
            currentTemplate: {
              ...prev.currentTemplate,
              imageSize: sizeOverride || prev.currentTemplate.imageSize,
              landmarks: res.landmarks,
              handedness:
                res.handedness ||
                handednessOverride ||
                prev.currentTemplate.handedness ||
                "left",
            },
          };
        });
      } else {
        setHandState((prev) => ({
          ...prev,
          generateError:
            "Could not automatically detect hand contour. Proportional fallback placed.",
        }));
      }
    } catch (err: any) {
      console.warn("Auto-detect hand error:", err);
      setHandState((prev) => ({
        ...prev,
        generateError: err?.message || "Failed to auto-detect hand landmarks.",
      }));
    } finally {
      setIsAutoDetecting(false);
    }
  };

  const handleToggleHandedness = (newHandedness: "left" | "right") => {
    if (!currentTemplate) return;
    setHandState((prev) => {
      if (!prev.currentTemplate) return prev;
      return {
        ...prev,
        currentTemplate: {
          ...prev.currentTemplate,
          handedness: newHandedness,
        },
      };
    });
    handleAutoDetectFingers(undefined, undefined, newHandedness);
  };

  // Handle custom hand photo upload
  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (loadEvt) => {
      const src = loadEvt.target?.result as string;
      const img = new Image();
      img.onload = () => {
        const w = img.naturalWidth || 1536;
        const h = img.naturalHeight || 2752;

        // Auto-scale default landmarks as instant responsive fallback
        const scaleX = w / 1536;
        const scaleY = h / 2752;

        const scaledLandmarks: Record<FingerName, any> = JSON.parse(
          JSON.stringify(DEFAULT_STUDIO_LANDMARKS)
        );
        for (const f of FINGERS) {
          scaledLandmarks[f].c_l = [
            Math.round(scaledLandmarks[f].c_l[0] * scaleX),
            Math.round(scaledLandmarks[f].c_l[1] * scaleY),
          ];
          scaledLandmarks[f].c_c = [
            Math.round(scaledLandmarks[f].c_c[0] * scaleX),
            Math.round(scaledLandmarks[f].c_c[1] * scaleY),
          ];
          scaledLandmarks[f].c_r = [
            Math.round(scaledLandmarks[f].c_r[0] * scaleX),
            Math.round(scaledLandmarks[f].c_r[1] * scaleY),
          ];
          scaledLandmarks[f].d_l = [
            Math.round(scaledLandmarks[f].d_l[0] * scaleX),
            Math.round(scaledLandmarks[f].d_l[1] * scaleY),
          ];
          scaledLandmarks[f].d_r = [
            Math.round(scaledLandmarks[f].d_r[0] * scaleX),
            Math.round(scaledLandmarks[f].d_r[1] * scaleY),
          ];
        }

        setHandState((prev) => ({
          ...prev,
          currentTemplate: {
            name: file.name.replace(/\.[^/.]+$/, ""),
            imageSrc: src,
            imageSize: [w, h],
            landmarks: scaledLandmarks,
            cuticleMaskSrc: undefined,
            aoMapSrc: undefined,
            specularMapSrc: undefined,
            isDefault: false,
          },
        }));

        // Automatically compute and snap the anchors to the real fingers!
        handleAutoDetectFingers(src, [w, h]);
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  // Revert back to bundled default studio hand template
  const handleResetToDefaultTemplate = () => {
    setHandState((prev) => ({
      ...prev,
      currentTemplate: JSON.parse(JSON.stringify(INITIAL_DEFAULT_HAND_TEMPLATE)),
      generateError: null,
    }));
  };

  // Reset landmarks of active finger or all fingers
  const handleResetLandmarks = () => {
    if (!currentTemplate) return;
    setHandState((prev) => {
      if (!prev.currentTemplate) return prev;
      const updated = { ...prev.currentTemplate };
      if (activeFinger === "All") {
        updated.landmarks = JSON.parse(
          JSON.stringify(DEFAULT_STUDIO_LANDMARKS)
        );
      } else {
        updated.landmarks[activeFinger] = JSON.parse(
          JSON.stringify(DEFAULT_STUDIO_LANDMARKS[activeFinger])
        );
      }
      return {
        ...prev,
        currentTemplate: updated,
      };
    });
  };

  // Nudge selected landmark coordinates with D-pad
  const handleNudge = (dx: number, dy: number) => {
    if (!currentTemplate) return;
    setHandState((prev) => {
      if (!prev.currentTemplate) return prev;
      const finger = activeFingerName;
      const key = currentSelectedKey;
      const curr = prev.currentTemplate.landmarks[finger][key];
      const updatedCoords: [number, number] = [
        Math.round(curr[0] + dx),
        Math.round(curr[1] + dy),
      ];

      const nextTemplate = {
        ...prev.currentTemplate,
        landmarks: {
          ...prev.currentTemplate.landmarks,
          [finger]: {
            ...prev.currentTemplate.landmarks[finger],
            [key]: updatedCoords,
          },
        },
      };

      return {
        ...prev,
        currentTemplate: nextTemplate,
        selectedLandmark: { finger, key },
      };
    });
  };

  // Update slider parameters for active finger (C-curve, roll offset)
  const handleFingerParamChange = (
    param: "c_curve" | "roll_offset",
    val: number
  ) => {
    if (!currentTemplate) return;
    setHandState((prev) => {
      if (!prev.currentTemplate) return prev;
      const finger = activeFingerName;
      return {
        ...prev,
        currentTemplate: {
          ...prev.currentTemplate,
          landmarks: {
            ...prev.currentTemplate.landmarks,
            [finger]: {
              ...prev.currentTemplate.landmarks[finger],
              [param]: val,
            },
          },
        },
      };
    });
  };

  // Generate procedural maps via Python sidecar
  const handleGenerateRigMaps = async () => {
    if (!currentTemplate) return;
    setHandState((prev) => ({
      ...prev,
      isGeneratingMaps: true,
      generateError: null,
    }));

    try {
      const res = await callSidecar<{
        status: string;
        cuticle_mask_base64?: string;
        ao_map_base64?: string;
        specular_map_base64?: string;
      }>("generate_hand_maps", {
        image_base64: currentTemplate.imageSrc,
        landmarks: currentTemplate.landmarks,
      });

      if (res && res.status === "ok") {
        setHandState((prev) => {
          if (!prev.currentTemplate) return prev;
          return {
            ...prev,
            isGeneratingMaps: false,
            generateError: null,
            currentTemplate: {
              ...prev.currentTemplate,
              cuticleMaskSrc: res.cuticle_mask_base64,
              aoMapSrc: res.ao_map_base64,
              specularMapSrc: res.specular_map_base64,
            },
            overlayToggles: {
              ...prev.overlayToggles,
              showCuticleMask: true,
              showAoMap: true,
            },
          };
        });
      } else {
        setHandState((prev) => ({
          ...prev,
          isGeneratingMaps: false,
          generateError: "Failed to generate rig maps from sidecar.",
        }));
      }
    } catch (err: any) {
      setHandState((prev) => ({
        ...prev,
        isGeneratingMaps: false,
        generateError: err?.message || "Sidecar communication error.",
      }));
    }
  };

  // Export current template as .nailtemplate JSON file
  const handleExportTemplate = () => {
    if (!currentTemplate) return;
    const exportBundle = {
      format: "nailtemplate",
      version: "1.0.0",
      name: currentTemplate.name,
      image_size: currentTemplate.imageSize,
      landmarks: currentTemplate.landmarks,
      hand_image_base64: currentTemplate.imageSrc,
      cuticle_mask_base64: currentTemplate.cuticleMaskSrc,
      ao_map_base64: currentTemplate.aoMapSrc,
      specular_map_base64: currentTemplate.specularMapSrc,
    };

    const blob = new Blob([JSON.stringify(exportBundle, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentTemplate.name.toLowerCase().replace(/\s+/g, "_")}.nailtemplate`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import .nailtemplate JSON file
  const handleImportTemplate = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (loadEvt) => {
      try {
        const parsed = JSON.parse(loadEvt.target?.result as string);
        if (!parsed.landmarks) {
          throw new Error("Invalid .nailtemplate: missing landmarks structure.");
        }
        setHandState((prev) => ({
          ...prev,
          currentTemplate: {
            name: parsed.name || file.name.replace(/\.[^/.]+$/, ""),
            imageSrc: parsed.hand_image_base64 || "/templates/default_studio_hand/hand.jpg",
            imageSize: parsed.image_size || [1536, 2752],
            landmarks: parsed.landmarks,
            cuticleMaskSrc: parsed.cuticle_mask_base64,
            aoMapSrc: parsed.ao_map_base64,
            specularMapSrc: parsed.specular_map_base64,
            isDefault: false,
          },
        }));
      } catch (err: any) {
        setHandState((prev) => ({
          ...prev,
          generateError: `Failed to import template: ${err.message}`,
        }));
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  return (
    <div className="p-5 space-y-6">
      {/* Hidden File Inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handlePhotoUpload}
      />
      <input
        ref={templateInputRef}
        type="file"
        accept=".nailtemplate,.json"
        className="hidden"
        onChange={handleImportTemplate}
      />

      {/* Header */}
      <div>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Hand className="w-4 h-4 text-rose-400" />
            Hand Template Rigging
          </h2>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
            Phase 3
          </span>
        </div>
        <p className="text-xs text-neutral-400 mt-0.5">
          Calibrate 5-point anatomical anchors & procedural rig passes
        </p>
      </div>

      {/* Template Card */}
      <div className="p-3.5 rounded-xl bg-neutral-900 border border-neutral-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">Active Template</span>
          <span className="text-[11px] font-mono text-neutral-400">
            {currentTemplate?.imageSize[0]} × {currentTemplate?.imageSize[1]} px
          </span>
        </div>

        <div className="flex items-center gap-3 p-2 rounded-lg bg-neutral-950 border border-neutral-800/80">
          <div className="w-12 h-14 rounded-md bg-neutral-900 overflow-hidden border border-neutral-800 shrink-0 flex items-center justify-center">
            {currentTemplate?.imageSrc ? (
              <img
                src={currentTemplate.imageSrc}
                alt="Hand"
                className="w-full h-full object-cover"
              />
            ) : (
              <Hand className="w-5 h-5 text-neutral-600" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-xs font-semibold text-neutral-200 block truncate">
              {currentTemplate?.name || "Default Studio Hand"}
            </span>
            <span className="text-[11px] text-emerald-400 flex items-center gap-1 mt-0.5">
              <CheckCircle2 className="w-3 h-3" />
              Pre-Calibrated (5/5 Nails Rigged)
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="py-1.5 px-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium rounded-lg border border-neutral-700 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5 text-neutral-400" />
            Custom Photo
          </button>
          <button
            onClick={handleResetToDefaultTemplate}
            disabled={currentTemplate?.isDefault}
            className="py-1.5 px-2 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed text-neutral-200 text-xs font-medium rounded-lg border border-neutral-700 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-neutral-400" />
            Studio Hand
          </button>
        </div>

        {/* Handedness Selection */}
        <div className="flex items-center justify-between bg-neutral-950 p-2 rounded-lg border border-neutral-800 text-xs">
          <span className="text-neutral-400 font-medium">Hand Side:</span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => handleToggleHandedness("left")}
              className={`px-2.5 py-1 rounded text-xs font-medium cursor-pointer transition-colors ${
                (currentTemplate?.handedness || "left") === "left"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "bg-neutral-800 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Left Hand
            </button>
            <button
              onClick={() => handleToggleHandedness("right")}
              className={`px-2.5 py-1 rounded text-xs font-medium cursor-pointer transition-colors ${
                currentTemplate?.handedness === "right"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "bg-neutral-800 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              Right Hand
            </button>
          </div>
        </div>

        {/* Bounding Box Assistive Refit Button */}
        <button
          onClick={() =>
            setHandState((p) => ({
              ...p,
              isBoxToolActive: !p.isBoxToolActive,
            }))
          }
          className={`w-full py-2 text-xs font-medium rounded-lg border flex items-center justify-center gap-2 transition-all cursor-pointer ${
            handState.isBoxToolActive
              ? "bg-rose-600/30 text-rose-200 border-rose-500 shadow-sm animate-pulse"
              : "bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border-neutral-700"
          }`}
          title="Drag a bounding box over any fingertip on the canvas to auto-compute its 5 nail anchors"
        >
          <Crop className="w-3.5 h-3.5 text-rose-400" />
          {handState.isBoxToolActive
            ? `Box Tool Active: Drag Box Over ${activeFingerName}`
            : `Box Select & Refit: ${activeFingerName}`}
        </button>

        {/* Auto-Detect Hand Landmarks Button */}
        <button
          onClick={() => handleAutoDetectFingers()}
          disabled={isAutoDetecting}
          className="w-full py-2 bg-gradient-to-r from-amber-600/25 to-rose-600/25 hover:from-amber-600/40 hover:to-rose-600/40 text-amber-200 text-xs font-medium rounded-lg border border-amber-500/30 flex items-center justify-center gap-2 transition-all cursor-pointer shadow-sm disabled:opacity-50"
        >
          {isAutoDetecting ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
              Detecting Hand & Nails (CV)...
            </>
          ) : (
            <>
              <Wand2 className="w-3.5 h-3.5 text-amber-400" />
              Auto-Detect Hand & Nails (CV)
            </>
          )}
        </button>
      </div>

      {/* Finger Navigation Chips */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">Target Finger</span>
          <span className="text-[11px] text-neutral-400">
            {activeFinger === "All" ? "Showing all 5 fingers" : `Editing ${activeFinger}`}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-1.5">
          <button
            onClick={() => setHandState((p) => ({ ...p, activeFinger: "All" }))}
            className={`py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors cursor-pointer text-center ${
              activeFinger === "All"
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                : "bg-neutral-900 text-neutral-400 border-neutral-800 hover:bg-neutral-800 hover:text-neutral-200"
            }`}
          >
            All Fingers
          </button>
          {FINGERS.map((fname) => {
            const col = FINGER_COLORS[fname];
            const isActive = activeFinger === fname;
            return (
              <button
                key={fname}
                onClick={() => setHandState((p) => ({ ...p, activeFinger: fname }))}
                className={`py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors cursor-pointer flex items-center justify-center gap-1.5 ${
                  isActive
                    ? "bg-neutral-800 text-neutral-100 border-neutral-600 shadow-sm"
                    : "bg-neutral-900 text-neutral-400 border-neutral-800 hover:bg-neutral-800 hover:text-neutral-200"
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: col.hex }}
                />
                {fname}
              </button>
            );
          })}
        </div>
      </div>

      {/* 5-Point Landmark Precision Coordinates & D-Pad */}
      <div className="p-3.5 rounded-xl bg-neutral-900 border border-neutral-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300">
            5-Point Anchors ({activeFingerName})
          </span>
          <div className="flex items-center gap-1 bg-neutral-950 px-1.5 py-0.5 rounded border border-neutral-800 text-[10px] font-mono">
            <span className="text-neutral-500">Step:</span>
            <button
              onClick={() => setNudgeStep(1)}
              className={`px-1 rounded ${
                nudgeStep === 1 ? "bg-rose-500/30 text-rose-300" : "text-neutral-400"
              }`}
            >
              1px
            </button>
            <button
              onClick={() => setNudgeStep(5)}
              className={`px-1 rounded ${
                nudgeStep === 5 ? "bg-rose-500/30 text-rose-300" : "text-neutral-400"
              }`}
            >
              5px
            </button>
          </div>
        </div>

        {/* Landmark Selection List */}
        <div className="space-y-1.5">
          {LANDMARK_KEYS.map((key) => {
            const info = LANDMARK_LABELS[key];
            const isSelected = currentSelectedKey === key;
            const coords = currentLandmarks[key] || [0, 0];

            return (
              <button
                key={key}
                onClick={() =>
                  setHandState((p) => ({
                    ...p,
                    selectedLandmark: { finger: activeFingerName, key },
                  }))
                }
                className={`w-full py-1.5 px-2.5 rounded-lg text-xs flex items-center justify-between border transition-all cursor-pointer ${
                  isSelected
                    ? "bg-rose-500/10 border-rose-500/40 text-rose-200 font-medium"
                    : "bg-neutral-950/60 border-neutral-800/80 text-neutral-300 hover:bg-neutral-800/50"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isSelected ? "bg-rose-400" : "bg-neutral-500"
                    }`}
                  />
                  <span className="truncate">{info.title}</span>
                </div>
                <span className="font-mono text-[11px] text-neutral-400 shrink-0">
                  {Math.round(coords[0])}, {Math.round(coords[1])}
                </span>
              </button>
            );
          })}
        </div>

        {/* Directional Nudge D-Pad */}
        <div className="pt-2 border-t border-neutral-800/80 flex items-center justify-between">
          <div className="text-[11px] text-neutral-400">
            Nudge <span className="font-semibold text-neutral-200">{LANDMARK_LABELS[currentSelectedKey]?.short}</span>
            <span className="font-mono text-[10px] text-neutral-500 block">
              ({Math.round(currentCoords[0])}, {Math.round(currentCoords[1])})
            </span>
          </div>

          <div className="grid grid-cols-3 gap-1 w-24">
            <div />
            <button
              onClick={() => handleNudge(0, -nudgeStep)}
              className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 flex items-center justify-center cursor-pointer border border-neutral-700"
              title={`Up ${nudgeStep}px`}
            >
              <ArrowUp className="w-3.5 h-3.5" />
            </button>
            <div />
            <button
              onClick={() => handleNudge(-nudgeStep, 0)}
              className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 flex items-center justify-center cursor-pointer border border-neutral-700"
              title={`Left ${nudgeStep}px`}
            >
              <ArrowLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => handleNudge(0, nudgeStep)}
              className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 flex items-center justify-center cursor-pointer border border-neutral-700"
              title={`Down ${nudgeStep}px`}
            >
              <ArrowDown className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => handleNudge(nudgeStep, 0)}
              className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 flex items-center justify-center cursor-pointer border border-neutral-700"
              title={`Right ${nudgeStep}px`}
            >
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Anatomical Finger Calibration */}
      <div className="p-3.5 rounded-xl bg-neutral-900 border border-neutral-800 space-y-3">
        <span className="text-xs font-medium text-neutral-300 flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-neutral-400" />
          Anatomical Rig ({activeFingerName})
        </span>

        {/* Transverse C-Curve */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-neutral-400">Transverse C-Curve</span>
            <span className="font-mono text-neutral-200">
              {(currentLandmarks.c_curve ?? 0.28).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.65"
            step="0.02"
            value={currentLandmarks.c_curve ?? 0.28}
            onChange={(e) =>
              handleFingerParamChange("c_curve", parseFloat(e.target.value))
            }
            className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
          />
        </div>

        {/* Yaw / Roll Perspective Offset */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-neutral-400">Yaw / Profile Roll</span>
            <span className="font-mono text-neutral-200">
              {(currentLandmarks.roll_offset ?? 0.0).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="-0.50"
            max="0.50"
            step="0.05"
            value={currentLandmarks.roll_offset ?? 0.0}
            onChange={(e) =>
              handleFingerParamChange("roll_offset", parseFloat(e.target.value))
            }
            className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
          />
          <span className="text-[10px] text-neutral-500 block">
            Used for 3/4 side profile fingers (e.g. Thumb).
          </span>
        </div>
      </div>

      {/* Procedural Rig Maps Generator */}
      <div className="p-3.5 rounded-xl bg-neutral-900 border border-neutral-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-neutral-300 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Procedural Rig Passes
          </span>
        </div>

        {generateError && (
          <div className="p-2 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-[11px]">
            {generateError}
          </div>
        )}

        <button
          onClick={handleGenerateRigMaps}
          disabled={isGeneratingMaps}
          className="w-full py-2 bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white font-medium text-xs rounded-lg transition-all shadow-md flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {isGeneratingMaps ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Computing Rig Maps...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              Generate Rig Maps
            </>
          )}
        </button>

        {/* Map Inspection Cards */}
        <div className="space-y-2 pt-1 text-xs">
          {/* Cuticle Occlusion Mask */}
          <div className="p-2 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded bg-neutral-900 border border-neutral-800 overflow-hidden shrink-0 flex items-center justify-center">
                {currentTemplate?.cuticleMaskSrc ? (
                  <img
                    src={currentTemplate.cuticleMaskSrc}
                    alt="Cuticle Mask"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-3 h-3 bg-neutral-700 rounded-sm" />
                )}
              </div>
              <div className="min-w-0">
                <span className="text-neutral-200 block truncate font-medium text-[11px]">
                  Cuticle Occlusion Mask
                </span>
                <span className="text-[10px] text-neutral-500 block truncate">
                  Eponychium skin fold tuck
                </span>
              </div>
            </div>
            <button
              onClick={() =>
                setHandState((p) => ({
                  ...p,
                  overlayToggles: {
                    ...p.overlayToggles,
                    showCuticleMask: !p.overlayToggles.showCuticleMask,
                  },
                }))
              }
              className={`p-1 rounded border transition-colors cursor-pointer ${
                overlayToggles.showCuticleMask
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  : "bg-neutral-800 text-neutral-500 border-neutral-700"
              }`}
              title="Toggle Cuticle Mask Overlay"
            >
              {overlayToggles.showCuticleMask ? (
                <Eye className="w-3.5 h-3.5" />
              ) : (
                <EyeOff className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {/* Contact AO Map */}
          <div className="p-2 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded bg-neutral-900 border border-neutral-800 overflow-hidden shrink-0 flex items-center justify-center">
                {currentTemplate?.aoMapSrc ? (
                  <img
                    src={currentTemplate.aoMapSrc}
                    alt="AO Map"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-3 h-3 bg-neutral-700 rounded-sm" />
                )}
              </div>
              <div className="min-w-0">
                <span className="text-neutral-200 block truncate font-medium text-[11px]">
                  Contact AO Shadow Map
                </span>
                <span className="text-[10px] text-neutral-500 block truncate">
                  Cuticle seam ambient depth
                </span>
              </div>
            </div>
            <button
              onClick={() =>
                setHandState((p) => ({
                  ...p,
                  overlayToggles: {
                    ...p.overlayToggles,
                    showAoMap: !p.overlayToggles.showAoMap,
                  },
                }))
              }
              className={`p-1 rounded border transition-colors cursor-pointer ${
                overlayToggles.showAoMap
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  : "bg-neutral-800 text-neutral-500 border-neutral-700"
              }`}
              title="Toggle Contact AO Shadow Overlay"
            >
              {overlayToggles.showAoMap ? (
                <Eye className="w-3.5 h-3.5" />
              ) : (
                <EyeOff className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {/* Specular Gloss Map */}
          <div className="p-2 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded bg-neutral-900 border border-neutral-800 overflow-hidden shrink-0 flex items-center justify-center">
                {currentTemplate?.specularMapSrc ? (
                  <img
                    src={currentTemplate.specularMapSrc}
                    alt="Specular Map"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-3 h-3 bg-neutral-700 rounded-sm" />
                )}
              </div>
              <div className="min-w-0">
                <span className="text-neutral-200 block truncate font-medium text-[11px]">
                  Specular Gloss Ribbon
                </span>
                <span className="text-[10px] text-neutral-500 block truncate">
                  Salon gel topcoat highlight
                </span>
              </div>
            </div>
            <button
              onClick={() =>
                setHandState((p) => ({
                  ...p,
                  overlayToggles: {
                    ...p.overlayToggles,
                    showSpecular: !p.overlayToggles.showSpecular,
                  },
                }))
              }
              className={`p-1 rounded border transition-colors cursor-pointer ${
                overlayToggles.showSpecular
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  : "bg-neutral-800 text-neutral-500 border-neutral-700"
              }`}
              title="Toggle Specular Gloss Overlay"
            >
              {overlayToggles.showSpecular ? (
                <Eye className="w-3.5 h-3.5" />
              ) : (
                <EyeOff className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Template Persistence & Actions */}
      <div className="pt-2 border-t border-neutral-800 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => templateInputRef.current?.click()}
            className="py-1.5 px-2 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 text-xs font-medium rounded-lg border border-neutral-800 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <FolderOpen className="w-3.5 h-3.5 text-neutral-400" />
            Import .nailtemplate
          </button>
          <button
            onClick={handleExportTemplate}
            className="py-1.5 px-2 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 text-xs font-medium rounded-lg border border-neutral-800 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-neutral-400" />
            Export Bundle
          </button>
        </div>

        <button
          onClick={handleResetLandmarks}
          className="w-full py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors text-neutral-200 cursor-pointer"
        >
          Reset {activeFinger === "All" ? "All Anchors" : `${activeFinger} Anchors`}
        </button>
      </div>
    </div>
  );
};
