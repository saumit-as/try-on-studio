import React, { useRef, useState, DragEvent, ChangeEvent } from "react";
import {
  Sparkles,
  Layers,
  Hand,
  Download,
  Upload,
  Loader2,
  AlertCircle,
  RotateCw,
} from "lucide-react";
import { ActiveTab, TrayState, DetectedSlot, FingerName } from "../../types";
import { callSidecar } from "../../services/sidecar";

interface CanvasViewportProps {
  activeTab: ActiveTab;
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
}

const SLOT_PALETTE = [
  "#ec4899", // Pink
  "#eab308", // Yellow
  "#84cc16", // Lime
  "#3b82f6", // Blue
  "#f97316", // Orange
  "#a855f7", // Purple
  "#10b981", // Emerald
  "#06b6d4", // Cyan
  "#f43f5e", // Rose
  "#6366f1", // Indigo
];

export const CanvasViewport: React.FC<CanvasViewportProps> = ({
  activeTab,
  trayState,
  setTrayState,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setTrayState((prev) => ({
        ...prev,
        segmentError: "Please upload a valid image file (.png, .jpg, .jpeg, .webp).",
      }));
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = reader.result as string;
      setTrayState((prev) => ({
        ...prev,
        trayImageSrc: base64Data,
        isSegmenting: true,
        segmentError: null,
        detectedSlots: [],
        extractedPreviews: {},
        shapeOverrides: {},
        nailRotation: {},
        slotInverted: {},
        selectedSlot: 1,
      }));

      try {
        const res = await callSidecar<{
          status: string;
          slot_count: number;
          slots: DetectedSlot[];
          image_size: [number, number];
        }>("segment_tray", {
          image_base64: base64Data,
          max_slots: 15,
        });

        if (res && res.status === "ok") {
          // Default mapping: map first 5 slots to Thumb, Index, Middle, Ring, Pinky
          const defaultMapping: Record<number, FingerName> = {};
          const fingers: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
          res.slots.slice(0, 5).forEach((s, idx) => {
            defaultMapping[s.slot_id] = fingers[idx];
          });

          setTrayState((prev) => ({
            ...prev,
            isSegmenting: false,
            trayImageSize: res.image_size,
            detectedSlots: res.slots,
            selectedSlot: res.slots[0]?.slot_id ?? 1,
            slotFinger: defaultMapping,
          }));

          // Fetch preview for initial slot
          if (res.slots.length > 0) {
            fetchNailPreview(base64Data, res.slots[0]);
          }
        } else {
          throw new Error("Failed to detect nail slots in tray photo.");
        }
      } catch (err: any) {
        setTrayState((prev) => ({
          ...prev,
          isSegmenting: false,
          segmentError:
            err?.message ||
            "Error contacting Python sidecar. Make sure the sidecar daemon is active.",
        }));
      }
    };
    reader.readAsDataURL(file);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const fetchNailPreview = async (imageSrc: string, slot: DetectedSlot) => {
    try {
      const ext = await callSidecar<{
        status: string;
        width: number;
        height: number;
        image_base64: string;
      }>("extract_nail", {
        image_base64: imageSrc,
        contour: slot.contour,
        deskew: true,
        auto_orient_cuticle_down: true,
      });

      if (ext && ext.status === "ok" && ext.image_base64) {
        setTrayState((prev) => ({
          ...prev,
          extractedPreviews: {
            ...prev.extractedPreviews,
            [slot.slot_id]: ext.image_base64.startsWith("data:")
              ? ext.image_base64
              : `data:image/png;base64,${ext.image_base64}`,
          },
        }));
      }
    } catch {
      // preview failure is non-blocking
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleSlotClick = (slot: DetectedSlot) => {
    setTrayState((prev) => ({ ...prev, selectedSlot: slot.slot_id }));
    if (trayState.trayImageSrc && !trayState.extractedPreviews[slot.slot_id]) {
      fetchNailPreview(trayState.trayImageSrc, slot);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-neutral-950 relative overflow-hidden border-r border-neutral-800/80">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="flex-1 flex items-center justify-center p-6 relative">
        {/* Viewport Canvas Frame */}
        <div className="relative w-full max-w-4xl h-full max-h-[85vh] bg-neutral-900/60 rounded-2xl border border-neutral-800 flex items-center justify-center shadow-2xl overflow-hidden group">
          {activeTab === "tray" && (
            <div className="relative w-full h-full flex flex-col items-center justify-center">
              {!trayState.trayImageSrc ? (
                /* Empty / Upload State */
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                  className={`w-full h-full flex flex-col items-center justify-center text-center p-8 cursor-pointer transition-all ${
                    isDragOver
                      ? "bg-amber-500/10 border-2 border-dashed border-amber-500"
                      : "hover:bg-neutral-900/80 border border-transparent"
                  }`}
                >
                  <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-4">
                    <Layers className="w-8 h-8 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-neutral-200 mb-1">
                    Upload Packaging Tray Photo
                  </h3>
                  <p className="text-xs text-neutral-400 max-w-sm mb-5">
                    Drag and drop your press-on packaging photo, or click to browse files.
                    Our CV engine will automatically isolate slots, deskew angles, and fit $G^2$ almond/coffin contours.
                  </p>
                  <div className="flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-xs font-medium rounded-lg border border-neutral-700 transition-colors">
                    <Upload className="w-4 h-4 text-amber-400" />
                    <span>Choose Image File</span>
                  </div>
                </div>
              ) : (
                /* Active Tray Canvas with Interactive SVG Contour Overlay */
                <div className="relative w-full h-full flex items-center justify-center p-4">
                  {trayState.isSegmenting && (
                    <div className="absolute inset-0 bg-neutral-950/70 backdrop-blur-sm z-30 flex flex-col items-center justify-center">
                      <Loader2 className="w-8 h-8 text-amber-400 animate-spin mb-3" />
                      <p className="text-sm font-medium text-neutral-200">
                        Analyzing Tray Geometry...
                      </p>
                      <p className="text-xs text-neutral-400 mt-1">
                        Detecting contours, card homography & $G^2$ continuous super-ellipses
                      </p>
                    </div>
                  )}

                  {trayState.segmentError && (
                    <div className="absolute top-4 left-4 right-4 z-40 bg-rose-950/80 border border-rose-800/80 backdrop-blur-md p-3 rounded-xl flex items-center gap-3 text-xs text-rose-200">
                      <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                      <div className="flex-1">{trayState.segmentError}</div>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-2.5 py-1 bg-rose-900 hover:bg-rose-800 text-[11px] font-medium rounded-lg border border-rose-700 cursor-pointer"
                      >
                        Try Another Image
                      </button>
                    </div>
                  )}

                  {/* Tray Image & SVG Contour Overlay */}
                  <div className="relative max-w-full max-h-full flex items-center justify-center shadow-lg rounded-xl overflow-hidden border border-neutral-800">
                    <img
                      src={trayState.trayImageSrc}
                      alt="Packaging Tray"
                      className="max-w-full max-h-[75vh] object-contain select-none pointer-events-none"
                    />

                    {trayState.trayImageSize && (
                      <svg
                        className="absolute inset-0 w-full h-full"
                        viewBox={`0 0 ${trayState.trayImageSize[0]} ${trayState.trayImageSize[1]}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {trayState.detectedSlots.map((slot, index) => {
                          const isSelected = trayState.selectedSlot === slot.slot_id;
                          const isHovered = trayState.hoveredSlot === slot.slot_id;
                          const color = SLOT_PALETTE[index % SLOT_PALETTE.length];
                          const pointsString = slot.contour
                            .map((pt) => `${pt[0]},${pt[1]}`)
                            .join(" ");

                          const [bx, by, bw, bh] = slot.bbox;
                          const centerX = bx + bw / 2;
                          const centerY = by + bh / 2;
                          const assignedFinger = trayState.slotFinger[slot.slot_id];

                          return (
                            <g
                              key={slot.slot_id}
                              onClick={() => handleSlotClick(slot)}
                              onMouseEnter={() =>
                                setTrayState((prev) => ({
                                  ...prev,
                                  hoveredSlot: slot.slot_id,
                                }))
                              }
                              onMouseLeave={() =>
                                setTrayState((prev) => ({
                                  ...prev,
                                  hoveredSlot: null,
                                }))
                              }
                              className="cursor-pointer transition-all"
                            >
                              {/* Glowing Contour Polygon */}
                              <polygon
                                points={pointsString}
                                fill={
                                  isSelected
                                    ? "rgba(245, 158, 11, 0.42)"
                                    : isHovered
                                    ? "rgba(245, 158, 11, 0.22)"
                                    : "rgba(0, 0, 0, 0.15)"
                                }
                                stroke={isSelected ? "#f59e0b" : isHovered ? "#fbbf24" : color}
                                strokeWidth={isSelected ? 4 : isHovered ? 3 : 2}
                                strokeLinejoin="round"
                                className="transition-all"
                              />

                              {/* Center Number & Finger Badge */}
                              <g transform={`translate(${centerX}, ${centerY})`}>
                                <rect
                                  x="-24"
                                  y="-12"
                                  width="48"
                                  height="24"
                                  rx="12"
                                  fill={isSelected ? "#f59e0b" : "#171717"}
                                  stroke={isSelected ? "#ffffff" : color}
                                  strokeWidth="1.5"
                                  className="shadow-md"
                                />
                                <text
                                  textAnchor="middle"
                                  dominantBaseline="central"
                                  fill={isSelected ? "#000000" : "#ffffff"}
                                  fontSize="10"
                                  fontWeight="bold"
                                  fontFamily="sans-serif"
                                >
                                  {assignedFinger ? assignedFinger[0] : `#${slot.slot_id}`}
                                </text>
                              </g>
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>

                  {/* Floating Action Controls */}
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-xs text-neutral-300 bg-neutral-900/85 backdrop-blur-md px-4 py-2 rounded-xl border border-neutral-800 shadow-xl">
                    <div className="flex items-center gap-3">
                      <span className="font-semibold text-neutral-200">
                        {trayState.detectedSlots.length} Slots Isolated
                      </span>
                      <span className="text-neutral-500">•</span>
                      <span className="text-amber-400">
                        {Object.keys(trayState.slotFinger).length} Assigned to Hand
                      </span>
                    </div>

                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-3 py-1 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded-lg border border-neutral-700 transition-colors cursor-pointer flex items-center gap-1.5"
                    >
                      <RotateCw className="w-3.5 h-3.5 text-neutral-400" />
                      <span>Replace Photo</span>
                    </button>
                  </div>
                </div>
              )}
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
