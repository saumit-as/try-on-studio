import React, {
  useRef,
  useState,
  useEffect,
  useCallback,
  DragEvent,
  ChangeEvent,
  WheelEvent,
  MouseEvent as ReactMouseEvent,
} from "react";
import {
  Sparkles,
  Download,
  Upload,
  Loader2,
  AlertCircle,
  RotateCw,
  Sliders,
  ZoomIn,
  ZoomOut,
  Maximize,
  MousePointer,
  PenTool,
  Hand,
  Undo2,
  Check,
  X,
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

interface DragState {
  mode: "move" | "resize" | "landmark";
  handle?: string;
  landmarkIdx?: number;
  startX: number;
  startY: number;
  origBbox: [number, number, number, number];
  origContour: [number, number][];
}

export const CanvasViewport: React.FC<CanvasViewportProps> = ({
  activeTab,
  trayState,
  setTrayState,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragState = useRef<DragState | null>(null);

  const [isDragOver, setIsDragOver] = useState(false);

  // Pan & Zoom Engine State
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ startX: number; startY: number; initPanX: number; initPanY: number } | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState<boolean>(false);

  // Pen Tool Drawing State
  const [penPoints, setPenPoints] = useState<[number, number][]>([]);
  const [currentMousePos, setCurrentMousePos] = useState<[number, number] | null>(null);
  const [isNearStartPoint, setIsNearStartPoint] = useState<boolean>(false);

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
        isManualEditMode: false,
        originalSlots: {},
        activeTool: "select",
      }));

      // Reset zoom & pan on new image
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setPenPoints([]);

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
          const defaultMapping: Record<number, FingerName> = {};
          const fingers: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
          res.slots.slice(0, 5).forEach((s, idx) => {
            defaultMapping[s.slot_id] = fingers[idx];
          });

          const origSlotsMap: Record<number, DetectedSlot> = {};
          res.slots.forEach((s) => {
            origSlotsMap[s.slot_id] = JSON.parse(JSON.stringify(s));
          });

          setTrayState((prev) => ({
            ...prev,
            isSegmenting: false,
            trayImageSize: res.image_size,
            detectedSlots: res.slots,
            selectedSlot: res.slots[0]?.slot_id ?? 1,
            slotFinger: defaultMapping,
            originalSlots: origSlotsMap,
            isManualEditMode: false,
            activeTool: "select",
          }));

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
      // preview error handled silently
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
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

  const handleSlotClick = (slot: DetectedSlot) => {
    if (dragState.current || trayState.activeTool === "pen") return;
    setTrayState((prev) => ({ ...prev, selectedSlot: slot.slot_id }));
    if (!trayState.extractedPreviews[slot.slot_id] && trayState.trayImageSrc) {
      fetchNailPreview(trayState.trayImageSrc, slot);
    }
  };

  // Convert client cursor coordinates into image-space coordinates via SVG matrix transform
  const getSvgPoint = useCallback(
    (e: ReactMouseEvent | MouseEvent): [number, number] | null => {
      if (!svgRef.current) return null;
      const svg = svgRef.current;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const transformed = pt.matrixTransform(ctm.inverse());
      return [transformed.x, transformed.y];
    },
    []
  );

  // Wheel Zoom Engine (smooth, clamping 0.3x to 6.0x)
  const handleWheel = (e: WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((prev) => Math.min(6.0, Math.max(0.3, Number((prev * factor).toFixed(2)))));
  };

  // Zoom control buttons
  const handleZoomIn = () => setZoom((prev) => Math.min(6.0, Number((prev * 1.25).toFixed(2))));
  const handleZoomOut = () => setZoom((prev) => Math.max(0.3, Number((prev * 0.8).toFixed(2))));
  const handleZoomReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Start panning viewport
  const startPan = (clientX: number, clientY: number) => {
    setIsPanning(true);
    panStartRef.current = {
      startX: clientX,
      startY: clientY,
      initPanX: pan.x,
      initPanY: pan.y,
    };
  };

  // Start dragging a slot or handle
  const startDrag = (
    e: ReactMouseEvent,
    mode: "move" | "resize" | "landmark",
    handle?: string,
    landmarkIdx?: number
  ) => {
    if (trayState.activeTool === "pen" || isSpacePressed) return;
    e.stopPropagation();
    e.preventDefault();
    const pt = getSvgPoint(e);
    if (!pt) return;
    const activeSlot = trayState.detectedSlots.find(
      (s) => s.slot_id === trayState.selectedSlot
    );
    if (!activeSlot) return;

    dragState.current = {
      mode,
      handle,
      landmarkIdx,
      startX: pt[0],
      startY: pt[1],
      origBbox: [...activeSlot.bbox],
      origContour: activeSlot.contour.map(([x, y]) => [x, y]),
    };
  };

  // Svg mouse move handler: handles transform drag and Pen tool preview
  const handleSvgMouseMove = (e: ReactMouseEvent<SVGSVGElement>) => {
    const pt = getSvgPoint(e);
    if (!pt) return;

    // 1. Pen tool active: track live cursor and check snapping to start point
    if (trayState.activeTool === "pen") {
      setCurrentMousePos(pt);
      if (penPoints.length >= 3) {
        const snapRadius = Math.max(16, (trayState.trayImageSize?.[0] || 1000) * 0.02) / zoom;
        const d = Math.hypot(pt[0] - penPoints[0][0], pt[1] - penPoints[0][1]);
        setIsNearStartPoint(d <= snapRadius);
      } else {
        setIsNearStartPoint(false);
      }
      return;
    }

    // 2. Selection transform drag active
    if (!dragState.current) return;

    const { mode, handle, landmarkIdx, startX, startY, origBbox, origContour } =
      dragState.current;
    const dx = Math.round(pt[0] - startX);
    const dy = Math.round(pt[1] - startY);

    if (mode === "move") {
      const newBbox: [number, number, number, number] = [
        origBbox[0] + dx,
        origBbox[1] + dy,
        origBbox[2],
        origBbox[3],
      ];
      const newContour = origContour.map(
        ([x, y]) => [x + dx, y + dy] as [number, number]
      );
      setTrayState((prev) => ({
        ...prev,
        detectedSlots: prev.detectedSlots.map((s) =>
          s.slot_id === prev.selectedSlot
            ? { ...s, bbox: newBbox, contour: newContour }
            : s
        ),
      }));
    } else if (mode === "resize" && handle) {
      let newBx = origBbox[0];
      let newBy = origBbox[1];
      let newBw = origBbox[2];
      let newBh = origBbox[3];

      if (handle.includes("e")) {
        newBw = Math.max(16, origBbox[2] + dx);
      }
      if (handle.includes("s")) {
        newBh = Math.max(16, origBbox[3] + dy);
      }
      if (handle.includes("w")) {
        newBw = Math.max(16, origBbox[2] - dx);
        newBx = origBbox[0] + dx;
      }
      if (handle.includes("n")) {
        newBh = Math.max(16, origBbox[3] - dy);
        newBy = origBbox[1] + dy;
      }

      const scaledContour = origContour.map(([x, y]) => [
        Math.round(newBx + ((x - origBbox[0]) / Math.max(1, origBbox[2])) * newBw),
        Math.round(newBy + ((y - origBbox[1]) / Math.max(1, origBbox[3])) * newBh),
      ] as [number, number]);

      setTrayState((prev) => ({
        ...prev,
        detectedSlots: prev.detectedSlots.map((s) =>
          s.slot_id === prev.selectedSlot
            ? {
                ...s,
                bbox: [newBx, newBy, newBw, newBh],
                contour: scaledContour,
              }
            : s
        ),
      }));
    } else if (mode === "landmark" && landmarkIdx !== undefined) {
      const newContour = origContour.map(([x, y], idx) => {
        if (idx === landmarkIdx) {
          return [x + dx, y + dy] as [number, number];
        }
        const dist = Math.min(
          Math.abs(idx - landmarkIdx),
          origContour.length - Math.abs(idx - landmarkIdx)
        );
        if (dist <= 2) {
          const weight = dist === 1 ? 0.5 : 0.25;
          return [Math.round(x + dx * weight), Math.round(y + dy * weight)] as [
            number,
            number
          ];
        }
        return [x, y] as [number, number];
      });

      const xs = newContour.map((p) => p[0]);
      const ys = newContour.map((p) => p[1]);
      const minX = Math.min(...xs);
      const minY = Math.min(...ys);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);

      setTrayState((prev) => ({
        ...prev,
        detectedSlots: prev.detectedSlots.map((s) =>
          s.slot_id === prev.selectedSlot
            ? {
                ...s,
                bbox: [minX, minY, maxX - minX, maxY - minY],
                contour: newContour,
              }
            : s
        ),
      }));
    }
  };

  // Pen tool point click handler
  const handleSvgClick = (e: ReactMouseEvent<SVGSVGElement>) => {
    if (trayState.activeTool !== "pen") return;
    const pt = getSvgPoint(e);
    if (!pt) return;

    // Check if clicking near first point to close path
    if (penPoints.length >= 3 && isNearStartPoint) {
      finishPenPath();
      return;
    }

    setPenPoints((prev) => [...prev, [Math.round(pt[0]), Math.round(pt[1])]]);
  };

  // Complete and apply custom pen drawn contour
  const finishPenPath = () => {
    if (penPoints.length < 3) return;

    const xs = penPoints.map((p) => p[0]);
    const ys = penPoints.map((p) => p[1]);
    const minX = Math.round(Math.min(...xs));
    const minY = Math.round(Math.min(...ys));
    const maxX = Math.round(Math.max(...xs));
    const maxY = Math.round(Math.max(...ys));
    const bw = Math.max(16, maxX - minX);
    const bh = Math.max(16, maxY - minY);
    const newBbox: [number, number, number, number] = [minX, minY, bw, bh];

    setTrayState((prev) => {
      const updatedSlots = prev.detectedSlots.map((s) =>
        s.slot_id === prev.selectedSlot
          ? {
              ...s,
              bbox: newBbox,
              contour: penPoints,
            }
          : s
      );

      const activeSlot = updatedSlots.find((s) => s.slot_id === prev.selectedSlot);
      if (activeSlot && prev.trayImageSrc) {
        fetchNailPreview(prev.trayImageSrc, activeSlot);
      }

      return {
        ...prev,
        detectedSlots: updatedSlots,
        extractedPreviews: {
          ...prev.extractedPreviews,
          [prev.selectedSlot]: "",
        },
        activeTool: "select",
      };
    });

    setPenPoints([]);
    setCurrentMousePos(null);
    setIsNearStartPoint(false);
  };

  const cancelPenDrawing = () => {
    setPenPoints([]);
    setCurrentMousePos(null);
    setIsNearStartPoint(false);
    setTrayState((prev) => ({ ...prev, activeTool: "select" }));
  };

  const undoLastPenPoint = () => {
    setPenPoints((prev) => prev.slice(0, -1));
  };

  const endDrag = () => {
    if (!dragState.current) return;
    dragState.current = null;

    setTrayState((prev) => {
      const activeSlot = prev.detectedSlots.find(
        (s) => s.slot_id === prev.selectedSlot
      );
      if (activeSlot && prev.trayImageSrc) {
        fetchNailPreview(prev.trayImageSrc, activeSlot);
      }
      return {
        ...prev,
        extractedPreviews: {
          ...prev.extractedPreviews,
          [prev.selectedSlot]: "",
        },
      };
    });
  };

  // Keyboard shortcut listeners (Space for pan, P for pen, V for select, Esc/Backspace)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.code === "Space" && !e.repeat) {
        setIsSpacePressed(true);
      } else if (e.key === "p" || e.key === "P") {
        setTrayState((prev) => ({ ...prev, activeTool: "pen" }));
      } else if (e.key === "v" || e.key === "V") {
        setTrayState((prev) => ({ ...prev, activeTool: "select" }));
      } else if (e.key === "h" || e.key === "H") {
        setTrayState((prev) => ({ ...prev, activeTool: "pan" }));
      } else if (e.key === "Escape") {
        if (penPoints.length > 0) {
          cancelPenDrawing();
        }
      } else if (e.key === "Backspace" || e.key === "Delete") {
        if (penPoints.length > 0) {
          undoLastPenPoint();
        }
      } else if (e.key === "Enter") {
        if (penPoints.length >= 3) {
          finishPenPath();
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setIsSpacePressed(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [penPoints, setTrayState]);

  // Global mouse listeners for panning and ending drags outside the SVG
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isPanning && panStartRef.current) {
        const dx = e.clientX - panStartRef.current.startX;
        const dy = e.clientY - panStartRef.current.startY;
        setPan({
          x: panStartRef.current.initPanX + dx,
          y: panStartRef.current.initPanY + dy,
        });
      }
    };

    const handleGlobalMouseUp = () => {
      if (isPanning) {
        setIsPanning(false);
        panStartRef.current = null;
      }
      if (dragState.current) {
        endDrag();
      }
    };

    window.addEventListener("mousemove", handleGlobalMouseMove);
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove);
      window.removeEventListener("mouseup", handleGlobalMouseUp);
    };
  }, [isPanning]);

  const handleContainerMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (e.button === 1 || isSpacePressed || trayState.activeTool === "pan") {
      e.preventDefault();
      startPan(e.clientX, e.clientY);
    }
  };

  const isPanModeActive = isSpacePressed || trayState.activeTool === "pan";

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleContainerMouseDown}
      className={`flex-1 h-full flex flex-col bg-neutral-950 relative overflow-hidden select-none ${
        isPanModeActive
          ? isPanning
            ? "cursor-grabbing"
            : "cursor-grab"
          : trayState.activeTool === "pen"
          ? "cursor-crosshair"
          : "cursor-default"
      }`}
    >
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
      />

      {/* Main Viewport Content Area */}
      <div className="flex-1 w-full h-full flex items-center justify-center p-6 relative">
        {activeTab === "tray" && (
          <div className="w-full h-full flex items-center justify-center relative">
            {!trayState.trayImageSrc ? (
              /* Empty Upload Dropzone */
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`w-full max-w-xl p-10 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
                  isDragOver
                    ? "border-amber-500 bg-amber-500/10 scale-102"
                    : "border-neutral-800 bg-neutral-900/40 hover:border-neutral-700 hover:bg-neutral-900/70"
                }`}
              >
                <div className="w-16 h-16 rounded-full bg-neutral-800/80 border border-neutral-700/80 flex items-center justify-center mb-4 text-amber-400 shadow-md">
                  <Upload className="w-8 h-8" />
                </div>
                <h3 className="text-base font-semibold text-neutral-200 mb-1">
                  Upload Packaging Tray Photo
                </h3>
                <p className="text-xs text-neutral-400 max-w-md mb-4 leading-relaxed">
                  Drag and drop your press-on packaging photo, or click to browse files.
                  Use Pan & Zoom and the Pen Tool to customize any nail contour.
                </p>
                <button className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-neutral-950 text-xs font-semibold rounded-xl shadow-lg shadow-amber-500/20 transition-all cursor-pointer">
                  Browse Files
                </button>
              </div>
            ) : (
              /* Active Tray Canvas with Pan & Zoom Transform Matrix */
              <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
                {trayState.isSegmenting && (
                  <div className="absolute inset-0 z-50 bg-neutral-950/75 backdrop-blur-sm flex flex-col items-center justify-center rounded-2xl">
                    <Loader2 className="w-10 h-10 animate-spin text-amber-500 mb-3" />
                    <p className="text-sm font-semibold text-neutral-200">
                      Analyzing Tray Photo & Segmenting Slots...
                    </p>
                    <p className="text-xs text-neutral-400 mt-1">
                      Detecting contours, card homography & G2 continuous super-ellipses
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

                {/* Transform Container with Pan and Zoom */}
                <div
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: "center center",
                    transition: isPanning ? "none" : "transform 0.05s ease-out",
                  }}
                  className="relative max-w-full max-h-full flex items-center justify-center shadow-2xl rounded-xl overflow-visible border border-neutral-800/80"
                >
                  <img
                    src={trayState.trayImageSrc}
                    alt="Packaging Tray"
                    className="max-w-full max-h-[75vh] object-contain select-none pointer-events-none"
                  />

                  {trayState.trayImageSize && (
                    <svg
                      ref={svgRef}
                      className="absolute inset-0 w-full h-full select-none"
                      viewBox={`0 0 ${trayState.trayImageSize[0]} ${trayState.trayImageSize[1]}`}
                      preserveAspectRatio="xMidYMid meet"
                      onClick={handleSvgClick}
                      onMouseMove={handleSvgMouseMove}
                      onMouseUp={endDrag}
                    >
                      {/* Detected Slot Contours */}
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

                        const handleSize = Math.max(
                          10,
                          Math.round((trayState.trayImageSize?.[0] || 1500) * 0.01)
                        );
                        const halfHandle = handleSize / 2;

                        let cuticleIdx = 0;
                        let tipIdx = 0;
                        let leftIdx = 0;
                        let rightIdx = 0;

                        if (slot.contour.length > 0) {
                          slot.contour.forEach(([x, y], idx) => {
                            if (y < slot.contour[cuticleIdx][1]) cuticleIdx = idx;
                            if (y > slot.contour[tipIdx][1]) tipIdx = idx;
                            if (x < slot.contour[leftIdx][0]) leftIdx = idx;
                            if (x > slot.contour[rightIdx][0]) rightIdx = idx;
                          });
                        }

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
                            className={`${
                              trayState.activeTool === "pen"
                                ? "pointer-events-none"
                                : "cursor-pointer"
                            } transition-all`}
                          >
                            {/* Contour Polygon */}
                            <polygon
                              points={pointsString}
                              fill={
                                isSelected
                                  ? "rgba(245, 158, 11, 0.38)"
                                  : isHovered
                                  ? "rgba(245, 158, 11, 0.22)"
                                  : "rgba(0, 0, 0, 0.15)"
                              }
                              stroke={isSelected ? "#f59e0b" : isHovered ? "#fbbf24" : color}
                              strokeWidth={isSelected ? 3.5 : isHovered ? 3 : 2}
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

                            {/* Manual Selection Bounding Box & Handles */}
                            {trayState.isManualEditMode &&
                              isSelected &&
                              trayState.activeTool !== "pen" && (
                                <g className="manual-edit-transform-box">
                                  <rect
                                    x={bx}
                                    y={by}
                                    width={bw}
                                    height={bh}
                                    fill="rgba(245, 158, 11, 0.05)"
                                    stroke="#f59e0b"
                                    strokeWidth="2"
                                    strokeDasharray="6,4"
                                    className="pointer-events-none"
                                  />

                                  <g
                                    transform={`translate(${centerX}, ${by - 18})`}
                                    onMouseDown={(e) => startDrag(e, "move")}
                                    className="cursor-move"
                                  >
                                    <rect
                                      x="-28"
                                      y="-9"
                                      width="56"
                                      height="18"
                                      rx="9"
                                      fill="#f59e0b"
                                      stroke="#ffffff"
                                      strokeWidth="1.5"
                                    />
                                    <text
                                      textAnchor="middle"
                                      dominantBaseline="central"
                                      fill="#000000"
                                      fontSize="9"
                                      fontWeight="bold"
                                      fontFamily="sans-serif"
                                    >
                                      DRAG
                                    </text>
                                  </g>

                                  {[
                                    { id: "nw", x: bx, y: by, cursor: "nwse-resize" },
                                    { id: "n", x: bx + bw / 2, y: by, cursor: "ns-resize" },
                                    { id: "ne", x: bx + bw, y: by, cursor: "nesw-resize" },
                                    { id: "e", x: bx + bw, y: by + bh / 2, cursor: "ew-resize" },
                                    { id: "se", x: bx + bw, y: by + bh, cursor: "nwse-resize" },
                                    { id: "s", x: bx + bw / 2, y: by + bh, cursor: "ns-resize" },
                                    { id: "sw", x: bx, y: by + bh, cursor: "nesw-resize" },
                                    { id: "w", x: bx, y: by + bh / 2, cursor: "ew-resize" },
                                  ].map((h) => (
                                    <rect
                                      key={h.id}
                                      x={h.x - halfHandle}
                                      y={h.y - halfHandle}
                                      width={handleSize}
                                      height={handleSize}
                                      rx="2.5"
                                      fill="#f59e0b"
                                      stroke="#ffffff"
                                      strokeWidth="1.5"
                                      style={{ cursor: h.cursor }}
                                      onMouseDown={(e) => startDrag(e, "resize", h.id)}
                                    />
                                  ))}

                                  {[
                                    { name: "cuticle", idx: cuticleIdx, label: "C" },
                                    { name: "tip", idx: tipIdx, label: "T" },
                                    { name: "left", idx: leftIdx, label: "L" },
                                    { name: "right", idx: rightIdx, label: "R" },
                                  ].map((lm) => {
                                    const pt = slot.contour[lm.idx];
                                    if (!pt) return null;
                                    return (
                                      <circle
                                        key={lm.name}
                                        cx={pt[0]}
                                        cy={pt[1]}
                                        r={handleSize * 0.65}
                                        fill="#06b6d4"
                                        stroke="#ffffff"
                                        strokeWidth="2"
                                        className="cursor-crosshair shadow-sm hover:scale-125 transition-transform"
                                        onMouseDown={(e) =>
                                          startDrag(e, "landmark", undefined, lm.idx)
                                        }
                                      />
                                    );
                                  })}
                                </g>
                              )}
                          </g>
                        );
                      })}

                      {/* Pen Tool Interactive Path Rendering */}
                      {penPoints.length > 0 && (
                        <g className="pen-tool-layer pointer-events-none">
                          {/* Closed/filled preview polygon if >= 3 points */}
                          {penPoints.length >= 3 && (
                            <polygon
                              points={penPoints.map((pt) => `${pt[0]},${pt[1]}`).join(" ")}
                              fill="rgba(245, 158, 11, 0.20)"
                              stroke="none"
                            />
                          )}

                          {/* Polyline connecting placed points */}
                          <polyline
                            points={penPoints.map((pt) => `${pt[0]},${pt[1]}`).join(" ")}
                            fill="none"
                            stroke="#f59e0b"
                            strokeWidth="3"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                          />

                          {/* Elastic dashed preview line to current cursor */}
                          {currentMousePos && (
                            <line
                              x1={penPoints[penPoints.length - 1][0]}
                              y1={penPoints[penPoints.length - 1][1]}
                              x2={
                                isNearStartPoint ? penPoints[0][0] : currentMousePos[0]
                              }
                              y2={
                                isNearStartPoint ? penPoints[0][1] : currentMousePos[1]
                              }
                              stroke={isNearStartPoint ? "#10b981" : "#f59e0b"}
                              strokeWidth="2.5"
                              strokeDasharray="6,4"
                            />
                          )}

                          {/* Placed vertex circles */}
                          {penPoints.map((pt, idx) => {
                            const isStart = idx === 0;
                            const isSnapping = isStart && isNearStartPoint;
                            return (
                              <circle
                                key={idx}
                                cx={pt[0]}
                                cy={pt[1]}
                                r={
                                  isSnapping
                                    ? 8
                                    : Math.max(
                                        4,
                                        Math.round(
                                          (trayState.trayImageSize?.[0] || 1500) *
                                            0.005
                                        )
                                      )
                                }
                                fill={isSnapping ? "#10b981" : isStart ? "#f59e0b" : "#ffffff"}
                                stroke={isSnapping ? "#ffffff" : "#171717"}
                                strokeWidth="2"
                                className={isSnapping ? "animate-pulse" : ""}
                              />
                            );
                          })}
                        </g>
                      )}
                    </svg>
                  )}
                </div>

                {/* Floating Pen Tool Active Help Banner */}
                {trayState.activeTool === "pen" && (
                  <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 bg-neutral-900/95 backdrop-blur-md px-4 py-2 rounded-xl border border-amber-500/50 shadow-2xl flex items-center gap-3 text-xs text-neutral-200">
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-ping" />
                    <span className="font-medium">
                      Pen Tool Active: Click around Slot #{trayState.selectedSlot} ({penPoints.length} points placed)
                    </span>
                    <span className="text-neutral-500">|</span>
                    <button
                      onClick={finishPenPath}
                      disabled={penPoints.length < 3}
                      className="px-2 py-0.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-neutral-950 rounded font-semibold text-[11px] flex items-center gap-1 cursor-pointer transition-all"
                    >
                      <Check className="w-3 h-3" />
                      <span>Complete Path</span>
                    </button>
                    {penPoints.length > 0 && (
                      <button
                        onClick={undoLastPenPoint}
                        className="px-2 py-0.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded text-[11px] flex items-center gap-1 cursor-pointer"
                      >
                        <Undo2 className="w-3 h-3" />
                        <span>Undo Point</span>
                      </button>
                    )}
                    <button
                      onClick={cancelPenDrawing}
                      className="p-1 hover:text-rose-400 text-neutral-400 cursor-pointer"
                      title="Cancel (Esc)"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}

                {/* Floating Viewport Toolbar (Zoom, Pan, Pen, Selection) */}
                <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-xs text-neutral-300 bg-neutral-900/85 backdrop-blur-md px-4 py-2 rounded-xl border border-neutral-800 shadow-xl">
                  {/* Slots isolated readout */}
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-neutral-200">
                      {trayState.detectedSlots.length} Slots Isolated
                    </span>
                    <span className="text-neutral-500">•</span>
                    <span className="text-amber-400">
                      {Object.keys(trayState.slotFinger).length} Assigned to Hand
                    </span>
                  </div>

                  {/* Middle: Tool Switcher (Select, Pen Tool, Pan) */}
                  <div className="flex items-center bg-neutral-950 p-0.5 rounded-lg border border-neutral-800">
                    <button
                      onClick={() => {
                        cancelPenDrawing();
                        setTrayState((prev) => ({ ...prev, activeTool: "select" }));
                      }}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                        trayState.activeTool === "select"
                          ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                          : "text-neutral-400 hover:text-neutral-200"
                      }`}
                      title="Select & Transform Tool (V)"
                    >
                      <MousePointer className="w-3 h-3" />
                      <span>Select</span>
                    </button>

                    <button
                      onClick={() => {
                        setTrayState((prev) => ({ ...prev, activeTool: "pen" }));
                      }}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                        trayState.activeTool === "pen"
                          ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                          : "text-neutral-400 hover:text-neutral-200"
                      }`}
                      title="Draw Custom Contour Pen Tool (P)"
                    >
                      <PenTool className="w-3 h-3" />
                      <span>Pen Tool</span>
                    </button>

                    <button
                      onClick={() => {
                        cancelPenDrawing();
                        setTrayState((prev) => ({ ...prev, activeTool: "pan" }));
                      }}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                        trayState.activeTool === "pan"
                          ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                          : "text-neutral-400 hover:text-neutral-200"
                      }`}
                      title="Pan / Hand Tool (H or hold Spacebar)"
                    >
                      <Hand className="w-3 h-3" />
                      <span>Pan</span>
                    </button>
                  </div>

                  {/* Right: Zoom Controls & Manual Handles Toggle */}
                  <div className="flex items-center gap-2">
                    {/* Zoom Widget */}
                    <div className="flex items-center gap-1 bg-neutral-950 px-1 py-0.5 rounded-lg border border-neutral-800 text-[11px]">
                      <button
                        onClick={handleZoomOut}
                        className="p-1 hover:text-neutral-100 text-neutral-400 cursor-pointer"
                        title="Zoom Out (-)"
                      >
                        <ZoomOut className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={handleZoomReset}
                        className="px-1.5 font-mono text-neutral-300 hover:text-amber-400 cursor-pointer"
                        title="Click to reset zoom to 100%"
                      >
                        {Math.round(zoom * 100)}%
                      </button>
                      <button
                        onClick={handleZoomIn}
                        className="p-1 hover:text-neutral-100 text-neutral-400 cursor-pointer"
                        title="Zoom In (+)"
                      >
                        <ZoomIn className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={handleZoomReset}
                        className="p-1 hover:text-neutral-100 text-neutral-400 cursor-pointer border-l border-neutral-800 ml-0.5 pl-1.5"
                        title="Reset to Fit Screen"
                      >
                        <Maximize className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Manual Selection Mode Toggle */}
                    <button
                      onClick={() =>
                        setTrayState((prev) => ({
                          ...prev,
                          isManualEditMode: !prev.isManualEditMode,
                        }))
                      }
                      className={`px-2.5 py-1 rounded-lg border transition-all cursor-pointer flex items-center gap-1.5 ${
                        trayState.isManualEditMode
                          ? "bg-amber-500/25 border-amber-500 text-amber-300 font-semibold shadow-xs"
                          : "bg-neutral-800 hover:bg-neutral-700 border-neutral-700 text-neutral-300"
                      }`}
                      title="Toggle interactive on-canvas handles to nudge or resize the selection box"
                    >
                      <Sliders className="w-3 h-3" />
                      <span>
                        {trayState.isManualEditMode ? "Handles: ON" : "Handles"}
                      </span>
                    </button>

                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-2.5 py-1 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded-lg border border-neutral-700 transition-colors cursor-pointer flex items-center gap-1"
                    >
                      <RotateCw className="w-3 h-3 text-neutral-400" />
                      <span>Replace</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hand Rigging Placeholder */}
        {activeTab === "hand" && (
          <div className="flex flex-col items-center justify-center text-center max-w-md p-8 border border-neutral-800 rounded-2xl bg-neutral-900/40">
            <div className="w-16 h-16 rounded-full bg-neutral-800 border border-neutral-700 flex items-center justify-center mb-4 text-amber-400 shadow-md">
              <Hand className="w-8 h-8" />
            </div>
            <h3 className="text-base font-semibold text-neutral-200 mb-1">
              Phase 3: Hand Rigging Canvas
            </h3>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Place 5 anchor points on each finger to define cuticle curves, sidewalls, and fingertip length vectors. Ships with built-in Default Studio Hand template.
            </p>
          </div>
        )}

        {/* Composite Mockup Placeholder */}
        {activeTab === "composite" && (
          <div className="flex flex-col items-center justify-center text-center max-w-md p-8 border border-neutral-800 rounded-2xl bg-neutral-900/40">
            <div className="w-16 h-16 rounded-full bg-neutral-800 border border-neutral-700 flex items-center justify-center mb-4 text-amber-400 shadow-md">
              <Sparkles className="w-8 h-8" />
            </div>
            <h3 className="text-base font-semibold text-neutral-200 mb-1">
              Phase 4: Real-Time PBR Compositor
            </h3>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Photorealistic 6-pass physical blending: Glare Healing, Studio Softbox Relighting, Specular Ribbons, Anatomical Pocket Tuck, and Contact Ambient Occlusion.
            </p>
          </div>
        )}

        {/* Production Export Placeholder */}
        {activeTab === "export" && (
          <div className="flex flex-col items-center justify-center text-center max-w-md p-8 border border-neutral-800 rounded-2xl bg-neutral-900/40">
            <div className="w-16 h-16 rounded-full bg-neutral-800 border border-neutral-700 flex items-center justify-center mb-4 text-amber-400 shadow-md">
              <Download className="w-8 h-8" />
            </div>
            <h3 className="text-base font-semibold text-neutral-200 mb-1">
              Phase 5: High-Res Production Export
            </h3>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Render print-ready 300 DPI composite PNGs and multi-layer PSD files with preserved layer hierarchy and isolated blend modes.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};