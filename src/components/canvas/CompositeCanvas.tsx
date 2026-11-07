import React, {
  useRef,
  useState,
  useEffect,
  useCallback,
  WheelEvent,
  MouseEvent as ReactMouseEvent,
} from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Sparkles,
  Loader2,
  Eye,
  EyeOff,
  Sun,
  Wand2,
  Sliders,
} from "lucide-react";
import {
  FingerName,
  FingerNailAdjustment,
  HandState,
  LightingState,
  TrayState,
} from "../../types";
import {
  DEFAULT_FINGER_ADJUSTMENTS,
  FINGER_COLORS,
} from "../../constants/handDefaults";
import { callSidecar } from "../../services/sidecar";

interface CompositeCanvasProps {
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
  handState: HandState;
  lightingState: LightingState;
  setLightingState?: React.Dispatch<React.SetStateAction<LightingState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

interface DragHandleTarget {
  finger: FingerName;
  mode: "length" | "width" | "shift";
  handleSide?: "left" | "right";
  startStageX: number;
  startStageY: number;
  initVal: number;
  bedLength: number;
  halfWidth: number;
  axisUnit: [number, number];
  perpUnit: [number, number];
}

export const CompositeCanvas: React.FC<CompositeCanvasProps> = ({
  trayState,
  setTrayState,
  handState,
  lightingState,
  setLightingState,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Viewport Pan & Zoom State
  const [zoom, setZoom] = useState<number>(0.32);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [isSpacePressed, setIsSpacePressed] = useState<boolean>(false);
  const panStartRef = useRef<{
    startX: number;
    startY: number;
    initPanX: number;
    initPanY: number;
  } | null>(null);

  // Interactive On-Canvas Handles State
  const [showNailHandles, setShowNailHandles] = useState<boolean>(true);
  const [hoveredFinger, setHoveredFinger] = useState<FingerName | null>(null);
  const dragHandleRef = useRef<DragHandleTarget | null>(null);
  const clickStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const wasDraggingRef = useRef<boolean>(false);

  // Rendering state
  const [compositeImageSrc, setCompositeImageSrc] = useState<string | null>(null);
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [renderLatency, setRenderLatency] = useState<number | null>(null);
  const [renderSize, setRenderSize] = useState<[number, number] | null>(null);
  const [showBareHand, setShowBareHand] = useState<boolean>(false);
  const [isFullQuality, setIsFullQuality] = useState<boolean>(false);

  const selectedFinger = lightingState.selectedFinger || "All";
  const fingerAdjustments =
    lightingState.fingerAdjustments || DEFAULT_FINGER_ADJUSTMENTS;

  const updateLightingState = useCallback(
    (partial: Partial<LightingState>) => {
      if (setLightingState) {
        setLightingState((prev) => ({ ...prev, ...partial }));
      }
    },
    [setLightingState]
  );

  const updateFingerAdjustment = useCallback(
    (f: FingerName, partial: Partial<FingerNailAdjustment>) => {
      if (!setLightingState) return;
      setLightingState((prev) => {
        const adjs = {
          ...(prev.fingerAdjustments || DEFAULT_FINGER_ADJUSTMENTS),
        };
        adjs[f] = { ...(adjs[f] || DEFAULT_FINGER_ADJUSTMENTS[f]), ...partial };
        return {
          ...prev,
          selectedFinger: f,
          fingerAdjustments: adjs,
        };
      });
    },
    [setLightingState]
  );

  // Auto-fit mockup into viewport
  const fitToContainer = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth <= 0 || clientHeight <= 0) return;

    const imgW = handState.currentTemplate?.imageSize[0] || 1536;
    const imgH = handState.currentTemplate?.imageSize[1] || 2752;

    const padding = 48;
    const availW = clientWidth - padding;
    const availH = clientHeight - padding;

    const scale = Math.min(availW / imgW, availH / imgH, 1.0);
    const initialZoom = Math.max(0.15, Math.min(scale, 2.0));

    setZoom(initialZoom);
    setPan({
      x: (clientWidth - imgW * initialZoom) / 2,
      y: (clientHeight - imgH * initialZoom) / 2,
    });
  }, [handState.currentTemplate?.imageSize]);

  useEffect(() => {
    fitToContainer();
  }, [fitToContainer]);

  // Handle Spacebar hotkey for panning or bare-hand comparison
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        const target = e.target as HTMLElement;
        if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
          setIsSpacePressed(true);
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setIsSpacePressed(false);
        setIsPanning(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Smooth mouse wheel zoom
  const handleWheel = (e: WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    const newZoom = Math.min(Math.max(0.12, zoom * zoomFactor), 4.5);

    const newPanX = mouseX - (mouseX - pan.x) * (newZoom / zoom);
    const newPanY = mouseY - (mouseY - pan.y) * (newZoom / zoom);

    setZoom(newZoom);
    setPan({ x: newPanX, y: newPanY });
  };

  const getStageCoords = useCallback(
    (clientX: number, clientY: number) => {
      if (!containerRef.current) return { x: 0, y: 0 };
      const rect = containerRef.current.getBoundingClientRect();
      return {
        x: (clientX - rect.left - pan.x) / zoom,
        y: (clientY - rect.top - pan.y) / zoom,
      };
    },
    [pan, zoom]
  );

  // Handle Escape key to deselect active finger
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedFinger !== "All") {
          updateLightingState({ selectedFinger: "All" });
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedFinger, updateLightingState]);

  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    clickStartPosRef.current = { x: e.clientX, y: e.clientY };
    wasDraggingRef.current = false;

    if (e.button === 1 || isSpacePressed) {
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        initPanX: pan.x,
        initPanY: pan.y,
      };
    }
  };

  const handleMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (clickStartPosRef.current) {
      const dist = Math.hypot(
        e.clientX - clickStartPosRef.current.x,
        e.clientY - clickStartPosRef.current.y
      );
      if (dist > 5) {
        wasDraggingRef.current = true;
      }
    }

    if (isPanning && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.startX;
      const dy = e.clientY - panStartRef.current.startY;
      setPan({
        x: panStartRef.current.initPanX + dx,
        y: panStartRef.current.initPanY + dy,
      });
      return;
    }

    // Handle interactive on-nail dragging
    if (dragHandleRef.current) {
      const {
        finger,
        mode,
        handleSide,
        startStageX,
        startStageY,
        initVal,
        bedLength,
        halfWidth,
        axisUnit,
        perpUnit,
      } = dragHandleRef.current;
      const { x: curX, y: curY } = getStageCoords(e.clientX, e.clientY);

      const deltaX = curX - startStageX;
      const deltaY = curY - startStageY;

      // Project motion along the finger axis and perp vectors
      const projAxis = deltaX * axisUnit[0] + deltaY * axisUnit[1];
      const projPerp = deltaX * perpUnit[0] + deltaY * perpUnit[1];

      if (mode === "length") {
        // Dragging along axis vector extends length
        const scaleDelta = (projAxis / Math.max(1, bedLength)) * 1.5;
        const newLen = Math.max(
          0.5,
          Math.min(2.5, Number((initVal + scaleDelta).toFixed(2)))
        );
        updateFingerAdjustment(finger, { lengthScale: newLen });
      } else if (mode === "width") {
        // Dragging outward from center widens; inward narrows
        const dir = handleSide === "left" ? -1 : 1;
        const wDelta = ((projPerp * dir) / Math.max(1, halfWidth)) * 0.9;
        const newW = Math.max(
          0.7,
          Math.min(1.4, Number((initVal + wDelta).toFixed(2)))
        );
        updateFingerAdjustment(finger, { widthScale: newW });
      } else if (mode === "shift") {
        // Shifting along axis moves nail up or down (tuck / extension)
        const newShift = Math.max(
          -30,
          Math.min(30, Math.round(initVal + projAxis))
        );
        updateFingerAdjustment(finger, { axialShift: newShift });
      }
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    panStartRef.current = null;
    dragHandleRef.current = null;
  };

  // Click outside to deselect
  const handleCanvasClick = () => {
    if (wasDraggingRef.current) return;
    if (selectedFinger !== "All") {
      updateLightingState({ selectedFinger: "All" });
    }
  };

  // Compile mapped nails dictionary
  const getMappedNailsDict = useCallback(() => {
    const dict: Record<string, string> = {};
    for (const [slotIdStr, finger] of Object.entries(trayState.slotFinger)) {
      const sid = Number(slotIdStr);
      if (trayState.extractedPreviews[sid]) {
        dict[finger] = trayState.extractedPreviews[sid];
      }
    }
    return dict;
  }, [trayState.slotFinger, trayState.extractedPreviews]);

  const mappedNailsCount = Object.keys(getMappedNailsDict()).length;

  // Execute Composite Render via Python Sidecar
  const renderComposite = useCallback(async () => {
    const mappedNails = getMappedNailsDict();
    setIsRendering(true);

    try {
      const res = await callSidecar<{
        status: string;
        composite_base64: string;
        image_size: [number, number];
        render_time_ms: number;
      }>("render_composite", {
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
        downscale: isFullQuality ? 1.0 : 0.5,
      });

      if (res && res.status === "ok" && res.composite_base64) {
        setCompositeImageSrc(res.composite_base64);
        setRenderLatency(res.render_time_ms);
        setRenderSize(res.image_size);
      }
    } catch (err) {
      console.warn("Render composite error:", err);
    } finally {
      setIsRendering(false);
    }
  }, [
    getMappedNailsDict,
    handState.currentTemplate,
    lightingState,
    isFullQuality,
  ]);

  // Debounced auto-render when lighting, length, or template changes
  useEffect(() => {
    const timer = setTimeout(() => {
      renderComposite();
    }, 120);
    return () => clearTimeout(timer);
  }, [renderComposite]);

  // 1-Click Helper: Auto-load default demo tray if user hasn't segmented one yet
  const handleLoadDemoTray = async () => {
    setIsRendering(true);
    try {
      const segRes = await callSidecar<{
        status: string;
        slots: any[];
      }>("segment_tray", {
        image_path: "test_assets/target_tray.jpg",
        max_slots: 5,
      });

      if (segRes && segRes.status === "ok" && segRes.slots.length > 0) {
        const previews: Record<number, string> = {};
        const mapping: Record<number, FingerName> = {};

        for (let i = 0; i < Math.min(5, segRes.slots.length); i++) {
          const s = segRes.slots[i];
          const extRes = await callSidecar<{
            status: string;
            image_base64: string;
          }>("extract_nail", {
            image_path: "test_assets/target_tray.jpg",
            contour: s.contour,
            deskew: true,
          });

          if (extRes && extRes.status === "ok") {
            previews[s.slot_id] = extRes.image_base64;
            mapping[s.slot_id] = FINGERS[i];
          }
        }

        setTrayState((p) => ({
          ...p,
          trayImageSrc: "test_assets/target_tray.jpg",
          detectedSlots: segRes.slots,
          extractedPreviews: previews,
          slotFinger: mapping,
        }));
      }
    } catch (e) {
      console.warn("Demo tray extraction error:", e);
    } finally {
      setIsRendering(false);
    }
  };

  const currentDisplaySrc = showBareHand
    ? handState.currentTemplate?.imageSrc
    : compositeImageSrc || handState.currentTemplate?.imageSrc;

  const templateWidth = handState.currentTemplate?.imageSize[0] || 1536;
  const templateHeight = handState.currentTemplate?.imageSize[1] || 2752;

  // Compute On-Canvas geometry for each finger
  const fingerGeometries = FINGERS.map((fname) => {
    const lm = handState.currentTemplate?.landmarks?.[fname];
    if (!lm) return null;

    const adj = fingerAdjustments[fname] || DEFAULT_FINGER_ADJUSTMENTS[fname];
    const lenScale = (adj.lengthScale ?? 1.05) * (lightingState.nailLengthScale ?? 1.05) / 1.05;
    const wScale = adj.widthScale ?? 1.0;
    const axialShift = adj.axialShift ?? 0;
    const angleOffset = adj.angleOffset ?? 0;
    const tipTaper = adj.tipTaper ?? 1.0;

    const c_c = lm.c_c;
    const c_l = lm.c_l;
    const c_r = lm.c_r;
    const d_l = lm.d_l;
    const d_r = lm.d_r;

    const distalMid = [(d_l[0] + d_r[0]) * 0.5, (d_l[1] + d_r[1]) * 0.5];
    const axisVec = [distalMid[0] - c_c[0], distalMid[1] - c_c[1]];
    const bedLen = Math.max(1, Math.hypot(axisVec[0], axisVec[1]));
    let axisUnit: [number, number] = [axisVec[0] / bedLen, axisVec[1] / bedLen];
    let perpUnit: [number, number] = [-axisUnit[1], axisUnit[0]];

    // Ensure perpUnit consistently points from left to right (matching Python warping)
    const dDiff = [d_r[0] - d_l[0], d_r[1] - d_l[1]];
    if (perpUnit[0] * dDiff[0] + perpUnit[1] * dDiff[1] < 0) {
      perpUnit = [-perpUnit[0], -perpUnit[1]];
    }

    // Apply angle offset rotation
    if (Math.abs(angleOffset) > 0.01) {
      const rad = (angleOffset * Math.PI) / 180;
      const cosT = Math.cos(rad);
      const sinT = Math.sin(rad);
      const rotAxis: [number, number] = [
        cosT * axisUnit[0] + sinT * perpUnit[0],
        cosT * axisUnit[1] + sinT * perpUnit[1],
      ];
      const rLen = Math.hypot(rotAxis[0], rotAxis[1]) || 1;
      axisUnit = [rotAxis[0] / rLen, rotAxis[1] / rLen];
      perpUnit = [-axisUnit[1], axisUnit[0]];
      if (perpUnit[0] * dDiff[0] + perpUnit[1] * dDiff[1] < 0) {
        perpUnit = [-perpUnit[0], -perpUnit[1]];
      }
    }

    const distW = Math.hypot(d_r[0] - d_l[0], d_r[1] - d_l[1]);
    const cutW = Math.hypot(c_r[0] - c_l[0], c_r[1] - c_l[1]);
    const bedWidth = Math.max(distW, cutW);
    const halfWidth = bedWidth * 0.545 * wScale;
    const extDist = bedLen * 1.34 * lenScale;

    // Shifted base center
    const baseCenter: [number, number] = [
      c_c[0] + axisUnit[0] * axialShift,
      c_c[1] + axisUnit[1] * axialShift,
    ];

    const baseLeft: [number, number] = [
      baseCenter[0] - perpUnit[0] * halfWidth,
      baseCenter[1] - perpUnit[1] * halfWidth,
    ];
    const baseRight: [number, number] = [
      baseCenter[0] + perpUnit[0] * halfWidth,
      baseCenter[1] + perpUnit[1] * halfWidth,
    ];

    const tipCenter: [number, number] = [
      baseCenter[0] + axisUnit[0] * extDist,
      baseCenter[1] + axisUnit[1] * extDist,
    ];
    const tipHalfW = halfWidth * tipTaper;
    const tipLeft: [number, number] = [
      tipCenter[0] - perpUnit[0] * tipHalfW,
      tipCenter[1] - perpUnit[1] * tipHalfW,
    ];
    const tipRight: [number, number] = [
      tipCenter[0] + perpUnit[0] * tipHalfW,
      tipCenter[1] + perpUnit[1] * tipHalfW,
    ];

    const midLeft: [number, number] = [
      (baseLeft[0] + tipLeft[0]) * 0.5,
      (baseLeft[1] + tipLeft[1]) * 0.5,
    ];
    const midRight: [number, number] = [
      (baseRight[0] + tipRight[0]) * 0.5,
      (baseRight[1] + tipRight[1]) * 0.5,
    ];

    return {
      fname,
      adj,
      bedLen,
      halfWidth,
      axisUnit,
      perpUnit,
      baseCenter,
      baseLeft,
      baseRight,
      tipCenter,
      tipLeft,
      tipRight,
      midLeft,
      midRight,
    };
  }).filter(Boolean);

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleCanvasClick}
      className={`w-full h-full relative overflow-hidden bg-neutral-950 select-none ${
        isPanning || isSpacePressed
          ? "cursor-grab active:cursor-grabbing"
          : "cursor-default"
      }`}
    >
      {/* Transformed Stage Container */}
      <div
        className="absolute origin-top-left transition-transform duration-75 ease-out"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          width: `${templateWidth}px`,
          height: `${templateHeight}px`,
        }}
      >
        {currentDisplaySrc && (
          <img
            src={currentDisplaySrc}
            alt="Composited Try-On Hand"
            className="w-full h-full object-contain pointer-events-none drop-shadow-2xl"
            draggable={false}
          />
        )}

        {/* Interactive SVG Overlay for on-top-of-finger editing */}
        {showNailHandles && !showBareHand && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-auto"
            viewBox={`0 0 ${templateWidth} ${templateHeight}`}
            onClick={handleCanvasClick}
          >
            {fingerGeometries.map((geom) => {
              if (!geom) return null;
              const {
                fname,
                adj,
                bedLen,
                halfWidth,
                axisUnit,
                perpUnit,
                baseCenter,
                baseLeft,
                baseRight,
                tipCenter,
                tipLeft,
                tipRight,
                midLeft,
                midRight,
              } = geom;

              const isSelected = selectedFinger === fname || selectedFinger === "All";
              const isDirectlySelected = selectedFinger === fname;
              const isHovered = hoveredFinger === fname;
              const color = FINGER_COLORS[fname];

              // Polygon path for nail outline
              const polyPoints = `${baseLeft[0]},${baseLeft[1]} ${tipLeft[0]},${tipLeft[1]} ${tipRight[0]},${tipRight[1]} ${baseRight[0]},${baseRight[1]}`;

              return (
                <g
                  key={fname}
                  className="transition-opacity"
                  onMouseEnter={() => setHoveredFinger(fname)}
                  onMouseLeave={() => setHoveredFinger(null)}
                >
                  {/* Clickable Nail Body Frame */}
                  <polygon
                    points={polyPoints}
                    onClick={(e) => {
                      e.stopPropagation();
                      updateLightingState({ selectedFinger: fname });
                    }}
                    fill={
                      isDirectlySelected
                        ? `${color.hex}22`
                        : isHovered
                        ? `${color.hex}15`
                        : "transparent"
                    }
                    stroke={
                      isDirectlySelected
                        ? color.hex
                        : isHovered
                        ? color.lightHex
                        : isSelected
                        ? `${color.hex}55`
                        : "transparent"
                    }
                    strokeWidth={isDirectlySelected ? 3 : 1.5}
                    strokeDasharray={isDirectlySelected ? "none" : "5 5"}
                    className="cursor-pointer"
                  />

                  {/* Handles shown when nail is selected */}
                  {(isDirectlySelected || isHovered) && (
                    <>
                      {/* 1. Tip Length Handle (pull along axis to scale length) */}
                      <g
                        className="cursor-ns-resize"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          const { x, y } = getStageCoords(e.clientX, e.clientY);
                          dragHandleRef.current = {
                            finger: fname,
                            mode: "length",
                            startStageX: x,
                            startStageY: y,
                            initVal: adj.lengthScale,
                            bedLength: bedLen,
                            halfWidth,
                            axisUnit,
                            perpUnit,
                          };
                          updateLightingState({ selectedFinger: fname });
                        }}
                      >
                        <circle
                          cx={tipCenter[0]}
                          cy={tipCenter[1]}
                          r={9}
                          fill="#10b981"
                          stroke="#ffffff"
                          strokeWidth={2}
                          className="hover:scale-125 transition-transform drop-shadow"
                        />
                        <line
                          x1={tipCenter[0] - 4}
                          y1={tipCenter[1]}
                          x2={tipCenter[0] + 4}
                          y2={tipCenter[1]}
                          stroke="#ffffff"
                          strokeWidth={1.5}
                        />
                      </g>

                      {/* 2. Left Width Handle */}
                      <g
                        className="cursor-ew-resize"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          const { x, y } = getStageCoords(e.clientX, e.clientY);
                          dragHandleRef.current = {
                            finger: fname,
                            mode: "width",
                            handleSide: "left",
                            startStageX: x,
                            startStageY: y,
                            initVal: adj.widthScale,
                            bedLength: bedLen,
                            halfWidth,
                            axisUnit,
                            perpUnit,
                          };
                          updateLightingState({ selectedFinger: fname });
                        }}
                      >
                        <circle
                          cx={midLeft[0]}
                          cy={midLeft[1]}
                          r={7}
                          fill="#06b6d4"
                          stroke="#ffffff"
                          strokeWidth={2}
                          className="hover:scale-125 transition-transform drop-shadow"
                        />
                      </g>

                      {/* 3. Right Width Handle */}
                      <g
                        className="cursor-ew-resize"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          const { x, y } = getStageCoords(e.clientX, e.clientY);
                          dragHandleRef.current = {
                            finger: fname,
                            mode: "width",
                            handleSide: "right",
                            startStageX: x,
                            startStageY: y,
                            initVal: adj.widthScale,
                            bedLength: bedLen,
                            halfWidth,
                            axisUnit,
                            perpUnit,
                          };
                          updateLightingState({ selectedFinger: fname });
                        }}
                      >
                        <circle
                          cx={midRight[0]}
                          cy={midRight[1]}
                          r={7}
                          fill="#06b6d4"
                          stroke="#ffffff"
                          strokeWidth={2}
                          className="hover:scale-125 transition-transform drop-shadow"
                        />
                      </g>

                      {/* 4. Cuticle Base Tuck Handle */}
                      <g
                        className="cursor-ns-resize"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          const { x, y } = getStageCoords(e.clientX, e.clientY);
                          dragHandleRef.current = {
                            finger: fname,
                            mode: "shift",
                            startStageX: x,
                            startStageY: y,
                            initVal: adj.axialShift,
                            bedLength: bedLen,
                            halfWidth,
                            axisUnit,
                            perpUnit,
                          };
                          updateLightingState({ selectedFinger: fname });
                        }}
                      >
                        <circle
                          cx={baseCenter[0]}
                          cy={baseCenter[1]}
                          r={7}
                          fill="#f43f5e"
                          stroke="#ffffff"
                          strokeWidth={2}
                          className="hover:scale-125 transition-transform drop-shadow"
                        />
                      </g>

                      {/* Floating Name Badge */}
                      <g
                        transform={`translate(${tipCenter[0]}, ${tipCenter[1] - 22})`}
                        className="pointer-events-none"
                      >
                        <rect
                          x={-45}
                          y={-14}
                          width={90}
                          height={20}
                          rx={10}
                          fill="#171717"
                          stroke={color.hex}
                          strokeWidth={1.5}
                          className="drop-shadow-lg"
                        />
                        <text
                          x={0}
                          y={0}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fill="#ffffff"
                          fontSize={11}
                          fontWeight="600"
                        >
                          {fname} {adj.lengthScale}x
                        </text>
                      </g>
                    </>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* Floating Status Pill (Top Left) */}
      <div className="absolute top-4 left-4 flex items-center gap-2 z-20">
        <div className="flex items-center gap-2 bg-neutral-900/90 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-neutral-800 text-xs shadow-lg">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-neutral-200 font-medium">
            PBR 6-Pass Live Compositor
          </span>
          <span className="text-neutral-500">•</span>
          <span className="text-neutral-400 font-mono text-[11px]">
            {isRendering ? (
              <span className="text-amber-400 flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" /> Rendering...
              </span>
            ) : renderLatency ? (
              `${renderLatency}ms latency`
            ) : (
              "Ready"
            )}
          </span>
          {renderSize && (
            <>
              <span className="text-neutral-500">•</span>
              <span className="text-neutral-400 font-mono text-[11px]">
                {renderSize[0]}x{renderSize[1]}
              </span>
            </>
          )}
        </div>

        {/* Quality Mode Badge */}
        <button
          onClick={() => setIsFullQuality((q) => !q)}
          className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors cursor-pointer shadow-lg flex items-center gap-1.5 ${
            isFullQuality
              ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
              : "bg-neutral-900/90 text-neutral-400 border-neutral-800 hover:text-neutral-200"
          }`}
          title="Toggle Full 300 DPI Resolution Render"
        >
          <Sparkles className="w-3.5 h-3.5" />
          {isFullQuality ? "Full 300 DPI" : "Fast Preview (0.5x)"}
        </button>
      </div>

      {/* Helper Prompt if Nails are not mapped yet */}
      {mappedNailsCount < 5 && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-neutral-900/95 border border-amber-500/40 px-4 py-2.5 rounded-2xl text-xs text-amber-200 shadow-2xl z-20 backdrop-blur-md">
          <Sun className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            {mappedNailsCount === 0
              ? "No tray nails mapped yet."
              : `${mappedNailsCount}/5 nails mapped.`}{" "}
            Map slots in Tray Extractor, or load the studio tray:
          </span>
          <button
            onClick={handleLoadDemoTray}
            className="px-2.5 py-1 bg-amber-500 hover:bg-amber-400 text-neutral-950 font-semibold rounded-lg transition-colors cursor-pointer flex items-center gap-1 text-[11px]"
          >
            <Wand2 className="w-3 h-3" />
            Auto-Map Studio Nails
          </button>
        </div>
      )}

      {/* Floating Bottom Viewport Toolbar */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-neutral-900/90 backdrop-blur-md px-3.5 py-2 rounded-full border border-neutral-800 shadow-2xl z-20">
        {/* Zoom Controls */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          <button
            onClick={() => setZoom((z) => Math.max(0.12, z * 0.85))}
            className="p-1.5 rounded-full hover:bg-neutral-800 text-neutral-300 transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-[11px] font-mono text-neutral-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(4.5, z * 1.18))}
            className="p-1.5 rounded-full hover:bg-neutral-800 text-neutral-300 transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={fitToContainer}
            className="p-1.5 rounded-full hover:bg-neutral-800 text-neutral-300 transition-colors cursor-pointer"
            title="Fit to Screen"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>

        {/* On-Canvas Direct Edit Handles Toggle */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          <button
            onClick={() => setShowNailHandles((s) => !s)}
            className={`px-2.5 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              showNailHandles
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Toggle interactive on-canvas drag handles for each finger"
          >
            <Sliders className="w-3.5 h-3.5" />
            {showNailHandles ? "Nail Handles: On" : "Nail Handles: Off"}
          </button>
        </div>

        {/* Before / After Bare Hand Comparison */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          <button
            onMouseDown={() => setShowBareHand(true)}
            onMouseUp={() => setShowBareHand(false)}
            onMouseLeave={() => setShowBareHand(false)}
            className={`px-2.5 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              showBareHand
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Hold down to compare with the bare hand"
          >
            {showBareHand ? (
              <EyeOff className="w-3.5 h-3.5" />
            ) : (
              <Eye className="w-3.5 h-3.5" />
            )}
            Hold to Compare Bare Hand
          </button>
        </div>

        {/* Refresh Render */}
        <button
          onClick={renderComposite}
          disabled={isRendering}
          className="px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer flex items-center gap-1.5 bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 disabled:opacity-50"
          title="Re-render 6-pass PBR composite"
        >
          <Sparkles className={`w-3.5 h-3.5 ${isRendering ? "animate-spin" : ""}`} />
          Render Composite
        </button>
      </div>
    </div>
  );
};
