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
  Layers,
  Wand2,
  Crop,
  Loader2,
} from "lucide-react";
import { FingerName, HandState, LandmarkKey } from "../../types";
import {
  FINGER_COLORS,
  LANDMARK_LABELS,
} from "../../constants/handDefaults";
import { callSidecar } from "../../services/sidecar";

interface HandRigCanvasProps {
  handState: HandState;
  setHandState: React.Dispatch<React.SetStateAction<HandState>>;
}

const FINGERS: FingerName[] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const LANDMARK_KEYS: LandmarkKey[] = ["c_c", "c_l", "c_r", "d_l", "d_r"];

export const HandRigCanvas: React.FC<HandRigCanvasProps> = ({
  handState,
  setHandState,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { currentTemplate, activeFinger, selectedLandmark, overlayToggles } =
    handState;

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

  // Landmark Dragging State
  const [draggingLandmark, setDraggingLandmark] = useState<{
    finger: FingerName;
    key: LandmarkKey;
  } | null>(null);

  const [hoveredLandmark, setHoveredLandmark] = useState<{
    finger: FingerName;
    key: LandmarkKey;
  } | null>(null);

  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [boxStart, setBoxStart] = useState<[number, number] | null>(null);
  const [boxCurrent, setBoxCurrent] = useState<[number, number] | null>(null);
  const [isRefittingFinger, setIsRefittingFinger] = useState<boolean>(false);

  const handleAutoDetect = async () => {
    if (!currentTemplate?.imageSrc) return;
    setIsDetecting(true);
    try {
      const res = await callSidecar<{
        status: string;
        landmarks: Record<FingerName, any>;
      }>("auto_detect_hand", {
        image_base64: currentTemplate.imageSrc,
      });
      if (res && res.status === "ok" && res.landmarks) {
        setHandState((p) => {
          if (!p.currentTemplate) return p;
          return {
            ...p,
            currentTemplate: {
              ...p.currentTemplate,
              landmarks: res.landmarks,
            },
          };
        });
      }
    } catch (e) {
      console.warn("Auto-detect failed:", e);
    } finally {
      setIsDetecting(false);
    }
  };

  const imgW = currentTemplate?.imageSize[0] || 1536;
  const imgH = currentTemplate?.imageSize[1] || 2752;

  // Auto-fit hand template into viewport on mount or when template changes
  const fitToContainer = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth <= 0 || clientHeight <= 0) return;

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
  }, [imgW, imgH]);

  useEffect(() => {
    fitToContainer();
  }, [fitToContainer, currentTemplate?.imageSrc]);

  // Handle Spacebar hotkey for panning
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setHandState((p) => ({ ...p, activeFinger: "All" }));
      } else if (e.code === "Space" && !e.repeat) {
        const target = e.target as HTMLElement;
        if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
          setIsSpacePressed(true);
        }
      } else if (e.key === "l" || e.key === "L") {
        const target = e.target as HTMLElement;
        if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
          setHandState((p) => ({
            ...p,
            overlayToggles: {
              ...p.overlayToggles,
              showLandmarks: !p.overlayToggles.showLandmarks,
            },
          }));
        }
      } else if (e.key === "g" || e.key === "G") {
        const target = e.target as HTMLElement;
        if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
          setHandState((p) => ({
            ...p,
            overlayToggles: {
              ...p.overlayToggles,
              showGeometry: !p.overlayToggles.showGeometry,
            },
          }));
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
  }, [setHandState]);

  // Smooth mouse wheel zoom anchored at cursor
  const handleWheel = (e: WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    const newZoom = Math.min(Math.max(0.12, zoom * zoomFactor), 4.5);

    // Zoom anchored to mouse position
    const newPanX = mouseX - (mouseX - pan.x) * (newZoom / zoom);
    const newPanY = mouseY - (mouseY - pan.y) * (newZoom / zoom);

    setZoom(newZoom);
    setPan({ x: newPanX, y: newPanY });
  };

  // Convert client viewport coordinates to unscaled image pixels
  const clientToImageCoords = (
    clientX: number,
    clientY: number
  ): [number, number] => {
    if (!containerRef.current) return [0, 0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = (clientX - rect.left - pan.x) / zoom;
    const y = (clientY - rect.top - pan.y) / zoom;
    return [Math.round(x), Math.round(y)];
  };

  // Pointer Down Handler
  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (e.button === 1 || isSpacePressed) {
      // Pan with middle mouse or spacebar
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        initPanX: pan.x,
        initPanY: pan.y,
      };
      return;
    }

    if (handState.isBoxToolActive && e.button === 0) {
      const coords = clientToImageCoords(e.clientX, e.clientY);
      setBoxStart(coords);
      setBoxCurrent(coords);
      return;
    }
  };

  // Pointer Move Handler (Pan, Landmark Drag, or Box Marquee)
  const handleMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (isPanning && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.startX;
      const dy = e.clientY - panStartRef.current.startY;
      setPan({
        x: panStartRef.current.initPanX + dx,
        y: panStartRef.current.initPanY + dy,
      });
      return;
    }

    if (boxStart) {
      const coords = clientToImageCoords(e.clientX, e.clientY);
      setBoxCurrent(coords);
      return;
    }

    if (draggingLandmark && currentTemplate) {
      const [imgX, imgY] = clientToImageCoords(e.clientX, e.clientY);
      const clampedX = Math.max(0, Math.min(imgW, imgX));
      const clampedY = Math.max(0, Math.min(imgH, imgY));

      const { finger, key } = draggingLandmark;
      setHandState((prev) => {
        if (!prev.currentTemplate) return prev;
        return {
          ...prev,
          currentTemplate: {
            ...prev.currentTemplate,
            landmarks: {
              ...prev.currentTemplate.landmarks,
              [finger]: {
                ...prev.currentTemplate.landmarks[finger],
                [key]: [clampedX, clampedY],
              },
            },
          },
        };
      });
    }
  };

  // Pointer Up Handler (Pan End, Landmark Drop, or Box Refit Execution)
  const handleMouseUp = async () => {
    setIsPanning(false);
    panStartRef.current = null;
    if (draggingLandmark) {
      setDraggingLandmark(null);
    }

    if (boxStart && boxCurrent && currentTemplate) {
      const bx = Math.min(boxStart[0], boxCurrent[0]);
      const by = Math.min(boxStart[1], boxCurrent[1]);
      const bw = Math.abs(boxCurrent[0] - boxStart[0]);
      const bh = Math.abs(boxCurrent[1] - boxStart[1]);
      setBoxStart(null);
      setBoxCurrent(null);

      if (bw >= 15 && bh >= 15) {
        const targetFinger =
          activeFinger === "All" ? "Middle" : activeFinger;
        setIsRefittingFinger(true);
        try {
          const res = await callSidecar<{
            status: string;
            finger_name: FingerName;
            landmarks: any;
          }>("refit_finger", {
            image_base64: currentTemplate.imageSrc,
            finger_name: targetFinger,
            bbox: [
              Math.round(bx),
              Math.round(by),
              Math.round(bw),
              Math.round(bh),
            ],
            handedness: currentTemplate.handedness || "left",
          });

          if (res && res.status === "ok" && res.landmarks) {
            setHandState((prev) => {
              if (!prev.currentTemplate) return prev;
              return {
                ...prev,
                isBoxToolActive: false,
                currentTemplate: {
                  ...prev.currentTemplate,
                  landmarks: {
                    ...prev.currentTemplate.landmarks,
                    [targetFinger]: {
                      ...prev.currentTemplate.landmarks[targetFinger],
                      ...res.landmarks,
                    },
                  },
                },
              };
            });
          }
        } catch (err) {
          console.warn("Refit finger error:", err);
        } finally {
          setIsRefittingFinger(false);
        }
      }
    }
  };

  // Landmark Pin Down Handler
  const handleLandmarkMouseDown = (
    e: ReactMouseEvent,
    finger: FingerName,
    key: LandmarkKey
  ) => {
    e.stopPropagation();
    e.preventDefault();
    setDraggingLandmark({ finger, key });
    setHandState((p) => ({
      ...p,
      selectedLandmark: { finger, key },
      activeFinger: finger,
    }));
  };

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={() => {
        if (activeFinger !== "All" && !isPanning && !draggingLandmark && !boxStart) {
          setHandState((p) => ({ ...p, activeFinger: "All" }));
        }
      }}
      className={`w-full h-full relative overflow-hidden bg-neutral-950 select-none ${
        isPanning || isSpacePressed
          ? "cursor-grab active:cursor-grabbing"
          : handState.isBoxToolActive
          ? "cursor-crosshair"
          : draggingLandmark
          ? "cursor-crosshair"
          : "cursor-default"
      }`}
    >
      {/* Loading banner when refitting finger */}
      {isRefittingFinger && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-neutral-900/95 border border-rose-500/50 px-4 py-2 rounded-full text-xs text-rose-300 shadow-xl z-30 animate-pulse">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-rose-400" />
          <span>
            Refitting {activeFinger === "All" ? "Middle" : activeFinger} nail anchors...
          </span>
        </div>
      )}

      {/* Transformed Stage Container */}
      <div
        className="absolute origin-top-left"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          width: `${imgW}px`,
          height: `${imgH}px`,
        }}
      >
        {/* Layer 1: Base Hand Photograph */}
        {currentTemplate?.imageSrc && (
          <img
            src={currentTemplate.imageSrc}
            alt="Hand Template"
            draggable={false}
            className="w-full h-full object-contain pointer-events-none drop-shadow-2xl"
          />
        )}

        {/* Layer 2: Procedural Cuticle Mask Overlay */}
        {overlayToggles.showCuticleMask && currentTemplate?.cuticleMaskSrc && (
          <img
            src={currentTemplate.cuticleMaskSrc}
            alt="Cuticle Mask Overlay"
            draggable={false}
            className="w-full h-full object-contain pointer-events-none absolute inset-0 opacity-70 mix-blend-screen"
            style={{ filter: "drop-shadow(0 0 6px rgba(16, 185, 129, 0.6))" }}
          />
        )}

        {/* Layer 3: Procedural Contact AO Shadow Overlay */}
        {overlayToggles.showAoMap && currentTemplate?.aoMapSrc && (
          <img
            src={currentTemplate.aoMapSrc}
            alt="Contact AO Overlay"
            draggable={false}
            className="w-full h-full object-contain pointer-events-none absolute inset-0 opacity-85 mix-blend-multiply"
          />
        )}

        {/* Layer 4: Procedural Specular Gloss Ribbon Overlay */}
        {overlayToggles.showSpecular && currentTemplate?.specularMapSrc && (
          <img
            src={currentTemplate.specularMapSrc}
            alt="Specular Ribbon Overlay"
            draggable={false}
            className="w-full h-full object-contain pointer-events-none absolute inset-0 opacity-90 mix-blend-screen"
          />
        )}

        {/* Layer 5: SVG Rigging & Landmarks Overlay */}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${imgW} ${imgH}`}
          className="w-full h-full absolute inset-0 pointer-events-none"
        >
          {/* Render Finger Rig Geometry and Landmarks */}
          {FINGERS.map((fname) => {
            const isTargetFinger = activeFinger === "All" || activeFinger === fname;
            const fingerColor = FINGER_COLORS[fname];
            const lm = currentTemplate?.landmarks?.[fname];
            if (!lm) return null;

            const c_l = lm.c_l;
            const c_c = lm.c_c;
            const c_r = lm.c_r;
            const d_l = lm.d_l;
            const d_r = lm.d_r;

            const distal_mid: [number, number] = [
              (d_l[0] + d_r[0]) * 0.5,
              (d_l[1] + d_r[1]) * 0.5,
            ];

            const opacity = isTargetFinger ? 1.0 : 0.35;

            return (
              <g key={fname} style={{ opacity }}>
                {/* 1. Shaded Nail-Bed Polygon & Wireframe */}
                {overlayToggles.showGeometry && (
                  <>
                    {/* Nail Bed Fill */}
                    <path
                      d={`M ${c_l[0]} ${c_l[1]} Q ${c_c[0]} ${c_c[1]} ${c_r[0]} ${c_r[1]} L ${d_r[0]} ${d_r[1]} L ${d_l[0]} ${d_l[1]} Z`}
                      fill={fingerColor.hex}
                      fillOpacity={0.16}
                      stroke="none"
                    />

                    {/* Cuticle Proximal U-Arch Curve */}
                    <path
                      d={`M ${c_l[0]} ${c_l[1]} Q ${c_c[0]} ${c_c[1]} ${c_r[0]} ${c_r[1]}`}
                      fill="none"
                      stroke={fingerColor.hex}
                      strokeWidth={3}
                      strokeLinecap="round"
                    />

                    {/* Lateral Sidewalls */}
                    <line
                      x1={c_l[0]}
                      y1={c_l[1]}
                      x2={d_l[0]}
                      y2={d_l[1]}
                      stroke={fingerColor.hex}
                      strokeWidth={2.5}
                      strokeLinecap="round"
                    />
                    <line
                      x1={c_r[0]}
                      y1={c_r[1]}
                      x2={d_r[0]}
                      y2={d_r[1]}
                      stroke={fingerColor.hex}
                      strokeWidth={2.5}
                      strokeLinecap="round"
                    />

                    {/* Distal Free-Edge Baseline */}
                    <line
                      x1={d_l[0]}
                      y1={d_l[1]}
                      x2={d_r[0]}
                      y2={d_r[1]}
                      stroke={fingerColor.hex}
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      strokeLinecap="round"
                    />

                    {/* Longitudinal Finger Axis Vector */}
                    <line
                      x1={c_c[0]}
                      y1={c_c[1]}
                      x2={distal_mid[0]}
                      y2={distal_mid[1]}
                      stroke={fingerColor.lightHex}
                      strokeWidth={2}
                      strokeDasharray="4 3"
                    />
                    <circle
                      cx={distal_mid[0]}
                      cy={distal_mid[1]}
                      r={3.5}
                      fill={fingerColor.lightHex}
                    />
                  </>
                )}

                {/* 2. Interactive Draggable 5-Point Landmark Pins */}
                {overlayToggles.showLandmarks &&
                  LANDMARK_KEYS.map((k) => {
                    const coords = lm[k];
                    if (!coords) return null;

                    const isSelected =
                      selectedLandmark?.finger === fname &&
                      selectedLandmark.key === k;
                    const isHovered =
                      hoveredLandmark?.finger === fname &&
                      hoveredLandmark.key === k;
                    const isApex = k === "c_c";

                    const hitRadius = 18;
                    const ringRadius = isSelected || isHovered ? 10 : 7;
                    const innerRadius = isApex ? 4.5 : 3.5;

                    return (
                      <g
                        key={k}
                        className="pointer-events-auto"
                        onMouseEnter={() =>
                          setHoveredLandmark({ finger: fname, key: k })
                        }
                        onMouseLeave={() => setHoveredLandmark(null)}
                        onMouseDown={(e) =>
                          handleLandmarkMouseDown(e, fname, k)
                        }
                      >
                        {/* Invisible Large Hit Target */}
                        <circle
                          cx={coords[0]}
                          cy={coords[1]}
                          r={hitRadius}
                          fill="transparent"
                          className="cursor-pointer"
                        />

                        {/* Outer Glow Halo */}
                        {(isSelected || isHovered) && (
                          <circle
                            cx={coords[0]}
                            cy={coords[1]}
                            r={ringRadius + 5}
                            fill={fingerColor.hex}
                            fillOpacity={0.25}
                            className="animate-pulse"
                          />
                        )}

                        {/* Outer Ring */}
                        <circle
                          cx={coords[0]}
                          cy={coords[1]}
                          r={ringRadius}
                          fill="#0a0a0a"
                          stroke={
                            isSelected
                              ? "#ffffff"
                              : isHovered
                              ? fingerColor.lightHex
                              : fingerColor.hex
                          }
                          strokeWidth={isSelected ? 2.5 : 2}
                          className="drop-shadow-md"
                        />

                        {/* Inner Core Dot or Diamond for Apex */}
                        {isApex ? (
                          <rect
                            x={coords[0] - innerRadius}
                            y={coords[1] - innerRadius}
                            width={innerRadius * 2}
                            height={innerRadius * 2}
                            transform={`rotate(45 ${coords[0]} ${coords[1]})`}
                            fill={isSelected ? "#ffffff" : "#f59e0b"}
                          />
                        ) : (
                          <circle
                            cx={coords[0]}
                            cy={coords[1]}
                            r={innerRadius}
                            fill={isSelected ? "#ffffff" : fingerColor.hex}
                          />
                        )}

                        {/* Sub-pixel Coordinate Tooltip on Hover/Active */}
                        {(isSelected || isHovered) && (
                          <g transform={`translate(${coords[0]}, ${coords[1] - 18})`}>
                            <rect
                              x={-44}
                              y={-20}
                              width={88}
                              height={20}
                              rx={4}
                              fill="#171717"
                              stroke="#404040"
                              strokeWidth={1}
                              className="drop-shadow-lg"
                            />
                            <text
                              x={0}
                              y={-7}
                              textAnchor="middle"
                              fill="#f5f5f5"
                              fontSize={10}
                              fontFamily="monospace"
                              fontWeight="600"
                            >
                              {LANDMARK_LABELS[k].short}: {Math.round(coords[0])},{Math.round(coords[1])}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}
              </g>
            );
          })}

          {/* Active Bounding Box Marquee for ROI Refit */}
          {boxStart && boxCurrent && (
            <g>
              <rect
                x={Math.min(boxStart[0], boxCurrent[0])}
                y={Math.min(boxStart[1], boxCurrent[1])}
                width={Math.abs(boxCurrent[0] - boxStart[0])}
                height={Math.abs(boxCurrent[1] - boxStart[1])}
                fill={
                  FINGER_COLORS[
                    activeFinger === "All" ? "Middle" : activeFinger
                  ].hex
                }
                fillOpacity={0.25}
                stroke={
                  FINGER_COLORS[
                    activeFinger === "All" ? "Middle" : activeFinger
                  ].hex
                }
                strokeWidth={2}
                strokeDasharray="6 3"
                rx={4}
              />
              <text
                x={Math.min(boxStart[0], boxCurrent[0]) + 6}
                y={Math.max(16, Math.min(boxStart[1], boxCurrent[1]) - 8)}
                fill={
                  FINGER_COLORS[
                    activeFinger === "All" ? "Middle" : activeFinger
                  ].lightHex
                }
                fontSize={12}
                fontFamily="sans-serif"
                fontWeight="bold"
              >
                Refit {activeFinger === "All" ? "Middle" : activeFinger}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Floating Bottom Control Toolbar */}
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

        {/* Overlay Pass Toggles */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          {/* Toggle Landmarks */}
          <button
            onClick={() =>
              setHandState((p) => ({
                ...p,
                overlayToggles: {
                  ...p.overlayToggles,
                  showLandmarks: !p.overlayToggles.showLandmarks,
                },
              }))
            }
            className={`px-2 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              overlayToggles.showLandmarks
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Toggle Landmarks (L)"
          >
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            Anchors
          </button>

          {/* Toggle Geometry */}
          <button
            onClick={() =>
              setHandState((p) => ({
                ...p,
                overlayToggles: {
                  ...p.overlayToggles,
                  showGeometry: !p.overlayToggles.showGeometry,
                },
              }))
            }
            className={`px-2 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              overlayToggles.showGeometry
                ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Toggle Wireframe Geometry (G)"
          >
            <Layers className="w-3.5 h-3.5" />
            Wireframe
          </button>

          {/* Cuticle Mask Toggle */}
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
            className={`px-2 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              overlayToggles.showCuticleMask
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Toggle Cuticle Skin Occlusion Pass"
          >
            Cuticle Tuck
          </button>

          {/* Contact AO Toggle */}
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
            className={`px-2 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer flex items-center gap-1.5 ${
              overlayToggles.showAoMap
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:text-neutral-200"
            }`}
            title="Toggle Contact AO Shadow Pass"
          >
            AO Shadow
          </button>
        </div>

        {/* Quick Finger Switcher */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          <button
            onClick={() => setHandState((p) => ({ ...p, activeFinger: "All" }))}
            className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
              activeFinger === "All"
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            All
          </button>
          {FINGERS.map((f) => (
            <button
              key={f}
              onClick={() => setHandState((p) => ({ ...p, activeFinger: f }))}
              className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer flex items-center gap-1 ${
                activeFinger === f
                  ? "bg-neutral-700 text-neutral-100"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ backgroundColor: FINGER_COLORS[f].hex }}
              />
              {f[0]}
            </button>
          ))}
        </div>

        {/* Box Refit Mode Toggle */}
        <button
          onClick={() =>
            setHandState((p) => ({ ...p, isBoxToolActive: !p.isBoxToolActive }))
          }
          className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer flex items-center gap-1 border ${
            handState.isBoxToolActive
              ? "bg-rose-500/30 text-rose-200 border-rose-500/60 shadow-sm animate-pulse"
              : "bg-neutral-800 text-neutral-300 border-neutral-700/60 hover:text-white"
          }`}
          title={`Draw bounding box over ${
            activeFinger === "All" ? "target finger" : activeFinger
          } to refit nail anchors`}
        >
          <Crop className="w-3.5 h-3.5 text-rose-400" />
          Box Refit
        </button>

        {/* 1-Click Auto-Detect Hand Rig */}
        <button
          onClick={handleAutoDetect}
          disabled={isDetecting}
          className="px-2 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer flex items-center gap-1 bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 disabled:opacity-50"
          title="Auto-Detect Hand & Fingernail Anchors (CV)"
        >
          <Wand2 className={`w-3.5 h-3.5 ${isDetecting ? "animate-spin" : ""}`} />
          Auto-Detect
        </button>
      </div>
    </div>
  );
};
