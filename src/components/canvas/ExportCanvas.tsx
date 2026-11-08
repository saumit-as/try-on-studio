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
  Loader2,
  Download,
} from "lucide-react";
import {
  HandState,
  LightingState,
  TrayState,
} from "../../types";
import { callSidecar } from "../../services/sidecar";

export type ExportAspectRatio = "original" | "1:1" | "4:5" | "16:9";

interface ExportCanvasProps {
  trayState: TrayState;
  handState: HandState;
  lightingState: LightingState;
  aspectRatio: ExportAspectRatio;
  setAspectRatio: (r: ExportAspectRatio) => void;
  onQuickExport?: () => void;
  isExporting?: boolean;
}

export const ExportCanvas: React.FC<ExportCanvasProps> = ({
  trayState,
  handState,
  lightingState,
  aspectRatio,
  setAspectRatio,
  onQuickExport,
  isExporting = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Viewport Pan & Zoom State
  const [zoom, setZoom] = useState<number>(0.28);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [isSpacePressed, setIsSpacePressed] = useState<boolean>(false);
  const panStartRef = useRef<{
    startX: number;
    startY: number;
    initPanX: number;
    initPanY: number;
  } | null>(null);

  // Rendered composite preview image
  const [compositeImageSrc, setCompositeImageSrc] = useState<string | null>(null);
  const [isRendering, setIsRendering] = useState<boolean>(false);

  const templateWidth = handState.currentTemplate?.imageSize[0] || 1536;
  const templateHeight = handState.currentTemplate?.imageSize[1] || 2752;

  // Calculate target framed dimensions based on selected aspect ratio
  const getFramedDimensions = useCallback((): [number, number] => {
    const w = templateWidth;
    const h = templateHeight;
    if (aspectRatio === "1:1") {
      const dim = Math.max(w, h);
      return [dim, dim];
    } else if (aspectRatio === "4:5") {
      const reqW = Math.round(h * 0.8);
      return reqW >= w ? [reqW, h] : [w, Math.round(w / 0.8)];
    } else if (aspectRatio === "16:9") {
      const reqW = Math.round(h * (16.0 / 9.0));
      return reqW >= w ? [reqW, h] : [w, Math.round(w * (9.0 / 16.0))];
    }
    return [w, h];
  }, [templateWidth, templateHeight, aspectRatio]);

  const [framedWidth, framedHeight] = getFramedDimensions();

  // Print dimensions at 300 DPI in inches
  const printWidthInches = (framedWidth / 300).toFixed(2);
  const printHeightInches = (framedHeight / 300).toFixed(2);

  // Auto-fit mockup into viewport
  const fitToContainer = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth <= 0 || clientHeight <= 0) return;

    const padding = 64;
    const availW = clientWidth - padding;
    const availH = clientHeight - padding;

    const scale = Math.min(availW / framedWidth, availH / framedHeight, 1.0);
    const initialZoom = Math.max(0.12, Math.min(scale, 2.0));

    setZoom(initialZoom);
    setPan({
      x: (clientWidth - framedWidth * initialZoom) / 2,
      y: (clientHeight - framedHeight * initialZoom) / 2,
    });
  }, [framedWidth, framedHeight]);

  useEffect(() => {
    fitToContainer();
  }, [fitToContainer]);

  // Handle Spacebar hotkey for panning
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
    const newZoom = Math.min(Math.max(0.1, zoom * zoomFactor), 4.0);

    const newPanX = mouseX - (mouseX - pan.x) * (newZoom / zoom);
    const newPanY = mouseY - (mouseY - pan.y) * (newZoom / zoom);

    setZoom(newZoom);
    setPan({ x: newPanX, y: newPanY });
  };

  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
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
    if (isPanning && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.startX;
      const dy = e.clientY - panStartRef.current.startY;
      setPan({
        x: panStartRef.current.initPanX + dx,
        y: panStartRef.current.initPanY + dy,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    panStartRef.current = null;
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

  // Render high-quality composite preview
  useEffect(() => {
    let active = true;
    const fetchComposite = async () => {
      setIsRendering(true);
      try {
        const mappedNails = getMappedNailsDict();
        const res = await callSidecar<{
          status: string;
          composite_base64: string;
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
          downscale: 1.0, // Full 300 DPI resolution for export tab
        });
        if (active && res && res.status === "ok" && res.composite_base64) {
          setCompositeImageSrc(res.composite_base64);
        }
      } catch (err) {
        console.warn("Export preview render error:", err);
      } finally {
        if (active) setIsRendering(false);
      }
    };
    fetchComposite();
    return () => {
      active = false;
    };
  }, [getMappedNailsDict, handState.currentTemplate, lightingState]);

  const currentDisplaySrc =
    compositeImageSrc || handState.currentTemplate?.imageSrc;

  // Offset inside the framed canvas
  const innerOffsetX = (framedWidth - templateWidth) / 2;
  const innerOffsetY = (framedHeight - templateHeight) / 2;

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      className={`w-full h-full relative overflow-hidden bg-neutral-950 select-none ${
        isPanning || isSpacePressed
          ? "cursor-grab active:cursor-grabbing"
          : "cursor-default"
      }`}
    >
      {/* Transformed Stage Container (Framed to Target Aspect Ratio) */}
      <div
        className="absolute origin-top-left transition-transform duration-75 ease-out shadow-2xl"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          width: `${framedWidth}px`,
          height: `${framedHeight}px`,
          backgroundColor: "#828183", // Neutral studio gray background matching hand template
        }}
      >
        {/* Centered Base Hand & Nails Image */}
        {currentDisplaySrc && (
          <img
            src={currentDisplaySrc}
            alt="Production Export Preview"
            className="absolute object-contain pointer-events-none drop-shadow-xl"
            style={{
              left: `${innerOffsetX}px`,
              top: `${innerOffsetY}px`,
              width: `${templateWidth}px`,
              height: `${templateHeight}px`,
            }}
            draggable={false}
          />
        )}

        {/* Outer Framing Outline & Crop Safe Area */}
        <div className="absolute inset-0 border-2 border-amber-500/60 pointer-events-none">
          {/* Subtle Corner Crop Marks */}
          <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-amber-400" />
          <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-amber-400" />
          <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-amber-400" />
          <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-amber-400" />

          {/* Rule of Thirds Grid (Subtle Guide) */}
          <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 pointer-events-none opacity-20">
            <div className="border-r border-b border-amber-300 border-dashed" />
            <div className="border-r border-b border-amber-300 border-dashed" />
            <div className="border-b border-amber-300 border-dashed" />
            <div className="border-r border-b border-amber-300 border-dashed" />
            <div className="border-r border-b border-amber-300 border-dashed" />
            <div className="border-b border-amber-300 border-dashed" />
            <div className="border-r border-amber-300 border-dashed" />
            <div className="border-r border-amber-300 border-dashed" />
            <div />
          </div>
        </div>
      </div>

      {/* Floating Header Banner (Top Left) */}
      <div className="absolute top-4 left-4 flex items-center gap-2 z-20">
        <div className="flex items-center gap-2.5 bg-neutral-900/90 backdrop-blur-md px-4 py-2 rounded-full border border-neutral-800 text-xs shadow-xl">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
          <span className="text-neutral-100 font-semibold tracking-wide">
            Phase 5: Production Export Stage
          </span>
          <span className="text-neutral-600">•</span>
          <span className="text-amber-300 font-mono text-[11px] font-medium">
            300 DPI Print Ready
          </span>
          <span className="text-neutral-600">•</span>
          <span className="text-neutral-300 font-mono text-[11px]">
            {framedWidth} × {framedHeight} px ({printWidthInches}" × {printHeightInches}")
          </span>
          {isRendering && (
            <span className="text-amber-400 flex items-center gap-1 text-[11px] ml-1">
              <Loader2 className="w-3 h-3 animate-spin" /> Rendering Full-Res...
            </span>
          )}
        </div>
      </div>

      {/* Aspect Ratio Framing Preset Pills (Top Center) */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-neutral-900/90 backdrop-blur-md p-1.5 rounded-full border border-neutral-800 shadow-xl z-20">
        {(
          [
            { id: "original", label: "Original (1536×2752)", desc: "Native" },
            { id: "1:1", label: "1:1 Square (2752×2752)", desc: "Feed" },
            { id: "4:5", label: "4:5 Portrait (2202×2752)", desc: "Post" },
            { id: "16:9", label: "16:9 Banner (4892×2752)", desc: "Hero" },
          ] as const
        ).map((preset) => (
          <button
            key={preset.id}
            onClick={() => setAspectRatio(preset.id)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all cursor-pointer ${
              aspectRatio === preset.id
                ? "bg-amber-500 text-neutral-950 font-semibold shadow-md"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Floating Bottom Viewport Toolbar */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-neutral-900/90 backdrop-blur-md px-3.5 py-2 rounded-full border border-neutral-800 shadow-2xl z-20">
        {/* Zoom Controls */}
        <div className="flex items-center gap-1 border-r border-neutral-800 pr-2">
          <button
            onClick={() => setZoom((z) => Math.max(0.1, z * 0.85))}
            className="p-1.5 rounded-full hover:bg-neutral-800 text-neutral-300 transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-[11px] font-mono text-neutral-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(4.0, z * 1.18))}
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

        {/* Quick Export Trigger */}
        {onQuickExport && (
          <button
            onClick={onQuickExport}
            disabled={isExporting}
            className="px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-neutral-950 shadow-lg disabled:opacity-50"
          >
            {isExporting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Exporting Production Assets...
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                Export Package Now
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
};
