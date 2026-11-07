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
  Wand2,
} from "lucide-react";
import { ActiveTab, TrayState, DetectedSlot, FingerName, PenAnchor, HandState, LightingState } from "../../types";
import { callSidecar } from "../../services/sidecar";
import { HandRigCanvas } from "./HandRigCanvas";
import { CompositeCanvas } from "./CompositeCanvas";

interface CanvasViewportProps {
  activeTab: ActiveTab;
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
  handState: HandState;
  setHandState: React.Dispatch<React.SetStateAction<HandState>>;
  lightingState: LightingState;
  setLightingState: React.Dispatch<React.SetStateAction<LightingState>>;
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

// Sample cubic Bezier curve segment into discrete points for OpenCV contour
function sampleCubicBezier(
  p0: [number, number],
  c1: [number, number],
  c2: [number, number],
  p1: [number, number],
  steps: number = 16
): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const t2 = t * t;
    const x =
      mt2 * mt * p0[0] +
      3 * mt2 * t * c1[0] +
      3 * mt * t2 * c2[0] +
      t2 * t * p1[0];
    const y =
      mt2 * mt * p0[1] +
      3 * mt2 * t * c1[1] +
      3 * mt * t2 * c2[1] +
      t2 * t * p1[1];
    pts.push([Math.round(x), Math.round(y)]);
  }
  return pts;
}

// Construct SVG cubic Bezier path 'd' string
function constructPathD(
  anchors: PenAnchor[],
  closed: boolean,
  mousePos: [number, number] | null,
  isNearStart: boolean
): string {
  if (anchors.length === 0) return "";
  let d = `M ${anchors[0].x} ${anchors[0].y}`;
  for (let i = 1; i < anchors.length; i++) {
    const prev = anchors[i - 1];
    const curr = anchors[i];
    d += ` C ${prev.handleOut[0]} ${prev.handleOut[1]}, ${curr.handleIn[0]} ${curr.handleIn[1]}, ${curr.x} ${curr.y}`;
  }
  if (closed) {
    const last = anchors[anchors.length - 1];
    const first = anchors[0];
    d += ` C ${last.handleOut[0]} ${last.handleOut[1]}, ${first.handleIn[0]} ${first.handleIn[1]}, ${first.x} ${first.y} Z`;
  } else if (mousePos) {
    const last = anchors[anchors.length - 1];
    if (isNearStart && anchors.length >= 3) {
      const first = anchors[0];
      d += ` C ${last.handleOut[0]} ${last.handleOut[1]}, ${first.handleIn[0]} ${first.handleIn[1]}, ${first.x} ${first.y}`;
    } else {
      d += ` C ${last.handleOut[0]} ${last.handleOut[1]}, ${mousePos[0]} ${mousePos[1]}, ${mousePos[0]} ${mousePos[1]}`;
    }
  }
  return d;
}

// Generate smooth default Bezier curve anchors from bounding box for instant editing
function generateDefaultPenAnchors(bbox: [number, number, number, number]): PenAnchor[] {
  const [bx, by, bw, bh] = bbox;
  const cx = Math.round(bx + bw / 2);
  const topY = by;
  const botY = by + bh;

  return [
    {
      x: cx,
      y: topY,
      handleIn: [Math.round(cx - bw * 0.28), topY],
      handleOut: [Math.round(cx + bw * 0.28), topY],
    },
    {
      x: bx + bw,
      y: Math.round(by + bh * 0.4),
      handleIn: [bx + bw, Math.round(by + bh * 0.22)],
      handleOut: [bx + bw, Math.round(by + bh * 0.65)],
    },
    {
      x: Math.round(bx + bw * 0.88),
      y: Math.round(by + bh * 0.88),
      handleIn: [Math.round(bx + bw * 0.94), Math.round(by + bh * 0.78)],
      handleOut: [Math.round(bx + bw * 0.75), Math.round(by + bh * 0.98)],
    },
    {
      x: cx,
      y: botY,
      handleIn: [Math.round(cx + bw * 0.2), botY],
      handleOut: [Math.round(cx - bw * 0.2), botY],
    },
    {
      x: Math.round(bx + bw * 0.12),
      y: Math.round(by + bh * 0.88),
      handleIn: [Math.round(bx + bw * 0.25), Math.round(by + bh * 0.98)],
      handleOut: [Math.round(bx + bw * 0.06), Math.round(by + bh * 0.78)],
    },
    {
      x: bx,
      y: Math.round(by + bh * 0.4),
      handleIn: [bx, Math.round(by + bh * 0.65)],
      handleOut: [bx, Math.round(by + bh * 0.22)],
    },
  ];
}

export const CanvasViewport: React.FC<CanvasViewportProps> = ({
  activeTab,
  trayState,
  setTrayState,
  handState,
  setHandState,
  lightingState,
  setLightingState,
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

  // Photoshop-Style Bezier Pen Tool State
  const [penAnchors, setPenAnchors] = useState<PenAnchor[]>([]);
  const [isPathClosed, setIsPathClosed] = useState<boolean>(false);
  const [currentMousePos, setCurrentMousePos] = useState<[number, number] | null>(null);
  const [isNearStartPoint, setIsNearStartPoint] = useState<boolean>(false);
  const [isAltPressed, setIsAltPressed] = useState<boolean>(false);
  const [selectedAnchorIdx, setSelectedAnchorIdx] = useState<number | null>(null);
  const penDragRef = useRef<{
    active: boolean;
    anchorIndex: number;
    dragType: "create-tangent" | "move-anchor" | "move-handle-in" | "move-handle-out" | "bend-segment";
    startX: number;
    startY: number;
    initialAnchor?: PenAnchor;
    initialNextAnchor?: PenAnchor;
  } | null>(null);

  // Auto-load saved pen pattern for selected slot when in Pen Tool (allows editing existing patterns)
  useEffect(() => {
    if (trayState.activeTool === "pen") {
      const saved = trayState.slotPenAnchors?.[trayState.selectedSlot];
      if (saved && saved.length > 0) {
        setPenAnchors(JSON.parse(JSON.stringify(saved)));
        setIsPathClosed(true);
      } else {
        setPenAnchors([]);
        setIsPathClosed(false);
      }
    }
  }, [trayState.selectedSlot, trayState.activeTool, trayState.slotPenAnchors]);

  // Convert auto-detected contour to 6 editable Bezier anchors
  const handleConvertAutoToPen = () => {
    const activeSlot = trayState.detectedSlots.find(
      (s) => s.slot_id === trayState.selectedSlot
    );
    if (!activeSlot) return;
    const defaultAnchors = generateDefaultPenAnchors(activeSlot.bbox);
    setPenAnchors(defaultAnchors);
    setIsPathClosed(true);
    setSelectedAnchorIdx(0);
  };

  // Toggle smooth curve / sharp corner on an anchor point (e.g. last point or closing point)
  const toggleAnchorSmooth = useCallback((idx: number) => {
    setPenAnchors((prev) => {
      if (!prev[idx]) return prev;
      const next = [...prev];
      const anchor = next[idx];
      const n = next.length;
      if (n < 2) return prev;

      const hasHandles =
        Math.hypot(anchor.handleOut[0] - anchor.x, anchor.handleOut[1] - anchor.y) > 2 ||
        Math.hypot(anchor.handleIn[0] - anchor.x, anchor.handleIn[1] - anchor.y) > 2;

      if (hasHandles) {
        // Convert smooth curve to sharp corner
        next[idx] = {
          ...anchor,
          handleIn: [anchor.x, anchor.y],
          handleOut: [anchor.x, anchor.y],
        };
      } else {
        // Convert sharp corner to smooth curve with natural tangent vectors
        const prevAnchor = next[(idx - 1 + n) % n];
        const nextAnchor = next[(idx + 1) % n];

        let vx = nextAnchor.x - prevAnchor.x;
        let vy = nextAnchor.y - prevAnchor.y;
        let len = Math.hypot(vx, vy);
        if (len < 1e-3) {
          vx = 1;
          vy = 0;
          len = 1;
        }
        const nx = vx / len;
        const ny = vy / len;

        const dPrev = Math.hypot(anchor.x - prevAnchor.x, anchor.y - prevAnchor.y);
        const dNext = Math.hypot(nextAnchor.x - anchor.x, nextAnchor.y - anchor.y);
        const hLenOut = Math.max(16, dNext * 0.33);
        const hLenIn = Math.max(16, dPrev * 0.33);

        next[idx] = {
          ...anchor,
          handleOut: [Math.round(anchor.x + nx * hLenOut), Math.round(anchor.y + ny * hLenOut)],
          handleIn: [Math.round(anchor.x - nx * hLenIn), Math.round(anchor.y - ny * hLenIn)],
        };
      }
      return next;
    });
  }, []);

  // Delete selected anchor point from pen path
  const deleteSelectedAnchor = useCallback(() => {
    if (selectedAnchorIdx === null) return;
    setPenAnchors((prev) => {
      if (prev.length <= 3) return prev;
      return prev.filter((_, i) => i !== selectedAnchorIdx);
    });
    setSelectedAnchorIdx(null);
  }, [selectedAnchorIdx]);

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
      setPenAnchors([]);

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
      const isCustomContour =
        !!(slot.pen_anchors && slot.pen_anchors.length > 0) ||
        !!(trayState.slotPenAnchors?.[slot.slot_id] && trayState.slotPenAnchors[slot.slot_id].length > 0);

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
        exact_mask: isCustomContour,
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

  // Handle clicking directly on an anchor point
  const handleAnchorMouseDown = (e: ReactMouseEvent, idx: number) => {
    e.stopPropagation();
    e.preventDefault();
    const pt = getSvgPoint(e);
    if (!pt) return;

    setSelectedAnchorIdx(idx);

    if (e.altKey || isAltPressed) {
      // Alt-click on an anchor: toggle handles (collapse to sharp corner)
      const anchor = penAnchors[idx];
      const hasHandles =
        Math.hypot(anchor.handleOut[0] - anchor.x, anchor.handleOut[1] - anchor.y) > 2 ||
        Math.hypot(anchor.handleIn[0] - anchor.x, anchor.handleIn[1] - anchor.y) > 2;

      if (hasHandles) {
        // Collapse handles to convert smooth point into a sharp corner
        setPenAnchors((prev) => {
          const next = [...prev];
          if (next[idx]) {
            next[idx] = {
              ...next[idx],
              handleIn: [next[idx].x, next[idx].y],
              handleOut: [next[idx].x, next[idx].y],
            };
          }
          return next;
        });
        return;
      } else {
        // Drag out new tangent handles from sharp corner
        penDragRef.current = {
          active: true,
          anchorIndex: idx,
          dragType: "create-tangent",
          startX: anchor.x,
          startY: anchor.y,
        };
        return;
      }
    }

    // Clicking start point when path is open and >= 3 points closes the loop and activates drag curving!
    if (!isPathClosed && idx === 0 && penAnchors.length >= 3) {
      setIsPathClosed(true);
      setIsNearStartPoint(false);
      setSelectedAnchorIdx(0);
      penDragRef.current = {
        active: true,
        anchorIndex: 0,
        dragType: "create-tangent",
        startX: penAnchors[0].x,
        startY: penAnchors[0].y,
      };
      return;
    }

    // Move anchor point
    penDragRef.current = {
      active: true,
      anchorIndex: idx,
      dragType: "move-anchor",
      startX: pt[0],
      startY: pt[1],
      initialAnchor: { ...penAnchors[idx] },
    };
  };

  // Handle clicking directly on a blue tangent handle (direction point)
  const handleHandleMouseDown = (
    e: ReactMouseEvent,
    idx: number,
    handleType: "handleOut" | "handleIn"
  ) => {
    e.stopPropagation();
    e.preventDefault();
    const pt = getSvgPoint(e);
    if (!pt) return;

    penDragRef.current = {
      active: true,
      anchorIndex: idx,
      dragType: handleType === "handleOut" ? "move-handle-out" : "move-handle-in",
      startX: pt[0],
      startY: pt[1],
    };
  };

  // Handle clicking/dragging directly on a curve or straight line segment
  const handleSegmentMouseDown = (e: ReactMouseEvent, segIdx: number) => {
    e.stopPropagation();
    e.preventDefault();
    const pt = getSvgPoint(e);
    if (!pt) return;

    const curr = penAnchors[segIdx];
    const nextIdx = (segIdx + 1) % penAnchors.length;
    const next = penAnchors[nextIdx];
    if (!curr || !next) return;

    // Check if curr.handleOut or next.handleIn need initial sprout
    const hasOut = Math.hypot(curr.handleOut[0] - curr.x, curr.handleOut[1] - curr.y) > 2;
    const hasIn = Math.hypot(next.handleIn[0] - next.x, next.handleIn[1] - next.y) > 2;

    const initHOut: [number, number] = hasOut
      ? [...curr.handleOut]
      : [Math.round(curr.x + (next.x - curr.x) * 0.33), Math.round(curr.y + (next.y - curr.y) * 0.33)];

    const initHIn: [number, number] = hasIn
      ? [...next.handleIn]
      : [Math.round(next.x - (next.x - curr.x) * 0.33), Math.round(next.y - (next.y - curr.y) * 0.33)];

    penDragRef.current = {
      active: true,
      anchorIndex: segIdx,
      dragType: "bend-segment",
      startX: pt[0],
      startY: pt[1],
      initialAnchor: {
        x: curr.x,
        y: curr.y,
        handleOut: initHOut,
        handleIn: [...curr.handleIn],
      },
      initialNextAnchor: {
        x: next.x,
        y: next.y,
        handleIn: initHIn,
        handleOut: [...next.handleOut],
      },
    };
    setSelectedAnchorIdx(segIdx);
  };

  // SVG mouse down handler: supports hold-and-drag Photoshop-style Pen Tool curving
  const handleSvgMouseDown = (e: ReactMouseEvent<SVGSVGElement>) => {
    if (e.button === 1 || isSpacePressed || trayState.activeTool === "pan") {
      e.preventDefault();
      startPan(e.clientX, e.clientY);
      return;
    }

    if (trayState.activeTool === "pen" && e.button === 0) {
      e.preventDefault();
      const pt = getSvgPoint(e);
      if (!pt) return;

      // If loop is already closed, don't add random points on empty click
      if (isPathClosed) return;

      // If clicking near start point and has >= 3 anchors, close loop with hold-drag curve!
      if (penAnchors.length >= 3 && isNearStartPoint) {
        setIsPathClosed(true);
        setIsNearStartPoint(false);
        setSelectedAnchorIdx(0);
        penDragRef.current = {
          active: true,
          anchorIndex: 0,
          dragType: "create-tangent",
          startX: penAnchors[0].x,
          startY: penAnchors[0].y,
        };
        return;
      }

      // Add new anchor and start tangent handle dragging
      const newAnchor: PenAnchor = {
        x: Math.round(pt[0]),
        y: Math.round(pt[1]),
        handleIn: [Math.round(pt[0]), Math.round(pt[1])],
        handleOut: [Math.round(pt[0]), Math.round(pt[1])],
      };

      const newIdx = penAnchors.length;
      setPenAnchors((prev) => [...prev, newAnchor]);
      setSelectedAnchorIdx(newIdx);
      penDragRef.current = {
        active: true,
        anchorIndex: newIdx,
        dragType: "create-tangent",
        startX: pt[0],
        startY: pt[1],
      };
    }
  };

  // Svg mouse move handler: handles transform drag and Pen tool preview
  const handleSvgMouseMove = (e: ReactMouseEvent<SVGSVGElement>) => {
    const pt = getSvgPoint(e);
    if (!pt) return;

    // 1. Pen tool active
    if (trayState.activeTool === "pen") {
      setCurrentMousePos(pt);

      if (penDragRef.current && penDragRef.current.active) {
        const { anchorIndex, dragType, startX, startY } = penDragRef.current;
        const dx = pt[0] - startX;
        const dy = pt[1] - startY;
        const isBreaking = e.altKey || isAltPressed;

        if (dragType === "create-tangent") {
          // Photoshop hold-and-drag: handleOut follows cursor, handleIn reflects symmetrically unless Alt held
          setPenAnchors((prev) => {
            const next = [...prev];
            if (next[anchorIndex]) {
              const anchor = next[anchorIndex];
              if (isBreaking) {
                next[anchorIndex] = {
                  ...anchor,
                  handleOut: [Math.round(startX + dx), Math.round(startY + dy)],
                };
              } else {
                next[anchorIndex] = {
                  ...anchor,
                  handleOut: [Math.round(startX + dx), Math.round(startY + dy)],
                  handleIn: [Math.round(startX - dx), Math.round(startY - dy)],
                };
              }
            }
            return next;
          });
        } else if (dragType === "move-handle-out") {
          setPenAnchors((prev) => {
            const next = [...prev];
            if (next[anchorIndex]) {
              const anchor = next[anchorIndex];
              const newHandleOut: [number, number] = [Math.round(pt[0]), Math.round(pt[1])];
              if (isBreaking) {
                // BREAK CURVE: only move handleOut, leave handleIn unchanged!
                next[anchorIndex] = {
                  ...anchor,
                  handleOut: newHandleOut,
                };
              } else {
                // Symmetrical tangent handle adjustment
                const hdx = newHandleOut[0] - anchor.x;
                const hdy = newHandleOut[1] - anchor.y;
                next[anchorIndex] = {
                  ...anchor,
                  handleOut: newHandleOut,
                  handleIn: [Math.round(anchor.x - hdx), Math.round(anchor.y - hdy)],
                };
              }
            }
            return next;
          });
        } else if (dragType === "move-handle-in") {
          setPenAnchors((prev) => {
            const next = [...prev];
            if (next[anchorIndex]) {
              const anchor = next[anchorIndex];
              const newHandleIn: [number, number] = [Math.round(pt[0]), Math.round(pt[1])];
              if (isBreaking) {
                // BREAK CURVE: only move handleIn, leave handleOut unchanged!
                next[anchorIndex] = {
                  ...anchor,
                  handleIn: newHandleIn,
                };
              } else {
                // Symmetrical tangent handle adjustment
                const hdx = newHandleIn[0] - anchor.x;
                const hdy = newHandleIn[1] - anchor.y;
                next[anchorIndex] = {
                  ...anchor,
                  handleIn: newHandleIn,
                  handleOut: [Math.round(anchor.x - hdx), Math.round(anchor.y - hdy)],
                };
              }
            }
            return next;
          });
        } else if (dragType === "move-anchor" && penDragRef.current.initialAnchor) {
          const init = penDragRef.current.initialAnchor;
          setPenAnchors((prev) => {
            const next = [...prev];
            if (next[anchorIndex]) {
              next[anchorIndex] = {
                x: Math.round(init.x + dx),
                y: Math.round(init.y + dy),
                handleIn: [Math.round(init.handleIn[0] + dx), Math.round(init.handleIn[1] + dy)],
                handleOut: [Math.round(init.handleOut[0] + dx), Math.round(init.handleOut[1] + dy)],
              };
            }
            return next;
          });
        } else if (dragType === "bend-segment" && penDragRef.current.initialAnchor && penDragRef.current.initialNextAnchor) {
          const initA = penDragRef.current.initialAnchor;
          const initB = penDragRef.current.initialNextAnchor;
          const nextIdx = (anchorIndex + 1) % penAnchors.length;

          setPenAnchors((prev) => {
            const nextList = [...prev];
            if (nextList[anchorIndex] && nextList[nextIdx]) {
              nextList[anchorIndex] = {
                ...nextList[anchorIndex],
                handleOut: [Math.round(initA.handleOut[0] + dx * 0.75), Math.round(initA.handleOut[1] + dy * 0.75)],
              };
              nextList[nextIdx] = {
                ...nextList[nextIdx],
                handleIn: [Math.round(initB.handleIn[0] + dx * 0.75), Math.round(initB.handleIn[1] + dy * 0.75)],
              };
            }
            return nextList;
          });
        }
        return;
      }

      // Check snapping to start point (only if path is not yet closed)
      if (!isPathClosed && penAnchors.length >= 3) {
        const snapRadius = Math.max(14, 22 / Math.max(0.3, zoom));
        const d = Math.hypot(pt[0] - penAnchors[0].x, pt[1] - penAnchors[0].y);
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

  // SVG mouse up handler: commits handle curving or finishes drag
  const handleSvgMouseUp = (e: ReactMouseEvent<SVGSVGElement>) => {
    if (trayState.activeTool === "pen") {
      if (penDragRef.current && penDragRef.current.active) {
        const pt = getSvgPoint(e);
        if (pt && penDragRef.current.dragType === "create-tangent") {
          const { anchorIndex, startX, startY } = penDragRef.current;
          const dist = Math.hypot(pt[0] - startX, pt[1] - startY);
          // If clicked and released with minimal drag (< 4px screen equivalent) on a point without handles, keep as corner
          if (dist < 4 / Math.max(0.3, zoom)) {
            const anchor = penAnchors[anchorIndex];
            if (anchor) {
              const alreadyHadHandles =
                Math.hypot(anchor.handleOut[0] - anchor.x, anchor.handleOut[1] - anchor.y) > 2 ||
                Math.hypot(anchor.handleIn[0] - anchor.x, anchor.handleIn[1] - anchor.y) > 2;
              if (!alreadyHadHandles) {
                setPenAnchors((prev) => {
                  const next = [...prev];
                  if (next[anchorIndex]) {
                    next[anchorIndex] = {
                      ...next[anchorIndex],
                      handleIn: [next[anchorIndex].x, next[anchorIndex].y],
                      handleOut: [next[anchorIndex].x, next[anchorIndex].y],
                    };
                  }
                  return next;
                });
              }
            }
          }
        } else if (pt && penDragRef.current.dragType === "bend-segment") {
          const { anchorIndex, startX, startY } = penDragRef.current;
          const dist = Math.hypot(pt[0] - startX, pt[1] - startY);
          // If segment was merely clicked (< 4px drag), insert a new anchor point right at cursor!
          if (dist < 4 / Math.max(0.3, zoom)) {
            const insertIdx = anchorIndex + 1;
            const newPt: PenAnchor = {
              x: Math.round(pt[0]),
              y: Math.round(pt[1]),
              handleIn: [Math.round(pt[0]), Math.round(pt[1])],
              handleOut: [Math.round(pt[0]), Math.round(pt[1])],
            };
            setPenAnchors((prev) => {
              const nextList = [...prev];
              nextList.splice(insertIdx, 0, newPt);
              return nextList;
            });
            setSelectedAnchorIdx(insertIdx);
          }
        }
        penDragRef.current = null;
      }
    }
    endDrag();
  };

  // Complete and apply custom pen drawn curved contour (persists to slotPenAnchors)
  const finishPenPath = () => {
    if (penAnchors.length < 3) return;

    // Sample all cubic Bezier segments into dense polygon contour
    const denseContour: [number, number][] = [];
    for (let i = 0; i < penAnchors.length; i++) {
      const curr = penAnchors[i];
      const next = penAnchors[(i + 1) % penAnchors.length];
      const segment = sampleCubicBezier(
        [curr.x, curr.y],
        curr.handleOut,
        next.handleIn,
        [next.x, next.y],
        16
      );
      // Exclude the last point of each segment to avoid duplicate with the start of next
      denseContour.push(...segment.slice(0, -1));
    }

    if (denseContour.length < 3) return;

    const xs = denseContour.map((p) => p[0]);
    const ys = denseContour.map((p) => p[1]);
    const minX = Math.round(Math.min(...xs));
    const minY = Math.round(Math.min(...ys));
    const maxX = Math.round(Math.max(...xs));
    const maxY = Math.round(Math.max(...ys));
    const bw = Math.max(16, maxX - minX);
    const bh = Math.max(16, maxY - minY);
    const newBbox: [number, number, number, number] = [minX, minY, bw, bh];

    const currentSlotId = trayState.selectedSlot;
    const savedAnchors = JSON.parse(JSON.stringify(penAnchors));

    setTrayState((prev) => {
      const updatedSlots = prev.detectedSlots.map((s) =>
        s.slot_id === currentSlotId
          ? {
              ...s,
              bbox: newBbox,
              contour: denseContour,
              pen_anchors: savedAnchors,
            }
          : s
      );

      const activeSlot = updatedSlots.find((s) => s.slot_id === currentSlotId);
      if (activeSlot && prev.trayImageSrc) {
        fetchNailPreview(prev.trayImageSrc, activeSlot);
      }

      return {
        ...prev,
        detectedSlots: updatedSlots,
        slotPenAnchors: {
          ...prev.slotPenAnchors,
          [currentSlotId]: savedAnchors,
        },
        extractedPreviews: {
          ...prev.extractedPreviews,
          [currentSlotId]: "",
        },
        activeTool: "select",
      };
    });

    setIsPathClosed(false);
    setCurrentMousePos(null);
    setIsNearStartPoint(false);
    setSelectedAnchorIdx(null);
    penDragRef.current = null;
  };

  const cancelPenDrawing = () => {
    setPenAnchors([]);
    setIsPathClosed(false);
    setCurrentMousePos(null);
    setIsNearStartPoint(false);
    setSelectedAnchorIdx(null);
    penDragRef.current = null;
    setTrayState((prev) => ({ ...prev, activeTool: "select" }));
  };

  const undoLastPenPoint = () => {
    setSelectedAnchorIdx(null);
    if (isPathClosed) {
      // Re-open closed loop so user can adjust
      setIsPathClosed(false);
    } else {
      setPenAnchors((prev) => prev.slice(0, -1));
    }
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

  // Keyboard shortcut listeners (Ctrl+Z, Space for pan, P for pen, V for select, Alt for break, Esc/Backspace)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === "Alt") {
        setIsAltPressed(true);
      }
      if (e.code === "Space" && !e.repeat) {
        setIsSpacePressed(true);
      } else if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
        // Control + Z: Undo last point / re-open loop
        if (trayState.activeTool === "pen") {
          e.preventDefault();
          undoLastPenPoint();
        }
      } else if (e.key === "p" || e.key === "P") {
        setTrayState((prev) => ({ ...prev, activeTool: "pen", isManualEditMode: false }));
      } else if (e.key === "v" || e.key === "V") {
        setTrayState((prev) => ({ ...prev, activeTool: "select" }));
      } else if (e.key === "h" || e.key === "H") {
        setTrayState((prev) => ({ ...prev, activeTool: "pan" }));
      } else if (e.key === "Escape") {
        if (penAnchors.length > 0) {
          cancelPenDrawing();
        }
      } else if (e.key === "Backspace" || e.key === "Delete") {
        if (trayState.activeTool === "pen") {
          if (selectedAnchorIdx !== null && penAnchors.length > 3) {
            e.preventDefault();
            deleteSelectedAnchor();
          } else if (penAnchors.length > 0) {
            e.preventDefault();
            undoLastPenPoint();
          }
        }
      } else if (e.key === "Enter") {
        if (penAnchors.length >= 3) {
          finishPenPath();
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Alt") {
        setIsAltPressed(false);
      }
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
  }, [penAnchors, isPathClosed, isAltPressed, setTrayState]);

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
                      onMouseDown={handleSvgMouseDown}
                      onMouseMove={handleSvgMouseMove}
                      onMouseUp={handleSvgMouseUp}
                    >
                      {/* Detected Slot Contours */}
                      {trayState.detectedSlots.map((slot, index) => {
                        const isSelected = trayState.selectedSlot === slot.slot_id;
                        const isHovered = trayState.hoveredSlot === slot.slot_id;

                        // When Pen Tool is active: HIDE ALL OTHER SLOTS to completely remove visual clutter!
                        if (trayState.activeTool === "pen" && !isSelected) {
                          return null;
                        }

                        const color = SLOT_PALETTE[index % SLOT_PALETTE.length];
                        const pointsString = slot.contour
                          .map((pt) => `${pt[0]},${pt[1]}`)
                          .join(" ");

                        const [bx, by, bw, bh] = slot.bbox;
                        const centerX = bx + bw / 2;
                        const centerY = by + bh / 2;
                        const assignedFinger = trayState.slotFinger[slot.slot_id];

                        // Scale handles inversely with zoom so they remain a constant ~10-14 screen pixels
                        const screenScale = 1 / Math.max(0.2, zoom);
                        const handleSize = Math.max(8, 14 * screenScale);
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
                            {trayState.activeTool === "pen" ? (
                              penAnchors.length === 0 ? (
                                <polygon
                                  points={pointsString}
                                  fill="none"
                                  stroke="rgba(245, 158, 11, 0.45)"
                                  strokeWidth={Math.max(1, 1.5 * screenScale)}
                                  strokeDasharray={`${Math.max(4, 6 * screenScale)},${Math.max(3, 4 * screenScale)}`}
                                  className="pointer-events-none"
                                />
                              ) : null
                            ) : (
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
                                strokeWidth={
                                  isSelected
                                    ? Math.max(2, 3.5 * screenScale)
                                    : isHovered
                                    ? Math.max(1.5, 3 * screenScale)
                                    : Math.max(1, 2 * screenScale)
                                }
                                strokeLinejoin="round"
                                className="transition-all"
                              />
                            )}

                            {/* Center Number & Finger Badge: Hide when Pen Tool is active */}
                            {trayState.activeTool !== "pen" && (
                              <g transform={`translate(${centerX}, ${centerY})`}>
                                <rect
                                  x={-18 * screenScale}
                                  y={-9 * screenScale}
                                  width={36 * screenScale}
                                  height={18 * screenScale}
                                  rx={9 * screenScale}
                                  fill={isSelected ? "#f59e0b" : "#171717"}
                                  stroke={isSelected ? "#ffffff" : color}
                                  strokeWidth={1.2 * screenScale}
                                  className="shadow-md"
                                />
                                <text
                                  textAnchor="middle"
                                  dominantBaseline="central"
                                  fill={isSelected ? "#000000" : "#ffffff"}
                                  fontSize={Math.max(8, 9 * screenScale)}
                                  fontWeight="bold"
                                  fontFamily="sans-serif"
                                >
                                  {assignedFinger ? assignedFinger[0] : `#${slot.slot_id}`}
                                </text>
                              </g>
                            )}

                            {/* Manual Selection Bounding Box & Handles: NEVER show when activeTool === "pen" */}
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
                                    strokeWidth={1.5 * screenScale}
                                    strokeDasharray={`${Math.max(4, 6 * screenScale)},${Math.max(3, 4 * screenScale)}`}
                                    className="pointer-events-none"
                                  />

                                  <g
                                    transform={`translate(${centerX}, ${by - 14 * screenScale})`}
                                    onMouseDown={(e) => startDrag(e, "move")}
                                    className="cursor-move"
                                  >
                                    <rect
                                      x={-22 * screenScale}
                                      y={-7 * screenScale}
                                      width={44 * screenScale}
                                      height={14 * screenScale}
                                      rx={7 * screenScale}
                                      fill="#f59e0b"
                                      stroke="#ffffff"
                                      strokeWidth={screenScale}
                                    />
                                    <text
                                      textAnchor="middle"
                                      dominantBaseline="central"
                                      fill="#000000"
                                      fontSize={Math.max(7, 8 * screenScale)}
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
                                      rx={2 * screenScale}
                                      fill="#f59e0b"
                                      stroke="#ffffff"
                                      strokeWidth={screenScale}
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
                                        r={handleSize * 0.55}
                                        fill="#06b6d4"
                                        stroke="#ffffff"
                                        strokeWidth={1.5 * screenScale}
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

                      {/* Photoshop-Style Bezier Pen Tool Layer */}
                      {trayState.activeTool === "pen" && (
                        <g className="pen-tool-bezier-layer">
                          {/* Dynamic Cubic Bezier Spline Path */}
                          {penAnchors.length > 0 && (
                            <path
                              d={constructPathD(penAnchors, isPathClosed, null, false)}
                              fill={isPathClosed || penAnchors.length >= 3 ? "rgba(245, 158, 11, 0.22)" : "none"}
                              stroke="#f59e0b"
                              strokeWidth={Math.max(1.5, 2.5 / Math.max(0.3, zoom))}
                              strokeLinejoin="round"
                              strokeLinecap="round"
                              className="pointer-events-none"
                            />
                          )}

                          {/* Interactive Curve Segments: Hover to highlight, Drag to bend, Click to insert point */}
                          {penAnchors.map((anchor, i) => {
                            if (!isPathClosed && i >= penAnchors.length - 1) return null;
                            const nextAnchor = penAnchors[(i + 1) % penAnchors.length];
                            const segD = `M ${anchor.x} ${anchor.y} C ${anchor.handleOut[0]} ${anchor.handleOut[1]}, ${nextAnchor.handleIn[0]} ${nextAnchor.handleIn[1]}, ${nextAnchor.x} ${nextAnchor.y}`;
                            const screenScale = 1 / Math.max(0.2, zoom);
                            return (
                              <path
                                key={`seg-hit-${i}`}
                                d={segD}
                                fill="none"
                                stroke="transparent"
                                strokeWidth={Math.max(14, 20 * screenScale)}
                                strokeLinecap="round"
                                className="cursor-grab pointer-events-auto hover:stroke-amber-400/50 transition-colors"
                                onMouseDown={(e) => handleSegmentMouseDown(e, i)}
                              />
                            );
                          })}

                          {/* Live Rubberband Bezier Curve to Cursor (Only while path is open) */}
                          {penAnchors.length > 0 && !isPathClosed && currentMousePos && !penDragRef.current?.active && (
                            <path
                              d={`M ${penAnchors[penAnchors.length - 1].x} ${
                                penAnchors[penAnchors.length - 1].y
                              } C ${penAnchors[penAnchors.length - 1].handleOut[0]} ${
                                penAnchors[penAnchors.length - 1].handleOut[1]
                              }, ${
                                isNearStartPoint && penAnchors.length >= 3
                                  ? penAnchors[0].x
                                  : currentMousePos[0]
                              } ${
                                isNearStartPoint && penAnchors.length >= 3
                                  ? penAnchors[0].y
                                  : currentMousePos[1]
                              }, ${
                                isNearStartPoint && penAnchors.length >= 3
                                  ? penAnchors[0].x
                                  : currentMousePos[0]
                              } ${
                                isNearStartPoint && penAnchors.length >= 3
                                  ? penAnchors[0].y
                                  : currentMousePos[1]
                              }`}
                              fill="none"
                              stroke={isNearStartPoint && penAnchors.length >= 3 ? "#10b981" : "#f59e0b"}
                              strokeWidth={Math.max(1.5, 2 / Math.max(0.3, zoom))}
                              strokeDasharray={`${Math.max(4, 6 / Math.max(0.3, zoom))},${Math.max(3, 4 / Math.max(0.3, zoom))}`}
                              className="pointer-events-none"
                            />
                          )}

                          {/* Tangent Handles and Anchor Points */}
                          {penAnchors.map((anchor, idx) => {
                            const screenScale = 1 / Math.max(0.2, zoom);
                            const ptSize = Math.max(6, 9 * screenScale);
                            const halfPt = ptSize / 2;
                            const handleRadius = Math.max(3.5, 5 * screenScale);
                            const hitRadius = Math.max(12, 18 * screenScale);
                            const isStart = idx === 0;
                            const isSelected = selectedAnchorIdx === idx;
                            const isSnapping = !isPathClosed && isStart && isNearStartPoint && penAnchors.length >= 3;
                            const hasOutHandle =
                              Math.hypot(anchor.handleOut[0] - anchor.x, anchor.handleOut[1] - anchor.y) > 1.5;
                            const hasInHandle =
                              Math.hypot(anchor.handleIn[0] - anchor.x, anchor.handleIn[1] - anchor.y) > 1.5;

                            return (
                              <g key={idx}>
                                {/* Outgoing Tangent Bar & Blue Handle Dot */}
                                {hasOutHandle && (
                                  <g className="tangent-out-group">
                                    <line
                                      x1={anchor.x}
                                      y1={anchor.y}
                                      x2={anchor.handleOut[0]}
                                      y2={anchor.handleOut[1]}
                                      stroke="#38bdf8"
                                      strokeWidth={Math.max(1, 1.2 * screenScale)}
                                      className="pointer-events-none"
                                    />
                                    {/* Generous invisible hit target for effortless grabbing */}
                                    <circle
                                      cx={anchor.handleOut[0]}
                                      cy={anchor.handleOut[1]}
                                      r={hitRadius}
                                      fill="transparent"
                                      className="cursor-pointer pointer-events-auto"
                                      onMouseDown={(e) => handleHandleMouseDown(e, idx, "handleOut")}
                                    />
                                    {/* Visible blue handle dot */}
                                    <circle
                                      cx={anchor.handleOut[0]}
                                      cy={anchor.handleOut[1]}
                                      r={handleRadius}
                                      fill="#0284c7"
                                      stroke="#ffffff"
                                      strokeWidth={Math.max(1.5, 2 * screenScale)}
                                      className="pointer-events-none drop-shadow-md"
                                    />
                                  </g>
                                )}

                                {/* Incoming Tangent Bar & Blue Handle Dot */}
                                {hasInHandle && (
                                  <g className="tangent-in-group">
                                    <line
                                      x1={anchor.x}
                                      y1={anchor.y}
                                      x2={anchor.handleIn[0]}
                                      y2={anchor.handleIn[1]}
                                      stroke="#38bdf8"
                                      strokeWidth={Math.max(1, 1.2 * screenScale)}
                                      className="pointer-events-none"
                                    />
                                    {/* Generous invisible hit target for effortless grabbing */}
                                    <circle
                                      cx={anchor.handleIn[0]}
                                      cy={anchor.handleIn[1]}
                                      r={hitRadius}
                                      fill="transparent"
                                      className="cursor-pointer pointer-events-auto"
                                      onMouseDown={(e) => handleHandleMouseDown(e, idx, "handleIn")}
                                    />
                                    {/* Visible blue handle dot */}
                                    <circle
                                      cx={anchor.handleIn[0]}
                                      cy={anchor.handleIn[1]}
                                      r={handleRadius}
                                      fill="#0284c7"
                                      stroke="#ffffff"
                                      strokeWidth={Math.max(1.5, 2 * screenScale)}
                                      className="pointer-events-none drop-shadow-md"
                                    />
                                  </g>
                                )}

                                {/* Anchor Vertex Node (Square or Snapping Circle) */}
                                {isSnapping ? (
                                  <circle
                                    cx={anchor.x}
                                    cy={anchor.y}
                                    r={ptSize * 1.3}
                                    fill="#10b981"
                                    stroke="#ffffff"
                                    strokeWidth={Math.max(1.5, 2 * screenScale)}
                                    className="animate-pulse cursor-pointer pointer-events-auto shadow-md"
                                    onMouseDown={(e) => {
                                      e.stopPropagation();
                                      e.preventDefault();
                                      setIsPathClosed(true);
                                      setIsNearStartPoint(false);
                                      setSelectedAnchorIdx(0);
                                      penDragRef.current = {
                                        active: true,
                                        anchorIndex: 0,
                                        dragType: "create-tangent",
                                        startX: penAnchors[0].x,
                                        startY: penAnchors[0].y,
                                      };
                                    }}
                                  />
                                ) : (
                                  <g className="anchor-node-group">
                                    {/* Generous invisible hit target for anchor */}
                                    <circle
                                      cx={anchor.x}
                                      cy={anchor.y}
                                      r={hitRadius}
                                      fill="transparent"
                                      className="cursor-move pointer-events-auto"
                                      onMouseDown={(e) => handleAnchorMouseDown(e, idx)}
                                      onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        toggleAnchorSmooth(idx);
                                      }}
                                    />

                                    {/* Selected Anchor Pulsing Glow Ring */}
                                    {isSelected && (
                                      <rect
                                        x={anchor.x - halfPt - 3 * screenScale}
                                        y={anchor.y - halfPt - 3 * screenScale}
                                        width={ptSize + 6 * screenScale}
                                        height={ptSize + 6 * screenScale}
                                        rx={2.5 * screenScale}
                                        fill="none"
                                        stroke="#f59e0b"
                                        strokeWidth={1.5 * screenScale}
                                        strokeDasharray={`${Math.max(3, 4 * screenScale)},${Math.max(2, 3 * screenScale)}`}
                                        className="pointer-events-none animate-pulse"
                                      />
                                    )}

                                    {/* Visible Anchor Square */}
                                    <rect
                                      x={anchor.x - halfPt}
                                      y={anchor.y - halfPt}
                                      width={ptSize}
                                      height={ptSize}
                                      rx={1.5 * screenScale}
                                      fill={isSelected ? "#f59e0b" : isStart ? "#fbbf24" : "#ffffff"}
                                      stroke="#171717"
                                      strokeWidth={Math.max(1, 1.5 * screenScale)}
                                      className="pointer-events-none shadow-xs"
                                    />
                                  </g>
                                )}
                              </g>
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
                    <div className="flex flex-col">
                      <span className="font-semibold text-neutral-100 flex items-center gap-1.5">
                        <span>Pen Tool: Slot #{trayState.selectedSlot}</span>
                        {isPathClosed && (
                          <span className="px-1.5 py-0.2 bg-emerald-500/20 text-emerald-400 text-[10px] rounded font-bold border border-emerald-500/40">
                            Closed Loop (Editable)
                          </span>
                        )}
                      </span>
                      <span className="text-[10px] text-neutral-400">
                        {isPathClosed
                          ? "Drag points, blue handles, or line segments to tweak. Double-click point to curve/sharpen."
                          : "Click & hold-drag to curve. Hold Alt to break handles. Click start point to close loop."}
                      </span>
                    </div>

                    <div className="h-6 w-px bg-neutral-800" />

                    {/* Selected Anchor Quick Actions: Curve / Make Corner & Delete Point */}
                    {selectedAnchorIdx !== null && penAnchors[selectedAnchorIdx] && (
                      <div className="flex items-center gap-2 bg-neutral-950/80 px-2 py-1 rounded-lg border border-amber-500/30">
                        <span className="text-[11px] font-semibold text-amber-400">
                          Point #{selectedAnchorIdx + 1}
                        </span>

                        <button
                          onClick={() => toggleAnchorSmooth(selectedAnchorIdx)}
                          className="px-2 py-0.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-all border border-amber-500/40"
                          title="Toggle between smooth curved tangent handles and sharp corner (or double-click point)"
                        >
                          <Sliders className="w-3 h-3 text-amber-400" />
                          <span>
                            {Math.hypot(
                              penAnchors[selectedAnchorIdx].handleOut[0] - penAnchors[selectedAnchorIdx].x,
                              penAnchors[selectedAnchorIdx].handleOut[1] - penAnchors[selectedAnchorIdx].y
                            ) > 2 ||
                            Math.hypot(
                              penAnchors[selectedAnchorIdx].handleIn[0] - penAnchors[selectedAnchorIdx].x,
                              penAnchors[selectedAnchorIdx].handleIn[1] - penAnchors[selectedAnchorIdx].y
                            ) > 2
                              ? "Make Corner"
                              : "Curve Point"}
                          </span>
                        </button>

                        {penAnchors.length > 3 && (
                          <button
                            onClick={deleteSelectedAnchor}
                            className="px-1.5 py-0.5 bg-rose-950/50 hover:bg-rose-900/70 text-rose-300 rounded text-[11px] flex items-center gap-1 cursor-pointer border border-rose-800/40"
                            title="Delete this anchor point (Del / Backspace)"
                          >
                            <X className="w-3 h-3 text-rose-400" />
                            <span>Delete</span>
                          </button>
                        )}
                      </div>
                    )}

                    {/* Convert Auto-Detection Button (if empty) */}
                    {penAnchors.length === 0 && (
                      <button
                        onClick={handleConvertAutoToPen}
                        className="px-2.5 py-1 bg-neutral-800 hover:bg-neutral-700 text-amber-300 rounded text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-all border border-neutral-700 hover:border-amber-500/50"
                        title="Generate 6 editable Bezier curve anchors from current nail detection"
                      >
                        <Wand2 className="w-3 h-3 text-amber-400" />
                        <span>Convert to Bezier</span>
                      </button>
                    )}

                    {/* Close Path button (if >= 3 points and open) */}
                    {!isPathClosed && penAnchors.length >= 3 && (
                      <button
                        onClick={() => setIsPathClosed(true)}
                        className="px-2.5 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded text-[11px] font-medium flex items-center gap-1 cursor-pointer transition-all border border-amber-500/40"
                        title="Close path loop to preview and adjust the complete shape"
                      >
                        <Check className="w-3 h-3" />
                        <span>Close Loop</span>
                      </button>
                    )}

                    {/* Re-open button (if closed) */}
                    {isPathClosed && (
                      <button
                        onClick={() => setIsPathClosed(false)}
                        className="px-2 py-1 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded text-[11px] flex items-center gap-1 cursor-pointer"
                        title="Re-open loop to add more points"
                      >
                        <span>Re-open</span>
                      </button>
                    )}

                    {/* Save & Apply Button */}
                    <button
                      onClick={finishPenPath}
                      disabled={penAnchors.length < 3}
                      className="px-3 py-1 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-neutral-950 rounded-lg font-bold text-[11px] flex items-center gap-1.5 cursor-pointer transition-all shadow-md"
                      title="Save contour to nail and update preview (Enter)"
                    >
                      <Check className="w-3.5 h-3.5" />
                      <span>Save to Nail (Enter)</span>
                    </button>

                    {/* Undo Point (Ctrl+Z) */}
                    {penAnchors.length > 0 && (
                      <button
                        onClick={undoLastPenPoint}
                        className="px-2 py-1 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded text-[11px] flex items-center gap-1 cursor-pointer"
                        title="Undo last point / re-open (Ctrl+Z)"
                      >
                        <Undo2 className="w-3 h-3" />
                        <span>Undo (Ctrl+Z)</span>
                      </button>
                    )}

                    {/* Cancel Button */}
                    <button
                      onClick={cancelPenDrawing}
                      className="p-1.5 hover:text-rose-400 text-neutral-400 cursor-pointer"
                      title="Exit Pen Tool (Esc)"
                    >
                      <X className="w-4 h-4" />
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
                        setTrayState((prev) => ({ ...prev, activeTool: "pen", isManualEditMode: false }));
                      }}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                        trayState.activeTool === "pen"
                          ? "bg-amber-500 text-neutral-950 font-bold shadow-xs"
                          : "text-neutral-400 hover:text-neutral-200"
                      }`}
                      title="Draw Custom Curved Contour Pen Tool (P)"
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

        {/* Hand Rigging Canvas */}
        {activeTab === "hand" && (
          <div className="w-full h-full relative">
            <HandRigCanvas
              handState={handState}
              setHandState={setHandState}
            />
          </div>
        )}

        {/* Phase 4: Live PBR Composite Viewport */}
        {activeTab === "composite" && (
          <div className="w-full h-full relative">
            <CompositeCanvas
              trayState={trayState}
              setTrayState={setTrayState}
              handState={handState}
              lightingState={lightingState}
              setLightingState={setLightingState}
            />
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