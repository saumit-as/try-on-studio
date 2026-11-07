export type ActiveTab = "tray" | "hand" | "composite" | "export";

export type SidecarStatus = "checking" | "connected" | "disconnected";

export type LightingPresetId = "studio" | "ring" | "flash" | "natural";

export type FingerName = "Thumb" | "Index" | "Middle" | "Ring" | "Pinky";

export type NailShape =
  | "Almond"
  | "Coffin"
  | "Square"
  | "Squoval"
  | "Stiletto"
  | "Oval";

export interface DetectedSlot {
  slot_id: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  contour: [number, number][];
  area: number;
  aspect_ratio: number;
  pca_angle: number;
  shape_info: {
    shape: string;
    confidence: number;
    metrics: Record<string, any>;
  };
  card_quad?: [number, number][];
  pen_anchors?: PenAnchor[];
}

export interface LightingState {
  preset: string;
  glossIntensity: number;
  shadowDepth: number;
  lightAngle: number;
  cCurveIntensity: number;
  nailLengthScale: number;
  showAdvanced: boolean;
}

export interface PenAnchor {
  x: number;
  y: number;
  handleIn: [number, number];
  handleOut: [number, number];
}

export interface TrayState {
  trayImageSrc: string | null;
  trayImageSize: [number, number] | null;
  isSegmenting: boolean;
  segmentError: string | null;
  detectedSlots: DetectedSlot[];
  selectedSlot: number;
  slotFinger: Record<number, FingerName>;
  nailRotation: Record<number, number>;
  slotInverted: Record<number, boolean>;
  shapeOverrides: Record<number, string>;
  extractedPreviews: Record<number, string>;
  hoveredSlot: number | null;
  isManualEditMode: boolean;
  originalSlots: Record<number, DetectedSlot>;
  activeTool: "select" | "pen" | "pan";
  slotPenAnchors: Record<number, PenAnchor[]>;
}
