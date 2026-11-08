export type ActiveTab = "tray" | "hand" | "composite" | "export";

export type SidecarStatus = "checking" | "connected" | "disconnected";

export type LightingPresetId = "studio" | "ring" | "flash" | "natural";

export type FingerName = "Thumb" | "Index" | "Middle" | "Ring" | "Pinky";

export type LandmarkKey = "c_l" | "c_c" | "c_r" | "d_l" | "d_r";

export type NailShape =
  | "Almond"
  | "Coffin"
  | "Square"
  | "Squoval"
  | "Stiletto"
  | "Oval";

export interface ExtractionDebugInfo {
  slot_id: number;
  method: string;
  contour_points: number;
  crop_bbox?: [number, number, number, number];
  deskew_angle?: number;
  is_inverted?: boolean;
  dematting_metrics?: {
    choke_px: number;
    area_removed_pct: number;
    core_px: number;
  };
  extracted_size?: [number, number];
  duration_ms?: number;
  warnings?: string[];
  debug_dump_path?: string;
}

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
  method?: string;
  method_note?: string;
  debug_info?: ExtractionDebugInfo;
}

export interface FingerNailAdjustment {
  lengthScale: number; // Longitudinal tip extension (0.5x - 2.5x)
  widthScale: number;  // Lateral width fit (0.7x - 1.4x)
  axialShift: number;  // Cuticle tuck / extension (-30px to +30px)
  angleOffset: number; // Angular rotation (-15° to +15°)
  cCurve: number;      // Transverse curvature (0 - 100%)
  tipTaper: number;    // Tip width taper (0.6x - 1.3x)
}

export interface LightingState {
  preset: string;
  glossIntensity: number;
  shadowDepth: number;
  lightAngle: number;
  cCurveIntensity: number;
  nailLengthScale: number;
  showAdvanced: boolean;
  selectedFinger: "All" | FingerName;
  fingerAdjustments: Record<FingerName, FingerNailAdjustment>;
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
  extractedDebug: Record<number, ExtractionDebugInfo>;
  hoveredSlot: number | null;
  isManualEditMode: boolean;
  originalSlots: Record<number, DetectedSlot>;
  activeTool: "select" | "pen" | "pan";
  slotPenAnchors: Record<number, PenAnchor[]>;
}

export interface FingerLandmarks {
  c_l: [number, number]; // Left Lateral Sidewall
  c_c: [number, number]; // Cuticle Apex (top center proximal fold)
  c_r: [number, number]; // Right Lateral Sidewall
  d_l: [number, number]; // Distal Edge Left
  d_r: [number, number]; // Distal Edge Right
  c_curve?: number;
  roll_offset?: number;
  length_scale?: number;
  tip_taper?: number;
  width_scale?: number;
  axial_shift?: number;
  angle_offset?: number;
}

export interface HandTemplate {
  name: string;
  imageSrc: string; // URL or base64
  imageSize: [number, number]; // [width, height]
  landmarks: Record<FingerName, FingerLandmarks>;
  handedness?: "left" | "right";
  cuticleMaskSrc?: string;
  aoMapSrc?: string;
  specularMapSrc?: string;
  isDefault?: boolean;
}

export interface HandOverlayToggles {
  showLandmarks: boolean;
  showGeometry: boolean;
  showCuticleMask: boolean;
  showAoMap: boolean;
  showSpecular: boolean;
}

export interface HandState {
  currentTemplate: HandTemplate | null;
  activeFinger: FingerName | "All";
  selectedLandmark: { finger: FingerName; key: LandmarkKey } | null;
  isGeneratingMaps: boolean;
  generateError: string | null;
  overlayToggles: HandOverlayToggles;
  isBoxToolActive?: boolean;
}
