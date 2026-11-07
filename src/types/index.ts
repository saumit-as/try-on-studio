export type ActiveTab = "tray" | "hand" | "composite" | "export";

export type SidecarStatus = "checking" | "connected" | "disconnected";

export type LightingPresetId = "studio" | "ring" | "flash" | "natural";

export type FingerName = "Thumb" | "Index" | "Middle" | "Ring" | "Pinky";

export interface LightingState {
  preset: string;
  glossIntensity: number;
  shadowDepth: number;
  lightAngle: number;
  cCurveIntensity: number;
  nailLengthScale: number;
  showAdvanced: boolean;
}

export interface TrayState {
  selectedSlot: number;
  slotFinger: Record<number, string>;
  nailRotation: number;
}
