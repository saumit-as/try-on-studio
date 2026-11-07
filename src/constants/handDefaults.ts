import { FingerLandmarks, FingerNailAdjustment, FingerName, HandTemplate, LandmarkKey } from "../types";

export const DEFAULT_FINGER_ADJUSTMENTS: Record<FingerName, FingerNailAdjustment> = {
  Thumb: { lengthScale: 1.05, widthScale: 1.0, axialShift: 0, angleOffset: 0, cCurve: 50, tipTaper: 1.0 },
  Index: { lengthScale: 1.05, widthScale: 1.0, axialShift: 0, angleOffset: 0, cCurve: 50, tipTaper: 1.0 },
  Middle: { lengthScale: 1.05, widthScale: 1.0, axialShift: 0, angleOffset: 0, cCurve: 50, tipTaper: 1.0 },
  Ring: { lengthScale: 1.05, widthScale: 1.0, axialShift: 0, angleOffset: 0, cCurve: 50, tipTaper: 1.0 },
  Pinky: { lengthScale: 1.05, widthScale: 1.0, axialShift: 0, angleOffset: 0, cCurve: 50, tipTaper: 1.0 },
};

export const DEFAULT_STUDIO_LANDMARKS: Record<FingerName, FingerLandmarks> = {
  Thumb: {
    c_l: [200.0, 1425.0],
    c_c: [236.0, 1442.0],
    c_r: [275.0, 1416.0],
    d_l: [190.0, 1270.0],
    d_r: [268.0, 1276.0],
    c_curve: 0.44,
    roll_offset: -0.35,
    length_scale: 1.0,
    tip_taper: 1.0,
  },
  Index: {
    c_l: [412.0, 714.0],
    c_c: [466.0, 738.0],
    c_r: [519.0, 712.0],
    d_l: [399.0, 562.0],
    d_r: [518.0, 560.0],
    c_curve: 0.28,
    roll_offset: 0.0,
    length_scale: 1.0,
    tip_taper: 1.0,
  },
  Middle: {
    c_l: [693.0, 566.0],
    c_c: [747.0, 592.0],
    c_r: [799.0, 566.0],
    d_l: [681.0, 410.0],
    d_r: [798.0, 410.0],
    c_curve: 0.28,
    roll_offset: 0.0,
    length_scale: 1.0,
    tip_taper: 1.0,
  },
  Ring: {
    c_l: [948.0, 686.0],
    c_c: [1000.0, 710.0],
    c_r: [1051.0, 686.0],
    d_l: [936.0, 540.0],
    d_r: [1048.0, 540.0],
    c_curve: 0.28,
    roll_offset: 0.0,
    length_scale: 1.0,
    tip_taper: 1.0,
  },
  Pinky: {
    c_l: [1222.0, 990.0],
    c_c: [1262.0, 1012.0],
    c_r: [1301.0, 988.0],
    d_l: [1214.0, 866.0],
    d_r: [1301.0, 864.0],
    c_curve: 0.32,
    roll_offset: 0.0,
    length_scale: 1.0,
    tip_taper: 1.0,
  },
};

export const FINGER_COLORS: Record<FingerName, { hex: string; lightHex: string; label: string }> = {
  Thumb: { hex: "#f43f5e", lightHex: "#fda4af", label: "Thumb" },
  Index: { hex: "#f59e0b", lightHex: "#fde68a", label: "Index" },
  Middle: { hex: "#10b981", lightHex: "#a7f3d0", label: "Middle" },
  Ring: { hex: "#3b82f6", lightHex: "#bfdbfe", label: "Ring" },
  Pinky: { hex: "#a855f7", lightHex: "#e9d5ff", label: "Pinky" },
};

export const LANDMARK_LABELS: Record<LandmarkKey, { title: string; short: string; desc: string }> = {
  c_c: {
    title: "Cuticle Apex",
    short: "Apex",
    desc: "Top center of the proximal nail fold",
  },
  c_l: {
    title: "Left Sidewall",
    short: "Left Fold",
    desc: "Left lateral junction of proximal fold and sidewall",
  },
  c_r: {
    title: "Right Sidewall",
    short: "Right Fold",
    desc: "Right lateral junction of proximal fold and sidewall",
  },
  d_l: {
    title: "Distal Left",
    short: "Distal L",
    desc: "Left corner of hyponychium / natural nail free-edge",
  },
  d_r: {
    title: "Distal Right",
    short: "Distal R",
    desc: "Right corner of hyponychium / natural nail free-edge",
  },
};

export const INITIAL_DEFAULT_HAND_TEMPLATE: HandTemplate = {
  name: "Default Studio Hand",
  imageSrc: "/templates/default_studio_hand/hand.jpg",
  imageSize: [1536, 2752],
  landmarks: JSON.parse(JSON.stringify(DEFAULT_STUDIO_LANDMARKS)),
  cuticleMaskSrc: "/templates/default_studio_hand/cuticle_mask.png",
  aoMapSrc: "/templates/default_studio_hand/ao_map.png",
  specularMapSrc: "/templates/default_studio_hand/specular_map.png",
  isDefault: true,
};
