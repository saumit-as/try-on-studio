import { useState, useEffect } from "react";
import { Header } from "./components/layout/Header";
import { CanvasViewport } from "./components/canvas/CanvasViewport";
import { InspectorPanel } from "./components/inspector/InspectorPanel";
import { pingSidecar } from "./services/sidecar";
import {
  ActiveTab,
  SidecarStatus,
  LightingState,
  TrayState,
  HandState,
} from "./types";
import { INITIAL_DEFAULT_HAND_TEMPLATE, DEFAULT_FINGER_ADJUSTMENTS } from "./constants/handDefaults";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("tray");
  const [sidecarStatus, setSidecarStatus] = useState<SidecarStatus>("checking");
  const [sidecarVersion, setSidecarVersion] = useState<string>("");

  // Composite & Lighting State
  const [lightingState, setLightingState] = useState<LightingState>({
    preset: "studio",
    glossIntensity: 75,
    shadowDepth: 60,
    lightAngle: 45,
    cCurveIntensity: 65,
    nailLengthScale: 1.05,
    showAdvanced: false,
    selectedFinger: "All",
    fingerAdjustments: JSON.parse(JSON.stringify(DEFAULT_FINGER_ADJUSTMENTS)),
  });

  // Tray State
  const [trayState, setTrayState] = useState<TrayState>({
    trayImageSrc: null,
    trayImageSize: null,
    isSegmenting: false,
    segmentError: null,
    detectedSlots: [],
    selectedSlot: 1,
    slotFinger: {
      1: "Thumb",
      2: "Index",
      3: "Middle",
      4: "Ring",
      5: "Pinky",
    },
    nailRotation: {},
    slotInverted: {},
    shapeOverrides: {},
    extractedPreviews: {},
    hoveredSlot: null,
    isManualEditMode: false,
    originalSlots: {},
    activeTool: "select",
    slotPenAnchors: {},
  });

  // Phase 3: Hand Rigging State
  const [handState, setHandState] = useState<HandState>({
    currentTemplate: INITIAL_DEFAULT_HAND_TEMPLATE,
    activeFinger: "All",
    selectedLandmark: { finger: "Index", key: "c_c" },
    isGeneratingMaps: false,
    generateError: null,
    overlayToggles: {
      showLandmarks: true,
      showGeometry: true,
      showCuticleMask: false,
      showAoMap: false,
      showSpecular: false,
    },
  });

  // Check Python sidecar connection
  useEffect(() => {
    async function testConnection() {
      try {
        const res = await pingSidecar();
        if (res && res.status === "pong") {
          setSidecarStatus("connected");
          setSidecarVersion(res.version || "0.1.0");
        } else {
          setSidecarStatus("disconnected");
        }
      } catch (err) {
        // In browser dev mode without Tauri runner, mark graceful fallback
        setSidecarStatus("disconnected");
      }
    }
    testConnection();
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-neutral-950 text-neutral-100 font-sans select-none overflow-hidden">
      {/* Top Application Bar */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        sidecarStatus={sidecarStatus}
        sidecarVersion={sidecarVersion}
      />

      {/* Main Workspace Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left / Center Interactive Canvas Viewport */}
        <CanvasViewport
          activeTab={activeTab}
          trayState={trayState}
          setTrayState={setTrayState}
          handState={handState}
          setHandState={setHandState}
          lightingState={lightingState}
          setLightingState={setLightingState}
        />

        {/* Right Parameter & Tooling Inspector */}
        <InspectorPanel
          activeTab={activeTab}
          lightingState={lightingState}
          setLightingState={setLightingState}
          trayState={trayState}
          setTrayState={setTrayState}
          handState={handState}
          setHandState={setHandState}
        />
      </div>
    </div>
  );
}
