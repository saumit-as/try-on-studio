import React from "react";
import { ActiveTab, HandState, LightingState, TrayState } from "../../types";
import { TrayControls } from "./TrayControls";
import { LightingControls } from "./LightingControls";
import { HandControls } from "./HandControls";
import { ExportControls } from "./ExportControls";
import { ExportAspectRatio } from "../canvas/ExportCanvas";

interface InspectorPanelProps {
  activeTab: ActiveTab;
  lightingState: LightingState;
  setLightingState: React.Dispatch<React.SetStateAction<LightingState>>;
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
  handState: HandState;
  setHandState: React.Dispatch<React.SetStateAction<HandState>>;
  exportAspectRatio: ExportAspectRatio;
  setExportAspectRatio: (r: ExportAspectRatio) => void;
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  activeTab,
  lightingState,
  setLightingState,
  trayState,
  setTrayState,
  handState,
  setHandState,
  exportAspectRatio,
  setExportAspectRatio,
}) => {
  return (
    <aside className="w-96 bg-neutral-900/40 backdrop-blur-md flex flex-col overflow-y-auto border-l border-neutral-800">
      {activeTab === "tray" && (
        <TrayControls trayState={trayState} setTrayState={setTrayState} />
      )}
      {activeTab === "hand" && (
        <HandControls handState={handState} setHandState={setHandState} />
      )}
      {activeTab === "composite" && (
        <LightingControls
          lightingState={lightingState}
          setLightingState={setLightingState}
        />
      )}
      {activeTab === "export" && (
        <ExportControls
          trayState={trayState}
          handState={handState}
          lightingState={lightingState}
          aspectRatio={exportAspectRatio}
          setAspectRatio={setExportAspectRatio}
        />
      )}
    </aside>
  );
};
