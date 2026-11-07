import React from "react";
import { ActiveTab, LightingState, TrayState } from "../../types";
import { TrayControls } from "./TrayControls";
import { LightingControls } from "./LightingControls";
import { HandControls } from "./HandControls";
import { ExportControls } from "./ExportControls";

interface InspectorPanelProps {
  activeTab: ActiveTab;
  lightingState: LightingState;
  setLightingState: React.Dispatch<React.SetStateAction<LightingState>>;
  trayState: TrayState;
  setTrayState: React.Dispatch<React.SetStateAction<TrayState>>;
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  activeTab,
  lightingState,
  setLightingState,
  trayState,
  setTrayState,
}) => {
  return (
    <aside className="w-96 bg-neutral-900/40 backdrop-blur-md flex flex-col overflow-y-auto border-l border-neutral-800">
      {activeTab === "tray" && (
        <TrayControls trayState={trayState} setTrayState={setTrayState} />
      )}
      {activeTab === "hand" && <HandControls />}
      {activeTab === "composite" && (
        <LightingControls
          lightingState={lightingState}
          setLightingState={setLightingState}
        />
      )}
      {activeTab === "export" && <ExportControls />}
    </aside>
  );
};
