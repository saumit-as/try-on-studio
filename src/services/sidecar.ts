import { invoke } from "@tauri-apps/api/core";

export interface SidecarResponse<T = any> {
  status?: string;
  result?: T;
  error?: string;
}

/**
 * Communicates with the persistent Python sidecar through Tauri's JSON-RPC bridge.
 */
export async function callSidecar<T = any>(
  method: string,
  params: Record<string, any> = {}
): Promise<T> {
  return await invoke<T>("call_sidecar", {
    method,
    params,
  });
}

/**
 * Performs a health-check ping against the Python sidecar.
 */
export async function pingSidecar(): Promise<{ status: string; version: string }> {
  return await callSidecar("ping", {});
}
