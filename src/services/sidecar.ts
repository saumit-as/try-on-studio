import { invoke } from "@tauri-apps/api/core";

export interface SidecarResponse<T = any> {
  status?: string;
  result?: T;
  error?: string;
}

/**
 * Communicates with the persistent Python sidecar through Tauri's JSON-RPC bridge.
 * Instruments timing and rich console logging for debuggability.
 */
export async function callSidecar<T = any>(
  method: string,
  params: Record<string, any> = {}
): Promise<T> {
  const t0 = performance.now();
  // Strip large base64 data for concise console logs unless needed
  const previewParams = { ...params };
  if (previewParams.image_base64 && typeof previewParams.image_base64 === "string") {
    previewParams.image_base64 = `[base64 string ${previewParams.image_base64.length} chars]`;
  }
  if (previewParams.nail_base64 && typeof previewParams.nail_base64 === "string") {
    previewParams.nail_base64 = `[base64 string ${previewParams.nail_base64.length} chars]`;
  }

  try {
    const result = await invoke<T>("call_sidecar", {
      method,
      params,
    });
    const elapsed = (performance.now() - t0).toFixed(1);
    console.log(`[Sidecar RPC] ${method} (${elapsed}ms):`, {
      params: previewParams,
      result: (result && typeof result === "object" && (result as any).image_base64)
        ? { ...(result as any), image_base64: `[base64 ${(result as any).image_base64.length} chars]` }
        : result,
    });
    return result;
  } catch (err) {
    const elapsed = (performance.now() - t0).toFixed(1);
    console.error(`[Sidecar RPC ERROR] ${method} (${elapsed}ms):`, {
      params: previewParams,
      error: err,
    });
    throw err;
  }
}

/**
 * Performs a health-check ping against the Python sidecar.
 */
export async function pingSidecar(): Promise<{ status: string; version: string }> {
  return await callSidecar("ping", {});
}

