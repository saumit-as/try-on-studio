mod sidecar;

use sidecar::{SidecarClient, SidecarState};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
fn call_sidecar(
    state: State<'_, SidecarState>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    state.call(&method, params)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sidecar_client = Arc::new(SidecarClient::new());
    let sidecar_for_setup = Arc::clone(&sidecar_client);
    let sidecar_for_exit = Arc::clone(&sidecar_client);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(sidecar_client)
        .setup(move |app| {
            if let Err(e) = sidecar_for_setup.start(&app.handle()) {
                eprintln!("[Sidecar Warning] Failed to start sidecar on launch: {}", e);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![call_sidecar])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                sidecar_for_exit.stop();
            }
        });
}
