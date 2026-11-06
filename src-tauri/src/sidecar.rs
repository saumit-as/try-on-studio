use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};

#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcRequest {
    id: u64,
    method: String,
    params: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<Value>,
    error: Option<Value>,
}

pub struct SidecarClient {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    stdout_reader: Mutex<Option<BufReader<ChildStdout>>>,
    req_counter: AtomicU64,
}

impl SidecarClient {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            stdout_reader: Mutex::new(None),
            req_counter: AtomicU64::new(1),
        }
    }

    pub fn start(&self, app: &AppHandle) -> Result<(), String> {
        let mut child_guard = self.child.lock().map_err(|e| e.to_string())?;
        if child_guard.is_some() {
            return Ok(()); // already started
        }

        // 1. Try to find packaged sidecar externalBin first
        // 2. Fall back to python3 script in dev mode
        let mut command = if let Ok(sidecar_path) = Self::find_sidecar_binary(app) {
            Command::new(sidecar_path)
        } else {
            let script_path = Self::find_dev_script(app);
            let mut cmd = Command::new("python3");
            cmd.arg(script_path);
            cmd
        };

        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        #[cfg(target_os = "windows")]
        {
            // On Windows, prevent console window popup
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut spawned = command.spawn().map_err(|e| {
            format!("Failed to spawn Python sidecar: {}", e)
        })?;

        let stdin = spawned.stdin.take().ok_or("Failed to open sidecar stdin")?;
        let stdout = spawned.stdout.take().ok_or("Failed to open sidecar stdout")?;
        let reader = BufReader::new(stdout);

        *self.stdin.lock().map_err(|e| e.to_string())? = Some(stdin);
        *self.stdout_reader.lock().map_err(|e| e.to_string())? = Some(reader);
        *child_guard = Some(spawned);

        Ok(())
    }

    fn find_sidecar_binary(app: &AppHandle) -> Result<PathBuf, String> {
        // In production / packaged app, Tauri places sidecars adjacent or in resource directory
        if let Ok(res_dir) = app.path().resource_dir() {
            let possible_binary = res_dir.join("binaries").join("nail-sidecar");
            if possible_binary.exists() {
                return Ok(possible_binary);
            }
        }
        Err("Packaged binary not found".to_string())
    }

    fn find_dev_script(_app: &AppHandle) -> PathBuf {
        // Dev path fallback: ../python_sidecar/main.py relative to src-tauri or current exe
        let candidates = [
            PathBuf::from("../python_sidecar/main.py"),
            PathBuf::from("./python_sidecar/main.py"),
            PathBuf::from("../../python_sidecar/main.py"),
        ];

        for path in candidates {
            if path.exists() {
                return path;
            }
        }
        PathBuf::from("../python_sidecar/main.py")
    }

    pub fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let req_id = self.req_counter.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest {
            id: req_id,
            method: method.to_string(),
            params,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        // Lock stdin and write
        {
            let mut stdin_guard = self.stdin.lock().map_err(|e| e.to_string())?;
            let stdin = stdin_guard.as_mut().ok_or("Sidecar stdin not available")?;
            writeln!(stdin, "{}", request_json)
                .map_err(|e| format!("Failed to write to sidecar stdin: {}", e))?;
            stdin.flush().map_err(|e| format!("Failed to flush sidecar stdin: {}", e))?;
        }

        // Lock stdout and read response line
        let mut line = String::new();
        {
            let mut stdout_guard = self.stdout_reader.lock().map_err(|e| e.to_string())?;
            let reader = stdout_guard.as_mut().ok_or("Sidecar stdout not available")?;
            reader.read_line(&mut line)
                .map_err(|e| format!("Failed to read sidecar response: {}", e))?;
        }

        if line.is_empty() {
            return Err("Sidecar process closed pipe unexpectedly".to_string());
        }

        let resp: JsonRpcResponse = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse response JSON: {} (raw: '{}')", e, line))?;

        if let Some(err) = resp.error {
            return Err(format!("Sidecar error: {}", err));
        }

        resp.result.ok_or_else(|| "Empty result from sidecar".to_string())
    }

    pub fn stop(&self) {
        if let Ok(mut child_guard) = self.child.lock() {
            if let Some(mut child) = child_guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

pub type SidecarState = Arc<SidecarClient>;
