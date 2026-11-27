//! One CUDA model per bridge. Only the bridge owns the worker's private pipes.
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::native_inference::InferenceError;

type Request = (String, Sender<Option<String>>);
const LOST: InferenceError = (
    "503 Service Unavailable",
    "resident engine stopped before responding",
);

pub struct ResidentEngine {
    pub model: PathBuf,
    child: Child,
    requests: Option<Sender<Request>>,
    reader: Option<JoinHandle<()>>,
}

impl ResidentEngine {
    pub fn start(binary: &Path, model: &Path) -> Result<Self, InferenceError> {
        let mut command = Command::new(binary);
        command
            .arg("serve-token-ids")
            .arg(model)
            .env_remove("INKLING2_BRIDGE_TOKEN")
            .env_remove("INKLING2_INFERENCE_TOKEN")
            .env_remove("INKLING2_UPSTREAM_API_KEY")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn().map_err(|_| LOST)?;
        let (Some(mut input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(LOST);
        };
        let (sender, receiver): (_, Receiver<Request>) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut output = BufReader::new(output);
            for (request, result) in receiver {
                let mut line = String::new();
                let ok = input
                    .write_all(request.as_bytes())
                    .and_then(|()| input.flush())
                    .is_ok()
                    && (&mut output).take(65536).read_line(&mut line).is_ok()
                    && line.ends_with('\n');
                let _ = result.send(ok.then_some(line));
                if !ok {
                    break;
                }
            }
        });
        Ok(Self {
            model: model.into(),
            child,
            requests: Some(sender),
            reader: Some(reader),
        })
    }

    pub fn live(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn exchange(&self, request: String) -> Result<String, InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.requests
            .as_ref()
            .ok_or(LOST)?
            .send((request, tx))
            .map_err(|_| LOST)?;
        let line = rx
            .recv_timeout(Duration::from_secs(600))
            .map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => {
                    ("504 Gateway Timeout", "resident engine timed out")
                }
                mpsc::RecvTimeoutError::Disconnected => LOST,
            })?
            .ok_or(LOST)?;
        match line.trim_end() {
            "error\t2" => Err((
                "503 Service Unavailable",
                "model load failed; check available system memory and GPU memory",
            )),
            "error\t3" => Err((
                "503 Service Unavailable",
                "CUDA inference failed during decoding",
            )),
            _ => Ok(line),
        }
    }
}

impl Drop for ResidentEngine {
    fn drop(&mut self) {
        self.requests.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
