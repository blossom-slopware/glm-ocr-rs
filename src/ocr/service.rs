use std::sync::Arc;

use tokio::sync::Mutex;

use super::abort::AbortSignal;
use super::engine::OcrEngine;
use super::request::{OcrRequest, OcrRunResult};

/// Tracks one in-flight request.
struct ActiveRequest {
    abort: AbortSignal,
}

struct ActiveRequestGuard {
    active: Arc<Mutex<Option<ActiveRequest>>>,
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        let mut active_guard = self.active.blocking_lock();
        *active_guard = None;
    }
}

/// Error type for the OCR service layer.
#[derive(Debug)]
pub enum OcrError {
    Busy,
    Internal(String),
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrError::Busy => write!(f, "Server is busy with another request"),
            OcrError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

/// Single-active-request controller wrapping the OCR engine.
pub struct OcrService {
    engine: Arc<Mutex<OcrEngine>>,
    active: Arc<Mutex<Option<ActiveRequest>>>,
}

impl OcrService {
    pub fn new(engine: OcrEngine) -> Self {
        OcrService {
            engine: Arc::new(Mutex::new(engine)),
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// Run an OCR request. Rejects with `OcrError::Busy` if a request is already in progress.
    ///
    /// The caller provides the `AbortSignal` and is responsible for setting it on client
    /// disconnect. The active request slot is cleared automatically when the blocking task
    /// finishes.
    ///
    /// Returns a `JoinHandle` for the blocking generation task.
    pub async fn run(
        &self,
        req: OcrRequest,
        abort: AbortSignal,
        on_text_chunk: impl FnMut(String) + Send + 'static,
    ) -> Result<tokio::task::JoinHandle<OcrRunResult>, OcrError> {
        // Check if busy and register active request atomically
        {
            let mut active = self.active.lock().await;
            if active.is_some() {
                return Err(OcrError::Busy);
            }
            *active = Some(ActiveRequest {
                abort: abort.clone(),
            });
        }

        // Run generation in blocking task
        let engine = self.engine.clone();
        let active = self.active.clone();

        let handle = tokio::task::spawn_blocking(move || {
            let _active_guard = ActiveRequestGuard {
                active: active.clone(),
            };
            let mut engine = engine.blocking_lock();
            engine.run(&req, &abort, on_text_chunk)
        });

        Ok(handle)
    }

    /// Check if a request is currently in progress.
    pub async fn is_busy(&self) -> bool {
        self.active.lock().await.is_some()
    }
}
