use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::task::JoinHandle;

use super::abort::AbortSignal;
use super::engine::OcrEngine;
use super::error::{EngineError, OcrError, ServiceStateKind, ServiceStatusSnapshot};
use super::request::{OcrRequest, OcrRunResult};

enum ServiceState {
    Idle,
    Busy,
    Faulted {
        reason: &'static str,
    },
}

/// Single-active-request controller wrapping the OCR engine.
pub struct OcrService {
    engine: Arc<Mutex<OcrEngine>>,
    state: Arc<Mutex<ServiceState>>,
}

impl OcrService {
    pub fn new(engine: OcrEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            state: Arc::new(Mutex::new(ServiceState::Idle)),
        }
    }

    /// Run an OCR request. Rejects when the service is busy or faulted.
    ///
    /// Returns a JoinHandle for the blocking generation task.
    pub fn run(
        &self,
        req: OcrRequest,
        abort: AbortSignal,
        on_text_chunk: impl FnMut(String) + Send + 'static,
    ) -> Result<JoinHandle<Result<OcrRunResult, OcrError>>, OcrError> {
        {
            let mut state = lock_service_state(&self.state);
            match &*state {
                ServiceState::Idle => {
                    *state = ServiceState::Busy;
                }
                ServiceState::Busy => {
                    return Err(OcrError::Busy);
                }
                ServiceState::Faulted { reason } => {
                    return Err(OcrError::Faulted { reason });
                }
            }
        }

        let engine = Arc::clone(&self.engine);
        let state = Arc::clone(&self.state);

        let handle = tokio::task::spawn_blocking(move || {
            let run_result = run_engine_request(&engine, &req, &abort, on_text_chunk);
            finalize_request_state(&state, &run_result);
            run_result.map_err(OcrError::from)
        });

        Ok(handle)
    }

    pub fn status(&self) -> ServiceStatusSnapshot {
        let state = lock_service_state(&self.state);
        match &*state {
            ServiceState::Idle => ServiceStatusSnapshot {
                state: ServiceStateKind::Idle,
                fault_reason: None,
            },
            ServiceState::Busy => ServiceStatusSnapshot {
                state: ServiceStateKind::Busy,
                fault_reason: None,
            },
            ServiceState::Faulted { reason } => ServiceStatusSnapshot {
                state: ServiceStateKind::Faulted,
                fault_reason: Some(reason),
            },
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.status().state, ServiceStateKind::Busy)
    }
}

fn run_engine_request(
    engine: &Arc<Mutex<OcrEngine>>,
    req: &OcrRequest,
    abort: &AbortSignal,
    on_text_chunk: impl FnMut(String),
) -> Result<OcrRunResult, EngineError> {
    let mut engine = match engine.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return Err(EngineError::StateInvariant {
                code: "engine_mutex_poisoned",
                message: "engine mutex is poisoned".to_string(),
            });
        }
    };

    match panic::catch_unwind(AssertUnwindSafe(|| engine.run(req, abort, on_text_chunk))) {
        Ok(result) => result,
        Err(payload) => Err(EngineError::WorkerPanic {
            message: panic_payload_to_string(payload),
        }),
    }
}

fn finalize_request_state(
    state: &Arc<Mutex<ServiceState>>,
    result: &Result<OcrRunResult, EngineError>,
) {
    let mut state_guard = lock_service_state(state);
    *state_guard = match result {
        Ok(_) => ServiceState::Idle,
        Err(error) if error.should_fault_service() => {
            let reason = error.fault_reason().unwrap_or("internal_error");
            log::error!(
                "OCR request faulted service (reason={}): {:?}",
                reason,
                error
            );
            ServiceState::Faulted { reason }
        }
        Err(error) => {
            log::warn!("OCR request failed without faulting service: {:?}", error);
            ServiceState::Idle
        }
    };
}

fn lock_service_state(state: &Arc<Mutex<ServiceState>>) -> MutexGuard<'_, ServiceState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::error!("Service state mutex is poisoned; recovering state lock");
            poisoned.into_inner()
        }
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
