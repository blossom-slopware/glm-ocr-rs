use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::oneshot;

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

struct PendingRequest {
    req: OcrRequest,
    abort: AbortSignal,
    on_text_chunk: Box<dyn FnMut(String) + Send>,
    result_tx: oneshot::Sender<Result<OcrRunResult, OcrError>>,
}

/// Preemptive single-active-request controller wrapping the OCR engine.
///
/// When a new request arrives while the engine is busy:
/// 1. The current in-flight request is aborted.
/// 2. Any previously pending request is evicted with `Err(Busy)`.
/// 3. The new request becomes the sole pending request.
/// 4. Once the in-flight request finishes aborting, the pending request is
///    picked up and executed immediately.
pub struct OcrService {
    engine: Arc<Mutex<OcrEngine>>,
    state: Arc<Mutex<ServiceState>>,
    /// Holds at most one pending request (the most recent).
    pending: Arc<Mutex<Option<PendingRequest>>>,
    /// Abort signal of the currently executing request, if any.
    current_abort: Arc<Mutex<Option<AbortSignal>>>,
}

impl OcrService {
    pub fn new(engine: OcrEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            state: Arc::new(Mutex::new(ServiceState::Idle)),
            pending: Arc::new(Mutex::new(None)),
            current_abort: Arc::new(Mutex::new(None)),
        }
    }

    /// Submit an OCR request.
    ///
    /// - If idle: starts immediately.
    /// - If busy: aborts the current request, evicts any existing pending
    ///   request (returns `Err(Busy)` to it), and queues this one.
    /// - If faulted: returns `Err(Faulted)`.
    ///
    /// Returns a `oneshot::Receiver` that will eventually deliver the result.
    pub fn run(
        &self,
        req: OcrRequest,
        abort: AbortSignal,
        on_text_chunk: impl FnMut(String) + Send + 'static,
    ) -> Result<oneshot::Receiver<Result<OcrRunResult, OcrError>>, OcrError> {
        let (tx, rx) = oneshot::channel();

        {
            let state = lock_mutex(&self.state);
            match &*state {
                ServiceState::Faulted { reason } => {
                    return Err(OcrError::Faulted { reason });
                }
                ServiceState::Idle => {
                    drop(state);
                    self.start_request(req, abort, Box::new(on_text_chunk), tx);
                    return Ok(rx);
                }
                ServiceState::Busy => {
                    // Fall through to preemption path below.
                    // Drop state lock first.
                }
            }
        }

        // Preemption path: we are Busy.
        // 1. Evict any existing pending request.
        {
            let mut pending = lock_mutex(&self.pending);
            if let Some(old) = pending.take() {
                let _ = old.result_tx.send(Err(OcrError::Busy));
                log::info!("Evicted pending request (replaced by newer request)");
            }
            // 2. Store ourselves as the new pending request.
            *pending = Some(PendingRequest {
                req,
                abort,
                on_text_chunk: Box::new(on_text_chunk),
                result_tx: tx,
            });
        }

        // 3. Abort the currently executing request.
        {
            let current_abort = lock_mutex(&self.current_abort);
            if let Some(signal) = current_abort.as_ref() {
                signal.set();
                log::info!("Abort signal sent to current in-flight request");
            }
        }

        Ok(rx)
    }

    pub fn status(&self) -> ServiceStatusSnapshot {
        let state = lock_mutex(&self.state);
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

    // ── Internal ──

    /// Spawn a blocking task to run the given request on the engine.
    fn start_request(
        &self,
        req: OcrRequest,
        abort: AbortSignal,
        on_text_chunk: Box<dyn FnMut(String) + Send>,
        result_tx: oneshot::Sender<Result<OcrRunResult, OcrError>>,
    ) {
        // Transition to Busy and record abort signal.
        {
            let mut state = lock_mutex(&self.state);
            *state = ServiceState::Busy;
        }
        {
            let mut ca = lock_mutex(&self.current_abort);
            *ca = Some(abort.clone());
        }

        let engine = Arc::clone(&self.engine);
        let state = Arc::clone(&self.state);
        let pending = Arc::clone(&self.pending);
        let current_abort = Arc::clone(&self.current_abort);

        // Capture `self`'s Arcs for drain_pending to call start_request again.
        let engine_for_drain = Arc::clone(&self.engine);
        let state_for_drain = Arc::clone(&self.state);
        let pending_for_drain = Arc::clone(&self.pending);
        let current_abort_for_drain = Arc::clone(&self.current_abort);

        tokio::task::spawn_blocking(move || {
            let run_result = run_engine_request(&engine, &req, &abort, on_text_chunk);
            let ocr_result = match &run_result {
                Ok(result) => Ok(result.clone()),
                Err(engine_err) => Err(OcrError::from_engine_error_ref(engine_err)),
            };

            // Send result to the waiting receiver (ignore error if receiver dropped).
            let _ = result_tx.send(ocr_result);

            // Finalize state based on the engine result, then check for pending.
            finalize_and_drain(
                &state,
                &run_result,
                &pending,
                &current_abort,
                &engine_for_drain,
                &state_for_drain,
                &pending_for_drain,
                &current_abort_for_drain,
            );
        });
    }
}

/// After a request finishes, update state and start the next pending request if any.
fn finalize_and_drain(
    state: &Arc<Mutex<ServiceState>>,
    result: &Result<OcrRunResult, EngineError>,
    pending: &Arc<Mutex<Option<PendingRequest>>>,
    current_abort: &Arc<Mutex<Option<AbortSignal>>>,
    engine: &Arc<Mutex<OcrEngine>>,
    state_for_next: &Arc<Mutex<ServiceState>>,
    pending_for_next: &Arc<Mutex<Option<PendingRequest>>>,
    current_abort_for_next: &Arc<Mutex<Option<AbortSignal>>>,
) {
    // Determine next state based on result.
    let faulted = match result {
        Ok(_) => false,
        Err(error) if error.should_fault_service() => {
            let reason = error.fault_reason().unwrap_or("internal_error");
            log::error!(
                "OCR request faulted service (reason={}): {:?}",
                reason,
                error
            );
            let mut state_guard = lock_mutex(state);
            *state_guard = ServiceState::Faulted { reason };
            true
        }
        Err(error) => {
            log::warn!("OCR request failed without faulting service: {:?}", error);
            false
        }
    };

    if faulted {
        // If faulted, reject any pending request too.
        let mut pend = lock_mutex(pending);
        if let Some(old) = pend.take() {
            let _ = old.result_tx.send(Err(OcrError::Faulted {
                reason: "service faulted during previous request",
            }));
        }
        let mut ca = lock_mutex(current_abort);
        *ca = None;
        return;
    }

    // Not faulted — check if there's a pending request to pick up.
    let next = {
        let mut pend = lock_mutex(pending);
        pend.take()
    };

    match next {
        Some(next_req) => {
            // Stay Busy, start the next request.
            log::info!("Draining pending request — starting next OCR inference");
            {
                let mut ca = lock_mutex(current_abort);
                *ca = Some(next_req.abort.clone());
            }

            let engine_clone = Arc::clone(engine);
            let state_clone = Arc::clone(state_for_next);
            let pending_clone = Arc::clone(pending_for_next);
            let current_abort_clone = Arc::clone(current_abort_for_next);

            // We're already on a blocking thread, so run directly instead of
            // spawning another blocking task.
            let run_result = run_engine_request(
                &engine_clone,
                &next_req.req,
                &next_req.abort,
                next_req.on_text_chunk,
            );
            let ocr_result = match &run_result {
                Ok(result) => Ok(result.clone()),
                Err(engine_err) => Err(OcrError::from_engine_error_ref(engine_err)),
            };
            let _ = next_req.result_tx.send(ocr_result);

            // Recurse to drain again (in case yet another request arrived).
            finalize_and_drain(
                &state_clone,
                &run_result,
                &pending_clone,
                &current_abort_clone,
                &engine_clone,
                &state_clone,
                &pending_clone,
                &current_abort_clone,
            );
        }
        None => {
            // No pending — go idle.
            let mut state_guard = lock_mutex(state);
            *state_guard = ServiceState::Idle;
            let mut ca = lock_mutex(current_abort);
            *ca = None;
        }
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

fn lock_mutex<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::error!("Mutex is poisoned; recovering lock");
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
