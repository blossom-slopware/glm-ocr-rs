use std::sync::Arc;

use axum::extract::State;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json as AxumJson, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::ocr::{AbortSignal, OcrError, OcrRequest, OcrService, StopReason};

/// Shared application state.
pub struct AppState {
    pub service: OcrService,
    pub model_name: String,
}

// ─── SSE event types ───

#[derive(Serialize)]
struct SseDelta {
    delta: String,
}

#[derive(Serialize)]
struct SseDone {
    done: bool,
    stop_reason: String,
    generated_tokens: usize,
}

// ─── Routes ───

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ocr", post(ocr_handler))
        .route("/ocr/stream", post(ocr_stream_handler))
        .route("/ocr/status", get(ocr_status_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    log::debug!("GET /health");
    AxumJson(serde_json::json!({
        "status": "ok",
        "model": state.model_name,
    }))
}

async fn ocr_status_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let busy = state.service.is_busy().await;
    log::debug!("GET /ocr/status busy={}", busy);
    AxumJson(serde_json::json!({ "busy": busy }))
}

/// POST /ocr — full (non-streaming) OCR response.
/// If the client disconnects, the AbortOnDrop guard aborts generation.
async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    AxumJson(req): AxumJson<OcrRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    log::info!(
        "POST /ocr max_tokens={} temp={}",
        req.max_tokens, req.temperature
    );

    let abort = AbortSignal::new();
    let _abort_guard = AbortOnDrop::new(abort.clone(), req.image_description());

    let handle = state.service.run(req, abort, |_chunk| {}).await
        .map_err(map_ocr_error)?;

    let result = handle.await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation task panicked: {}", e)))?;

    log::info!(
        "OCR completed: {} tokens, stop_reason={:?}",
        result.generated_tokens, result.stop_reason
    );
    _abort_guard.abort.set();
    Ok(AxumJson(result).into_response())
}

/// POST /ocr/stream — SSE streaming OCR response.
/// Client disconnect is detected both by:
/// 1. tx.blocking_send() failure in the callback (sets abort immediately)
/// 2. AbortOnDrop guard when the spawned task is cleaned up
async fn ocr_stream_handler(
    State(state): State<Arc<AppState>>,
    AxumJson(req): AxumJson<OcrRequest>,
) -> Result<Response, (StatusCode, String)> {
    log::info!(
        "POST /ocr/stream max_tokens={} temp={}",
        req.max_tokens, req.temperature
    );

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(128);

    // Create abort signal — shared between callback and guard
    let abort = AbortSignal::new();
    let image_desc = req.image_description();

    // The callback detects client disconnect: if tx.blocking_send fails,
    // the client is gone, so we set the abort signal immediately.
    let abort_for_callback = abort.clone();
    let image_desc_for_callback = image_desc.clone();
    let on_text_chunk = {
        let tx = tx.clone();
        move |chunk: String| {
            let event_data = SseDelta { delta: chunk };
            let send_result = tx.blocking_send(Ok(
                Event::default().data(serde_json::to_string(&event_data).unwrap())
            ));
            if send_result.is_err() {
                // Client disconnected — abort generation
                abort_for_callback.set();
                log::info!("Client disconnected (send failed) — aborting generation for {}", image_desc_for_callback);
            }
        }
    };

    let handle = state.service.run(req, abort.clone(), on_text_chunk).await
        .map_err(map_ocr_error)?;

    // Spawn task to wait for generation and send done event
    let tx_done = tx.clone();
    let image_desc_clone = image_desc.clone();
    tokio::spawn(async move {
        // Guard: if this task is dropped, abort generation
        let _abort_guard = AbortOnDrop::new(abort, image_desc_clone);

        let result = match handle.await {
            Ok(result) => result,
            Err(e) => {
                log::error!("Generation task panicked: {}", e);
                let _ = tx_done.send(Ok(
                    Event::default().data(format!(r#"{{"error":"generation panicked: {}"}}"#, e))
                )).await;
                return;
            }
        };

        let stop_reason_str = match &result.stop_reason {
            StopReason::StopToken => "stop_token",
            StopReason::MaxTokens => "max_tokens",
            StopReason::Aborted => "aborted",
        };
        let done_event = SseDone {
            done: true,
            stop_reason: stop_reason_str.to_string(),
            generated_tokens: result.generated_tokens,
        };
        let _ = tx_done.send(Ok(
            Event::default().data(serde_json::to_string(&done_event).unwrap())
        )).await;
        log::info!(
            "OCR stream completed: {} tokens, stop_reason={}",
            result.generated_tokens, stop_reason_str
        );

        // Mark abort so the AbortOnDrop guard won't log a spurious
        // "Client disconnected" message on normal completion.
        _abort_guard.abort.set();
    });

    let stream = ReceiverStream::new(rx);
    let mut response = Sse::new(stream).into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    response.headers_mut().insert(
        CACHE_CONTROL,
        "no-cache".parse().unwrap(),
    );
    Ok(response)
}

fn map_ocr_error(e: OcrError) -> (StatusCode, String) {
    match e {
        OcrError::Busy => (StatusCode::CONFLICT, "Server is busy with another request".to_string()),
        OcrError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
    }
}

/// Guard that sets the abort signal when dropped.
/// Ensures generation is aborted when the client disconnects.
struct AbortOnDrop {
    abort: AbortSignal,
    image_desc: String,
}

impl AbortOnDrop {
    fn new(abort: AbortSignal, image_desc: String) -> Self {
        AbortOnDrop { abort, image_desc }
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if !self.abort.is_set() {
            log::info!("Client disconnected — aborting generation for {}", self.image_desc);
            self.abort.set();
        }
    }
}
