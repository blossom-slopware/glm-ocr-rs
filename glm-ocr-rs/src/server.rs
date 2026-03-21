use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json as AxumJson, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::ocr::{
    AbortSignal, OcrError, OcrRequest, OcrService, ServiceStateKind, StopReason,
};

/// Shared application state.
pub struct AppState {
    pub service: OcrService,
    pub model_name: String,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: &'static str,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

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

#[derive(Serialize)]
struct SseError {
    error: &'static str,
    code: &'static str,
}

type SseSender = mpsc::Sender<Result<Event, Infallible>>;
type ApiErrorResponse = (StatusCode, AxumJson<ErrorResponse>);

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ocr", post(ocr_handler))
        .route("/ocr/stream", post(ocr_stream_handler))
        .route("/ocr/status", get(ocr_status_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let status = state.service.status();
    log::debug!("GET /health state={:?}", status.state);

    let http_status = if status.state == ServiceStateKind::Faulted {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };

    (
        http_status,
        AxumJson(serde_json::json!({
            "status": if status.state == ServiceStateKind::Faulted { "faulted" } else { "ok" },
            "model": state.model_name.as_str(),
            "service_state": status.state,
            "fault_reason": status.fault_reason,
        })),
    )
}

async fn ocr_status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let status = state.service.status();
    log::debug!("GET /ocr/status state={:?}", status.state);
    AxumJson(serde_json::json!({
        "busy": status.state == ServiceStateKind::Busy,
        "state": status.state,
        "fault_reason": status.fault_reason,
    }))
}

/// POST /ocr — full (non-streaming) OCR response.
/// If the client disconnects, the AbortOnDrop guard aborts generation.
async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    AxumJson(req): AxumJson<OcrRequest>,
) -> Result<Response, ApiErrorResponse> {
    log::info!(
        "POST /ocr max_tokens={} temp={}",
        req.max_tokens,
        req.temperature
    );

    let abort = AbortSignal::new();
    let abort_guard = AbortOnDrop::new(abort.clone(), req.image_description());

    let rx = state
        .service
        .run(req, abort, |_chunk| {})
        .map_err(map_ocr_error)?;

    let result = match rx.await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            abort_guard.mark_completed();
            return Err(map_ocr_error(error));
        }
        Err(_recv_error) => {
            log::error!("OCR result channel closed unexpectedly");
            abort_guard.mark_completed();
            return Err(internal_channel_error_response());
        }
    };

    log::info!(
        "OCR completed: {} tokens, stop_reason={:?}",
        result.generated_tokens,
        result.stop_reason
    );
    abort_guard.mark_completed();
    Ok(AxumJson(result).into_response())
}

/// POST /ocr/stream — SSE streaming OCR response.
/// Client disconnect is detected both by:
/// 1. tx.blocking_send() failure in the callback (sets abort immediately)
/// 2. AbortOnDrop guard when the spawned task is cleaned up
async fn ocr_stream_handler(
    State(state): State<Arc<AppState>>,
    AxumJson(req): AxumJson<OcrRequest>,
) -> Result<Response, ApiErrorResponse> {
    log::info!(
        "POST /ocr/stream max_tokens={} temp={}",
        req.max_tokens,
        req.temperature
    );

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(1024);

    let abort = AbortSignal::new();
    let image_desc = req.image_description();

    let abort_for_callback = abort.clone();
    let image_desc_for_callback = image_desc.clone();
    let on_text_chunk = {
        let tx = tx.clone();
        move |chunk: String| {
            let event_data = SseDelta { delta: chunk };
            if !blocking_send_sse_json(&tx, &event_data) {
                abort_for_callback.set();
                log::info!(
                    "Client disconnected (send failed) — aborting generation for {}",
                    image_desc_for_callback
                );
            }
        }
    };

    let result_rx = state
        .service
        .run(req, abort.clone(), on_text_chunk)
        .map_err(map_ocr_error)?;

    let tx_done = tx.clone();
    let image_desc_clone = image_desc.clone();
    tokio::spawn(async move {
        let abort_guard = AbortOnDrop::new(abort, image_desc_clone);

        match result_rx.await {
            Ok(Ok(result)) => {
                let stop_reason_str = stop_reason_label(&result.stop_reason);
                let done_event = SseDone {
                    done: true,
                    stop_reason: stop_reason_str.to_string(),
                    generated_tokens: result.generated_tokens,
                };
                let _ = send_sse_json(&tx_done, &done_event).await;
                log::info!(
                    "OCR stream completed: {} tokens, stop_reason={}",
                    result.generated_tokens,
                    stop_reason_str
                );
                abort_guard.mark_completed();
            }
            Ok(Err(OcrError::Aborted)) => {
                let done_event = SseDone {
                    done: true,
                    stop_reason: stop_reason_label(&StopReason::Aborted).to_string(),
                    generated_tokens: 0,
                };
                let _ = send_sse_json(&tx_done, &done_event).await;
                abort_guard.mark_completed();
            }
            Ok(Err(error)) => {
                log::error!("OCR stream request failed: {}", error);
                let _ = send_sse_json(
                    &tx_done,
                    &SseError {
                        error: error.message(),
                        code: error.code(),
                    },
                )
                .await;
                abort_guard.mark_completed();
            }
            Err(_recv_error) => {
                log::error!("OCR stream result channel closed unexpectedly");
                let _ = send_sse_json(
                    &tx_done,
                    &SseError {
                        error: "result channel closed",
                        code: "channel_closed",
                    },
                )
                .await;
                abort_guard.mark_completed();
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let mut response = Sse::new(stream).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Ok(response)
}

fn map_ocr_error(error: OcrError) -> ApiErrorResponse {
    let status = match &error {
        OcrError::Busy => StatusCode::CONFLICT,
        OcrError::Faulted { .. } => StatusCode::SERVICE_UNAVAILABLE,
        OcrError::BadRequest { .. } => StatusCode::BAD_REQUEST,
        OcrError::Aborted => StatusCode::REQUEST_TIMEOUT,
        OcrError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    };

    (
        status,
        AxumJson(ErrorResponse {
            error: ErrorBody {
                code: error.code(),
                message: error.message(),
            },
        }),
    )
}

fn internal_channel_error_response() -> ApiErrorResponse {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        AxumJson(ErrorResponse {
            error: ErrorBody {
                code: "channel_closed",
                message: "result channel closed unexpectedly",
            },
        }),
    )
}

fn blocking_send_sse_json<T: Serialize>(tx: &SseSender, payload: &T) -> bool {
    let event = match serialize_sse_event(payload) {
        Ok(event) => event,
        Err(error) => {
            log::error!("Failed to serialize SSE payload: {}", error);
            return false;
        }
    };

    tx.blocking_send(Ok(event)).is_ok()
}

async fn send_sse_json<T: Serialize>(tx: &SseSender, payload: &T) -> bool {
    let event = match serialize_sse_event(payload) {
        Ok(event) => event,
        Err(error) => {
            log::error!("Failed to serialize SSE payload: {}", error);
            return false;
        }
    };

    tx.send(Ok(event)).await.is_ok()
}

fn serialize_sse_event<T: Serialize>(payload: &T) -> Result<Event, serde_json::Error> {
    let json = serde_json::to_string(payload)?;
    Ok(Event::default().data(json))
}

fn stop_reason_label(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::StopToken => "stop_token",
        StopReason::MaxTokens => "max_tokens",
        StopReason::Aborted => "aborted",
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

    fn mark_completed(&self) {
        self.abort.set();
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
