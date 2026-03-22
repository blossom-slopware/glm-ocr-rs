use serde::Serialize;
use thiserror::Error;

use glm_ocr_rs::engine::EngineError;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("evicted by newer request")]
    Evicted,

    #[error("service is faulted")]
    Faulted {
        reason: &'static str,
    },

    #[error("bad request: {message}")]
    BadRequest {
        code: &'static str,
        message: &'static str,
    },

    #[error("request aborted")]
    Aborted,

    #[error("internal server error")]
    Internal {
        code: &'static str,
        message: &'static str,
    },
}

impl OcrError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Evicted => "evicted",
            Self::Faulted { reason } => reason,
            Self::BadRequest { code, .. } => code,
            Self::Aborted => "request_aborted",
            Self::Internal { code, .. } => code,
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Self::Evicted => "evicted by newer request",
            Self::Faulted { .. } => "service is faulted and requires restart",
            Self::BadRequest { message, .. } => message,
            Self::Aborted => "request aborted",
            Self::Internal { message, .. } => message,
        }
    }

    pub fn from_engine_error_ref(error: &EngineError) -> Self {
        match error {
            EngineError::Aborted => Self::Aborted,
            EngineError::InvalidRequest { code, message } => Self::BadRequest { code, message },
            EngineError::ImageLoad { .. } => Self::BadRequest {
                code: "image_load_failed",
                message: "failed to load input image",
            },
            EngineError::ImageDecode { .. } => Self::BadRequest {
                code: "image_decode_failed",
                message: "failed to decode input image",
            },
            EngineError::Preprocess { .. } => Self::Internal {
                code: "image_preprocess_failed",
                message: "image preprocessing failed",
            },
            EngineError::PromptRender { .. } => Self::Internal {
                code: "prompt_render_failed",
                message: "prompt rendering failed",
            },
            EngineError::Tokenization { .. } => Self::Internal {
                code: "tokenization_failed",
                message: "tokenization failed",
            },
            EngineError::StreamDecode { .. } => Self::Internal {
                code: "stream_decode_failed",
                message: "stream decode failed",
            },
            EngineError::Generation { .. } => Self::Internal {
                code: "generation_failed",
                message: "generation failed",
            },
            EngineError::StateInvariant { code, .. } => Self::Internal {
                code,
                message: "internal state invariant violated",
            },
            EngineError::WorkerPanic { .. } => Self::Internal {
                code: "worker_panicked",
                message: "internal worker panicked",
            },
        }
    }
}

impl From<EngineError> for OcrError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::Aborted => Self::Aborted,
            EngineError::InvalidRequest { code, message } => Self::BadRequest { code, message },
            EngineError::ImageLoad { .. } => Self::BadRequest {
                code: "image_load_failed",
                message: "failed to load input image",
            },
            EngineError::ImageDecode { .. } => Self::BadRequest {
                code: "image_decode_failed",
                message: "failed to decode input image",
            },
            EngineError::Preprocess { .. } => Self::Internal {
                code: "image_preprocess_failed",
                message: "image preprocessing failed",
            },
            EngineError::PromptRender { .. } => Self::Internal {
                code: "prompt_render_failed",
                message: "prompt rendering failed",
            },
            EngineError::Tokenization { .. } => Self::Internal {
                code: "tokenization_failed",
                message: "tokenization failed",
            },
            EngineError::StreamDecode { .. } => Self::Internal {
                code: "stream_decode_failed",
                message: "stream decode failed",
            },
            EngineError::Generation { .. } => Self::Internal {
                code: "generation_failed",
                message: "generation failed",
            },
            EngineError::StateInvariant { code, .. } => Self::Internal {
                code,
                message: "internal state invariant violated",
            },
            EngineError::WorkerPanic { .. } => Self::Internal {
                code: "worker_panicked",
                message: "internal worker panicked",
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceStateKind {
    Idle,
    Busy,
    Faulted,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServiceStatusSnapshot {
    pub state: ServiceStateKind,
    pub fault_reason: Option<&'static str>,
}
