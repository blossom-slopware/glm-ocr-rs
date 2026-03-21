use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("request aborted")]
    Aborted,

    #[error("invalid request: {message}")]
    InvalidRequest {
        code: &'static str,
        message: &'static str,
    },

    #[error("failed to load input image")]
    ImageLoad {
        #[source]
        source: anyhow::Error,
    },

    #[error("failed to decode input image")]
    ImageDecode {
        #[source]
        source: anyhow::Error,
    },

    #[error("image preprocessing failed")]
    Preprocess {
        #[source]
        source: anyhow::Error,
    },

    #[error("prompt rendering failed")]
    PromptRender {
        #[source]
        source: anyhow::Error,
    },

    #[error("tokenization failed")]
    Tokenization {
        #[source]
        source: anyhow::Error,
    },

    #[error("stream decode failed")]
    StreamDecode {
        #[source]
        source: anyhow::Error,
    },

    #[error("generation failed")]
    Generation {
        #[source]
        source: anyhow::Error,
    },

    #[error("state invariant violated: {message}")]
    StateInvariant {
        code: &'static str,
        message: String,
    },

    #[error("worker panicked: {message}")]
    WorkerPanic {
        message: String,
    },
}

impl EngineError {
    pub fn should_fault_service(&self) -> bool {
        matches!(
            self,
            Self::StreamDecode { .. }
                | Self::Generation { .. }
                | Self::StateInvariant { .. }
                | Self::WorkerPanic { .. }
        )
    }

    pub fn fault_reason(&self) -> Option<&'static str> {
        match self {
            Self::StreamDecode { .. } => Some("stream_decode_failed"),
            Self::Generation { .. } => Some("generation_failed"),
            Self::StateInvariant { code, .. } => Some(code),
            Self::WorkerPanic { .. } => Some("worker_panicked"),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("server is busy")]
    Busy,

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
            Self::Busy => "busy",
            Self::Faulted { reason } => reason,
            Self::BadRequest { code, .. } => code,
            Self::Aborted => "request_aborted",
            Self::Internal { code, .. } => code,
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Self::Busy => "server is busy with another request",
            Self::Faulted { .. } => "service is faulted and requires restart",
            Self::BadRequest { message, .. } => message,
            Self::Aborted => "request aborted",
            Self::Internal { message, .. } => message,
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
