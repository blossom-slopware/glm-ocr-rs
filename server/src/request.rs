use serde::{Deserialize, Serialize};

use glm_ocr_rs::engine::EngineError;

/// How the image is supplied by the caller.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Local file path, http(s):// URL, or data: URI.
    Url { url: String },
    /// Raw bytes (for direct Rust API use, not deserializable from JSON).
    #[serde(skip)]
    Bytes(Vec<u8>),
}

fn default_max_tokens() -> usize {
    4096
}

fn default_temperature() -> f32 {
    0.01
}

/// A single OCR request.
#[derive(Debug, Deserialize)]
pub struct OcrRequest {
    pub image: ImageSource,
    /// The OCR instruction prompt. If None, uses the default OCR prompt.
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

const MAX_TOKENS_LIMIT: usize = 16384;

impl OcrRequest {
    pub fn validate(&self) -> Result<usize, EngineError> {
        match &self.image {
            ImageSource::Url { url } if url.trim().is_empty() => {
                return Err(EngineError::InvalidRequest {
                    code: "empty_image_url",
                    message: "image URL must not be empty",
                });
            }
            ImageSource::Bytes(bytes) if bytes.is_empty() => {
                return Err(EngineError::InvalidRequest {
                    code: "empty_image_bytes",
                    message: "image bytes must not be empty",
                });
            }
            _ => {}
        }

        if self.max_tokens == 0 {
            return Err(EngineError::InvalidRequest {
                code: "invalid_max_tokens",
                message: "max_tokens must be greater than zero",
            });
        }

        let effective_max_tokens = if self.max_tokens > MAX_TOKENS_LIMIT {
            log::warn!(
                "max_tokens {} exceeds limit {}, clamping to {}",
                self.max_tokens,
                MAX_TOKENS_LIMIT,
                MAX_TOKENS_LIMIT
            );
            MAX_TOKENS_LIMIT
        } else {
            self.max_tokens
        };

        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::InvalidRequest {
                code: "invalid_temperature",
                message: "temperature must be a finite non-negative number",
            });
        }

        Ok(effective_max_tokens)
    }

    pub fn image_description(&self) -> String {
        match &self.image {
            ImageSource::Url { url } => {
                if url.starts_with("data:") {
                    format!("data:...({} bytes)", url.len())
                } else if url.len() > 120 {
                    format!("{}...", &url[..120])
                } else {
                    url.clone()
                }
            }
            ImageSource::Bytes(bytes) => format!("<{} bytes>", bytes.len()),
        }
    }

    /// Convert to inference-level input for the engine.
    pub fn to_inference_input(&self) -> glm_ocr_rs::engine::InferenceInput {
        let image = match &self.image {
            ImageSource::Url { url } => glm_ocr_rs::engine::ImageSource::Url(url.clone()),
            ImageSource::Bytes(bytes) => glm_ocr_rs::engine::ImageSource::Bytes(bytes.clone()),
        };
        glm_ocr_rs::engine::InferenceInput {
            image,
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    StopToken,
    MaxTokens,
    Aborted,
}

impl From<glm_ocr_rs::engine::StopReason> for StopReason {
    fn from(reason: glm_ocr_rs::engine::StopReason) -> Self {
        match reason {
            glm_ocr_rs::engine::StopReason::StopToken => Self::StopToken,
            glm_ocr_rs::engine::StopReason::MaxTokens => Self::MaxTokens,
            glm_ocr_rs::engine::StopReason::Aborted => Self::Aborted,
        }
    }
}

/// Result of a completed OCR run.
#[derive(Debug, Clone, Serialize)]
pub struct OcrRunResult {
    pub text: String,
    pub generated_tokens: usize,
    pub stop_reason: StopReason,
}

impl From<glm_ocr_rs::engine::InferenceResult> for OcrRunResult {
    fn from(result: glm_ocr_rs::engine::InferenceResult) -> Self {
        Self {
            text: result.text,
            generated_tokens: result.generated_tokens,
            stop_reason: result.stop_reason.into(),
        }
    }
}
