use serde::{Deserialize, Serialize};

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

impl OcrRequest {
    /// Short description of the image source for logging.
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
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    StopToken,
    MaxTokens,
    Aborted,
}

/// Result of a completed OCR run.
#[derive(Debug, Serialize)]
pub struct OcrRunResult {
    pub text: String,
    pub generated_tokens: usize,
    pub stop_reason: StopReason,
}
