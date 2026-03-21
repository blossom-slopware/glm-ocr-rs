pub mod abort;
pub mod engine;
pub mod error;
pub mod request;
pub mod service;

pub use abort::AbortSignal;
pub use engine::OcrEngine;
pub use error::{EngineError, OcrError, ServiceStateKind, ServiceStatusSnapshot};
pub use request::{ImageSource, OcrRequest, OcrRunResult, StopReason};
pub use service::OcrService;
