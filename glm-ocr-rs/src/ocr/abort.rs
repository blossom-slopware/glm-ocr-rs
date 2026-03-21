use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct AbortSignal(Arc<AtomicBool>);

impl AbortSignal {
    pub fn new() -> Self {
        AbortSignal(Arc::new(AtomicBool::new(false)))
    }

    /// Create a signal that is never set — used by `generate()` wrapper.
    pub fn none() -> Self {
        AbortSignal::new()
    }

    pub fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
