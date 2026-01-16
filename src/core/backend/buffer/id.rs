use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

impl BufferId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}
