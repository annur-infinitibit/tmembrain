#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Memory compression and summarization for Membrain.
//!
//! Reduces storage footprint by compressing older memories
//! while preserving semantic fidelity.

pub mod decay;
pub mod distillation;

pub use decay::{DecayEngine, DecayPolicy};
pub use distillation::{DistillationConfig, DistillationEngine};

/// Configuration for compression operations
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Minimum confidence before memory is deleted
    pub min_confidence: f64,
    /// Whether to auto-delete very low confidence memories
    pub auto_delete: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.1,
            auto_delete: true,
            batch_size: 100,
        }
    }
}
