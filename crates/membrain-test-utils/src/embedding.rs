//! Deterministic embedding provider for tests.
//!
//! Not a mock framework. A real `EmbeddingProvider` impl that returns
//! deterministic vectors so tests can assert on exact values. Use `fail_next`
//! to exercise error paths.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use parking_lot::RwLock;

use membrain_core::error::{Error, Result};
use membrain_core::traits::EmbeddingProvider;
use membrain_core::types::Embedding;

/// Deterministic `EmbeddingProvider` for tests.
///
/// Default behavior: for a given input text `t`, produces a stable vector of
/// `dimension` floats derived from a byte-hash of `t`. Use `with_fixed` to
/// always return the same vector, or `with_mapping` for per-text overrides.
pub struct DeterministicEmbeddingProvider {
    dimension: usize,
    fixed: Option<Vec<f32>>,
    mapping: RwLock<HashMap<String, Vec<f32>>>,
    embed_calls: AtomicUsize,
    health_calls: AtomicUsize,
    fail_next_embed: AtomicBool,
}

impl DeterministicEmbeddingProvider {
    /// New provider producing hash-stable vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            fixed: None,
            mapping: RwLock::new(HashMap::new()),
            embed_calls: AtomicUsize::new(0),
            health_calls: AtomicUsize::new(0),
            fail_next_embed: AtomicBool::new(false),
        }
    }

    /// New provider that always returns `vector`.
    pub fn with_fixed(dimension: usize, vector: Vec<f32>) -> Self {
        let mut provider = Self::new(dimension);
        provider.fixed = Some(vector);
        provider
    }

    /// New provider that returns preset vectors for specific inputs.
    /// Unknown inputs fall back to the hash-stable default.
    pub fn with_mapping(dimension: usize, mapping: HashMap<String, Vec<f32>>) -> Self {
        let provider = Self::new(dimension);
        *provider.mapping.write() = mapping;
        provider
    }

    /// Inject a single failure into the next `embed` call.
    pub fn fail_next(&self) {
        self.fail_next_embed.store(true, Ordering::SeqCst);
    }

    /// Number of times `embed`/`embed_batch` have been invoked.
    pub fn embed_call_count(&self) -> usize {
        self.embed_calls.load(Ordering::SeqCst)
    }

    /// Number of times `health_check` has been invoked.
    pub fn health_check_call_count(&self) -> usize {
        self.health_calls.load(Ordering::SeqCst)
    }

    fn vector_for(&self, text: &str) -> Vec<f32> {
        if let Some(ref fixed) = self.fixed {
            return fixed.clone();
        }
        if let Some(preset) = self.mapping.read().get(text) {
            return preset.clone();
        }
        hash_vector(text, self.dimension)
    }
}

/// Deterministic hash-based vector: stable for a given (text, dimension) pair.
fn hash_vector(text: &str, dimension: usize) -> Vec<f32> {
    let bytes = text.as_bytes();
    (0..dimension)
        .map(|index| {
            let mut acc: u64 = 0x9E37_79B9_7F4A_7C15_u64.wrapping_add(index as u64);
            for &byte in bytes {
                acc = acc
                    .wrapping_mul(1_000_003)
                    .wrapping_add(u64::from(byte))
                    .rotate_left(7);
            }
            let norm = ((acc & 0xFFFF) as f32) / 65_535.0;
            norm * 2.0 - 1.0
        })
        .collect()
}

#[async_trait]
impl EmbeddingProvider for DeterministicEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_next_embed.swap(false, Ordering::SeqCst) {
            return Err(Error::EmbeddingGeneration("injected failure".to_string()));
        }
        Ok(Embedding::new(self.vector_for(text)))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.embed_calls.fetch_add(texts.len(), Ordering::SeqCst);
        if self.fail_next_embed.swap(false, Ordering::SeqCst) {
            return Err(Error::EmbeddingGeneration("injected failure".to_string()));
        }
        Ok(texts
            .iter()
            .map(|text| Embedding::new(self.vector_for(text)))
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "deterministic-test"
    }

    fn model(&self) -> &str {
        "deterministic-hash"
    }

    fn max_input_length(&self) -> usize {
        usize::MAX
    }

    async fn health_check(&self) -> Result<()> {
        self.health_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
