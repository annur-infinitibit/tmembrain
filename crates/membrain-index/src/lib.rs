#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Full-text and metadata indexing for Membrain.
//!
//! Supplements vector search with keyword-based and
//! metadata-filtered retrieval capabilities.
//!
//! This crate also re-exports the full memscaledb API and provides a bridging
//! `VectorIndex` trait that operates on `membrain-core` types (`MemoryId`,
//! `Embedding`) rather than raw `VectorId` / `&[f32]` slices.

// Re-export everything from memscaledb for direct use.
pub use memscaledb::*;

use membrain_core::error::{Error as CoreError, Result as CoreResult};
use membrain_core::types::{Embedding, MemoryId};

/// Convert a `MemoryId` to a `VectorId` (both are UUID v7 under the hood).
pub fn memory_id_to_vector_id(id: MemoryId) -> memscaledb::VectorId {
    memscaledb::VectorId::from_bytes(*id.as_bytes())
}

/// Convert a `VectorId` to a `MemoryId`.
pub fn vector_id_to_memory_id(id: memscaledb::VectorId) -> MemoryId {
    MemoryId::from_bytes(*id.as_bytes())
}

/// Convert a memscaledb error to a membrain-core error.
fn convert_error(error: memscaledb::Error) -> CoreError {
    match error {
        memscaledb::Error::DimensionMismatch { expected, actual } => {
            CoreError::EmbeddingDimensionMismatch { expected, actual }
        }
        other => CoreError::IndexError(other.to_string()),
    }
}

/// Convert a memscaledb `VectorSearchResult` to one with `MemoryId`.
fn convert_result(result: memscaledb::VectorSearchResult) -> VectorSearchResultCompat {
    VectorSearchResultCompat {
        id: vector_id_to_memory_id(result.id),
        score: result.score,
        distance: result.distance,
    }
}

/// Search result using `MemoryId` for compatibility with the membrain ecosystem.
#[derive(Debug, Clone)]
pub struct VectorSearchResultCompat {
    /// The memory ID.
    pub id: MemoryId,
    /// Similarity score (higher is more similar).
    pub score: f32,
    /// Distance (lower is more similar).
    pub distance: f32,
}

/// Trait for vector indices using membrain-core types.
///
/// This bridges the memscaledb `VectorIndex` trait to the membrain ecosystem
/// by accepting `MemoryId` and `Embedding` instead of `VectorId` and `&[f32]`.
pub trait MembrainVectorIndex: Send + Sync {
    /// Add a vector to the index.
    fn add(&mut self, id: MemoryId, embedding: &Embedding) -> CoreResult<()>;

    /// Remove a vector from the index.
    fn remove(&mut self, id: &MemoryId) -> CoreResult<bool>;

    /// Search for nearest neighbors.
    fn search(&self, query: &Embedding, k: usize) -> CoreResult<Vec<VectorSearchResultCompat>>;

    /// Search with a filter predicate.
    fn search_with_filter(
        &self,
        query: &Embedding,
        k: usize,
        filter: &dyn Fn(&MemoryId) -> bool,
    ) -> CoreResult<Vec<VectorSearchResultCompat>>;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all vectors.
    fn clear(&mut self);

    /// Vector dimension.
    fn dimension(&self) -> usize;
}

/// Blanket implementation: any `memscaledb::VectorIndex` can be used as a
/// `MembrainVectorIndex` via automatic type conversion.
impl<T: memscaledb::VectorIndex> MembrainVectorIndex for T {
    fn add(&mut self, id: MemoryId, embedding: &Embedding) -> CoreResult<()> {
        let vector_id = memory_id_to_vector_id(id);
        memscaledb::VectorIndex::add(self, vector_id, embedding.values()).map_err(convert_error)
    }

    fn remove(&mut self, id: &MemoryId) -> CoreResult<bool> {
        let vector_id = memory_id_to_vector_id(*id);
        memscaledb::VectorIndex::remove(self, &vector_id).map_err(convert_error)
    }

    fn search(&self, query: &Embedding, k: usize) -> CoreResult<Vec<VectorSearchResultCompat>> {
        let results =
            memscaledb::VectorIndex::search(self, query.values(), k).map_err(convert_error)?;
        Ok(results.into_iter().map(convert_result).collect())
    }

    fn search_with_filter(
        &self,
        query: &Embedding,
        k: usize,
        filter: &dyn Fn(&MemoryId) -> bool,
    ) -> CoreResult<Vec<VectorSearchResultCompat>> {
        let vector_filter = |vid: &memscaledb::VectorId| {
            let mid = vector_id_to_memory_id(*vid);
            filter(&mid)
        };
        let results =
            memscaledb::VectorIndex::search_with_filter(self, query.values(), k, &vector_filter)
                .map_err(convert_error)?;
        Ok(results.into_iter().map(convert_result).collect())
    }

    fn len(&self) -> usize {
        memscaledb::VectorIndex::len(self)
    }

    fn clear(&mut self) {
        memscaledb::VectorIndex::clear(self);
    }

    fn dimension(&self) -> usize {
        memscaledb::VectorIndex::dimension(self)
    }
}
