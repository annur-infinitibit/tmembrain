//! Reranker trait for semantic reranking of search results
//!
//! This module defines the `Reranker` trait that language-level implementations
//! (Python/JS) can mirror. The trait provides a Rust-native interface for future
//! extensibility while the primary reranking happens in the language SDKs via
//! HTTP calls to cross encoder (Cohere, Jina) or LLM (OpenAI, Anthropic) APIs.

use async_trait::async_trait;

use membrain_core::error::Result;

/// Result of reranking a single document
#[derive(Debug, Clone)]
pub struct RerankScore {
    /// Index of the document in the original input list
    pub index: usize,
    /// Relevance score assigned by the reranker (0.0-1.0)
    pub relevance_score: f64,
}

/// Trait for reranking search results
///
/// Implementations accept a query and a list of document texts, then return
/// scored results ordered by relevance. The `top_k` parameter controls how
/// many results to keep.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank a list of documents against a query.
    ///
    /// Returns a list of `RerankScore` entries sorted by descending relevance,
    /// truncated to `top_k` results.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankScore>>;
}
