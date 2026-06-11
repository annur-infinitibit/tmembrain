//! Core traits for the Membrain memory system

mod embedding;
mod extraction;
mod openai_embedding;
mod openai_extractor;
mod storage;

pub use embedding::{EmbeddingConfig, EmbeddingProvider, NoOpEmbeddingProvider};
pub use extraction::{ExtractedFact, ExtractedFactType, ExtractionResult, MemoryExtractor};
pub use openai_embedding::{infer_embedding_dimension, OpenAiEmbeddingProvider};
pub use openai_extractor::OpenAiMemoryExtractor;
pub use storage::{
    MatchType, MemoryStorage, SearchFilters, SearchMode, SearchQuery, SearchResult, StorageStats,
    Transaction,
};
