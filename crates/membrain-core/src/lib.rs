#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Core types and traits for the Membrain memory system.
//!
//! Defines the memory model, configuration, storage traits, and
//! common types used across all Membrain crates.
//!
//! # Memory Types
//!
//! - **Semantic**: Facts, concepts, entities, preferences (for RAG)
//! - **Episodic**: Conversation history, events, observations
//! - **Procedural**: Workflows, skills, behavioral patterns
//! - **AgentState**: Goals, tasks, working memory

pub mod config;
pub mod error;
pub mod memory;
pub mod traits;
pub mod types;

pub use config::Config;
pub use error::{Error, Result};
pub use memory::{
    AgentStateMemory, CaseMemory, EpisodicMemory, Memory, MemoryCommon, ProceduralMemory,
    SemanticMemory,
};
pub use traits::{
    EmbeddingConfig, EmbeddingProvider, ExtractedFact, ExtractedFactType, ExtractionResult,
    MemoryExtractor, MemoryStorage, NoOpEmbeddingProvider, OpenAiMemoryExtractor,
};
pub use types::{AgentId, Confidence, Embedding, MemoryId, Provenance, SessionId, Version};
