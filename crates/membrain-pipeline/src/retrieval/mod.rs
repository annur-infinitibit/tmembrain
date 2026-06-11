//! Retrieval pipeline for memory search and context building

mod gating;
mod intent;
mod pipeline;
mod reranker;
mod scoring;

pub use gating::{GatingDecision, RetrievalGating};
pub use intent::{IntentDetector, IntentType, QueryIntent};
pub use membrain_graph::bridge::GraphAugmentedRetrieval;
pub use pipeline::{
    RetrievalFilters, RetrievalPipeline, RetrievalRequest, RetrievalResult, RetrievedMemory,
};
pub use reranker::{RerankScore, Reranker};
pub use scoring::{DiversityReranker, ScoreWeights, ScoringStrategy};
