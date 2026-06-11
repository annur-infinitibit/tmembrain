#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Write and retrieval pipelines for Membrain.
//!
//! Handles memory ingestion (deduplication, validation, embedding)
//! and retrieval (scoring, gating, reranking) as composable stages.
//!
//! Pipeline stages:
//! - Write pipeline: Salience -> Novelty -> Redundancy -> Budget -> Store
//! - Retrieval pipeline: Intent -> Gating -> Search -> Score -> Budget trim

pub mod adapters;
pub mod retrieval;
pub mod write;

pub use retrieval::{RerankScore, Reranker};
pub use retrieval::{RetrievalFilters, RetrievalPipeline, RetrievalRequest, RetrievalResult};
pub use write::{RejectionReason, WritePipeline, WriteResult};
