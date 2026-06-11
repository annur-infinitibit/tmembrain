#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! FFI bindings for Membrain
//!
//! Provides a shared C ABI consumed by:
//! - Python (via ctypes)
//! - Node.js (via koffi)
//! - Any C/C++ consumer

pub mod c_api;
mod client;
mod types;

#[cfg(test)]
mod tests;

pub use client::MembrainClient;
pub use types::{
    GraphInfoJson, GraphPruningResultJson, GraphQueryResultJson, GraphScoredNodeJson,
    GraphTraversalStepJson, MemoryInfo, SearchFiltersJson, SearchResult, SearchResults,
    StorageStatsResult, StoreResult, VectorBackendCapabilities, VectorBackendHealthResult,
    VectorBackendStatsResult,
};
