#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Storage backends for Membrain.
//!
//! Provides in-memory and SQLite storage implementations
//! for the `MemoryStorage` trait.
//!
//! Supported backends:
//! - SQLite with FTS5 for full-text search
//! - In-memory storage for testing
//! - PostgreSQL with pgvector support (future)

pub mod backend;

pub use backend::memory::InMemoryStorage;
pub use backend::memscaledb::MemscaleDbStorage;
pub use backend::sqlite::SqliteStorage;

use membrain_core::config::{RetrievalConfig, StorageConfig};
use membrain_core::error::{Error, Result};
use membrain_core::traits::MemoryStorage;
use std::sync::Arc;

/// Create a storage backend from configuration.
///
/// Supports "memscaledb" (default), "memory" (in-memory), and "sqlite" backends.
/// An optional `RetrievalConfig` propagates hybrid search settings to MemscaleDB.
/// The `embedding_dimension` parameter sets the vector dimension for the storage
/// backend. When `None`, defaults to 1536 (OpenAI text-embedding-ada-002).
pub async fn create_storage(
    config: &StorageConfig,
    retrieval_config: Option<&RetrievalConfig>,
    embedding_dimension: Option<usize>,
) -> Result<Arc<dyn MemoryStorage>> {
    let dimension = embedding_dimension.unwrap_or(1536);
    match config.backend.as_str() {
        "memscaledb" => {
            let path = config.path.as_deref().unwrap_or("memscaledb");
            let memscale_config = memscaledb::MemscaleStorageConfig::new(path)
                .with_indexed_metadata_keys(config.indexed_metadata_keys.clone());
            if let Some(retrieval) = retrieval_config {
                let storage =
                    MemscaleDbStorage::with_retrieval_config(memscale_config, dimension, retrieval)
                        .await?;
                Ok(Arc::new(storage))
            } else {
                let storage =
                    MemscaleDbStorage::with_config(memscale_config, dimension).await?;
                Ok(Arc::new(storage))
            }
        }
        "sqlite" => {
            let path = config.path.as_deref().unwrap_or("membrain.db");
            let storage = SqliteStorage::new(path).await?;
            Ok(Arc::new(storage))
        }
        "memory" => {
            let storage = InMemoryStorage::new();
            Ok(Arc::new(storage))
        }
        "postgres" => Err(Error::Configuration(
            "PostgreSQL backend not yet implemented".to_string(),
        )),
        other => Err(Error::InvalidConfigValue {
            key: "storage.backend".to_string(),
            message: format!("Unknown backend: {}", other),
        }),
    }
}
