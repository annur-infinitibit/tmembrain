#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]
//! Scoped-search integration: verify metadata-based pre-filtering.

use std::collections::HashMap;
use std::env;

use membrain_core::memory::{FactMemory, Memory, MemoryCommon, SemanticContent, SemanticMemory};
use membrain_core::traits::{MemoryStorage, SearchFilters};
use membrain_core::types::{AgentId, Confidence, Embedding, Provenance, Source};
use membrain_storage::MemscaleDbStorage;
use memscaledb::MemscaleStorageConfig;
use uuid::Uuid;

fn make_memory(
    agent_id: AgentId,
    text: &str,
    embedding: Vec<f32>,
    metadata: &[(&str, serde_json::Value)],
) -> Memory {
    let provenance = Provenance::new_direct(Source::user_input("test"), agent_id);
    let mut common = MemoryCommon::new(agent_id, provenance)
        .with_confidence(Confidence::new(0.9))
        .with_embedding(Embedding::new(embedding));
    for (key, value) in metadata {
        common
            .metadata
            .insert((*key).to_string(), value.clone());
    }
    Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(FactMemory {
            statement: text.to_string(),
            subject: Some("test".to_string()),
            predicate: Some("is".to_string()),
            object: Some("testing".to_string()),
        }),
    })
}

async fn make_storage(
    indexed_keys: Vec<String>,
    dim: usize,
) -> (MemscaleDbStorage, std::path::PathBuf) {
    let temp_dir = env::temp_dir().join(format!("scoped_search_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let config = MemscaleStorageConfig::new(temp_dir.join("db"))
        .with_indexed_metadata_keys(indexed_keys);
    let storage = MemscaleDbStorage::with_config(config, dim).await.unwrap();
    (storage, temp_dir)
}

#[tokio::test]
async fn vector_search_filters_by_indexed_metadata() {
    let temp_dir = env::temp_dir().join(format!("scoped_search_vs_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let config = MemscaleStorageConfig::new(temp_dir.join("db"))
        .with_indexed_metadata_keys(vec!["user_id".into(), "thread_id".into()]);
    let storage = MemscaleDbStorage::with_config(config, 4).await.unwrap();

    let agent = AgentId::new();
    for (text, user) in [
        ("alice likes rust", "alice"),
        ("alice likes go", "alice"),
        ("bob likes python", "bob"),
        ("bob likes rust", "bob"),
    ] {
        let memory = make_memory(
            agent,
            text,
            vec![1.0, 0.0, 0.0, 0.0],
            &[("user_id", serde_json::json!(user))],
        );
        storage.store(memory).await.unwrap();
    }

    let embedding = Embedding::new(vec![1.0, 0.0, 0.0, 0.0]);
    let filters = SearchFilters::new()
        .with_metadata_entry("user_id", serde_json::json!("alice"));
    let results = storage
        .vector_search(&embedding, 10, Some(filters))
        .await
        .unwrap();
    assert_eq!(results.len(), 2, "expected exactly alice's two memories");
    for result in &results {
        let meta = &result.memory.common().metadata;
        assert_eq!(meta["user_id"], serde_json::json!("alice"));
    }

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn count_applies_indexed_metadata_prefilter() {
    let temp_dir = env::temp_dir().join(format!("scoped_search_ct_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let config = MemscaleStorageConfig::new(temp_dir.join("db"))
        .with_indexed_metadata_keys(vec!["user_id".into()]);
    let storage = MemscaleDbStorage::with_config(config, 4).await.unwrap();
    let agent = AgentId::new();

    for user in ["alice", "alice", "alice", "bob", "bob"] {
        let memory = make_memory(
            agent,
            &format!("{user} fact"),
            vec![1.0, 0.0, 0.0, 0.0],
            &[("user_id", serde_json::json!(user))],
        );
        storage.store(memory).await.unwrap();
    }

    let filters = SearchFilters::new()
        .with_metadata_entry("user_id", serde_json::json!("alice"));
    let count = storage.count(Some(filters)).await.unwrap();
    assert_eq!(count, 3);

    let filters_bob = SearchFilters::new()
        .with_metadata_entry("user_id", serde_json::json!("bob"));
    assert_eq!(storage.count(Some(filters_bob)).await.unwrap(), 2);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn non_indexed_metadata_filter_still_works() {
    // Non-indexed key falls back to residual post-filter path — correctness intact.
    let (storage, temp_dir) = make_storage(vec!["user_id".into()], 4).await;
    let agent = AgentId::new();
    for role in ["admin", "viewer", "admin"] {
        let memory = make_memory(
            agent,
            &format!("{role} memory"),
            vec![1.0, 0.0, 0.0, 0.0],
            &[
                ("user_id", serde_json::json!("alice")),
                ("role", serde_json::json!(role)),
            ],
        );
        storage.store(memory).await.unwrap();
    }

    let mut filter_map = HashMap::new();
    filter_map.insert("role".to_string(), serde_json::json!("admin"));
    let filters = SearchFilters::new().with_metadata(filter_map);
    let count = storage.count(Some(filters)).await.unwrap();
    assert_eq!(count, 2);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn indexed_plus_non_indexed_filter_intersects_correctly() {
    let (storage, temp_dir) = make_storage(vec!["user_id".into()], 4).await;
    let agent = AgentId::new();
    for (user, role) in [
        ("alice", "admin"),
        ("alice", "viewer"),
        ("bob", "admin"),
    ] {
        let memory = make_memory(
            agent,
            &format!("{user}/{role}"),
            vec![1.0, 0.0, 0.0, 0.0],
            &[
                ("user_id", serde_json::json!(user)),
                ("role", serde_json::json!(role)),
            ],
        );
        storage.store(memory).await.unwrap();
    }

    let filters = SearchFilters::new()
        .with_metadata_entry("user_id", serde_json::json!("alice"))
        .with_metadata_entry("role", serde_json::json!("admin"));
    let results = storage
        .vector_search(&Embedding::new(vec![1.0, 0.0, 0.0, 0.0]), 10, Some(filters))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    let meta = &results[0].memory.common().metadata;
    assert_eq!(meta["user_id"], serde_json::json!("alice"));
    assert_eq!(meta["role"], serde_json::json!("admin"));

    std::fs::remove_dir_all(temp_dir).ok();
}
