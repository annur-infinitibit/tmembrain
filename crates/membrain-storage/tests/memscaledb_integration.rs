#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]
//! Integration tests for MemscaleDB storage backend.

use membrain_core::memory::{FactMemory, Memory, SemanticContent, SemanticMemory};
use membrain_core::traits::{MemoryStorage, SearchFilters, SearchMode, SearchQuery};
use membrain_core::types::{AgentId, Confidence, Embedding, Provenance, Source};
use membrain_storage::MemscaleDbStorage;
use std::env;
use uuid::Uuid;

/// Create a test memory with the given text.
fn create_test_memory(agent_id: AgentId, text: &str) -> Memory {
    let provenance = Provenance::new_direct(Source::user_input("test"), agent_id);
    let common = membrain_core::memory::MemoryCommon::new(agent_id, provenance)
        .with_confidence(Confidence::new(0.9))
        .with_tag("test");

    let fact = FactMemory {
        statement: text.to_string(),
        subject: Some("test".to_string()),
        predicate: Some("is".to_string()),
        object: Some("testing".to_string()),
    };

    Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(fact),
    })
}

/// Create a test memory with embedding.
fn create_memory_with_embedding(agent_id: AgentId, text: &str, embedding: Vec<f32>) -> Memory {
    let provenance = Provenance::new_direct(Source::user_input("test"), agent_id);
    let common = membrain_core::memory::MemoryCommon::new(agent_id, provenance)
        .with_confidence(Confidence::new(0.9))
        .with_embedding(Embedding::new(embedding))
        .with_tag("test");

    let fact = FactMemory {
        statement: text.to_string(),
        subject: Some("test".to_string()),
        predicate: Some("is".to_string()),
        object: Some("testing".to_string()),
    };

    Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(fact),
    })
}

#[tokio::test]
async fn test_store_and_get() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory = create_test_memory(agent_id, "Test fact");

    let memory_id = storage
        .store(memory.clone())
        .await
        .expect("Failed to store memory");

    let retrieved = storage
        .get(&memory_id)
        .await
        .expect("Failed to get memory")
        .expect("Memory not found");

    assert_eq!(retrieved.id(), memory.id());
    assert_eq!(retrieved.text_content(), memory.text_content());

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_update_memory() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let mut memory = create_test_memory(agent_id, "Original fact");

    let memory_id = storage
        .store(memory.clone())
        .await
        .expect("Failed to store memory");

    let version = memory.common().version;
    memory.common_mut().confidence = Confidence::new(0.95);

    storage
        .update(memory.clone(), version)
        .await
        .expect("Failed to update memory");

    let retrieved = storage
        .get(&memory_id)
        .await
        .expect("Failed to get memory")
        .expect("Memory not found");

    assert_eq!(retrieved.confidence().value(), 0.95);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_delete_memory() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory = create_test_memory(agent_id, "To be deleted");

    let memory_id = storage.store(memory).await.expect("Failed to store memory");

    let exists_before = storage
        .exists(&memory_id)
        .await
        .expect("Failed to check existence");
    assert!(exists_before);

    let deleted = storage
        .delete(&memory_id)
        .await
        .expect("Failed to delete memory");
    assert!(deleted);

    let exists_after = storage
        .exists(&memory_id)
        .await
        .expect("Failed to check existence");
    assert!(!exists_after);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_get_many() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory1 = create_test_memory(agent_id, "First fact");
    let memory2 = create_test_memory(agent_id, "Second fact");

    let id1 = storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    let id2 = storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    let memories = storage
        .get_many(&[id1, id2])
        .await
        .expect("Failed to get many");

    assert_eq!(memories.len(), 2);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_text_search() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory1 = create_test_memory(agent_id, "The quick brown fox jumps over the lazy dog");
    let memory2 = create_test_memory(agent_id, "A slow turtle walks under the bridge");

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let results = storage
        .text_search("fox", 10)
        .await
        .expect("Failed to search");

    assert!(!results.is_empty(), "Should find results for 'fox'");
    assert!(results[0].memory.text_content().contains("fox"));

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_vector_search() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();

    let embedding1 = vec![1.0; 1536];
    let embedding2 = vec![0.5; 1536];

    let memory1 = create_memory_with_embedding(agent_id, "First memory", embedding1.clone());
    let memory2 = create_memory_with_embedding(agent_id, "Second memory", embedding2);

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    let query_embedding = Embedding::new(embedding1);
    let results = storage
        .vector_search(&query_embedding, 2, None)
        .await
        .expect("Failed to vector search");

    assert!(!results.is_empty(), "Should find results");
    assert_eq!(results.len(), 2);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_hybrid_search() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();

    let embedding1 = vec![1.0; 1536];
    let embedding2 = vec![0.5; 1536];

    let memory1 =
        create_memory_with_embedding(agent_id, "The quick brown fox jumps", embedding1.clone());
    let memory2 = create_memory_with_embedding(agent_id, "A slow turtle walks", embedding2);

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let query = SearchQuery::new()
        .with_query("fox")
        .with_embedding(Embedding::new(embedding1))
        .with_mode(SearchMode::Hybrid)
        .with_limit(10);

    let results = storage.search(query).await.expect("Failed to search");

    assert!(!results.is_empty(), "Should find results");

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_count() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();

    let count_before = storage.count(None).await.expect("Failed to count");
    assert_eq!(count_before, 0);

    let memory1 = create_test_memory(agent_id, "First fact");
    let memory2 = create_test_memory(agent_id, "Second fact");

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    let count_after = storage.count(None).await.expect("Failed to count");
    assert_eq!(count_after, 2);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_get_by_agent() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent1 = AgentId::new();
    let agent2 = AgentId::new();

    let memory1 = create_test_memory(agent1, "Agent 1 memory");
    let memory2 = create_test_memory(agent2, "Agent 2 memory");

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    let agent1_memories = storage
        .get_by_agent(&agent1, 10, 0)
        .await
        .expect("Failed to get by agent");

    assert_eq!(agent1_memories.len(), 1);
    assert_eq!(agent1_memories[0].common().agent_id, agent1);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_get_by_type() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory = create_test_memory(agent_id, "Semantic fact");

    storage
        .store(memory.clone())
        .await
        .expect("Failed to store");

    let memories = storage
        .get_by_type(memory.memory_type(), 10, 0)
        .await
        .expect("Failed to get by type");

    assert!(!memories.is_empty());
    assert_eq!(memories[0].memory_type(), memory.memory_type());

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_record_access() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory = create_test_memory(agent_id, "Access tracking test");

    let memory_id = storage.store(memory).await.expect("Failed to store");

    let before = storage
        .get(&memory_id)
        .await
        .expect("Failed to get")
        .expect("Memory not found");
    let access_count_before = before.common().provenance.access_count;

    storage
        .record_access(&memory_id)
        .await
        .expect("Failed to record access");

    let after = storage
        .get(&memory_id)
        .await
        .expect("Failed to get")
        .expect("Memory not found");
    let access_count_after = after.common().provenance.access_count;

    assert_eq!(access_count_after, access_count_before + 1);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_stats() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let embedding = vec![1.0; 1536];
    let memory = create_memory_with_embedding(agent_id, "Stats test", embedding);

    storage.store(memory).await.expect("Failed to store");

    let stats = storage.stats().await.expect("Failed to get stats");

    assert_eq!(stats.total_memories, 1);
    assert_eq!(stats.embeddings_count, 1);

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_health_check() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .health_check()
        .await
        .expect("Health check should pass");

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_filter_by_confidence() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let embedding = vec![1.0; 1536];

    let mut memory1 = create_memory_with_embedding(agent_id, "High confidence", embedding.clone());
    memory1.common_mut().confidence = Confidence::new(0.9);

    let mut memory2 = create_memory_with_embedding(agent_id, "Low confidence", embedding);
    memory2.common_mut().confidence = Confidence::new(0.3);

    storage
        .store(memory1)
        .await
        .expect("Failed to store memory1");
    storage
        .store(memory2)
        .await
        .expect("Failed to store memory2");

    let filters = SearchFilters::new().with_min_confidence(Confidence::new(0.8));

    let count = storage
        .count(Some(filters))
        .await
        .expect("Failed to count with filter");

    assert_eq!(count, 1, "Should only count high confidence memory");

    std::fs::remove_dir_all(temp_dir).ok();
}

#[tokio::test]
async fn test_delete_many() {
    let temp_dir = env::temp_dir().join(format!("memscaledb_test_{}", Uuid::new_v4()));
    let storage = MemscaleDbStorage::new(temp_dir.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    let agent_id = AgentId::new();
    let memory1 = create_test_memory(agent_id, "First");
    let memory2 = create_test_memory(agent_id, "Second");
    let memory3 = create_test_memory(agent_id, "Third");

    let id1 = storage.store(memory1).await.expect("Failed to store");
    let id2 = storage.store(memory2).await.expect("Failed to store");
    let id3 = storage.store(memory3).await.expect("Failed to store");

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let count = storage
        .delete_many(&[id1, id2])
        .await
        .expect("Failed to delete many");

    assert_eq!(count, 2);

    assert!(!storage.exists(&id1).await.expect("Failed to check"));
    assert!(!storage.exists(&id2).await.expect("Failed to check"));
    assert!(storage.exists(&id3).await.expect("Failed to check"));

    std::fs::remove_dir_all(temp_dir).ok();
}
