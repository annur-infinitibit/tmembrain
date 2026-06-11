//! Contract tests for `MemoryStorage` using the `InMemoryStorageStub`.
//!
//! Any real backend must satisfy the same semantics; these tests are copied
//! into each backend's integration suite when wiring new implementations.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::sync::Arc;

use futures_util::future::join_all;
use membrain_core::memory::{FactMemory, Memory, MemoryType, SemanticContent, SemanticMemory};
use membrain_core::traits::{MemoryStorage, SearchFilters, SearchQuery};
use membrain_core::types::{AgentId, Confidence, MemoryId, Provenance, Source};
use membrain_test_utils::{common, semantic_fact, semantic_fact_for, InMemoryStorageStub};

fn fresh_memory(text: &str) -> Memory {
    semantic_fact(text)
}

#[tokio::test]
async fn store_and_get_returns_same_memory() {
    let storage = InMemoryStorageStub::new();
    let memory = fresh_memory("apple");
    let id = storage.store(memory.clone()).await.expect("store");
    let reloaded = storage.get(&id).await.expect("get");
    assert!(reloaded.is_some());
    assert_eq!(reloaded.expect("some").id(), &id);
}

#[tokio::test]
async fn get_nonexistent_returns_none() {
    let storage = InMemoryStorageStub::new();
    let reloaded = storage.get(&MemoryId::new()).await.expect("get");
    assert!(reloaded.is_none());
}

#[tokio::test]
async fn get_many_filters_missing_ids() {
    let storage = InMemoryStorageStub::new();
    let id_a = storage.store(fresh_memory("a")).await.expect("store");
    let id_b = storage.store(fresh_memory("b")).await.expect("store");
    let missing = MemoryId::new();

    let memories = storage
        .get_many(&[id_a, missing, id_b])
        .await
        .expect("get_many");
    assert_eq!(memories.len(), 2);
}

#[tokio::test]
async fn update_rejects_version_conflict() {
    let storage = InMemoryStorageStub::new();
    let memory = fresh_memory("a");
    let id = storage.store(memory.clone()).await.expect("store");

    let mut updated = memory.clone();
    updated.set_text_content("a!");
    let result = storage.update(updated, 999).await;
    assert!(result.is_err());
    let _ = id;
}

#[tokio::test]
async fn update_succeeds_with_correct_version() {
    let storage = InMemoryStorageStub::new();
    let memory = fresh_memory("original");
    let id = storage.store(memory.clone()).await.expect("store");

    let mut updated = memory.clone();
    updated.set_text_content("changed");
    updated.common_mut().increment_version();
    storage
        .update(updated.clone(), 1)
        .await
        .expect("version-1 update");

    let reloaded = storage.get(&id).await.expect("get").expect("some");
    assert!(reloaded.text_content().contains("changed"));
}

#[tokio::test]
async fn delete_removes_memory() {
    let storage = InMemoryStorageStub::new();
    let id = storage.store(fresh_memory("a")).await.expect("store");
    assert!(storage.delete(&id).await.expect("delete"));
    assert!(storage.get(&id).await.expect("get").is_none());
}

#[tokio::test]
async fn delete_many_returns_count() {
    let storage = InMemoryStorageStub::new();
    let id_a = storage.store(fresh_memory("a")).await.expect("store");
    let id_b = storage.store(fresh_memory("b")).await.expect("store");
    let missing = MemoryId::new();
    let removed = storage
        .delete_many(&[id_a, id_b, missing])
        .await
        .expect("delete_many");
    assert_eq!(removed, 2);
}

#[tokio::test]
async fn search_filters_by_agent() {
    let storage = InMemoryStorageStub::new();
    let alice = AgentId::new();
    let bob = AgentId::new();
    storage
        .store(semantic_fact_for(alice, "alice-memory"))
        .await
        .expect("store");
    storage
        .store(semantic_fact_for(bob, "bob-memory"))
        .await
        .expect("store");

    let filters = SearchFilters::new().with_agents(vec![alice]);
    let query = SearchQuery::new().with_filters(filters).with_limit(10);
    let results = storage.search(query).await.expect("search");
    assert_eq!(results.len(), 1);
    assert!(results[0].memory.text_content().contains("alice"));
}

#[tokio::test]
async fn search_filters_by_type() {
    let storage = InMemoryStorageStub::new();
    storage.store(fresh_memory("fact-a")).await.expect("store");

    let (_agent, common) = common(0.7);
    let memory = Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(FactMemory::new("fact-b")),
    });
    storage.store(memory).await.expect("store");

    let filters = SearchFilters::new().with_types(vec![MemoryType::SemanticFact]);
    let query = SearchQuery::new().with_filters(filters).with_limit(10);
    let results = storage.search(query).await.expect("search");
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn get_by_agent_paginates() {
    let storage = InMemoryStorageStub::new();
    let agent = AgentId::new();
    for index in 0..5 {
        storage
            .store(semantic_fact_for(agent, &format!("m-{index}")))
            .await
            .expect("store");
    }
    let page_one = storage.get_by_agent(&agent, 2, 0).await.expect("page 0");
    let page_two = storage.get_by_agent(&agent, 2, 2).await.expect("page 1");
    assert_eq!(page_one.len(), 2);
    assert_eq!(page_two.len(), 2);
    assert_ne!(page_one[0].id(), page_two[0].id());
}

#[tokio::test]
async fn get_by_type_paginates() {
    let storage = InMemoryStorageStub::new();
    for index in 0..4 {
        storage
            .store(fresh_memory(&format!("fact-{index}")))
            .await
            .expect("store");
    }
    let slice = storage
        .get_by_type(MemoryType::SemanticFact, 2, 1)
        .await
        .expect("slice");
    assert_eq!(slice.len(), 2);
}

#[tokio::test]
async fn count_matches_inserts() {
    let storage = InMemoryStorageStub::new();
    for _ in 0..3 {
        storage.store(fresh_memory("x")).await.expect("store");
    }
    let total = MemoryStorage::count(&storage, None).await.expect("count");
    assert_eq!(total, 3);
}

#[tokio::test]
async fn exists_reflects_state() {
    let storage = InMemoryStorageStub::new();
    let id = storage.store(fresh_memory("x")).await.expect("store");
    assert!(storage.exists(&id).await.expect("exists"));
    storage.delete(&id).await.expect("delete");
    assert!(!storage.exists(&id).await.expect("exists"));
}

#[tokio::test]
async fn record_access_updates_provenance() {
    let storage = InMemoryStorageStub::new();
    let id = storage.store(fresh_memory("x")).await.expect("store");
    storage.record_access(&id).await.expect("access");
    let memory = storage.get(&id).await.expect("get").expect("some");
    assert!(memory.common().provenance.access_count >= 1);
}

#[tokio::test]
async fn stats_reflects_counts() {
    let storage = InMemoryStorageStub::new();
    for _ in 0..3 {
        storage.store(fresh_memory("x")).await.expect("store");
    }
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_memories, 3);
    assert!(stats.by_type.values().sum::<usize>() == 3);
}

#[tokio::test]
async fn transaction_commit_persists_changes() {
    let storage = InMemoryStorageStub::new();
    let mut txn = storage.begin_transaction().await.expect("txn");
    let id = txn.store(fresh_memory("in-tx")).await.expect("tx store");
    txn.commit().await.expect("commit");
    assert!(storage.exists(&id).await.expect("exists"));
    assert!(storage.transaction_count() >= 1);
}

#[tokio::test]
async fn transaction_rollback_discards_changes() {
    let storage = InMemoryStorageStub::new();
    let mut txn = storage.begin_transaction().await.expect("txn");
    let id = txn.store(fresh_memory("dropped")).await.expect("tx store");
    txn.rollback().await.expect("rollback");
    assert!(!storage.exists(&id).await.expect("exists"));
}

#[tokio::test]
async fn injected_error_surfaces_on_store() {
    let storage = InMemoryStorageStub::new();
    storage.injected_error(membrain_core::error::Error::Storage(
        "disk full".to_string(),
    ));
    let result = storage.store(fresh_memory("x")).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn concurrent_stores_preserve_count() {
    let storage = Arc::new(InMemoryStorageStub::new());
    let tasks = (0..32).map(|index| {
        let storage = Arc::clone(&storage);
        tokio::spawn(async move {
            let agent = AgentId::new();
            let provenance =
                Provenance::new_direct(Source::user_input("stress"), agent);
            let common = membrain_core::memory::MemoryCommon::new(agent, provenance)
                .with_confidence(Confidence::new(0.5));
            let memory = Memory::Semantic(SemanticMemory {
                common,
                content: SemanticContent::Fact(FactMemory::new(format!("m-{index}"))),
            });
            storage.store(memory).await
        })
    });
    let outcomes = join_all(tasks).await;
    for outcome in outcomes {
        outcome.expect("join").expect("store");
    }
    assert_eq!(storage.count(), 32);
}
