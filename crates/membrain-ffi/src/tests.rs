use crate::MembrainClient;
use membrain_core::config::{Config, StorageConfig};

async fn create_test_client() -> MembrainClient {
    let mut config = Config::default();
    config.storage = StorageConfig::memory();
    MembrainClient::with_config(config).await.unwrap()
}

#[tokio::test]
async fn test_client_creation() {
    let client = create_test_client().await;
    assert_eq!(client.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_store_and_search() {
    let client = create_test_client().await;

    let result = client
        .store_fact("The user prefers dark mode", 0.9)
        .await
        .unwrap();
    assert!(result.success);

    let results = client.search("dark mode", 10).await.unwrap();
    assert!(!results.memories.is_empty());
}

#[tokio::test]
async fn test_store_preference() {
    let client = create_test_client().await;

    let result = client
        .store_preference("user", "theme", "dark mode", "strong")
        .await
        .unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_get_and_delete() {
    let client = create_test_client().await;

    let result = client.store_fact("Test fact", 0.8).await.unwrap();
    assert!(result.success);
    let id = result.id.unwrap();

    let memory = client.get(&id).await.unwrap();
    assert!(memory.is_some());

    let deleted = client.delete(&id).await.unwrap();
    assert!(deleted);

    let memory = client.get(&id).await.unwrap();
    assert!(memory.is_none());
}

#[tokio::test]
async fn test_store_event() {
    let client = create_test_client().await;
    let result = client.store_event("meeting", "Team standup meeting").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_observation() {
    let client = create_test_client().await;
    let result = client.store_observation("User tends to ask about Rust").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_concept() {
    let client = create_test_client().await;
    let result = client.store_concept("FFI", "Foreign Function Interface").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_entity() {
    let client = create_test_client().await;
    let result = client.store_entity("Rust", "technology").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_workflow() {
    let client = create_test_client().await;
    let result = client.store_workflow("deploy", "Deploy to production").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_skill() {
    let client = create_test_client().await;
    let result = client.store_skill("code_review", "Review pull requests").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_pattern() {
    let client = create_test_client().await;
    let result = client.store_pattern("morning_standup", "Daily standup pattern", "temporal").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_goal() {
    let client = create_test_client().await;
    let result = client.store_goal("Complete the FFI bindings").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_task() {
    let client = create_test_client().await;
    let result = client.store_task("Write unit tests").await.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_store_case() {
    let client = create_test_client().await;
    let result = client
        .store_case(
            "How to deploy a service",
            "Build image, push, apply manifest",
            "Service deployed successfully",
            1.0,
        )
        .await
        .unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_stats() {
    let client = create_test_client().await;
    client.store_fact("Test fact", 0.9).await.unwrap();
    let stats = client.stats().await.unwrap();
    assert!(stats.total_memories > 0);
}

#[tokio::test]
async fn test_search_uses_intent_detection() {
    let client = create_test_client().await;
    client.store_fact("The capital of France is Paris", 0.9).await.unwrap();

    // search() should use the full adaptive pipeline (intent detection ON)
    let results = client.search("What is the capital of France?", 10).await.unwrap();
    // The pipeline should not gate a factual question
    assert!(!results.was_gated);
}

#[tokio::test]
async fn test_search_hello_is_gated() {
    let client = create_test_client().await;
    client.store_fact("Some important fact", 0.9).await.unwrap();

    // "hello" should be gated as a greeting
    let results = client.search("hello", 10).await.unwrap();
    assert!(results.was_gated);
    assert!(results.memories.is_empty());
}

#[tokio::test]
async fn test_search_with_filters_bypasses_gating() {
    let client = create_test_client().await;
    client.store_fact("Some important fact", 0.9).await.unwrap();

    // search_with_filters() disables gating even for greetings
    let results = client.search_with_filters("hello", 10, None).await.unwrap();
    assert!(!results.was_gated);
}

#[tokio::test]
async fn test_search_result_has_created_at() {
    let client = create_test_client().await;
    client.store_fact("User prefers dark mode", 0.9).await.unwrap();

    let results = client.search("dark mode", 10).await.unwrap();
    assert!(!results.memories.is_empty());

    for memory in &results.memories {
        assert!(!memory.created_at.is_empty(), "created_at should be populated");
        // Should look like an RFC 3339 timestamp (contains 'T' separator and '+' or 'Z')
        assert!(
            memory.created_at.contains('T'),
            "created_at should be RFC 3339 format: {}",
            memory.created_at,
        );
    }
}

#[tokio::test]
async fn test_event_findable_by_fact_query() {
    let client = create_test_client().await;

    // Store as an episodic event
    let result = client.store_event("meeting", "Alice met Bob for coffee last Tuesday").await.unwrap();
    assert!(result.success);

    // Search with a fact-style query -- should still find the event
    let results = client.search("Who did Alice meet?", 10).await.unwrap();
    assert!(
        !results.memories.is_empty(),
        "Event memories should be findable by fact-style queries"
    );
}
