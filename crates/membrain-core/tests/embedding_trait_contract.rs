//! Contract tests for `EmbeddingProvider` via `DeterministicEmbeddingProvider`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::collections::HashMap;

use membrain_core::traits::EmbeddingProvider;
use membrain_test_utils::DeterministicEmbeddingProvider;

#[tokio::test]
async fn embed_returns_fixed_dimension() {
    let provider = DeterministicEmbeddingProvider::new(384);
    let embedding = provider.embed("hello").await.expect("embed");
    assert_eq!(embedding.dimension(), 384);
}

#[tokio::test]
async fn embed_is_stable_for_same_input() {
    let provider = DeterministicEmbeddingProvider::new(128);
    let first = provider.embed("stable").await.expect("first");
    let second = provider.embed("stable").await.expect("second");
    assert_eq!(first.values(), second.values());
}

#[tokio::test]
async fn embed_batch_preserves_order_and_dim() {
    let provider = DeterministicEmbeddingProvider::new(64);
    let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let embeddings = provider.embed_batch(&texts).await.expect("batch");
    assert_eq!(embeddings.len(), 3);
    for embedding in &embeddings {
        assert_eq!(embedding.dimension(), 64);
    }
    // Order: embedding[0] should differ from embedding[1] for distinct inputs.
    assert_ne!(embeddings[0].values(), embeddings[1].values());
}

#[tokio::test]
async fn embed_call_count_tracks_invocations() {
    let provider = DeterministicEmbeddingProvider::new(64);
    provider.embed("a").await.expect("embed");
    provider.embed("b").await.expect("embed");
    provider
        .embed_batch(&["c".to_string(), "d".to_string()])
        .await
        .expect("batch");
    assert_eq!(provider.embed_call_count(), 4);
}

#[tokio::test]
async fn fixed_vector_mode_returns_same_values() {
    let fixed = vec![0.25_f32; 8];
    let provider = DeterministicEmbeddingProvider::with_fixed(8, fixed.clone());
    let embedding = provider.embed("anything").await.expect("embed");
    assert_eq!(embedding.values(), fixed.as_slice());
}

#[tokio::test]
async fn mapping_mode_falls_back_to_hash() {
    let mut mapping = HashMap::new();
    mapping.insert("known".to_string(), vec![1.0_f32; 4]);
    let provider = DeterministicEmbeddingProvider::with_mapping(4, mapping);

    let known = provider.embed("known").await.expect("known");
    let unknown = provider.embed("unknown").await.expect("unknown");
    assert_eq!(known.values(), &[1.0_f32, 1.0, 1.0, 1.0]);
    assert_ne!(unknown.values(), known.values());
}

#[tokio::test]
async fn injected_failure_surfaces_then_recovers() {
    let provider = DeterministicEmbeddingProvider::new(16);
    provider.fail_next();
    assert!(provider.embed("a").await.is_err());
    assert!(provider.embed("a").await.is_ok());
}

#[tokio::test]
async fn health_check_counts_increments() {
    let provider = DeterministicEmbeddingProvider::new(16);
    provider.health_check().await.expect("health");
    provider.health_check().await.expect("health");
    assert_eq!(provider.health_check_call_count(), 2);
}
