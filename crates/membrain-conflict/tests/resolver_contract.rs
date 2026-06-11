//! Trait-object contract tests for `ConflictResolver` via `FakeConflictResolver`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::sync::Arc;

use membrain_conflict::resolver::{ConflictDecision, ConflictResolutionResult, ConflictResolver};
use membrain_test_utils::{semantic_fact, FakeConflictResolver};

#[tokio::test]
async fn fake_always_add_returns_add() {
    let resolver = FakeConflictResolver::always_add();
    let memory = semantic_fact("something new");
    let result = resolver.resolve(&memory, &[]).await.expect("resolve");
    assert!(matches!(result.decision, ConflictDecision::Add));
    assert_eq!(resolver.call_count(), 1);
}

#[tokio::test]
async fn fake_always_update_returns_configured_target() {
    let existing = semantic_fact("existing");
    let resolver = FakeConflictResolver::always_update(*existing.id(), "merged");
    let new_memory = semantic_fact("new");
    let result = resolver
        .resolve(&new_memory, &[existing.clone()])
        .await
        .expect("resolve");
    match result.decision {
        ConflictDecision::Update {
            target_id,
            merged_content,
        } => {
            assert_eq!(target_id, *existing.id());
            assert_eq!(merged_content, "merged");
        }
        other => panic!("expected Update, got {other:?}"),
    }
}

#[tokio::test]
async fn fake_always_delete_returns_delete() {
    let existing = semantic_fact("existing");
    let resolver = FakeConflictResolver::always_delete(*existing.id(), "bad");
    let result = resolver
        .resolve(&semantic_fact("new"), &[existing.clone()])
        .await
        .expect("resolve");
    assert!(matches!(result.decision, ConflictDecision::Delete { .. }));
}

#[tokio::test]
async fn fake_always_noop_returns_noop() {
    let resolver = FakeConflictResolver::always_noop("dup");
    let result = resolver
        .resolve(&semantic_fact("new"), &[])
        .await
        .expect("resolve");
    assert!(matches!(result.decision, ConflictDecision::Noop { .. }));
}

#[tokio::test]
async fn fake_scripted_consumes_in_order() {
    let resolver = FakeConflictResolver::scripted(vec![
        ConflictResolutionResult {
            decision: ConflictDecision::Add,
            confidence: 0.9,
            reasoning: "first".to_string(),
        },
        ConflictResolutionResult {
            decision: ConflictDecision::Noop {
                reason: "duplicate".to_string(),
            },
            confidence: 0.7,
            reasoning: "second".to_string(),
        },
    ]);

    let memory = semantic_fact("x");
    let first = resolver.resolve(&memory, &[]).await.expect("resolve 1");
    let second = resolver.resolve(&memory, &[]).await.expect("resolve 2");
    assert!(matches!(first.decision, ConflictDecision::Add));
    assert!(matches!(second.decision, ConflictDecision::Noop { .. }));
    assert!(resolver.resolve(&memory, &[]).await.is_err());
}

#[tokio::test]
async fn fake_reports_name_and_model() {
    let resolver = FakeConflictResolver::always_add();
    assert_eq!(resolver.name(), "fake-resolver");
    assert_eq!(resolver.model(), "fake-model");
}

#[tokio::test]
async fn fake_is_trait_object_safe() {
    let trait_object: Arc<dyn ConflictResolver> = Arc::new(FakeConflictResolver::always_add());
    let memory = semantic_fact("x");
    let result = trait_object.resolve(&memory, &[]).await.expect("resolve");
    assert!(matches!(result.decision, ConflictDecision::Add));
}
