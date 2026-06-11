//! End-to-end integration tests for `OpenAiConflictResolver` using a
//! scripted local HTTP server. No network access required.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::time::Duration;

use membrain_conflict::openai_resolver::OpenAiConflictResolver;
use membrain_conflict::resolver::{ConflictDecision, ConflictResolver};
use membrain_core::config::ConflictResolutionConfig;
use membrain_core::memory::Memory;
use membrain_test_utils::llm_fixtures::{
    chat_completion_response_json, conflict_add_json, conflict_delete_json, conflict_noop_json,
    conflict_update_json,
};
use membrain_test_utils::openai_mock_server::{ScriptedOpenAiServer, ScriptedResponse};
use membrain_test_utils::semantic_fact;

fn make_config(base_url: String, retries: u32, timeout_secs: u64) -> ConflictResolutionConfig {
    ConflictResolutionConfig {
        enabled: true,
        provider: "openai".to_string(),
        model: "gpt-test".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: Some(base_url),
        timeout_secs,
        retries,
        max_similar_to_compare: 3,
    }
}

async fn start_server() -> ScriptedOpenAiServer {
    ScriptedOpenAiServer::start()
        .await
        .expect("start scripted server")
}

fn new_memory() -> Memory {
    semantic_fact("David likes basketball")
}

fn existing_memory() -> Memory {
    semantic_fact("David likes football")
}

#[tokio::test]
async fn resolve_add_decision() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.95, "Genuinely new info"),
    )));

    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");

    let decision = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Add));
    assert!((decision.confidence - 0.95).abs() < 1e-6);

    server.shutdown().await;
}

#[tokio::test]
async fn resolve_update_decision() {
    let server = start_server().await;
    let existing = existing_memory();
    let target_id = *existing.id();
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_update_json(target_id, "David likes both football and basketball", 0.9, "merge"),
    )));

    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing])
        .await
        .expect("resolve");
    match decision.decision {
        ConflictDecision::Update {
            target_id: returned,
            merged_content,
        } => {
            assert_eq!(returned, target_id);
            assert!(merged_content.contains("basketball"));
        }
        other => panic!("expected Update, got {other:?}"),
    }

    server.shutdown().await;
}

#[tokio::test]
async fn resolve_delete_decision() {
    let server = start_server().await;
    let existing = existing_memory();
    let target_id = *existing.id();
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_delete_json(target_id, "Contradicts new preference", 0.9, "contradiction"),
    )));

    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing])
        .await
        .expect("resolve");
    match decision.decision {
        ConflictDecision::Delete {
            target_id: returned,
            reason,
        } => {
            assert_eq!(returned, target_id);
            assert!(reason.contains("preference"));
        }
        other => panic!("expected Delete, got {other:?}"),
    }

    server.shutdown().await;
}

#[tokio::test]
async fn resolve_noop_decision() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_noop_json("already known", 0.9, "duplicate"),
    )));

    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Noop { .. }));

    server.shutdown().await;
}

#[tokio::test]
async fn resolve_rejects_malformed_response_body() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(
        "{ this is not valid JSON ",
    ));
    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let result = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await;
    assert!(result.is_err());
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_retries_5xx_then_succeeds() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::status(500, "boom"));
    server.queue_response(ScriptedResponse::status(502, "boom"));
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.8, "ok"),
    )));

    let config = make_config(server.chat_completions_base_url(), 3, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Add));
    assert!(server.captured_requests().len() >= 3);

    server.shutdown().await;
}

#[tokio::test]
async fn resolve_fails_after_retry_exhaustion() {
    let server = start_server().await;
    for _ in 0..4 {
        server.queue_response(ScriptedResponse::status(500, "boom"));
    }
    let config = make_config(server.chat_completions_base_url(), 2, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let result = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await;
    assert!(result.is_err());
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_stops_on_4xx_except_429() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::status(401, "unauthorized"));
    let config = make_config(server.chat_completions_base_url(), 3, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let result = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await;
    assert!(result.is_err());
    // Only one request — retries must not fire for non-429 4xx.
    assert_eq!(server.captured_requests().len(), 1);
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_retries_429_then_succeeds() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::status(429, "slow down"));
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.7, "recovered"),
    )));
    let config = make_config(server.chat_completions_base_url(), 2, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Add));
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_timeout_surfaces_error() {
    let server = start_server().await;
    server.queue_response(
        ScriptedResponse::ok_json(chat_completion_response_json(&conflict_add_json(
            0.5,
            "slow",
        )))
        .with_delay(Duration::from_secs(3)),
    );
    let mut config = make_config(server.chat_completions_base_url(), 0, 1);
    config.retries = 0;
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let result = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await;
    assert!(result.is_err());
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_sends_authorization_header_when_key_present() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.9, "ok"),
    )));
    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");

    let captured = server.last_request().expect("request captured");
    let header = captured
        .header("authorization")
        .expect("authorization header");
    assert!(header.starts_with("Bearer sk-test"));
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_omits_authorization_when_key_absent() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.9, "ok"),
    )));
    let mut config = make_config(server.chat_completions_base_url(), 0, 5);
    config.api_key = None;
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");

    let captured = server.last_request().expect("captured");
    assert!(captured.header("authorization").is_none());
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_uses_configured_model_name_in_request() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.9, "ok"),
    )));
    let mut config = make_config(server.chat_completions_base_url(), 0, 5);
    config.model = "gpt-custom".to_string();
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");

    let captured = server.last_request().expect("captured");
    let body = captured.body_string();
    assert!(body.contains("gpt-custom"));
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_prompt_includes_memories() {
    let server = start_server().await;
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &conflict_add_json(0.9, "ok"),
    )));
    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    let captured = server.last_request().expect("captured");
    let body = captured.body_string();
    assert!(body.contains("basketball"));
    assert!(body.contains("football"));
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_empty_similar_memories_bypasses_llm() {
    let server = start_server().await;
    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Add));
    assert_eq!(server.captured_requests().len(), 0);
    server.shutdown().await;
}

#[tokio::test]
async fn resolve_strips_markdown_fences() {
    let server = start_server().await;
    let wrapped = format!(
        "```json\n{}\n```",
        conflict_add_json(0.85, "fenced")
    );
    server.queue_response(ScriptedResponse::ok_json(chat_completion_response_json(
        &wrapped,
    )));
    let config = make_config(server.chat_completions_base_url(), 0, 5);
    let resolver = OpenAiConflictResolver::from_config(&config).expect("resolver");
    let decision = resolver
        .resolve(&new_memory(), &[existing_memory()])
        .await
        .expect("resolve");
    assert!(matches!(decision.decision, ConflictDecision::Add));
    server.shutdown().await;
}
