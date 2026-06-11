//! JSON payload builders for LLM integration tests.
//!
//! Not a mock framework — these are plain string helpers that produce the
//! exact body bytes the `ScriptedOpenAiServer` should respond with. Keeps
//! per-test boilerplate to one line.

use serde_json::json;

use membrain_core::traits::ExtractedFact;
use membrain_core::types::MemoryId;

/// OpenAI `chat/completions` response wrapping `content`.
pub fn chat_completion_response_json(content: &str) -> String {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })
    .to_string()
}

/// Conflict resolution decision JSON: ADD.
pub fn conflict_add_json(confidence: f64, reasoning: &str) -> String {
    json!({
        "decision": "add",
        "confidence": confidence,
        "reasoning": reasoning
    })
    .to_string()
}

/// Conflict resolution decision JSON: UPDATE.
pub fn conflict_update_json(
    target_id: MemoryId,
    merged: &str,
    confidence: f64,
    reasoning: &str,
) -> String {
    json!({
        "decision": "update",
        "target_id": target_id.to_string(),
        "merged_content": merged,
        "confidence": confidence,
        "reasoning": reasoning
    })
    .to_string()
}

/// Conflict resolution decision JSON: DELETE.
pub fn conflict_delete_json(
    target_id: MemoryId,
    reason: &str,
    confidence: f64,
    reasoning: &str,
) -> String {
    json!({
        "decision": "delete",
        "target_id": target_id.to_string(),
        "reason": reason,
        "confidence": confidence,
        "reasoning": reasoning
    })
    .to_string()
}

/// Conflict resolution decision JSON: NOOP.
pub fn conflict_noop_json(reason: &str, confidence: f64, reasoning: &str) -> String {
    json!({
        "decision": "noop",
        "reason": reason,
        "confidence": confidence,
        "reasoning": reasoning
    })
    .to_string()
}

/// Extraction response JSON: a list of extracted facts.
pub fn extraction_response_json(facts: &[ExtractedFact]) -> String {
    serde_json::to_string(facts).unwrap_or_else(|_| "[]".to_string())
}
