//! OpenAI-based conflict resolution implementation.
//!
//! Uses the Chat Completions API to classify how a new memory relates to
//! existing similar memories, enabling automatic deduplication and
//! contradiction resolution.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::warn;

use membrain_core::config::ConflictResolutionConfig;
use membrain_core::error::{Error, Result};
use membrain_core::memory::Memory;
use membrain_core::types::MemoryId;

use crate::resolver::{ConflictDecision, ConflictResolutionResult, ConflictResolver};

const CONFLICT_RESOLUTION_SYSTEM_PROMPT: &str = r#"You are a memory conflict resolution system. You will receive a NEW memory and a list of EXISTING memories that are semantically similar.

Your job is to classify the relationship between the new memory and each existing memory, then decide the best action.

Rules:
1. ADD: The new memory contains genuinely new information not present in any existing memory.
2. UPDATE: The new memory augments, refines, or updates an existing memory. Provide the merged content that combines old and new information coherently.
3. DELETE: The new memory directly contradicts an existing memory (e.g., a preference changed, a fact was corrected). The old memory should be invalidated.
4. NOOP: The new memory is essentially a duplicate of an existing memory, or contains no meaningful new information.

When deciding between UPDATE and DELETE:
- Use UPDATE when the new information adds detail to or refines the existing memory without contradicting it.
- Use DELETE when the new information directly contradicts or supersedes the existing memory (e.g., "likes football" vs "likes basketball").

Respond with ONLY a JSON object:
{
  "decision": "add" | "update" | "delete" | "noop",
  "target_id": "<id of existing memory, required for update/delete>",
  "merged_content": "<merged text, required for update>",
  "reason": "<explanation, required for delete/noop>",
  "confidence": 0.0-1.0,
  "reasoning": "<brief explanation of your decision>"
}"#;

/// OpenAI-compatible Chat Completions-based conflict resolver.
///
/// Works with OpenAI, Ollama, and any OpenAI-compatible API. When no
/// `api_key` is provided the `Authorization` header is skipped.
pub struct OpenAiConflictResolver {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    max_retries: u32,
}

impl OpenAiConflictResolver {
    /// Create a new conflict resolver from configuration.
    pub fn from_config(config: &ConflictResolutionConfig) -> Result<Self> {
        let api_key = config.api_key.clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::Internal(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            model: config.model.clone(),
            base_url,
            max_retries: config.retries,
        })
    }

    /// Call the Chat Completions API with the conflict resolution prompt.
    async fn request_resolution(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: CONFLICT_RESOLUTION_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            temperature: 0.0,
        };

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let backoff = Duration::from_millis(100 * 2u64.saturating_pow(attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let mut request = self.client.post(&url).json(&body);
            if let Some(ref key) = self.api_key {
                request = request.header("Authorization", format!("Bearer {}", key));
            }
            let response = request.send().await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let parsed: ChatCompletionResponse = resp.json().await.map_err(|e| {
                        Error::Internal(format!("Failed to parse chat completion response: {}", e))
                    })?;

                    if let Some(choice) = parsed.choices.into_iter().next() {
                        return Ok(choice.message.content);
                    }
                    return Err(Error::Internal(
                        "Empty choices in chat completion response".to_string(),
                    ));
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    last_error = Some(format!("OpenAI API error {}: {}", status, body_text));

                    if status.is_client_error() && status.as_u16() != 429 {
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(format!("HTTP request failed: {}", e));
                }
            }
        }

        Err(Error::Internal(last_error.unwrap_or_else(|| {
            "Unknown conflict resolution error".to_string()
        })))
    }

    /// Build the user prompt presenting the new memory and existing memories.
    fn build_prompt(new_memory: &Memory, similar_memories: &[Memory]) -> String {
        let new_text = memory_to_text(new_memory);

        let mut prompt = format!("NEW MEMORY:\n{}\n\nEXISTING MEMORIES:\n", new_text);

        for (index, existing) in similar_memories.iter().enumerate() {
            let existing_text = memory_to_text(existing);
            prompt.push_str(&format!(
                "---\nMemory {} (ID: {}):\n{}\n",
                index + 1,
                existing.id(),
                existing_text,
            ));
        }

        if similar_memories.is_empty() {
            prompt.push_str("(none)\n");
        }

        prompt
    }

    /// Parse the LLM response JSON into a conflict resolution result.
    fn parse_response(
        &self,
        response: &str,
        similar_memories: &[Memory],
    ) -> Result<ConflictResolutionResult> {
        let trimmed = response.trim();

        // Strip markdown code fences if present
        let json_str = if trimmed.starts_with("```") {
            let without_prefix = trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```");
            without_prefix.trim_end_matches("```").trim()
        } else {
            trimmed
        };

        let raw: RawResolutionResponse = serde_json::from_str(json_str).map_err(|e| {
            warn!(response = json_str, error = %e, "Failed to parse conflict resolution response");
            Error::Internal(format!(
                "Failed to parse conflict resolution response: {}",
                e
            ))
        })?;

        let decision = match raw.decision.as_str() {
            "add" => ConflictDecision::Add,
            "update" => {
                let target_id = self.resolve_target_id(&raw.target_id, similar_memories)?;
                let merged_content = raw.merged_content.unwrap_or_default();
                ConflictDecision::Update {
                    target_id,
                    merged_content,
                }
            }
            "delete" => {
                let target_id = self.resolve_target_id(&raw.target_id, similar_memories)?;
                let reason = raw
                    .reason
                    .unwrap_or_else(|| "Contradicted by new information".to_string());
                ConflictDecision::Delete { target_id, reason }
            }
            "noop" => {
                let reason = raw.reason.unwrap_or_else(|| "Already known".to_string());
                ConflictDecision::Noop { reason }
            }
            other => {
                warn!(
                    decision = other,
                    "Unknown conflict decision, defaulting to ADD"
                );
                ConflictDecision::Add
            }
        };

        Ok(ConflictResolutionResult {
            decision,
            confidence: raw.confidence.unwrap_or(0.5),
            reasoning: raw.reasoning.unwrap_or_default(),
        })
    }

    /// Resolve the target_id from the LLM response.
    ///
    /// The LLM may return the full UUID string or a 1-based index like "Memory 1".
    /// We try UUID parsing first, then fall back to matching against known IDs.
    fn resolve_target_id(
        &self,
        raw_id: &Option<String>,
        similar_memories: &[Memory],
    ) -> Result<MemoryId> {
        let id_str = raw_id.as_deref().ok_or_else(|| {
            Error::Internal("Conflict resolution response missing target_id".to_string())
        })?;

        // Try parsing as a MemoryId (UUID) directly
        if let Ok(parsed) = id_str.parse::<uuid::Uuid>() {
            return Ok(MemoryId::from_uuid(parsed));
        }

        // Try matching against known similar memory IDs (the LLM may return
        // the ID string we showed it, which is the Display form)
        for memory in similar_memories {
            if memory.id().to_string() == id_str {
                return Ok(*memory.id());
            }
        }

        // Fall back: if there's exactly one similar memory, use it
        if similar_memories.len() == 1 {
            return Ok(*similar_memories[0].id());
        }

        Err(Error::Internal(format!(
            "Could not resolve target_id '{}' from conflict resolution response",
            id_str
        )))
    }
}

#[async_trait]
impl ConflictResolver for OpenAiConflictResolver {
    async fn resolve(
        &self,
        new_memory: &Memory,
        similar_memories: &[Memory],
    ) -> Result<ConflictResolutionResult> {
        // If no similar memories, it's definitely new
        if similar_memories.is_empty() {
            return Ok(ConflictResolutionResult {
                decision: ConflictDecision::Add,
                confidence: 1.0,
                reasoning: "No similar existing memories found".to_string(),
            });
        }

        let prompt = Self::build_prompt(new_memory, similar_memories);
        let response = self.request_resolution(&prompt).await?;
        self.parse_response(&response, similar_memories)
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Extract human-readable text from a memory for the LLM prompt.
fn memory_to_text(memory: &Memory) -> String {
    let common = memory.common();
    let memory_type = memory.memory_type();
    let content = memory.text_content();
    let confidence = common.confidence.value();

    let mut text = format!(
        "Type: {:?}\nConfidence: {:.2}\nContent: {}",
        memory_type, confidence, content
    );

    if !common.tags.is_empty() {
        text.push_str(&format!("\nTags: {}", common.tags.join(", ")));
    }

    text
}

/// Raw JSON response from the LLM (before validation).
#[derive(Debug, Deserialize)]
struct RawResolutionResponse {
    decision: String,
    target_id: Option<String>,
    merged_content: Option<String>,
    reason: Option<String>,
    confidence: Option<f64>,
    reasoning: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{AgentId, Confidence, Provenance, Source};

    fn make_fact_memory(statement: &str) -> Memory {
        let agent_id = AgentId::new();
        let common = MemoryCommon::new(
            agent_id,
            Provenance::new_direct(
                Source::AgentGenerated {
                    process: "test".to_string(),
                    context: None,
                },
                agent_id,
            ),
        )
        .with_confidence(Confidence::new(0.9));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory {
                statement: statement.to_string(),
                subject: None,
                predicate: None,
                object: None,
            }),
        })
    }

    fn test_resolver() -> OpenAiConflictResolver {
        OpenAiConflictResolver {
            client: reqwest::Client::new(),
            api_key: Some("test-key".to_string()),
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            max_retries: 2,
        }
    }

    #[test]
    fn parse_add_response() {
        let resolver = test_resolver();
        let response = r#"{
            "decision": "add",
            "confidence": 0.95,
            "reasoning": "This is genuinely new information"
        }"#;

        let result = resolver
            .parse_response(response, &[])
            .expect("should parse");
        assert!(matches!(result.decision, ConflictDecision::Add));
        assert!((result.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_delete_response() {
        let resolver = test_resolver();
        let existing = make_fact_memory("David likes football");
        let existing_id = existing.id().to_string();

        let response = format!(
            r#"{{
                "decision": "delete",
                "target_id": "{}",
                "reason": "Preference changed from football to basketball",
                "confidence": 0.9,
                "reasoning": "The new memory contradicts the existing preference"
            }}"#,
            existing_id
        );

        let result = resolver
            .parse_response(&response, &[existing])
            .expect("should parse");

        if let ConflictDecision::Delete { reason, .. } = &result.decision {
            assert!(reason.contains("football"));
        } else {
            panic!("Expected Delete decision");
        }
    }

    #[test]
    fn parse_update_response() {
        let resolver = test_resolver();
        let existing = make_fact_memory("David likes sports");
        let existing_id = existing.id().to_string();

        let response = format!(
            r#"{{
                "decision": "update",
                "target_id": "{}",
                "merged_content": "David likes sports, especially basketball",
                "confidence": 0.85,
                "reasoning": "The new memory adds detail to the existing one"
            }}"#,
            existing_id
        );

        let result = resolver
            .parse_response(&response, &[existing])
            .expect("should parse");

        if let ConflictDecision::Update { merged_content, .. } = &result.decision {
            assert!(merged_content.contains("basketball"));
        } else {
            panic!("Expected Update decision");
        }
    }

    #[test]
    fn parse_noop_response() {
        let resolver = test_resolver();
        let response = r#"{
            "decision": "noop",
            "reason": "This fact is already stored",
            "confidence": 0.92,
            "reasoning": "Duplicate of existing memory"
        }"#;

        let result = resolver
            .parse_response(response, &[])
            .expect("should parse");
        assert!(matches!(result.decision, ConflictDecision::Noop { .. }));
    }

    #[test]
    fn parse_markdown_wrapped_response() {
        let resolver = test_resolver();
        let response =
            "```json\n{\"decision\": \"add\", \"confidence\": 0.8, \"reasoning\": \"new\"}\n```";

        let result = resolver
            .parse_response(response, &[])
            .expect("should parse");
        assert!(matches!(result.decision, ConflictDecision::Add));
    }

    #[test]
    fn parse_unknown_decision_defaults_to_add() {
        let resolver = test_resolver();
        let response = r#"{"decision": "unknown_thing", "confidence": 0.5}"#;

        let result = resolver
            .parse_response(response, &[])
            .expect("should parse");
        assert!(matches!(result.decision, ConflictDecision::Add));
    }

    #[test]
    fn resolve_target_id_single_memory_fallback() {
        let resolver = test_resolver();
        let existing = make_fact_memory("Some fact");

        // Even with a bad target_id, if there's only one similar memory, use it
        let result =
            resolver.resolve_target_id(&Some("bad-id".to_string()), std::slice::from_ref(&existing));
        assert!(result.is_ok());
        assert_eq!(result.expect("should resolve"), *existing.id());
    }

    #[test]
    fn build_prompt_formats_correctly() {
        let new_memory = make_fact_memory("David likes basketball");
        let existing = make_fact_memory("David likes football");

        let prompt = OpenAiConflictResolver::build_prompt(&new_memory, &[existing]);
        assert!(prompt.contains("NEW MEMORY:"));
        assert!(prompt.contains("EXISTING MEMORIES:"));
        assert!(prompt.contains("basketball"));
        assert!(prompt.contains("football"));
        assert!(prompt.contains("Memory 1"));
    }

    #[test]
    fn empty_similar_memories_prompt() {
        let new_memory = make_fact_memory("David likes basketball");
        let prompt = OpenAiConflictResolver::build_prompt(&new_memory, &[]);
        assert!(prompt.contains("(none)"));
    }
}
