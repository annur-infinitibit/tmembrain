//! OpenAI-based memory extraction implementation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::warn;

use crate::config::ExtractionConfig;
use crate::error::{Error, Result};

use super::{ExtractedFact, ExtractionResult, MemoryExtractor};

const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a memory extraction system. Given a memory text that may contain speaker names, timestamps, and content in various formats, extract structured facts.

Rules:
1. Extract only facts, preferences, relationships, and temporal events worth remembering.
2. Skip greetings, filler words, small talk with no informational content.
3. Each extracted fact MUST be a standalone statement that includes the speaker's name when one is identifiable.
4. If the message contains a relative time reference ("last year", "yesterday"), resolve it to an absolute date using any timestamp present in the text.
5. Include the person's name as the subject in every extracted fact when a name is available.
6. For preferences, clearly state what the person likes, dislikes, or prefers.
7. Set confidence: direct statement = 0.9, strong implication = 0.7, weak inference = 0.5.

Respond with ONLY a JSON array. Each element:
{"type": "fact"|"preference"|"temporal_event"|"relationship", "content": "...", "confidence": 0.0-1.0}

If nothing worth extracting, respond with: []"#;

/// OpenAI Chat Completions-based memory extractor.
///
/// Calls the Chat Completions API to extract structured facts from raw
/// conversation messages. Supports any OpenAI-compatible endpoint via
/// the `base_url` configuration.
pub struct OpenAiMemoryExtractor {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    max_retries: u32,
    min_confidence: f64,
    max_facts_per_memory: usize,
}

impl OpenAiMemoryExtractor {
    /// Create a new memory extractor from configuration.
    ///
    /// Works with OpenAI, Ollama, and any OpenAI-compatible API. When no
    /// `api_key` is provided the `Authorization` header is skipped.
    pub fn from_config(config: &ExtractionConfig) -> Result<Self> {
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
            min_confidence: config.min_confidence,
            max_facts_per_memory: config.max_facts_per_memory,
        })
    }

    /// Call the Chat Completions API and return the raw response content.
    async fn request_extraction(&self, text: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: EXTRACTION_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: text.to_string(),
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

                    // Do not retry on client errors (4xx) except rate limits (429)
                    if status.is_client_error() && status.as_u16() != 429 {
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(format!("HTTP request failed: {}", e));
                }
            }
        }

        Err(Error::Internal(
            last_error.unwrap_or_else(|| "Unknown extraction error".to_string()),
        ))
    }

    /// Parse an LLM response string into a list of extracted facts.
    ///
    /// Handles several response formats gracefully:
    /// - A JSON array of fact objects (expected)
    /// - A JSON object with an array value (some models wrap responses)
    /// - Markdown code fences around JSON
    /// - Malformed JSON (returns empty vec)
    fn parse_facts(&self, response: &str) -> Vec<ExtractedFact> {
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

        // Try parsing as a JSON array directly
        if let Ok(facts) = serde_json::from_str::<Vec<ExtractedFact>>(json_str) {
            return self.filter_and_limit(facts);
        }

        // Try parsing as a JSON object and extract the first array value
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(obj_map) = obj.as_object() {
                for value in obj_map.values() {
                    if let Ok(facts) = serde_json::from_value::<Vec<ExtractedFact>>(value.clone()) {
                        return self.filter_and_limit(facts);
                    }
                }
            }
        }

        warn!(
            response = json_str,
            "Failed to parse extraction response as JSON, returning empty result"
        );
        Vec::new()
    }

    /// Filter facts by min_confidence and limit to max_facts_per_memory.
    fn filter_and_limit(&self, facts: Vec<ExtractedFact>) -> Vec<ExtractedFact> {
        facts
            .into_iter()
            .filter(|fact| fact.confidence >= self.min_confidence)
            .take(self.max_facts_per_memory)
            .collect()
    }
}

#[async_trait]
impl MemoryExtractor for OpenAiMemoryExtractor {
    async fn extract(&self, text: &str) -> Result<ExtractionResult> {
        let response = self.request_extraction(text).await?;
        let facts = self.parse_facts(&response);
        Ok(ExtractionResult { facts })
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }
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
    use crate::traits::ExtractedFactType;

    fn test_extractor() -> OpenAiMemoryExtractor {
        OpenAiMemoryExtractor {
            client: reqwest::Client::new(),
            api_key: Some("test-key".to_string()),
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            max_retries: 2,
            min_confidence: 0.5,
            max_facts_per_memory: 10,
        }
    }

    #[test]
    fn parse_valid_json_array() {
        let extractor = test_extractor();
        let response = r#"[
            {"type": "fact", "content": "Angela loves pizza", "confidence": 0.9},
            {"type": "temporal_event", "content": "Angela visited Italy in 2021", "confidence": 0.7}
        ]"#;

        let facts = extractor.parse_facts(response);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].fact_type, ExtractedFactType::Fact);
        assert_eq!(facts[0].content, "Angela loves pizza");
        assert_eq!(facts[1].fact_type, ExtractedFactType::TemporalEvent);
    }

    #[test]
    fn parse_empty_array() {
        let extractor = test_extractor();
        let facts = extractor.parse_facts("[]");
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_wrapped_in_object() {
        let extractor = test_extractor();
        let response = r#"{"facts": [
            {"type": "preference", "content": "Bob prefers tea over coffee", "confidence": 0.85}
        ]}"#;

        let facts = extractor.parse_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fact_type, ExtractedFactType::Preference);
    }

    #[test]
    fn parse_markdown_code_fence() {
        let extractor = test_extractor();
        let response = "```json\n[\n{\"type\": \"fact\", \"content\": \"Alice is a doctor\", \"confidence\": 0.9}\n]\n```";

        let facts = extractor.parse_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Alice is a doctor");
    }

    #[test]
    fn parse_malformed_json_returns_empty() {
        let extractor = test_extractor();
        let facts = extractor.parse_facts("this is not json");
        assert!(facts.is_empty());
    }

    #[test]
    fn filter_by_min_confidence() {
        let extractor = test_extractor();
        let response = r#"[
            {"type": "fact", "content": "High confidence fact", "confidence": 0.9},
            {"type": "fact", "content": "Low confidence fact", "confidence": 0.3}
        ]"#;

        let facts = extractor.parse_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "High confidence fact");
    }

    #[test]
    fn limit_max_facts() {
        let mut extractor = test_extractor();
        extractor.max_facts_per_memory = 2;

        let response = r#"[
            {"type": "fact", "content": "Fact 1", "confidence": 0.9},
            {"type": "fact", "content": "Fact 2", "confidence": 0.8},
            {"type": "fact", "content": "Fact 3", "confidence": 0.7}
        ]"#;

        let facts = extractor.parse_facts(response);
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn from_config_works_without_api_key() {
        let config = ExtractionConfig {
            api_key: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            ..Default::default()
        };

        let extractor =
            OpenAiMemoryExtractor::from_config(&config).expect("should succeed without api_key");
        assert!(extractor.api_key.is_none());
    }

    #[test]
    fn from_config_sets_fields() {
        let config = ExtractionConfig {
            api_key: Some("sk-test".to_string()),
            model: "gpt-4o".to_string(),
            base_url: Some("http://localhost:8080/v1".to_string()),
            min_confidence: 0.7,
            max_facts_per_memory: 5,
            ..Default::default()
        };

        let extractor = OpenAiMemoryExtractor::from_config(&config).expect("should create");
        assert_eq!(extractor.model, "gpt-4o");
        assert_eq!(extractor.base_url, "http://localhost:8080/v1");
        assert!((extractor.min_confidence - 0.7).abs() < f64::EPSILON);
        assert_eq!(extractor.max_facts_per_memory, 5);
    }
}
