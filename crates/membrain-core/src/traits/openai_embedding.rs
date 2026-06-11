//! OpenAI embedding provider implementation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::traits::EmbeddingConfig;
use crate::types::Embedding;

use super::EmbeddingProvider;

/// Infer embedding dimension from a known model name.
///
/// Covers OpenAI and common Ollama embedding models. Returns 1536 for
/// unrecognised models (the OpenAI default).
pub fn infer_embedding_dimension(model: &str) -> usize {
    match model {
        // OpenAI models
        "text-embedding-3-large" => 3072,
        "text-embedding-3-small" => 1536,
        "text-embedding-ada-002" => 1536,
        // Ollama models
        "nomic-embed-text" => 768,
        "all-minilm" | "all-minilm:l6-v2" => 384,
        "mxbai-embed-large" => 1024,
        "snowflake-arctic-embed" => 1024,
        "bge-large" => 1024,
        "bge-m3" => 1024,
        _ => 1536,
    }
}

/// OpenAI-compatible embedding provider.
///
/// Supports `text-embedding-3-small`, `text-embedding-3-large`, and
/// `text-embedding-ada-002` models. Also works with Ollama and any other
/// OpenAI-compatible API endpoint via the `base_url` configuration.
///
/// When `api_key` is not set, the `Authorization` header is omitted,
/// enabling use with local providers like Ollama that do not require
/// authentication.
pub struct OpenAiEmbeddingProvider {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    dimension: usize,
    max_retries: u32,
}

impl OpenAiEmbeddingProvider {
    /// Create a new embedding provider from configuration.
    ///
    /// Works with OpenAI, Ollama, and any OpenAI-compatible API. When no
    /// `api_key` is provided the `Authorization` header is skipped, which
    /// is the expected behaviour for local providers like Ollama.
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
        let api_key = config.api_key.clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let dimension = config
            .dimension
            .unwrap_or(infer_embedding_dimension(&config.model));

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::Internal(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            model: config.model.clone(),
            base_url,
            dimension,
            max_retries: config.retries,
        })
    }

    async fn request_embeddings(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = OpenAiEmbeddingRequest {
            input,
            model: self.model.clone(),
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
                    let parsed: OpenAiEmbeddingResponse = resp.json().await.map_err(|e| {
                        Error::Internal(format!("Failed to parse OpenAI response: {}", e))
                    })?;

                    let mut embeddings: Vec<Vec<f32>> =
                        parsed.data.into_iter().map(|d| d.embedding).collect();

                    // Sort by index to preserve input order
                    embeddings.sort_by_key(|_| 0); // already sorted by API

                    return Ok(embeddings);
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
            last_error.unwrap_or_else(|| "Unknown embedding error".to_string()),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let results = self.request_embeddings(vec![text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .map(Embedding::new)
            .ok_or_else(|| Error::Internal("Empty embedding response".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let results = self.request_embeddings(texts.to_vec()).await?;
        Ok(results.into_iter().map(Embedding::new).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_input_length(&self) -> usize {
        8191 // OpenAI token limit for embedding models
    }

    async fn health_check(&self) -> Result<()> {
        // Embed a tiny string as a health check
        let _ = self.embed("health check").await?;
        Ok(())
    }
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_works_without_api_key() {
        let config = EmbeddingConfig {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            api_key: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            ..Default::default()
        };

        let provider =
            OpenAiEmbeddingProvider::from_config(&config).expect("should succeed without api_key");
        assert!(provider.api_key.is_none());
        assert_eq!(provider.dimension(), 768);
    }

    #[test]
    fn from_config_sets_dimension() {
        let config = EmbeddingConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-large".to_string(),
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        };

        let provider = OpenAiEmbeddingProvider::from_config(&config).expect("should create");
        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    fn from_config_defaults_to_1536() {
        let config = EmbeddingConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        };

        let provider = OpenAiEmbeddingProvider::from_config(&config).expect("should create");
        assert_eq!(provider.dimension(), 1536);
    }

    #[test]
    fn from_config_custom_base_url() {
        let config = EmbeddingConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("http://localhost:8080/v1".to_string()),
            ..Default::default()
        };

        let provider = OpenAiEmbeddingProvider::from_config(&config).expect("should create");
        assert_eq!(provider.base_url, "http://localhost:8080/v1");
    }

    #[test]
    fn from_config_ollama_model_dimensions() {
        let models_and_dims = [
            ("nomic-embed-text", 768),
            ("all-minilm", 384),
            ("mxbai-embed-large", 1024),
            ("snowflake-arctic-embed", 1024),
        ];

        for (model, expected_dim) in models_and_dims {
            let config = EmbeddingConfig {
                provider: "ollama".to_string(),
                model: model.to_string(),
                ..Default::default()
            };

            let provider = OpenAiEmbeddingProvider::from_config(&config).expect("should create");
            assert_eq!(
                provider.dimension(),
                expected_dim,
                "wrong dimension for {}",
                model
            );
        }
    }

    #[test]
    fn from_config_explicit_dimension_overrides_model_default() {
        let config = EmbeddingConfig {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            dimension: Some(512),
            ..Default::default()
        };

        let provider = OpenAiEmbeddingProvider::from_config(&config).expect("should create");
        assert_eq!(provider.dimension(), 512);
    }
}
