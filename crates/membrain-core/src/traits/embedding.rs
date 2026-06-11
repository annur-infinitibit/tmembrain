//! Embedding provider trait for generating vector embeddings

use async_trait::async_trait;

use crate::error::Result;
use crate::types::Embedding;

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Embedding>;

    /// Generate embeddings for multiple texts
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>>;

    /// Get the embedding dimension for this provider
    fn dimension(&self) -> usize;

    /// Get the provider name
    fn name(&self) -> &str;

    /// Get the model identifier
    fn model(&self) -> &str;

    /// Maximum tokens/characters this provider can handle
    fn max_input_length(&self) -> usize;

    /// Check if the provider is available
    async fn health_check(&self) -> Result<()>;
}

/// A no-op embedding provider for testing or when embeddings are disabled
pub struct NoOpEmbeddingProvider {
    dimension: usize,
}

impl NoOpEmbeddingProvider {
    /// Create a new no-op provider with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for NoOpEmbeddingProvider {
    fn default() -> Self {
        Self::new(Embedding::OPENAI_ADA_002)
    }
}

#[async_trait]
impl EmbeddingProvider for NoOpEmbeddingProvider {
    async fn embed(&self, _text: &str) -> Result<Embedding> {
        // Return zero vector
        Ok(Embedding::zeros(self.dimension))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|_| Embedding::zeros(self.dimension))
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "no-op"
    }

    fn model(&self) -> &str {
        "none"
    }

    fn max_input_length(&self) -> usize {
        usize::MAX
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// Configuration for embedding providers
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Provider name (openai, cohere, local, etc.)
    pub provider: String,
    /// Model identifier
    pub model: String,
    /// API key (if needed)
    pub api_key: Option<String>,
    /// API base URL (for custom endpoints)
    pub base_url: Option<String>,
    /// Explicit embedding dimension override. When set, this takes precedence
    /// over the dimension inferred from the model name. Required when using
    /// custom or local embedding models (e.g. sentence-transformers).
    pub dimension: Option<usize>,
    /// Maximum batch size
    pub batch_size: usize,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Number of retries
    pub retries: u32,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "text-embedding-ada-002".to_string(),
            api_key: None,
            base_url: None,
            dimension: None,
            batch_size: 100,
            timeout_secs: 30,
            retries: 3,
        }
    }
}

impl EmbeddingConfig {
    /// Create config for OpenAI
    pub fn openai(model: impl Into<String>) -> Self {
        Self {
            provider: "openai".to_string(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create config for Cohere
    pub fn cohere(model: impl Into<String>) -> Self {
        Self {
            provider: "cohere".to_string(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create config for local/self-hosted model
    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: "local".to_string(),
            model: model.into(),
            base_url: Some(base_url.into()),
            api_key: None,
            ..Default::default()
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_provider_returns_zeros() {
        let provider = NoOpEmbeddingProvider::new(384);
        let embedding = provider.embed("test").await.unwrap();

        assert_eq!(embedding.dimension(), 384);
        assert!(embedding.values().iter().all(|&v| v == 0.0));
    }

    #[tokio::test]
    async fn noop_provider_batch() {
        let provider = NoOpEmbeddingProvider::default();
        let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let embeddings = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.dimension(), Embedding::OPENAI_ADA_002);
        }
    }

    #[test]
    fn embedding_config_builders() {
        let openai = EmbeddingConfig::openai("text-embedding-3-small")
            .with_api_key("sk-test")
            .with_batch_size(50);

        assert_eq!(openai.provider, "openai");
        assert_eq!(openai.batch_size, 50);
        assert!(openai.api_key.is_some());

        let local = EmbeddingConfig::local("http://localhost:8080", "bge-base");
        assert_eq!(local.provider, "local");
        assert!(local.base_url.is_some());
    }
}
