//! Configuration management for Membrain

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::memory::MemoryType;
use crate::traits::EmbeddingConfig;

/// Configuration for the Membrain memory system.
///
/// # Examples
///
/// ```
/// use membrain_core::config::Config;
///
/// let config = Config::default();
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Storage configuration
    pub storage: StorageConfig,
    /// Embedding configuration
    pub embedding: EmbeddingConfig,
    /// Write pipeline configuration
    pub write: WriteConfig,
    /// Retrieval pipeline configuration
    pub retrieval: RetrievalConfig,
    /// Jobs configuration
    pub jobs: JobsConfig,
    /// Audit configuration
    pub audit: AuditConfig,
    /// Multi-agent configuration
    pub multi_agent: MultiAgentConfig,
    /// Scope configuration (default metadata injected on write / applied on read)
    pub scope: ScopeConfig,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a TOML string
    pub fn from_toml(content: &str) -> Result<Self> {
        let config: Config = toml::from_str(content)?;
        config.validate()?;
        Ok(config)
    }

    /// Create configuration from environment variables and optional file
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        // Override with environment variables
        if let Ok(path) = std::env::var("MEMBRAIN_STORAGE_PATH") {
            config.storage.path = Some(path);
        }

        if let Ok(backend) = std::env::var("MEMBRAIN_STORAGE_BACKEND") {
            config.storage.backend = backend;
        }

        if let Ok(key) = std::env::var("MEMBRAIN_EMBEDDING_API_KEY") {
            config.embedding.api_key = Some(key);
        }

        if let Ok(provider) = std::env::var("MEMBRAIN_EMBEDDING_PROVIDER") {
            config.embedding.provider = provider;
        }

        if let Ok(model) = std::env::var("MEMBRAIN_EMBEDDING_MODEL") {
            config.embedding.model = model;
        }

        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate storage backend
        match self.storage.backend.as_str() {
            "memscaledb" | "sqlite" | "memory" | "postgres" => {}
            other => {
                return Err(Error::InvalidConfigValue {
                    key: "storage.backend".to_string(),
                    message: format!("Unknown backend: {}", other),
                })
            }
        }

        // Validate thresholds
        if self.write.salience.threshold < 0.0 || self.write.salience.threshold > 1.0 {
            return Err(Error::InvalidConfigValue {
                key: "write.salience.threshold".to_string(),
                message: "Must be between 0.0 and 1.0".to_string(),
            });
        }

        if self.write.novelty.threshold < 0.0 || self.write.novelty.threshold > 1.0 {
            return Err(Error::InvalidConfigValue {
                key: "write.novelty.threshold".to_string(),
                message: "Must be between 0.0 and 1.0".to_string(),
            });
        }

        let mut seen = std::collections::HashSet::new();
        for key in &self.storage.indexed_metadata_keys {
            if key.is_empty() {
                return Err(Error::InvalidConfigValue {
                    key: "storage.indexed_metadata_keys".to_string(),
                    message: "Indexed metadata key must not be empty".to_string(),
                });
            }
            if !seen.insert(key.as_str()) {
                return Err(Error::InvalidConfigValue {
                    key: "storage.indexed_metadata_keys".to_string(),
                    message: format!("Duplicate indexed metadata key: {}", key),
                });
            }
        }

        for key in self.scope.default_scope.keys() {
            if key.is_empty() {
                return Err(Error::InvalidConfigValue {
                    key: "scope.default_scope".to_string(),
                    message: "Scope key must not be empty".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content =
            toml::to_string_pretty(self).map_err(|e| Error::Configuration(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a builder for configuration
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

/// Builder for Config
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Set storage configuration
    pub fn storage(mut self, storage: StorageConfig) -> Self {
        self.config.storage = storage;
        self
    }

    /// Set embedding configuration
    pub fn embedding(mut self, embedding: EmbeddingConfig) -> Self {
        self.config.embedding = embedding;
        self
    }

    /// Set write pipeline configuration
    pub fn write(mut self, write: WriteConfig) -> Self {
        self.config.write = write;
        self
    }

    /// Set retrieval configuration
    pub fn retrieval(mut self, retrieval: RetrievalConfig) -> Self {
        self.config.retrieval = retrieval;
        self
    }

    /// Set jobs configuration
    pub fn jobs(mut self, jobs: JobsConfig) -> Self {
        self.config.jobs = jobs;
        self
    }

    /// Set audit configuration
    pub fn audit(mut self, audit: AuditConfig) -> Self {
        self.config.audit = audit;
        self
    }

    /// Set scope configuration
    pub fn scope(mut self, scope: ScopeConfig) -> Self {
        self.config.scope = scope;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<Config> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Storage backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Backend type: "memscaledb", "sqlite", "memory", "postgres"
    pub backend: String,
    /// Path for file-based storage (MemscaleDB directory or SQLite file)
    pub path: Option<String>,
    /// Connection string for database backends
    pub connection_string: Option<String>,
    /// Maximum connections in pool
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Enable WAL mode for SQLite
    pub sqlite_wal_mode: bool,
    /// Metadata keys to build a secondary index on. Filter queries constraining
    /// only these keys can pre-filter at the storage layer for O(matching)
    /// lookups instead of O(n) full scans. Non-indexed keys still filter
    /// correctly via the residual post-filter path.
    pub indexed_metadata_keys: Vec<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memscaledb".to_string(),
            path: Some("memscaledb".to_string()),
            connection_string: None,
            max_connections: 5,
            connection_timeout_secs: 30,
            sqlite_wal_mode: true,
            indexed_metadata_keys: Vec::new(),
        }
    }
}

impl StorageConfig {
    /// Create MemscaleDB configuration
    pub fn memscaledb(path: impl Into<String>) -> Self {
        Self {
            backend: "memscaledb".to_string(),
            path: Some(path.into()),
            ..Default::default()
        }
    }

    /// Create SQLite configuration
    pub fn sqlite(path: impl Into<String>) -> Self {
        Self {
            backend: "sqlite".to_string(),
            path: Some(path.into()),
            ..Default::default()
        }
    }

    /// Create in-memory configuration
    pub fn memory() -> Self {
        Self {
            backend: "memory".to_string(),
            path: None,
            ..Default::default()
        }
    }

    /// Create Postgres configuration
    pub fn postgres(connection_string: impl Into<String>) -> Self {
        Self {
            backend: "postgres".to_string(),
            connection_string: Some(connection_string.into()),
            path: None,
            ..Default::default()
        }
    }
}

/// Write pipeline configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WriteConfig {
    /// Salience checking configuration
    pub salience: SalienceConfig,
    /// Novelty checking configuration
    pub novelty: NoveltyConfig,
    /// Redundancy checking configuration
    pub redundancy: RedundancyConfig,
    /// Budget configuration
    pub budget: BudgetConfig,
    /// Memory extraction configuration (LLM-based fact extraction from episodic memories)
    pub extraction: ExtractionConfig,
    /// Conflict resolution configuration (LLM-based ADD/UPDATE/DELETE/NOOP classification)
    pub conflict_resolution: ConflictResolutionConfig,
}

/// LLM-based conflict resolution configuration.
///
/// When enabled, new memories are compared against similar existing memories
/// using an LLM to classify the relationship as ADD (new info), UPDATE (augment
/// existing), DELETE (contradicts existing), or NOOP (already known). This
/// enables automatic fact deduplication, contradiction resolution, and memory
/// merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConflictResolutionConfig {
    /// Whether conflict resolution is enabled
    pub enabled: bool,
    /// LLM provider for conflict resolution (e.g. "openai")
    pub provider: String,
    /// Model identifier (e.g. "gpt-4o-mini")
    pub model: String,
    /// API key (falls back to embedding.api_key if not set)
    pub api_key: Option<String>,
    /// API base URL (falls back to embedding.base_url if not set)
    pub base_url: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Number of retries on failure
    pub retries: u32,
    /// Maximum number of similar memories to compare against
    pub max_similar_to_compare: usize,
}

impl Default for ConflictResolutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            api_key: None,
            base_url: None,
            timeout_secs: 30,
            retries: 2,
            max_similar_to_compare: 5,
        }
    }
}

/// LLM-based memory extraction configuration.
///
/// When enabled, episodic memories (conversations, events, observations) are
/// processed by an LLM to extract structured facts, preferences, and temporal
/// events. These are stored as additional `SemanticFact` memories alongside the
/// original, improving retrieval quality for factual queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionConfig {
    /// Whether extraction is enabled
    pub enabled: bool,
    /// LLM provider for extraction (e.g. "openai")
    pub provider: String,
    /// Model identifier (e.g. "gpt-4o-mini")
    pub model: String,
    /// API key (falls back to embedding.api_key if not set)
    pub api_key: Option<String>,
    /// API base URL (falls back to embedding.base_url if not set)
    pub base_url: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Number of retries on failure
    pub retries: u32,
    /// Minimum confidence threshold for extracted facts (0.0-1.0)
    pub min_confidence: f64,
    /// Maximum number of facts to extract per memory
    pub max_facts_per_memory: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            api_key: None,
            base_url: None,
            timeout_secs: 30,
            retries: 2,
            min_confidence: 0.5,
            max_facts_per_memory: 10,
        }
    }
}

/// Salience checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SalienceConfig {
    /// Whether salience checking is enabled
    pub enabled: bool,
    /// Minimum salience threshold (0.0-1.0)
    pub threshold: f64,
    /// Memory types exempt from salience checking
    pub exempt_types: Vec<MemoryType>,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.3,
            exempt_types: vec![
                MemoryType::AgentStateGoal,
                MemoryType::AgentStateTask,
                MemoryType::EpisodicEvent,
                MemoryType::EpisodicConversation,
                MemoryType::EpisodicObservation,
            ],
        }
    }
}

/// Novelty checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NoveltyConfig {
    /// Whether novelty checking is enabled
    pub enabled: bool,
    /// Minimum novelty threshold (0.0-1.0)
    pub threshold: f64,
    /// Number of similar memories to check against
    pub check_count: usize,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.3,
            check_count: 10,
        }
    }
}

/// Redundancy checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RedundancyConfig {
    /// Whether redundancy checking is enabled
    pub enabled: bool,
    /// Similarity threshold for considering redundant
    pub similarity_threshold: f64,
    /// Whether to auto-merge similar memories
    pub auto_merge: bool,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.95,
            auto_merge: true,
        }
    }
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetConfig {
    /// Global maximum number of memories
    pub global_max_memories: usize,
    /// Per-type limits
    pub type_limits: HashMap<MemoryType, usize>,
    /// Action when budget exceeded
    pub on_exceeded: BudgetExceededAction,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        let mut type_limits = HashMap::new();
        type_limits.insert(MemoryType::EpisodicConversation, 10000);
        type_limits.insert(MemoryType::EpisodicEvent, 5000);
        type_limits.insert(MemoryType::EpisodicObservation, 5000);
        type_limits.insert(MemoryType::SemanticFact, 50000);
        type_limits.insert(MemoryType::SemanticPreference, 1000);
        type_limits.insert(MemoryType::SemanticEntity, 10000);
        type_limits.insert(MemoryType::ProceduralWorkflow, 1000);
        type_limits.insert(MemoryType::ProceduralSkill, 1000);
        type_limits.insert(MemoryType::ProceduralCase, 10000);

        Self {
            global_max_memories: 100000,
            type_limits,
            on_exceeded: BudgetExceededAction::Reject,
        }
    }
}

/// Action when budget is exceeded
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BudgetExceededAction {
    /// Reject the new memory
    #[default]
    Reject,
    /// Remove oldest memories to make room
    RemoveOldest,
    /// Remove lowest confidence memories
    RemoveLowestConfidence,
    /// Trigger consolidation
    Consolidate,
}

/// Method for fusing vector and text search results in hybrid search
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HybridFusionMethod {
    /// Simple weighted average of normalized scores
    WeightedAverage,
    /// Reciprocal Rank Fusion -- robust, parameter-free fusion
    #[default]
    ReciprocalRankFusion,
}

/// Retrieval pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalConfig {
    /// Default result limit
    pub default_limit: usize,
    /// Maximum result limit
    pub max_limit: usize,
    /// Default token budget for context
    pub default_token_budget: usize,
    /// Whether to enable gating
    pub gating_enabled: bool,
    /// Minimum confidence for results
    pub min_confidence: f64,
    /// Weight for vector search in hybrid mode
    pub vector_weight: f64,
    /// Weight for text search in hybrid mode
    pub text_weight: f64,
    /// Fusion method for combining vector and text search results
    pub fusion_method: HybridFusionMethod,
    /// RRF smoothing constant (only used when fusion_method is ReciprocalRankFusion)
    pub rrf_k: f64,
    /// Reranker configuration (used by language-level rerankers in Python/JS)
    pub reranker: RerankerConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            max_limit: 100,
            default_token_budget: 4000,
            gating_enabled: true,
            min_confidence: 0.1,
            vector_weight: 0.7,
            text_weight: 0.3,
            fusion_method: HybridFusionMethod::default(),
            rrf_k: 60.0,
            reranker: RerankerConfig::default(),
        }
    }
}

/// Reranker configuration (used by language-level rerankers in Python/JS)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// Whether reranking is enabled
    pub enabled: bool,
    /// Reranker type: "cross_encoder", "llm", "none"
    pub reranker_type: String,
    /// Provider name: "cohere", "jina", "openai", "anthropic"
    pub provider: String,
    /// Model identifier
    pub model: String,
    /// Number of results to keep after reranking
    pub top_k: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reranker_type: "none".to_string(),
            provider: String::new(),
            model: String::new(),
            top_k: 5,
        }
    }
}

/// Background jobs configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JobsConfig {
    /// Whether jobs are enabled
    pub enabled: bool,
    /// Maximum concurrent jobs
    pub max_concurrent: usize,
    /// Consolidation job configuration
    pub consolidation: JobScheduleConfig,
    /// Decay job configuration
    pub decay: JobScheduleConfig,
    /// Contradiction detection job configuration
    pub contradiction: JobScheduleConfig,
    /// Index maintenance job configuration
    pub index_maintenance: JobScheduleConfig,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent: 2,
            consolidation: JobScheduleConfig {
                enabled: true,
                interval_secs: 3600, // 1 hour
                batch_size: 100,
            },
            decay: JobScheduleConfig {
                enabled: true,
                interval_secs: 86400, // 24 hours
                batch_size: 1000,
            },
            contradiction: JobScheduleConfig {
                enabled: true,
                interval_secs: 3600,
                batch_size: 50,
            },
            index_maintenance: JobScheduleConfig {
                enabled: true,
                interval_secs: 86400,
                batch_size: 10000,
            },
        }
    }
}

/// Configuration for a scheduled job
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JobScheduleConfig {
    /// Whether this job is enabled
    pub enabled: bool,
    /// Interval between runs in seconds
    pub interval_secs: u64,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for JobScheduleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 3600,
            batch_size: 100,
        }
    }
}

impl JobScheduleConfig {
    /// Get interval as Duration
    pub fn interval(&self) -> Duration {
        Duration::from_secs(self.interval_secs)
    }
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    /// Whether audit logging is enabled
    pub enabled: bool,
    /// Retention period in days
    pub retention_days: u32,
    /// Whether to log storage decisions
    pub log_storage: bool,
    /// Whether to log retrieval decisions
    pub log_retrieval: bool,
    /// Whether to log rejections
    pub log_rejections: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_days: 30,
            log_storage: true,
            log_retrieval: true,
            log_rejections: true,
        }
    }
}

/// Multi-agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MultiAgentConfig {
    /// Whether multi-agent mode is enabled
    pub enabled: bool,
    /// Default trust level for new agents
    pub default_trust_level: String,
    /// Whether to allow cross-agent memory sharing
    pub allow_sharing: bool,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_trust_level: "read_only".to_string(),
            allow_sharing: false,
        }
    }
}

/// Scope configuration for default metadata injection.
///
/// When `default_scope` is non-empty, its entries are merged into every
/// memory's `metadata` on write and into every search's `filters.metadata`
/// on read. Per-call keys override defaults on conflict; keys absent from the
/// per-call map are injected from the default. This yields row-level-security
/// semantics for user/thread/tenant scoping.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScopeConfig {
    /// Default scope entries, merged on every write and applied on every read.
    pub default_scope: HashMap<String, serde_json::Value>,
}

impl ScopeConfig {
    /// Construct a `ScopeConfig` from an iterator of key/value pairs.
    pub fn from_entries<I, K, V>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        Self {
            default_scope: entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// True when no default scope entries are configured.
    pub fn is_empty(&self) -> bool {
        self.default_scope.is_empty()
    }

    /// Merge `default_scope` into `target`, preserving any keys already present
    /// in `target` (per-call overrides). Keys absent from `target` are inserted.
    pub fn apply_to(&self, target: &mut HashMap<String, serde_json::Value>) {
        for (key, value) in &self.default_scope {
            target
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_from_toml() {
        let toml = r#"
            [storage]
            backend = "sqlite"
            path = "test.db"

            [embedding]
            provider = "openai"
            model = "text-embedding-3-small"

            [write.salience]
            enabled = true
            threshold = 0.5

            [jobs]
            enabled = false
        "#;

        let config = Config::from_toml(toml).unwrap();
        assert_eq!(config.storage.path, Some("test.db".to_string()));
        assert_eq!(config.write.salience.threshold, 0.5);
        assert!(!config.jobs.enabled);
    }

    #[test]
    fn config_validation_rejects_invalid() {
        let mut config = Config::default();
        config.write.salience.threshold = 1.5;

        assert!(config.validate().is_err());
    }

    #[test]
    fn config_builder() {
        let config = Config::builder()
            .storage(StorageConfig::memory())
            .audit(AuditConfig {
                enabled: false,
                ..Default::default()
            })
            .build()
            .unwrap();

        assert_eq!(config.storage.backend, "memory");
        assert!(!config.audit.enabled);
    }

    #[test]
    fn storage_config_helpers() {
        let sqlite = StorageConfig::sqlite("test.db");
        assert_eq!(sqlite.backend, "sqlite");
        assert_eq!(sqlite.path, Some("test.db".to_string()));

        let memory = StorageConfig::memory();
        assert_eq!(memory.backend, "memory");

        let postgres = StorageConfig::postgres("postgresql://localhost/test");
        assert_eq!(postgres.backend, "postgres");
        assert!(postgres.connection_string.is_some());
    }

    #[test]
    fn hybrid_fusion_method_default_is_rrf() {
        let config = RetrievalConfig::default();
        assert_eq!(
            config.fusion_method,
            HybridFusionMethod::ReciprocalRankFusion
        );
        assert!((config.rrf_k - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hybrid_fusion_method_toml_roundtrip() {
        let toml_str = r#"
            [retrieval]
            fusion_method = "weighted_average"
            rrf_k = 42.0
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.retrieval.fusion_method,
            HybridFusionMethod::WeightedAverage
        );
        assert!((config.retrieval.rrf_k - 42.0).abs() < f64::EPSILON);

        // Serialize and deserialize
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized.retrieval.fusion_method,
            HybridFusionMethod::WeightedAverage
        );
    }

    #[test]
    fn hybrid_fusion_method_rrf_toml() {
        let toml_str = r#"
            [retrieval]
            fusion_method = "reciprocal_rank_fusion"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.retrieval.fusion_method,
            HybridFusionMethod::ReciprocalRankFusion
        );
    }

    #[test]
    fn extraction_config_defaults() {
        let config = ExtractionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.max_facts_per_memory, 10);
        assert!((config.min_confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn scope_config_apply_preserves_per_call_keys() {
        let scope = ScopeConfig::from_entries([
            ("user_id", serde_json::json!("alice")),
            ("tenant_id", serde_json::json!("acme")),
        ]);
        let mut target = HashMap::new();
        target.insert("user_id".to_string(), serde_json::json!("bob"));
        scope.apply_to(&mut target);
        assert_eq!(target["user_id"], serde_json::json!("bob"));
        assert_eq!(target["tenant_id"], serde_json::json!("acme"));
    }

    #[test]
    fn scope_config_empty_is_noop() {
        let scope = ScopeConfig::default();
        assert!(scope.is_empty());
        let mut target = HashMap::new();
        target.insert("user_id".to_string(), serde_json::json!("alice"));
        scope.apply_to(&mut target);
        assert_eq!(target.len(), 1);
    }

    #[test]
    fn validation_rejects_empty_indexed_key() {
        let mut config = Config::default();
        config.storage.indexed_metadata_keys = vec![String::new()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_rejects_duplicate_indexed_keys() {
        let mut config = Config::default();
        config.storage.indexed_metadata_keys = vec!["user_id".into(), "user_id".into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn scope_config_toml_roundtrip() {
        let toml_str = r#"
            [storage]
            indexed_metadata_keys = ["user_id", "thread_id"]

            [scope.default_scope]
            user_id = "alice"
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.storage.indexed_metadata_keys,
            vec!["user_id".to_string(), "thread_id".to_string()]
        );
        assert_eq!(
            config.scope.default_scope.get("user_id"),
            Some(&serde_json::json!("alice"))
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn extraction_config_toml_roundtrip() {
        let toml_str = r#"
            [write.extraction]
            enabled = true
            model = "gpt-4o"
            min_confidence = 0.7
            max_facts_per_memory = 5
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.write.extraction.enabled);
        assert_eq!(config.write.extraction.model, "gpt-4o");
        assert!((config.write.extraction.min_confidence - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.write.extraction.max_facts_per_memory, 5);
    }
}
