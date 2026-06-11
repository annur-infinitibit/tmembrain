//! Comprehensive error types for the Membrain memory system

use thiserror::Error;

use crate::types::MemoryId;

/// Result type alias for Membrain operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Membrain
#[derive(Error, Debug)]
pub enum Error {
    // ==================== Type Errors ====================
    /// Invalid confidence value (must be in [0.0, 1.0])
    #[error("Invalid confidence value: {0} (must be in [0.0, 1.0])")]
    InvalidConfidence(f64),

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },

    /// Invalid embedding bytes
    #[error("Invalid embedding bytes: length must be multiple of 4")]
    InvalidEmbeddingBytes,

    /// Empty embedding list for aggregation
    #[error("Cannot compute on empty embedding list")]
    EmptyEmbeddingList,

    // ==================== Storage Errors ====================
    /// Memory not found
    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),

    /// Storage backend error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Database connection error
    #[error("Database connection error: {0}")]
    DatabaseConnection(String),

    /// Transaction failed
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Schema migration error
    #[error("Schema migration error: {0}")]
    SchemaMigration(String),

    // ==================== Write Pipeline Errors ====================
    /// Memory rejected by write policy
    #[error("Memory rejected: {reason}")]
    Rejected { reason: String },

    /// Budget exceeded
    #[error("Memory budget exceeded: {memory_type} has {current} of {max} memories")]
    BudgetExceeded {
        memory_type: String,
        current: usize,
        max: usize,
    },

    /// Duplicate memory detected
    #[error("Duplicate memory detected: similar to {existing_id}")]
    Duplicate { existing_id: MemoryId },

    /// Write conflict (optimistic concurrency)
    #[error("Write conflict: memory {0} was modified by another process")]
    WriteConflict(MemoryId),

    // ==================== Retrieval Errors ====================
    /// Query parsing error
    #[error("Query parsing error: {0}")]
    QueryParsing(String),

    /// Search failed
    #[error("Search failed: {0}")]
    SearchFailed(String),

    /// Index error
    #[error("Index error: {0}")]
    IndexError(String),

    // ==================== Embedding Provider Errors ====================
    /// Embedding generation failed
    #[error("Embedding generation failed: {0}")]
    EmbeddingGeneration(String),

    /// Embedding provider not configured
    #[error("Embedding provider not configured")]
    EmbeddingProviderNotConfigured,

    /// Rate limited
    #[error("Rate limited: retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },

    // ==================== Configuration Errors ====================
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Invalid configuration value
    #[error("Invalid configuration value for {key}: {message}")]
    InvalidConfigValue { key: String, message: String },

    /// Missing required configuration
    #[error("Missing required configuration: {0}")]
    MissingConfig(String),

    // ==================== Multi-Agent Errors ====================
    /// Insufficient trust level
    #[error("Insufficient trust level: required {required}, has {actual}")]
    InsufficientTrust { required: String, actual: String },

    /// Agent not found
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    /// Memory visibility error
    #[error("Memory not visible to agent: {memory_id} is {visibility}")]
    VisibilityError {
        memory_id: MemoryId,
        visibility: String,
    },

    // ==================== Job Errors ====================
    /// Job scheduling error
    #[error("Job scheduling error: {0}")]
    JobScheduling(String),

    /// Job execution error
    #[error("Job execution error in {job_name}: {message}")]
    JobExecution { job_name: String, message: String },

    /// Job already running
    #[error("Job already running: {0}")]
    JobAlreadyRunning(String),

    // ==================== Compression Errors ====================
    /// Consolidation failed
    #[error("Consolidation failed: {0}")]
    ConsolidationFailed(String),

    /// Decay calculation error
    #[error("Decay calculation error: {0}")]
    DecayError(String),

    // ==================== Audit Errors ====================
    /// Audit log error
    #[error("Audit log error: {0}")]
    AuditLog(String),

    // ==================== FFI Errors ====================
    /// FFI conversion error
    #[error("FFI conversion error: {0}")]
    FfiConversion(String),

    // ==================== Graph Errors ====================
    /// Graph layer error
    #[error("Graph error: {0}")]
    GraphError(String),

    // ==================== Generic Errors ====================
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Operation cancelled
    #[error("Operation cancelled")]
    Cancelled,

    /// Timeout
    #[error("Operation timed out after {0} seconds")]
    Timeout(u64),
}

impl Error {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::RateLimited { .. }
                | Error::DatabaseConnection(_)
                | Error::Timeout(_)
                | Error::WriteConflict(_)
        )
    }

    /// Get the error code for external APIs
    pub fn error_code(&self) -> &'static str {
        match self {
            Error::InvalidConfidence(_) => "INVALID_CONFIDENCE",
            Error::EmbeddingDimensionMismatch { .. } => "EMBEDDING_DIMENSION_MISMATCH",
            Error::InvalidEmbeddingBytes => "INVALID_EMBEDDING_BYTES",
            Error::EmptyEmbeddingList => "EMPTY_EMBEDDING_LIST",
            Error::MemoryNotFound(_) => "MEMORY_NOT_FOUND",
            Error::Storage(_) => "STORAGE_ERROR",
            Error::DatabaseConnection(_) => "DATABASE_CONNECTION_ERROR",
            Error::TransactionFailed(_) => "TRANSACTION_FAILED",
            Error::Serialization(_) => "SERIALIZATION_ERROR",
            Error::Deserialization(_) => "DESERIALIZATION_ERROR",
            Error::SchemaMigration(_) => "SCHEMA_MIGRATION_ERROR",
            Error::Rejected { .. } => "REJECTED",
            Error::BudgetExceeded { .. } => "BUDGET_EXCEEDED",
            Error::Duplicate { .. } => "DUPLICATE",
            Error::WriteConflict(_) => "WRITE_CONFLICT",
            Error::QueryParsing(_) => "QUERY_PARSING_ERROR",
            Error::SearchFailed(_) => "SEARCH_FAILED",
            Error::IndexError(_) => "INDEX_ERROR",
            Error::EmbeddingGeneration(_) => "EMBEDDING_GENERATION_ERROR",
            Error::EmbeddingProviderNotConfigured => "EMBEDDING_PROVIDER_NOT_CONFIGURED",
            Error::RateLimited { .. } => "RATE_LIMITED",
            Error::Configuration(_) => "CONFIGURATION_ERROR",
            Error::InvalidConfigValue { .. } => "INVALID_CONFIG_VALUE",
            Error::MissingConfig(_) => "MISSING_CONFIG",
            Error::InsufficientTrust { .. } => "INSUFFICIENT_TRUST",
            Error::AgentNotFound(_) => "AGENT_NOT_FOUND",
            Error::VisibilityError { .. } => "VISIBILITY_ERROR",
            Error::JobScheduling(_) => "JOB_SCHEDULING_ERROR",
            Error::JobExecution { .. } => "JOB_EXECUTION_ERROR",
            Error::JobAlreadyRunning(_) => "JOB_ALREADY_RUNNING",
            Error::ConsolidationFailed(_) => "CONSOLIDATION_FAILED",
            Error::DecayError(_) => "DECAY_ERROR",
            Error::AuditLog(_) => "AUDIT_LOG_ERROR",
            Error::FfiConversion(_) => "FFI_CONVERSION_ERROR",
            Error::GraphError(_) => "GRAPH_ERROR",
            Error::Internal(_) => "INTERNAL_ERROR",
            Error::Io(_) => "IO_ERROR",
            Error::Cancelled => "CANCELLED",
            Error::Timeout(_) => "TIMEOUT",
        }
    }
}

// Implement conversions from common error types

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<rmp_serde::encode::Error> for Error {
    fn from(err: rmp_serde::encode::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<rmp_serde::decode::Error> for Error {
    fn from(err: rmp_serde::decode::Error) -> Self {
        Error::Deserialization(err.to_string())
    }
}

impl From<uuid::Error> for Error {
    fn from(err: uuid::Error) -> Self {
        Error::Deserialization(format!("Invalid UUID: {}", err))
    }
}

impl From<toml::de::Error> for Error {
    fn from(err: toml::de::Error) -> Self {
        Error::Configuration(err.to_string())
    }
}

impl From<config::ConfigError> for Error {
    fn from(err: config::ConfigError) -> Self {
        Error::Configuration(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_retryable() {
        assert!(Error::RateLimited { retry_after_secs: 60 }.is_retryable());
        assert!(Error::DatabaseConnection("test".into()).is_retryable());
        assert!(Error::Timeout(30).is_retryable());
        assert!(!Error::InvalidConfidence(1.5).is_retryable());
    }

    #[test]
    fn error_codes() {
        assert_eq!(Error::InvalidConfidence(1.5).error_code(), "INVALID_CONFIDENCE");
        assert_eq!(Error::MemoryNotFound(MemoryId::new()).error_code(), "MEMORY_NOT_FOUND");
    }

    #[test]
    fn error_display() {
        let err = Error::BudgetExceeded {
            memory_type: "episodic".into(),
            current: 1000,
            max: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("episodic"));
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }
}
