//! Storage trait for memory persistence

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::memory::{Memory, MemoryType};
use crate::types::{AgentId, Confidence, Embedding, MemoryId, Version};

/// Trait for memory storage backends
#[async_trait]
pub trait MemoryStorage: Send + Sync {
    /// Store a new memory
    async fn store(&self, memory: Memory) -> Result<MemoryId>;

    /// Get a memory by ID
    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>>;

    /// Get multiple memories by IDs
    async fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>>;

    /// Update an existing memory (with optimistic concurrency)
    async fn update(&self, memory: Memory, expected_version: Version) -> Result<()>;

    /// Delete a memory
    async fn delete(&self, id: &MemoryId) -> Result<bool>;

    /// Delete multiple memories
    async fn delete_many(&self, ids: &[MemoryId]) -> Result<usize>;

    /// Search memories by query
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>>;

    /// Vector similarity search
    async fn vector_search(
        &self,
        embedding: &Embedding,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>>;

    /// Full-text search
    async fn text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;

    /// Count memories matching filters
    async fn count(&self, filters: Option<SearchFilters>) -> Result<usize>;

    /// Check if a memory exists
    async fn exists(&self, id: &MemoryId) -> Result<bool>;

    /// Get memories for an agent
    async fn get_by_agent(
        &self,
        agent_id: &AgentId,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>>;

    /// Get memories by type
    async fn get_by_type(
        &self,
        memory_type: MemoryType,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>>;

    /// Record an access to a memory
    async fn record_access(&self, id: &MemoryId) -> Result<()>;

    /// Get memory statistics
    async fn stats(&self) -> Result<StorageStats>;

    /// Begin a transaction (for backends that support it)
    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>>;

    /// Perform a health check
    async fn health_check(&self) -> Result<()>;
}

/// Transaction trait for atomic operations
#[async_trait]
pub trait Transaction: Send + Sync {
    /// Store within transaction
    async fn store(&mut self, memory: Memory) -> Result<MemoryId>;

    /// Update within transaction
    async fn update(&mut self, memory: Memory, expected_version: Version) -> Result<()>;

    /// Delete within transaction
    async fn delete(&mut self, id: &MemoryId) -> Result<bool>;

    /// Commit the transaction
    async fn commit(self: Box<Self>) -> Result<()>;

    /// Rollback the transaction
    async fn rollback(self: Box<Self>) -> Result<()>;
}

/// Search query parameters
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Text query for semantic/keyword search
    pub query: Option<String>,
    /// Vector for similarity search
    pub embedding: Option<Embedding>,
    /// Filters to apply
    pub filters: SearchFilters,
    /// Maximum number of results
    pub limit: usize,
    /// Offset for pagination
    pub offset: usize,
    /// Minimum similarity score (for vector search)
    pub min_score: Option<f64>,
    /// Search mode
    pub mode: SearchMode,
}

impl SearchQuery {
    /// Create a new search query
    pub fn new() -> Self {
        Self {
            limit: 10,
            ..Default::default()
        }
    }

    /// Set the text query
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set the embedding for vector search
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set filters
    pub fn with_filters(mut self, filters: SearchFilters) -> Self {
        self.filters = filters;
        self
    }

    /// Set limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set search mode
    pub fn with_mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Filter by memory types
    pub memory_types: Option<Vec<MemoryType>>,
    /// Filter by minimum confidence
    pub min_confidence: Option<Confidence>,
    /// Filter by agent IDs
    pub agent_ids: Option<Vec<AgentId>>,
    /// Filter by tags (any match)
    pub tags: Option<Vec<String>>,
    /// Filter by required tags (all must match)
    pub required_tags: Option<Vec<String>>,
    /// Filter by time range (created after)
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by time range (created before)
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by last accessed (after)
    pub accessed_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by metadata key-value
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Exclude specific memory IDs
    pub exclude_ids: Option<Vec<MemoryId>>,
    /// Filter for facts that were valid at this point in time (event time).
    /// Matches memories where valid_from <= T AND (valid_until is None OR valid_until > T).
    pub valid_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter for facts that were known to the system at this point in time.
    /// Matches memories where created_at <= T.
    pub known_at: Option<chrono::DateTime<chrono::Utc>>,
    /// If true, exclude memories that have been invalidated (valid_until is set and in the past).
    /// Defaults to false for backward compatibility.
    pub exclude_invalidated: Option<bool>,
}

impl SearchFilters {
    /// Create new empty filters
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by memory types
    pub fn with_types(mut self, types: Vec<MemoryType>) -> Self {
        self.memory_types = Some(types);
        self
    }

    /// Filter by minimum confidence
    pub fn with_min_confidence(mut self, confidence: Confidence) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Filter by agent IDs
    pub fn with_agents(mut self, agents: Vec<AgentId>) -> Self {
        self.agent_ids = Some(agents);
        self
    }

    /// Filter by tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Filter by created time range
    pub fn created_between(
        mut self,
        after: chrono::DateTime<chrono::Utc>,
        before: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.created_after = Some(after);
        self.created_before = Some(before);
        self
    }

    /// Exclude specific IDs
    pub fn exclude(mut self, ids: Vec<MemoryId>) -> Self {
        self.exclude_ids = Some(ids);
        self
    }

    /// Filter by metadata key-value pairs (all must match)
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a single metadata key-value filter entry
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }

    /// Filter for facts valid at a specific point in time (event time)
    pub fn valid_at(mut self, at: chrono::DateTime<chrono::Utc>) -> Self {
        self.valid_at = Some(at);
        self
    }

    /// Filter for facts known to the system at a specific point in time
    pub fn known_at(mut self, at: chrono::DateTime<chrono::Utc>) -> Self {
        self.known_at = Some(at);
        self
    }

    /// Exclude memories that have been invalidated (valid_until is set and in the past)
    pub fn exclude_invalidated(mut self) -> Self {
        self.exclude_invalidated = Some(true);
        self
    }
}

/// Search mode
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Vector similarity search only
    Vector,
    /// Full-text search only
    Text,
    /// Combined vector and text search
    #[default]
    Hybrid,
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The memory
    pub memory: Memory,
    /// Relevance score (0.0-1.0)
    pub score: f64,
    /// How the result was found
    pub match_type: MatchType,
    /// Highlighted snippets (for text search)
    pub highlights: Vec<String>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(memory: Memory, score: f64, match_type: MatchType) -> Self {
        Self {
            memory,
            score,
            match_type,
            highlights: Vec::new(),
        }
    }

    /// Add highlights
    pub fn with_highlights(mut self, highlights: Vec<String>) -> Self {
        self.highlights = highlights;
        self
    }
}

/// How a search result was matched
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    /// Matched by vector similarity
    Vector,
    /// Matched by text search
    Text,
    /// Matched by both
    Hybrid,
    /// Exact ID match
    Exact,
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of memories
    pub total_memories: usize,
    /// Memories by type
    pub by_type: HashMap<MemoryType, usize>,
    /// Total storage size in bytes
    pub storage_bytes: u64,
    /// Number of embeddings stored
    pub embeddings_count: usize,
    /// Average confidence score
    pub avg_confidence: f64,
    /// Number of unique agents
    pub agent_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_query_builder() {
        let query = SearchQuery::new()
            .with_query("test query")
            .with_limit(20)
            .with_offset(10)
            .with_mode(SearchMode::Hybrid)
            .with_filters(
                SearchFilters::new()
                    .with_min_confidence(Confidence::new(0.5))
                    .with_tags(vec!["important".to_string()]),
            );

        assert_eq!(query.query, Some("test query".to_string()));
        assert_eq!(query.limit, 20);
        assert_eq!(query.offset, 10);
        assert_eq!(query.mode, SearchMode::Hybrid);
        assert!(query.filters.min_confidence.is_some());
    }

    #[test]
    fn search_filters_builder() {
        let filters = SearchFilters::new()
            .with_types(vec![MemoryType::SemanticFact])
            .with_min_confidence(Confidence::new(0.7))
            .with_tags(vec!["test".to_string()])
            .exclude(vec![MemoryId::new()]);

        assert!(filters.memory_types.is_some());
        assert!(filters.min_confidence.is_some());
        assert!(filters.exclude_ids.is_some());
    }
}
