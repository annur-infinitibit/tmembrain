//! Main retrieval pipeline

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use membrain_audit::{AuditEntry, AuditLog};
use membrain_core::config::RetrievalConfig;
use membrain_core::error::Result;
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::{
    EmbeddingProvider, MatchType, MemoryStorage, SearchFilters, SearchMode, SearchQuery,
    SearchResult as CoreSearchResult,
};
use membrain_core::types::{AgentId, Confidence, Embedding, MemoryId};
use membrain_graph::bridge::GraphAugmentedRetrieval;

use super::gating::RetrievalGating;
use super::intent::{IntentDetector, QueryIntent};
use super::scoring::{DiversityReranker, ScoreWeights, ScoringStrategy};

/// Request for memory retrieval
#[derive(Debug, Clone)]
pub struct RetrievalRequest {
    /// The query text
    pub query: String,
    /// Query embedding (if available)
    pub embedding: Option<Embedding>,
    /// Maximum number of results
    pub limit: usize,
    /// Token budget for context (optional)
    pub token_budget: Option<usize>,
    /// Filters to apply
    pub filters: RetrievalFilters,
    /// Context about the retrieval
    pub context: RetrievalContext,
}

/// Filters for retrieval
#[derive(Debug, Clone, Default)]
pub struct RetrievalFilters {
    /// Filter by memory types
    pub memory_types: Option<Vec<MemoryType>>,
    /// Minimum confidence
    pub min_confidence: Option<Confidence>,
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    /// Filter by agent
    pub agent_id: Option<AgentId>,
    /// Exclude specific memories
    pub exclude_ids: Option<Vec<MemoryId>>,
    /// Filter by metadata key-value pairs (all must match)
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Context for retrieval
#[derive(Debug, Clone, Default)]
pub struct RetrievalContext {
    /// Current session ID
    pub session_id: Option<membrain_core::types::SessionId>,
    /// Current agent ID
    pub agent_id: Option<AgentId>,
    /// Whether to use intent detection
    pub use_intent_detection: bool,
    /// Whether to use gating
    pub use_gating: bool,
    /// Whether to apply diversity reranking
    pub use_diversity: bool,
}

impl RetrievalRequest {
    /// Create a new retrieval request
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            embedding: None,
            limit: 10,
            token_budget: None,
            filters: RetrievalFilters::default(),
            context: RetrievalContext {
                use_intent_detection: true,
                use_gating: true,
                use_diversity: true,
                ..Default::default()
            },
        }
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the token budget
    pub fn with_token_budget(mut self, budget: usize) -> Self {
        self.token_budget = Some(budget);
        self
    }

    /// Set filters
    pub fn with_filters(mut self, filters: RetrievalFilters) -> Self {
        self.filters = filters;
        self
    }

    /// Set memory type filter
    pub fn with_types(mut self, types: Vec<MemoryType>) -> Self {
        self.filters.memory_types = Some(types);
        self
    }

    /// Set minimum confidence
    pub fn with_min_confidence(mut self, confidence: Confidence) -> Self {
        self.filters.min_confidence = Some(confidence);
        self
    }

    /// Set metadata filters
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.filters.metadata = Some(metadata);
        self
    }

    /// Disable gating for this request
    pub fn without_gating(mut self) -> Self {
        self.context.use_gating = false;
        self
    }
}

/// Result of retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Retrieved memories with scores
    pub memories: Vec<RetrievedMemory>,
    /// Whether retrieval was gated (skipped)
    pub was_gated: bool,
    /// Gating reason if gated
    pub gating_reason: Option<String>,
    /// Detected query intent
    pub intent: Option<QueryIntent>,
    /// Total processing time in milliseconds
    pub duration_ms: u64,
    /// Number of results before filtering
    pub raw_count: usize,
}

impl RetrievalResult {
    /// Get memory IDs
    pub fn ids(&self) -> Vec<MemoryId> {
        self.memories.iter().map(|m| m.id).collect()
    }

    /// Check if any results were returned
    pub fn has_results(&self) -> bool {
        !self.memories.is_empty()
    }

    /// Get total token count (estimated)
    pub fn total_tokens(&self) -> usize {
        self.memories.iter().map(|m| m.estimated_tokens).sum()
    }
}

/// A retrieved memory with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    /// Memory ID
    pub id: MemoryId,
    /// The memory content
    pub memory: Memory,
    /// Final relevance score
    pub score: f64,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Text content for display
    pub text_content: String,
}

impl RetrievedMemory {
    fn from_memory(memory: Memory, score: f64) -> Self {
        let text = memory.text_content();
        let tokens = text.split_whitespace().count() * 4 / 3; // Rough estimate

        Self {
            id: *memory.id(),
            memory,
            score,
            estimated_tokens: tokens,
            text_content: text,
        }
    }
}

/// The retrieval pipeline
pub struct RetrievalPipeline {
    storage: Arc<dyn MemoryStorage>,
    audit: Arc<AuditLog>,
    config: RetrievalConfig,
    intent_detector: IntentDetector,
    gating: RetrievalGating,
    scorer: ScoringStrategy,
    graph_bridge: Option<Arc<dyn GraphAugmentedRetrieval>>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

impl RetrievalPipeline {
    /// Create a new retrieval pipeline
    pub fn new(
        storage: Arc<dyn MemoryStorage>,
        audit: Arc<AuditLog>,
        config: RetrievalConfig,
    ) -> Self {
        let weights = ScoreWeights {
            relevance: config.vector_weight,
            confidence: 0.2,
            recency: 0.1,
            frequency: 0.0,
        };

        Self {
            storage,
            audit,
            config,
            intent_detector: IntentDetector::new(),
            gating: RetrievalGating::new(),
            scorer: ScoringStrategy::new(weights),
            graph_bridge: None,
            embedding_provider: None,
        }
    }

    /// Attach a graph bridge for graph-augmented retrieval
    pub fn with_graph(mut self, graph: Arc<dyn GraphAugmentedRetrieval>) -> Self {
        self.graph_bridge = Some(graph);
        self
    }

    /// Attach an embedding provider for automatic query embedding.
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Process a retrieval request
    pub async fn retrieve(&self, request: RetrievalRequest) -> Result<RetrievalResult> {
        let start = Instant::now();

        // Step 1: Intent detection
        let intent = if request.context.use_intent_detection {
            Some(self.intent_detector.detect(&request.query))
        } else {
            None
        };

        // Step 1b: Build intent-adaptive scorer and reranker
        let scorer = if let Some(ref i) = intent {
            ScoringStrategy::new(ScoreWeights::for_intent(i.intent_type))
        } else {
            ScoringStrategy::new(self.scorer.weights().clone())
        };

        let reranker = if let Some(ref i) = intent {
            DiversityReranker::for_intent(i.intent_type)
        } else {
            DiversityReranker::new()
        };

        // Step 2: Gating
        if request.context.use_gating && self.config.gating_enabled {
            if let Some(ref i) = intent {
                let decision = self.gating.evaluate(&request.query, i);
                if !decision.should_retrieve {
                    return Ok(RetrievalResult {
                        memories: vec![],
                        was_gated: true,
                        gating_reason: Some(decision.reason),
                        intent,
                        duration_ms: start.elapsed().as_millis() as u64,
                        raw_count: 0,
                    });
                }
            }
        }

        // Step 2b: Auto-embed query if no embedding is provided and a provider exists.
        // Failure is non-fatal: we warn and fall back to text-only search.
        let mut request = request;
        if request.embedding.is_none() {
            if let Some(ref provider) = self.embedding_provider {
                match provider.embed(&request.query).await {
                    Ok(embedding) => {
                        request.embedding = Some(embedding);
                    }
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            "Auto-embedding query failed, using text-only search"
                        );
                    }
                }
            } else {
                tracing::warn!(
                    "No embedding provider configured -- searching without vector embeddings. \
                     Only text-based matching will be used, which may reduce retrieval quality. \
                     Set embedding.api_key in your config to enable auto-embedding."
                );
            }
        }

        // Step 3: Build search query (with temporal filter propagation)
        let search_filters = self.build_filters(&request, intent.as_ref());
        let limit = request.limit.min(self.config.max_limit);

        let search_query = SearchQuery::new()
            .with_query(&request.query)
            .with_limit(limit * 3) // Over-fetch for reranking and post-filter losses
            .with_filters(search_filters)
            .with_mode(SearchMode::Hybrid);

        // Add embedding if available (may have been auto-generated above)
        let search_query = if let Some(ref emb) = request.embedding {
            search_query.with_embedding(emb.clone())
        } else {
            search_query
        };

        // Step 4: Search storage
        let mut search_results = self.storage.search(search_query).await?;

        // Step 4b: Graph-augmented retrieval
        if let (Some(ref graph), Some(ref embedding)) = (&self.graph_bridge, &request.embedding) {
            match graph.graph_query(embedding, 2, limit) {
                Ok(graph_result) => {
                    // Collect IDs already present in vector results
                    let existing_ids: HashMap<MemoryId, usize> = search_results
                        .iter()
                        .enumerate()
                        .map(|(idx, r)| (*r.memory.id(), idx))
                        .collect();

                    // Collect graph-discovered IDs not already in results
                    let new_ids: Vec<(MemoryId, f64)> = graph_result
                        .nodes
                        .iter()
                        .filter(|node| !existing_ids.contains_key(&node.memory_id))
                        .map(|node| (node.memory_id, node.score as f64))
                        .collect();

                    if !new_ids.is_empty() {
                        let ids_to_fetch: Vec<MemoryId> =
                            new_ids.iter().map(|(id, _)| *id).collect();
                        let graph_score_map: HashMap<MemoryId, f64> = new_ids.into_iter().collect();

                        if let Ok(fetched) = self.storage.get_many(&ids_to_fetch).await {
                            for memory in fetched {
                                let score =
                                    graph_score_map.get(memory.id()).copied().unwrap_or(0.0);
                                search_results.push(CoreSearchResult {
                                    memory,
                                    score,
                                    match_type: MatchType::Vector,
                                    highlights: vec![],
                                });
                            }
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        "Graph query failed, continuing with vector results only"
                    );
                }
            }
        }

        let raw_count = search_results.len();

        // Step 5: Score and rank with intent-adaptive weights
        let scored_results = scorer.rank(search_results);

        // Step 6: Intent-aware diversity reranking
        let final_results = if request.context.use_diversity {
            reranker.rerank(scored_results)
        } else {
            scored_results
        };

        // Step 7: Apply limit and token budget
        let mut memories = Vec::new();
        let mut total_tokens = 0;

        for (result, score) in final_results.into_iter().take(limit) {
            let retrieved = RetrievedMemory::from_memory(result.memory, score);

            // Check token budget
            if let Some(budget) = request.token_budget {
                if total_tokens + retrieved.estimated_tokens > budget {
                    break;
                }
                total_tokens += retrieved.estimated_tokens;
            }

            // Record access
            let _ = self.storage.record_access(&retrieved.id).await;

            memories.push(retrieved);
        }

        // Step 7b: Reinforce graph edges between co-retrieved memories
        if let Some(ref graph) = self.graph_bridge {
            let retrieved_ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();
            if let Err(error) = graph.on_memory_retrieved(&retrieved_ids) {
                tracing::warn!(?error, "Graph edge reinforcement failed");
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Log retrieval
        self.audit.log_retrieval(
            AuditEntry::retrieval_decision(&request.query, memories.len())
                .with_duration_ms(duration_ms),
        );

        Ok(RetrievalResult {
            memories,
            was_gated: false,
            gating_reason: None,
            intent,
            duration_ms,
            raw_count,
        })
    }

    fn build_filters(
        &self,
        request: &RetrievalRequest,
        _intent: Option<&QueryIntent>,
    ) -> SearchFilters {
        let mut filters = SearchFilters::new();

        // Apply explicit user-provided type filters only.
        // Intent-suggested types are NOT used as a hard filter because memories
        // stored as one type (e.g. EpisodicEvent) would become invisible to
        // queries classified as a different intent (e.g. FactLookup). Intent
        // detection still benefits retrieval through adaptive scoring weights
        // and temporal filtering.
        if let Some(ref types) = request.filters.memory_types {
            filters.memory_types = Some(types.clone());
        }

        if let Some(ref conf) = request.filters.min_confidence {
            filters.min_confidence = Some(*conf);
        } else {
            filters.min_confidence = Some(Confidence::new(self.config.min_confidence));
        }

        if let Some(ref tags) = request.filters.tags {
            filters.tags = Some(tags.clone());
        }

        if let Some(ref agents) = request.filters.agent_id {
            filters.agent_ids = Some(vec![*agents]);
        }

        if let Some(ref exclude) = request.filters.exclude_ids {
            filters.exclude_ids = Some(exclude.clone());
        }

        if let Some(ref metadata) = request.filters.metadata {
            filters.metadata = Some(metadata.clone());
        }

        // Note: Temporal filter propagation is intentionally disabled.
        // The `created_after`/`created_before` filters operate on memory
        // *storage* time, not the *event* time described in the content.
        // Applying them based on keywords like "yesterday" or "recently"
        // causes false negatives when the content describes past events
        // but was stored at a different time (e.g. bulk-loaded data).
        // Temporal keywords are still detected by the intent system and
        // available in `QueryIntent.time_reference` for future use once
        // event-time indexing is implemented.

        filters
    }

    /// Get storage reference
    pub fn storage(&self) -> &Arc<dyn MemoryStorage> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::config::AuditConfig;
    use membrain_storage::InMemoryStorage;
    use membrain_test_utils::semantic_fact as create_memory;

    #[tokio::test]
    async fn test_basic_retrieval() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        // Add some memories
        storage
            .store(create_memory("The user prefers dark mode"))
            .await
            .unwrap();
        storage
            .store(create_memory("Python is a programming language"))
            .await
            .unwrap();
        storage
            .store(create_memory("The user likes vim keybindings"))
            .await
            .unwrap();

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        let request = RetrievalRequest::new("What are the user preferences?");
        let result = pipeline.retrieve(request).await.unwrap();

        assert!(!result.was_gated);
        assert!(result.has_results());
    }

    #[tokio::test]
    async fn test_gated_retrieval() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig {
            gating_enabled: true,
            ..Default::default()
        };

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        let request = RetrievalRequest::new("hello");
        let result = pipeline.retrieve(request).await.unwrap();

        assert!(result.was_gated);
        assert!(result.memories.is_empty());
    }

    #[tokio::test]
    async fn test_token_budget() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        // Add many memories
        for i in 0..10 {
            storage
                .store(create_memory(&format!(
                    "Fact number {} about various topics",
                    i
                )))
                .await
                .unwrap();
        }

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        let request = RetrievalRequest::new("What are the facts?")
            .with_token_budget(50) // Very small budget
            .with_limit(10);

        let result = pipeline.retrieve(request).await.unwrap();

        // Should be limited by token budget
        assert!(result.total_tokens() <= 60); // Allow some slack
    }

    #[tokio::test]
    async fn test_intent_adaptive_scoring() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        storage
            .store(create_memory("The user prefers dark mode"))
            .await
            .unwrap();

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        // FactLookup query should use fact-optimized weights
        let request = RetrievalRequest::new("What is the user's preference?");
        let result = pipeline.retrieve(request).await.unwrap();

        assert!(result.intent.is_some());
        // Verify the pipeline ran successfully with intent-adaptive scoring
        assert!(!result.was_gated);
    }

    #[tokio::test]
    async fn test_temporal_filter_propagation() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        storage
            .store(create_memory("Important fact about system"))
            .await
            .unwrap();

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        // "recently" should trigger a TimeReference::Recent filter
        let request = RetrievalRequest::new("What did we discuss recently?");
        let result = pipeline.retrieve(request).await.unwrap();

        if let Some(ref intent) = result.intent {
            assert!(
                intent.time_reference.is_some(),
                "Expected a time reference for 'recently'"
            );
        }
    }

    #[tokio::test]
    async fn test_pipeline_without_graph() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        storage
            .store(create_memory("Fact about Rust programming"))
            .await
            .unwrap();

        // Pipeline without graph should still work normally
        let pipeline = RetrievalPipeline::new(storage, audit, config);

        let request = RetrievalRequest::new("Tell me about Rust");
        let result = pipeline.retrieve(request).await.unwrap();

        assert!(!result.was_gated);
        assert!(result.has_results());
    }

    #[tokio::test]
    async fn test_auto_embed_query_with_noop_provider() {
        use membrain_core::traits::NoOpEmbeddingProvider;

        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        storage
            .store(create_memory("Test fact about embedding"))
            .await
            .unwrap();

        let provider = Arc::new(NoOpEmbeddingProvider::new(1536));
        let pipeline =
            RetrievalPipeline::new(storage, audit, config).with_embedding_provider(provider);

        // Query without explicit embedding -- should auto-embed
        let request = RetrievalRequest::new("Tell me about embedding");
        let result = pipeline.retrieve(request).await.unwrap();

        // The pipeline should still succeed even with zero-vector embeddings
        assert!(!result.was_gated);
    }

    #[tokio::test]
    async fn test_episodic_event_findable_by_fact_query() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = RetrievalConfig::default();

        // Store an episodic event.
        let memory = membrain_test_utils::episodic_event("meeting", "Alice met Bob for coffee");
        storage.store(memory).await.unwrap();

        let pipeline = RetrievalPipeline::new(storage, audit, config);

        // A fact-style query should still find the event because we no longer
        // use intent-suggested types as a hard filter.
        let request = RetrievalRequest::new("Who did Alice meet?");
        let result = pipeline.retrieve(request).await.unwrap();

        assert!(!result.was_gated);
        assert!(
            result.has_results(),
            "Event memories should be findable by fact-style queries"
        );
    }
}
