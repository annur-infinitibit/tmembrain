use membrain_core::types::Confidence;
use membrain_pipeline::RetrievalFilters;

/// Result of a store operation.
///
/// Contains success status, the new memory ID, and optional merge information.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StoreResult {
    pub success: bool,
    pub id: Option<String>,
    pub merged_with: Option<String>,
    pub rejection_reason: Option<String>,
    pub duration_ms: u64,
}

/// Results from a search query.
///
/// Contains matching memories with scores and timing information.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResults {
    pub memories: Vec<SearchResult>,
    pub was_gated: bool,
    pub duration_ms: u64,
}

/// A single search result
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: f64,
    pub memory_type: String,
    pub created_at: String,
}

/// Information about a memory
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryInfo {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub confidence: f64,
}

/// Storage statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageStatsResult {
    pub total_memories: usize,
    pub by_type: std::collections::HashMap<String, usize>,
    pub storage_bytes: u64,
    pub embeddings_count: usize,
    pub avg_confidence: f64,
    pub agent_count: usize,
}

/// Vector backend health check result
#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorBackendHealthResult {
    pub status: String,
    pub backend: String,
}

/// Vector backend capabilities
#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorBackendCapabilities {
    pub supports_metadata_filtering: bool,
    pub supports_hybrid_search: bool,
    pub supports_batch_operations: bool,
    pub max_dimension: usize,
}

/// Vector backend statistics result
#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorBackendStatsResult {
    pub backend: String,
    pub total_vectors: usize,
    pub capabilities: VectorBackendCapabilities,
}

/// JSON-deserializable search filters for FFI.
///
/// Use these to narrow search results by memory type, confidence, tags, or metadata.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SearchFiltersJson {
    /// Filter by memory types (e.g. ["semantic_fact", "episodic_event"])
    pub memory_types: Option<Vec<String>>,
    /// Minimum confidence score (0.0-1.0)
    pub min_confidence: Option<f64>,
    /// Filter by tags (any match)
    pub tags: Option<Vec<String>>,
    /// Filter by agent ID (UUID string)
    pub agent_id: Option<String>,
    /// Filter by metadata key-value pairs (all must match)
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// Optional query embedding vector for semantic search
    pub embedding: Option<Vec<f32>>,
}

impl SearchFiltersJson {
    pub(crate) fn into_retrieval_filters(self) -> RetrievalFilters {
        use membrain_core::memory::MemoryType;

        let memory_types = self.memory_types.map(|types| {
            types
                .iter()
                .filter_map(|t| match t.as_str() {
                    "episodic_conversation" => Some(MemoryType::EpisodicConversation),
                    "episodic_event" => Some(MemoryType::EpisodicEvent),
                    "episodic_observation" => Some(MemoryType::EpisodicObservation),
                    "semantic_fact" => Some(MemoryType::SemanticFact),
                    "semantic_preference" => Some(MemoryType::SemanticPreference),
                    "semantic_concept" => Some(MemoryType::SemanticConcept),
                    "semantic_entity" => Some(MemoryType::SemanticEntity),
                    "procedural_workflow" => Some(MemoryType::ProceduralWorkflow),
                    "procedural_skill" => Some(MemoryType::ProceduralSkill),
                    "procedural_pattern" => Some(MemoryType::ProceduralPattern),
                    "procedural_case" => Some(MemoryType::ProceduralCase),
                    "agent_state_goal" => Some(MemoryType::AgentStateGoal),
                    "agent_state_task" => Some(MemoryType::AgentStateTask),
                    "agent_state_working_memory" => Some(MemoryType::AgentStateWorkingMemory),
                    _ => None,
                })
                .collect()
        });

        let min_confidence = self.min_confidence.map(Confidence::new);

        let agent_id = self
            .agent_id
            .and_then(|id_str| id_str.parse::<membrain_core::types::AgentId>().ok());

        RetrievalFilters {
            memory_types,
            min_confidence,
            tags: self.tags,
            agent_id,
            exclude_ids: None,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Graph types for JSON serialization
// ---------------------------------------------------------------------------

/// JSON-serializable graph query result
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphQueryResultJson {
    pub nodes: Vec<GraphScoredNodeJson>,
    pub traversed_edges: Vec<GraphTraversalStepJson>,
    pub hops_performed: usize,
    pub nodes_visited: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphScoredNodeJson {
    pub memory_id: String,
    pub score: f32,
    pub hop_distance: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphTraversalStepJson {
    pub from: String,
    pub to: String,
    pub edge_weight: f32,
    pub attention_score: f32,
    pub hop: usize,
}

/// JSON-serializable pruning result
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphPruningResultJson {
    pub edges_removed: usize,
    pub nodes_removed: usize,
    pub edges_remaining: usize,
    pub nodes_remaining: usize,
}

/// JSON-serializable graph info
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphInfoJson {
    pub node_count: usize,
    pub edge_count: usize,
}

impl From<membrain_graph::GraphQueryResult> for GraphQueryResultJson {
    fn from(r: membrain_graph::GraphQueryResult) -> Self {
        Self {
            nodes: r
                .nodes
                .into_iter()
                .map(|n| GraphScoredNodeJson {
                    memory_id: n.memory_id.to_string(),
                    score: n.score,
                    hop_distance: n.hop_distance,
                })
                .collect(),
            traversed_edges: r
                .traversed_edges
                .into_iter()
                .map(|s| GraphTraversalStepJson {
                    from: s.from.to_string(),
                    to: s.to.to_string(),
                    edge_weight: s.edge_weight,
                    attention_score: s.attention_score,
                    hop: s.hop,
                })
                .collect(),
            hops_performed: r.hops_performed,
            nodes_visited: r.nodes_visited,
        }
    }
}

impl From<membrain_graph::PruningResult> for GraphPruningResultJson {
    fn from(r: membrain_graph::PruningResult) -> Self {
        Self {
            edges_removed: r.edges_removed,
            nodes_removed: r.nodes_removed,
            edges_remaining: r.edges_remaining,
            nodes_remaining: r.nodes_remaining,
        }
    }
}
