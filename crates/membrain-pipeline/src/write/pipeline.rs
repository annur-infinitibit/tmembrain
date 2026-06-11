//! Main write pipeline orchestrator

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use membrain_audit::{AuditEntry, AuditLog, DecisionContext, DecisionOutcome};
use membrain_conflict::ConflictResolver;
use membrain_core::config::WriteConfig;
use membrain_core::error::Result;
use membrain_core::memory::{
    FactMemory, Memory, MemoryCategory, MemoryCommon, SemanticContent, SemanticMemory,
};
use membrain_core::traits::{EmbeddingProvider, ExtractedFactType, MemoryExtractor, MemoryStorage};
use membrain_core::types::{
    AgentId, Confidence, Derivation, DerivationType, MemoryId, Provenance, Source,
};
use membrain_graph::bridge::GraphAugmentedRetrieval;

use super::budget::BudgetPolicy;
use super::novelty::NoveltyPolicy;
use super::policy::{PolicyResult, WritePolicy};
use super::redundancy::RedundancyPolicy;
use super::salience::SaliencePolicy;

/// Result of write pipeline processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteResult {
    /// Memory was successfully stored
    Stored {
        /// ID of the stored memory
        id: MemoryId,
        /// Duration of processing
        duration_ms: u64,
    },
    /// Memory was merged with an existing one
    Merged {
        /// Original memory ID (the one submitted)
        original_id: MemoryId,
        /// ID of the memory it was merged into
        merged_into: MemoryId,
    },
    /// Memory was rejected
    Rejected(RejectionReason),
}

impl WriteResult {
    /// Check if the write was successful (stored or merged)
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            WriteResult::Stored { .. } | WriteResult::Merged { .. }
        )
    }

    /// Get the memory ID if stored
    pub fn memory_id(&self) -> Option<MemoryId> {
        match self {
            WriteResult::Stored { id, .. } => Some(*id),
            WriteResult::Merged { merged_into, .. } => Some(*merged_into),
            WriteResult::Rejected(_) => None,
        }
    }
}

/// Reason for memory rejection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Low salience score
    LowSalience {
        score: f64,
        threshold: f64,
        details: Option<String>,
    },
    /// Low novelty (too similar to existing)
    LowNovelty {
        score: f64,
        threshold: f64,
        similar_to: Vec<MemoryId>,
    },
    /// Redundant with existing memory
    Redundant {
        duplicate_of: MemoryId,
        similarity: f64,
    },
    /// Budget exceeded
    BudgetExceeded {
        current: usize,
        limit: usize,
        memory_type: String,
    },
    /// Storage error
    StorageError(String),
    /// Custom rejection
    Custom(String),
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectionReason::LowSalience {
                score, threshold, ..
            } => {
                write!(f, "Low salience: {:.2} < {:.2}", score, threshold)
            }
            RejectionReason::LowNovelty {
                score, threshold, ..
            } => {
                write!(f, "Low novelty: {:.2} < {:.2}", score, threshold)
            }
            RejectionReason::Redundant {
                duplicate_of,
                similarity,
            } => {
                write!(
                    f,
                    "Redundant with {} (similarity: {:.2})",
                    duplicate_of, similarity
                )
            }
            RejectionReason::BudgetExceeded {
                current,
                limit,
                memory_type,
            } => {
                write!(
                    f,
                    "Budget exceeded for {}: {}/{}",
                    memory_type, current, limit
                )
            }
            RejectionReason::StorageError(e) => write!(f, "Storage error: {}", e),
            RejectionReason::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

/// The write pipeline that processes memories before storage
pub struct WritePipeline {
    /// Storage backend
    storage: Arc<dyn MemoryStorage>,
    /// Audit log
    audit: Arc<AuditLog>,
    /// Policies to apply
    policies: Vec<Box<dyn WritePolicy>>,
    /// Configuration
    config: WriteConfig,
    /// Optional graph bridge for adding nodes on store
    graph_bridge: Option<Arc<dyn GraphAugmentedRetrieval>>,
    /// Optional embedding provider for auto-embedding on write
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Optional LLM-based memory extractor for deriving facts from episodic memories
    memory_extractor: Option<Arc<dyn MemoryExtractor>>,
    /// Optional LLM-based conflict resolver for ADD/UPDATE/DELETE/NOOP classification
    conflict_resolver: Option<Arc<dyn ConflictResolver>>,
    /// Agent ID used as owner for extracted fact memories
    agent_id: Option<AgentId>,
}

impl WritePipeline {
    /// Create a new write pipeline
    pub fn new(storage: Arc<dyn MemoryStorage>, audit: Arc<AuditLog>, config: WriteConfig) -> Self {
        let policies: Vec<Box<dyn WritePolicy>> = vec![
            Box::new(SaliencePolicy::new(config.salience.clone())),
            Box::new(NoveltyPolicy::new(config.novelty.clone())),
            Box::new(RedundancyPolicy::new(config.redundancy.clone())),
            Box::new(BudgetPolicy::new(config.budget.clone())),
        ];

        Self {
            storage,
            audit,
            policies,
            config,
            graph_bridge: None,
            embedding_provider: None,
            memory_extractor: None,
            conflict_resolver: None,
            agent_id: None,
        }
    }

    /// Attach a graph bridge so stored memories are added to the graph
    pub fn with_graph(mut self, graph: Arc<dyn GraphAugmentedRetrieval>) -> Self {
        self.graph_bridge = Some(graph);
        self
    }

    /// Attach an embedding provider for automatic embedding generation on write.
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Attach a memory extractor for LLM-based fact extraction from episodic memories.
    pub fn with_memory_extractor(mut self, extractor: Arc<dyn MemoryExtractor>) -> Self {
        self.memory_extractor = Some(extractor);
        self
    }

    /// Attach a conflict resolver for LLM-based ADD/UPDATE/DELETE/NOOP classification.
    ///
    /// When attached, the pipeline uses the resolver to intelligently handle
    /// redundancy: instead of silently merging or rejecting, the LLM decides
    /// whether to add, update, delete (invalidate), or skip the new memory.
    pub fn with_conflict_resolver(mut self, resolver: Arc<dyn ConflictResolver>) -> Self {
        self.conflict_resolver = Some(resolver);
        self
    }

    /// Set the agent ID used as the owner for extracted fact memories.
    pub fn with_agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Process a memory through the write pipeline
    pub async fn process(&self, memory: Memory) -> Result<WriteResult> {
        let start = Instant::now();
        let memory_id = *memory.id();
        let memory_type = memory.memory_type();

        let mut context = DecisionContext::new();

        // Run through each policy
        for policy in &self.policies {
            if !policy.is_enabled() {
                continue;
            }

            let result = policy.evaluate(&memory, self.storage.as_ref()).await?;

            match &result {
                PolicyResult::Pass { score, details } => {
                    // Update context with scores
                    match policy.name() {
                        "salience" => context.salience_score = Some(*score),
                        "novelty" => context.novelty_score = Some(*score),
                        "redundancy" => context.max_similarity = Some(1.0 - score),
                        _ => {}
                    }
                    if let Some(d) = details {
                        context.notes = Some(d.clone());
                    }
                }
                PolicyResult::Reject { reason, score } => {
                    let rejection = self.create_rejection(policy.name(), *score, reason);

                    // Log rejection
                    self.audit.log_rejection(
                        AuditEntry::rejection(memory_id, reason)
                            .with_memory_type(memory_type)
                            .with_context(context.clone().with_policy(policy.name()))
                            .with_duration_ms(start.elapsed().as_millis() as u64),
                    );

                    return Ok(WriteResult::Rejected(rejection));
                }
                PolicyResult::Merge {
                    merge_with,
                    similarity,
                } => {
                    // If a conflict resolver is available, use LLM to decide
                    // whether to ADD, UPDATE, DELETE, or NOOP.
                    if let Some(ref resolver) = self.conflict_resolver {
                        if let Some(target_memory) = self.storage.get(merge_with).await? {
                            match resolver
                                .resolve(&memory, std::slice::from_ref(&target_memory))
                                .await
                            {
                                Ok(resolution) => {
                                    use membrain_conflict::ConflictDecision;
                                    match resolution.decision {
                                        ConflictDecision::Add => {
                                            // LLM says it's actually new -- continue to store
                                        }
                                        ConflictDecision::Update { merged_content, .. } => {
                                            let result = self
                                                .execute_update(
                                                    &target_memory,
                                                    &merged_content,
                                                    &memory,
                                                    &start,
                                                    &context,
                                                )
                                                .await;
                                            return result;
                                        }
                                        ConflictDecision::Delete { reason, .. } => {
                                            // Invalidate old memory and continue to store new
                                            self.invalidate_memory(merge_with, &memory).await;
                                            self.audit.log_storage(
                                                AuditEntry::merge(memory_id, *merge_with)
                                                    .with_memory_type(memory_type)
                                                    .with_context(
                                                        context
                                                            .clone()
                                                            .with_max_similarity(*similarity)
                                                            .with_policy("conflict_resolution")
                                                            .with_notes(format!(
                                                                "Superseded: {}",
                                                                reason
                                                            )),
                                                    )
                                                    .with_duration_ms(
                                                        start.elapsed().as_millis() as u64
                                                    ),
                                            );
                                            // Continue to store the new memory below
                                        }
                                        ConflictDecision::Noop { reason } => {
                                            self.audit.log_rejection(
                                                AuditEntry::rejection(memory_id, &reason)
                                                    .with_memory_type(memory_type)
                                                    .with_context(
                                                        context
                                                            .clone()
                                                            .with_policy("conflict_resolution"),
                                                    )
                                                    .with_duration_ms(
                                                        start.elapsed().as_millis() as u64
                                                    ),
                                            );
                                            return Ok(WriteResult::Rejected(
                                                RejectionReason::Custom(reason),
                                            ));
                                        }
                                    }
                                }
                                Err(error) => {
                                    // Conflict resolution failed -- fall through to
                                    // default merge behavior (non-fatal)
                                    tracing::warn!(
                                        ?error,
                                        "Conflict resolution failed, using default merge behavior"
                                    );
                                    self.audit.log_storage(
                                        AuditEntry::merge(memory_id, *merge_with)
                                            .with_memory_type(memory_type)
                                            .with_context(
                                                context
                                                    .clone()
                                                    .with_max_similarity(*similarity)
                                                    .with_policy(policy.name()),
                                            )
                                            .with_duration_ms(start.elapsed().as_millis() as u64),
                                    );
                                    return Ok(WriteResult::Merged {
                                        original_id: memory_id,
                                        merged_into: *merge_with,
                                    });
                                }
                            }
                        }
                    } else {
                        // No conflict resolver -- use default merge behavior
                        self.audit.log_storage(
                            AuditEntry::merge(memory_id, *merge_with)
                                .with_memory_type(memory_type)
                                .with_context(
                                    context
                                        .clone()
                                        .with_max_similarity(*similarity)
                                        .with_policy(policy.name()),
                                )
                                .with_duration_ms(start.elapsed().as_millis() as u64),
                        );
                        return Ok(WriteResult::Merged {
                            original_id: memory_id,
                            merged_into: *merge_with,
                        });
                    }
                }
                PolicyResult::Skipped { .. } => {
                    // Policy was skipped, continue
                }
            }
        }

        // Auto-embed if the memory has no embedding and a provider is available.
        // Failure is non-fatal: we warn and continue without an embedding.
        let memory = if memory.embedding().is_none() {
            if let Some(ref provider) = self.embedding_provider {
                let text = memory.text_content();
                match provider.embed(&text).await {
                    Ok(embedding) => {
                        let mut memory = memory;
                        memory.set_embedding(embedding);
                        memory
                    }
                    Err(error) => {
                        tracing::warn!(?error, "Auto-embedding failed, storing without embedding");
                        memory
                    }
                }
            } else {
                tracing::warn!(
                    "No embedding provider configured -- storing memory without vector embedding. \
                     Vector search will not work for this memory. \
                     Set embedding.api_key in your config to enable auto-embedding."
                );
                memory
            }
        } else {
            memory
        };

        // All policies passed, store the memory
        match self.storage.store(memory.clone()).await {
            Ok(id) => {
                let duration_ms = start.elapsed().as_millis() as u64;

                // Add node to graph (non-fatal on error)
                if let Some(ref graph) = self.graph_bridge {
                    if let Err(error) = graph.on_memory_stored(&memory) {
                        tracing::warn!(?error, "Failed to add memory to graph");
                    }
                }

                // Log successful storage
                self.audit.log_storage(
                    AuditEntry::storage_decision(id, memory_type, DecisionOutcome::Stored)
                        .with_context(context)
                        .with_duration_ms(duration_ms),
                );

                // Extract structured facts from episodic memories (non-fatal on error)
                if memory_type.category() == MemoryCategory::Episodic {
                    self.extract_and_store_facts(&memory, id).await;
                }

                Ok(WriteResult::Stored { id, duration_ms })
            }
            Err(e) => {
                let reason = RejectionReason::StorageError(e.to_string());

                self.audit.log_rejection(
                    AuditEntry::rejection(memory_id, e.to_string())
                        .with_memory_type(memory_type)
                        .with_duration_ms(start.elapsed().as_millis() as u64),
                );

                Ok(WriteResult::Rejected(reason))
            }
        }
    }

    /// Run LLM extraction on an episodic memory and store the resulting facts
    /// as `SemanticFact` memories. Failures are non-fatal -- we log a warning
    /// and continue.
    async fn extract_and_store_facts(&self, source_memory: &Memory, source_id: MemoryId) {
        let extractor = match self.memory_extractor {
            Some(ref e) => e,
            None => return,
        };

        if !self.config.extraction.enabled {
            return;
        }

        let text = source_memory.text_content();
        let extraction_result = match extractor.extract(&text).await {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    %source_id,
                    "Memory extraction failed, continuing without extracted facts"
                );
                return;
            }
        };

        if extraction_result.facts.is_empty() {
            tracing::debug!(%source_id, "No facts extracted from memory");
            return;
        }

        let agent_id = self
            .agent_id
            .unwrap_or_else(|| source_memory.common().agent_id);

        let fact_count = extraction_result.facts.len();
        let mut stored_count = 0usize;

        for extracted_fact in extraction_result.facts {
            // Build provenance linking back to the source memory
            let derivation = Derivation::new(vec![source_id], DerivationType::Extraction)
                .with_context(format!(
                    "Extracted by {} using {}",
                    extractor.name(),
                    extractor.model()
                ));
            let provenance = Provenance::new_derived(
                Source::agent_generated("memory_extraction"),
                vec![derivation],
                agent_id,
            );

            let confidence = Confidence::new(extracted_fact.confidence);
            let common = MemoryCommon::new(agent_id, provenance)
                .with_confidence(confidence)
                .with_tag("extracted")
                .with_metadata(
                    "source_memory_id".to_string(),
                    serde_json::Value::String(source_id.to_string()),
                )
                .with_metadata(
                    "extraction_type".to_string(),
                    serde_json::Value::String(format!("{:?}", extracted_fact.fact_type)),
                );

            let semantic_content = match extracted_fact.fact_type {
                ExtractedFactType::Preference => {
                    // For preferences, try to parse "X prefers/likes Y" structure
                    SemanticContent::Fact(FactMemory::new(&extracted_fact.content))
                }
                _ => SemanticContent::Fact(FactMemory::new(&extracted_fact.content)),
            };

            let mut fact_memory = Memory::Semantic(SemanticMemory {
                common,
                content: semantic_content,
            });

            // Auto-embed the extracted fact if a provider is available
            if let Some(ref provider) = self.embedding_provider {
                match provider.embed(&extracted_fact.content).await {
                    Ok(embedding) => fact_memory.set_embedding(embedding),
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            content = %extracted_fact.content,
                            "Failed to embed extracted fact"
                        );
                    }
                }
            }

            // Store directly to storage, bypassing policies (extracted facts should
            // not be rejected by novelty/salience/redundancy checks)
            match self.storage.store(fact_memory.clone()).await {
                Ok(fact_id) => {
                    stored_count += 1;

                    // Add to graph if configured
                    if let Some(ref graph) = self.graph_bridge {
                        if let Err(error) = graph.on_memory_stored(&fact_memory) {
                            tracing::warn!(?error, %fact_id, "Failed to add extracted fact to graph");
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        content = %extracted_fact.content,
                        "Failed to store extracted fact"
                    );
                }
            }
        }

        tracing::info!(
            %source_id,
            fact_count,
            stored_count,
            "Extracted and stored facts from episodic memory"
        );
    }

    /// Execute an UPDATE conflict resolution: merge content into the target memory.
    ///
    /// Updates the target memory's text content with the LLM-provided merged text,
    /// bumps the version, records a Merge derivation, and re-embeds if a provider
    /// is available.
    async fn execute_update(
        &self,
        target: &Memory,
        merged_content: &str,
        source: &Memory,
        start: &Instant,
        context: &DecisionContext,
    ) -> Result<WriteResult> {
        let target_id = *target.id();
        let source_id = *source.id();
        let memory_type = target.memory_type();

        // Clone and update the target memory with merged content
        let mut updated = target.clone();
        updated.set_text_content(merged_content);

        // Record derivation provenance
        let derivation = Derivation::new(vec![source_id], DerivationType::Merge)
            .with_context("Merged via LLM conflict resolution".to_string());
        updated.common_mut().provenance.derivations.push(derivation);

        // Bump version
        updated.common_mut().increment_version();

        // Re-embed with merged content if provider available
        if let Some(ref provider) = self.embedding_provider {
            match provider.embed(merged_content).await {
                Ok(embedding) => updated.set_embedding(embedding),
                Err(error) => {
                    tracing::warn!(?error, "Failed to re-embed merged memory");
                }
            }
        }

        // Store the updated memory
        let expected_version = target.common().version;
        if let Err(error) = self.storage.update(updated.clone(), expected_version).await {
            tracing::warn!(?error, "Failed to update target memory, storing as new");
            // Fall back to storing as new memory
            return Ok(WriteResult::Stored {
                id: source_id,
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Update graph if configured
        if let Some(ref graph) = self.graph_bridge {
            if let Err(error) = graph.on_memory_stored(&updated) {
                tracing::warn!(?error, "Failed to update memory in graph");
            }
        }

        self.audit.log_storage(
            AuditEntry::merge(source_id, target_id)
                .with_memory_type(memory_type)
                .with_context(context.clone().with_policy("conflict_resolution"))
                .with_duration_ms(start.elapsed().as_millis() as u64),
        );

        Ok(WriteResult::Merged {
            original_id: source_id,
            merged_into: target_id,
        })
    }

    /// Invalidate a memory by setting its valid_until to now.
    ///
    /// Used when conflict resolution determines that a new memory supersedes
    /// an existing one (DELETE decision).
    async fn invalidate_memory(&self, target_id: &MemoryId, new_memory: &Memory) {
        let target = match self.storage.get(target_id).await {
            Ok(Some(memory)) => memory,
            _ => return,
        };

        let mut updated = target.clone();
        updated.common_mut().invalidate();

        // Record that this memory was superseded
        let derivation = Derivation::new(vec![*new_memory.id()], DerivationType::Resolution)
            .with_context(
                "Invalidated by conflict resolution -- superseded by newer information".to_string(),
            );
        updated.common_mut().provenance.derivations.push(derivation);
        updated.common_mut().increment_version();

        let expected_version = target.common().version;
        if let Err(error) = self.storage.update(updated, expected_version).await {
            tracing::warn!(?error, %target_id, "Failed to invalidate superseded memory");
        }

        // Create Contradicts edge in graph between old and new
        if let Some(ref graph) = self.graph_bridge {
            if let Err(error) = graph.on_memory_deleted(target_id) {
                tracing::warn!(?error, "Failed to remove invalidated memory from graph");
            }
        }
    }

    fn create_rejection(&self, policy_name: &str, score: f64, reason: &str) -> RejectionReason {
        match policy_name {
            "salience" => RejectionReason::LowSalience {
                score,
                threshold: self.config.salience.threshold,
                details: Some(reason.to_string()),
            },
            "novelty" => RejectionReason::LowNovelty {
                score,
                threshold: self.config.novelty.threshold,
                similar_to: Vec::new(),
            },
            "redundancy" => RejectionReason::Custom(reason.to_string()),
            "budget" => RejectionReason::Custom(reason.to_string()),
            _ => RejectionReason::Custom(reason.to_string()),
        }
    }

    /// Process multiple memories
    pub async fn process_batch(&self, memories: Vec<Memory>) -> Vec<Result<WriteResult>> {
        let mut results = Vec::with_capacity(memories.len());
        for memory in memories {
            results.push(self.process(memory).await);
        }
        results
    }

    /// Get the storage backend
    pub fn storage(&self) -> &Arc<dyn MemoryStorage> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::config::AuditConfig;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{AgentId, Confidence, Provenance, Source};
    use membrain_storage::InMemoryStorage;

    fn create_memory(text: &str) -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new(text)),
        })
    }

    #[tokio::test]
    async fn test_successful_storage() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        let pipeline = WritePipeline::new(storage.clone(), audit, config);

        let memory = create_memory("The user prefers dark mode for better readability");
        let result = pipeline.process(memory).await.unwrap();

        assert!(result.is_success());
        assert!(result.memory_id().is_some());
        assert_eq!(storage.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_rejection_low_salience() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let mut config = WriteConfig::default();
        config.salience.threshold = 0.9; // Very high threshold

        let pipeline = WritePipeline::new(storage.clone(), audit, config);

        let memory = create_memory("ok"); // Very low salience content
        let result = pipeline.process(memory).await.unwrap();

        assert!(!result.is_success());
        assert!(matches!(
            result,
            WriteResult::Rejected(RejectionReason::LowSalience { .. })
        ));
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        let pipeline = WritePipeline::new(storage.clone(), audit, config);

        let memories = vec![
            create_memory("First fact about user preferences"),
            create_memory("Second fact about system configuration"),
            create_memory("Third fact about project settings"),
        ];

        let results = pipeline.process_batch(memories).await;

        let successes = results
            .iter()
            .filter(|r| r.as_ref().map(|w| w.is_success()).unwrap_or(false))
            .count();
        assert!(successes >= 2); // At least 2 should succeed
    }

    #[tokio::test]
    async fn test_write_pipeline_graph_hook() {
        use membrain_core::types::Embedding;
        use membrain_graph::{GraphBridge, GraphConfig, MemoryGraph};

        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        let graph_config = GraphConfig {
            embedding_dim: 4,
            hidden_dim: 4,
            ..Default::default()
        };
        let graph = Arc::new(MemoryGraph::new(graph_config));
        let bridge: Arc<dyn GraphAugmentedRetrieval> = Arc::new(GraphBridge::new(graph.clone()));

        let pipeline = WritePipeline::new(storage.clone(), audit, config).with_graph(bridge);

        // Create a memory with an embedding so the graph hook adds a node
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));
        let mut memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new("Graph integration test fact")),
        });
        memory.set_embedding(Embedding::new(vec![1.0, 0.0, 0.0, 0.0]));

        let result = pipeline.process(memory).await.unwrap();
        assert!(result.is_success());

        // Verify the graph has a node
        assert_eq!(graph.node_count(), 1);
    }

    #[tokio::test]
    async fn test_write_pipeline_without_graph() {
        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        // Pipeline without graph should still work normally
        let pipeline = WritePipeline::new(storage.clone(), audit, config);

        let memory = create_memory("Pipeline without graph works fine");
        let result = pipeline.process(memory).await.unwrap();
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_auto_embedding_with_noop_provider() {
        use membrain_core::traits::NoOpEmbeddingProvider;

        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        let provider: Arc<dyn EmbeddingProvider> = Arc::new(NoOpEmbeddingProvider::new(4));

        let pipeline =
            WritePipeline::new(storage.clone(), audit, config).with_embedding_provider(provider);

        // Memory without an embedding should get auto-embedded
        let memory = create_memory("Auto-embed this fact about programming");
        assert!(memory.embedding().is_none());

        let result = pipeline.process(memory).await.unwrap();
        assert!(result.is_success());

        // Verify the stored memory now has an embedding
        if let Some(id) = result.memory_id() {
            let stored = storage.get(&id).await.unwrap().unwrap();
            assert!(
                stored.embedding().is_some(),
                "Memory should have been auto-embedded"
            );
        }
    }

    #[tokio::test]
    async fn test_existing_embedding_not_overwritten() {
        use membrain_core::traits::NoOpEmbeddingProvider;

        let storage = Arc::new(InMemoryStorage::new());
        let audit = Arc::new(AuditLog::new(AuditConfig::default()));
        let config = WriteConfig::default();

        let provider: Arc<dyn EmbeddingProvider> = Arc::new(NoOpEmbeddingProvider::new(4));

        let pipeline =
            WritePipeline::new(storage.clone(), audit, config).with_embedding_provider(provider);

        // Memory with an existing embedding should NOT be re-embedded
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));
        let mut memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new("Already embedded fact")),
        });
        let original_embedding = membrain_core::types::Embedding::new(vec![1.0, 2.0, 3.0, 4.0]);
        memory.set_embedding(original_embedding.clone());

        let result = pipeline.process(memory).await.unwrap();
        assert!(result.is_success());

        // Verify the original embedding is preserved
        if let Some(id) = result.memory_id() {
            let stored = storage.get(&id).await.unwrap().unwrap();
            let emb = stored.embedding().unwrap();
            assert_eq!(emb.values(), &[1.0, 2.0, 3.0, 4.0]);
        }
    }
}
