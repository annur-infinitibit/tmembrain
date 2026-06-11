use std::sync::Arc;

use membrain_audit::AuditLog;
use membrain_core::config::Config;
use membrain_core::error::Result;
use membrain_core::memory::{
    AgentStateContent, AgentStateMemory, CaseMemory, ConceptMemory, EntityMemory, EntityType,
    EpisodicContent, EpisodicMemory, EventMemory, FactMemory, Goal, Memory, MemoryCommon,
    ObservationMemory, PatternMemory, PatternType, PreferenceMemory, PreferenceStrength,
    ProceduralContent, ProceduralMemory, SemanticContent, SemanticMemory, SkillMemory, Task,
    WorkflowMemory,
};
use membrain_core::traits::{
    infer_embedding_dimension, EmbeddingProvider, MemoryExtractor, MemoryStorage,
    OpenAiEmbeddingProvider, OpenAiMemoryExtractor,
};
use membrain_core::types::{AgentId, Confidence, Embedding, MemoryId, Provenance, Source};
use membrain_conflict::OpenAiConflictResolver;
use membrain_graph::{GraphBridge, GraphConfig, MemoryGraph};
use membrain_pipeline::{RetrievalPipeline, RetrievalRequest, WritePipeline, WriteResult};

use crate::types::{
    MemoryInfo, SearchFiltersJson, SearchResult, SearchResults, StorageStatsResult, StoreResult,
    VectorBackendCapabilities, VectorBackendHealthResult, VectorBackendStatsResult,
};

/// Core client for Membrain, usable from FFI.
///
/// Provides the main API for storing and searching LLM memories.
/// Used directly from Rust, or via the C ABI from Python/JavaScript.
///
/// All public methods are `async` — callers choose how to drive them:
///   - **PyO3 (Python):** `pyo3-async-runtimes` bridges each future into a
///     Python coroutine.
///   - **C ABI (Node.js/JavaScript):** A `thread_local!` Tokio runtime in the
///     C API layer calls `block_on` so the extern "C" signatures stay
///     synchronous.
///   - **Rust consumers:** Just `.await` directly.
///
/// # Examples
///
/// **Rust:**
/// ```no_run
/// use membrain_ffi::MembrainClient;
///
/// # async fn example() -> membrain_core::error::Result<()> {
/// let client = MembrainClient::new().await?;
/// let result = client.store_fact("User prefers dark mode", 0.9).await?;
/// let results = client.search("dark mode", 10).await?;
/// for memory in &results.memories {
///     println!("{}: {}", memory.content, memory.score);
/// }
/// # Ok(())
/// # }
/// ```
///
/// **Python:**
/// ```python
/// from membrain import MembrainClient
///
/// client = MembrainClient()
/// client.store_fact("User prefers dark mode", confidence=0.9)
/// results = client.search("dark mode", limit=10)
/// for memory in results.memories:
///     print(f"{memory.content}: {memory.score}")
/// ```
///
/// **JavaScript:**
/// ```javascript
/// const { MembrainClient } = require("membrain");
///
/// const client = new MembrainClient();
/// client.storeFact("User prefers dark mode", 0.9);
/// const results = client.search("dark mode", 10);
/// results.memories.forEach(m => console.log(`${m.content}: ${m.score}`));
/// ```
pub struct MembrainClient {
    storage: Arc<dyn MemoryStorage>,
    write_pipeline: WritePipeline,
    retrieval_pipeline: RetrievalPipeline,
    agent_id: AgentId,
    graph_bridge: Option<Arc<GraphBridge>>,
}

impl MembrainClient {
    /// Create a new client with default configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use membrain_ffi::MembrainClient;
    ///
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// let client = MembrainClient::new().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new() -> Result<Self> {
        // Honour MEMBRAIN_* env vars so callers can redirect storage/backend
        // without constructing a Config explicitly.
        let config = Config::from_env().unwrap_or_else(|_| Config::default());
        Self::with_config(config).await
    }

    /// Create a new client with custom configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use membrain_ffi::MembrainClient;
    /// use membrain_core::config::Config;
    ///
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// let config = Config::default();
    /// let client = MembrainClient::with_config(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_config(config: Config) -> Result<Self> {
        let embed_dim = config
            .embedding
            .dimension
            .unwrap_or(infer_embedding_dimension(&config.embedding.model));

        let storage: Arc<dyn MemoryStorage> = membrain_storage::create_storage(
            &config.storage,
            Some(&config.retrieval),
            Some(embed_dim),
        )
        .await?;

        let audit = Arc::new(AuditLog::new(config.audit.clone()));

        // Create graph bridge for graph-augmented retrieval.
        // Use the explicit dimension from config if provided, otherwise
        // infer from the model name.
        let embedding_dim = embed_dim;
        let graph_config = GraphConfig {
            embedding_dim,
            ..Default::default()
        };
        let graph = Arc::new(MemoryGraph::new(graph_config));
        let graph_bridge = Arc::new(GraphBridge::new(graph));

        // Create embedding provider for OpenAI-compatible APIs (OpenAI, Ollama, etc.)
        let embedding_provider: Option<Arc<dyn EmbeddingProvider>> =
            match config.embedding.provider.as_str() {
                // OpenAI requires an API key; Ollama and other local providers do not
                "openai" if config.embedding.api_key.is_some() => {
                    Self::try_create_embedding_provider(&config.embedding)
                }
                "ollama" => {
                    let mut embedding_config = config.embedding.clone();
                    if embedding_config.base_url.is_none() {
                        embedding_config.base_url = Some("http://localhost:11434/v1".to_string());
                    }
                    Self::try_create_embedding_provider(&embedding_config)
                }
                _ => None,
            };

        // Create memory extractor if extraction is enabled
        let memory_extractor: Option<Arc<dyn MemoryExtractor>> =
            if config.write.extraction.enabled {
                let mut extraction_config = config.write.extraction.clone();

                // Fall back to embedding API key/base_url if not set on extraction
                if extraction_config.api_key.is_none() {
                    extraction_config.api_key = config.embedding.api_key.clone();
                }
                if extraction_config.base_url.is_none() {
                    extraction_config.base_url = config.embedding.base_url.clone();
                }
                // Default Ollama base_url for local providers
                if extraction_config.provider == "ollama" && extraction_config.base_url.is_none() {
                    extraction_config.base_url = Some("http://localhost:11434/v1".to_string());
                }

                match OpenAiMemoryExtractor::from_config(&extraction_config) {
                    Ok(extractor) => Some(Arc::new(extractor)),
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            "Failed to create memory extractor, extraction disabled"
                        );
                        None
                    }
                }
            } else {
                None
            };

        // Create conflict resolver if enabled
        let conflict_resolver: Option<Arc<dyn membrain_conflict::ConflictResolver>> =
            if config.write.conflict_resolution.enabled {
                let mut conflict_config = config.write.conflict_resolution.clone();

                // Fall back to embedding API key/base_url if not set on conflict resolution
                if conflict_config.api_key.is_none() {
                    conflict_config.api_key = config.embedding.api_key.clone();
                }
                if conflict_config.base_url.is_none() {
                    conflict_config.base_url = config.embedding.base_url.clone();
                }
                // Default Ollama base_url for local providers
                if conflict_config.provider == "ollama" && conflict_config.base_url.is_none() {
                    conflict_config.base_url = Some("http://localhost:11434/v1".to_string());
                }

                match OpenAiConflictResolver::from_config(&conflict_config) {
                    Ok(resolver) => Some(Arc::new(resolver)),
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            "Failed to create conflict resolver, conflict resolution disabled"
                        );
                        None
                    }
                }
            } else {
                None
            };

        let agent_id = AgentId::new();

        let mut write_pipeline = WritePipeline::new(storage.clone(), audit.clone(), config.write.clone())
            .with_graph(graph_bridge.clone())
            .with_agent_id(agent_id);
        let mut retrieval_pipeline = RetrievalPipeline::new(storage.clone(), audit, config.retrieval.clone())
            .with_graph(graph_bridge.clone());

        if let Some(ref provider) = embedding_provider {
            write_pipeline = write_pipeline.with_embedding_provider(provider.clone());
            retrieval_pipeline = retrieval_pipeline.with_embedding_provider(provider.clone());
        }

        if let Some(ref extractor) = memory_extractor {
            write_pipeline = write_pipeline.with_memory_extractor(extractor.clone());
        }

        if let Some(ref resolver) = conflict_resolver {
            write_pipeline = write_pipeline.with_conflict_resolver(resolver.clone());
        }

        Ok(Self {
            storage,
            write_pipeline,
            retrieval_pipeline,
            agent_id,
            graph_bridge: Some(graph_bridge),
        })
    }

    /// Store a fact in LLM memory.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use membrain_ffi::MembrainClient;
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// # let client = MembrainClient::new().await?;
    /// let result = client.store_fact("Rust is a systems language", 0.95).await?;
    /// assert!(result.success);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_fact(&self, statement: &str, confidence: f64) -> Result<StoreResult> {
        let memory = self.create_fact_memory(statement, confidence, None);
        self.store_memory(memory).await
    }

    /// Store a preference
    pub async fn store_preference(
        &self,
        holder: &str,
        subject: &str,
        preference: &str,
        strength: &str,
    ) -> Result<StoreResult> {
        let pref_strength = match strength.to_lowercase().as_str() {
            "weak" => PreferenceStrength::Weak,
            "moderate" => PreferenceStrength::Moderate,
            "strong" => PreferenceStrength::Strong,
            "absolute" => PreferenceStrength::Absolute,
            _ => PreferenceStrength::Moderate,
        };

        let pref = PreferenceMemory::new(holder, subject, preference).with_strength(pref_strength);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Preference(pref),
        });

        self.store_memory(memory).await
    }

    /// Store an event
    pub async fn store_event(&self, event_type: &str, description: &str) -> Result<StoreResult> {
        let event = EventMemory::new(event_type, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let memory = Memory::Episodic(EpisodicMemory {
            common,
            content: EpisodicContent::Event(event),
        });

        self.store_memory(memory).await
    }

    /// Store an observation
    pub async fn store_observation(&self, content: &str) -> Result<StoreResult> {
        let observation = ObservationMemory::new(content);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.6));

        let memory = Memory::Episodic(EpisodicMemory {
            common,
            content: EpisodicContent::Observation(observation),
        });

        self.store_memory(memory).await
    }

    /// Store a concept
    pub async fn store_concept(&self, name: &str, definition: &str) -> Result<StoreResult> {
        let concept = ConceptMemory::new(name, definition);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Concept(concept),
        });

        self.store_memory(memory).await
    }

    /// Store an entity
    pub async fn store_entity(&self, name: &str, entity_type: &str) -> Result<StoreResult> {
        let etype = match entity_type.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "organization" => EntityType::Organization,
            "place" => EntityType::Place,
            "product" => EntityType::Product,
            "event" => EntityType::Event,
            "concept" => EntityType::Concept,
            "document" => EntityType::Document,
            "technology" => EntityType::Technology,
            _ => EntityType::Other,
        };

        let entity = EntityMemory::new(name, etype);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Entity(entity),
        });

        self.store_memory(memory).await
    }

    /// Store a workflow
    pub async fn store_workflow(&self, name: &str, description: &str) -> Result<StoreResult> {
        let workflow = WorkflowMemory::new(name, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Workflow(workflow),
        });

        self.store_memory(memory).await
    }

    /// Store a skill
    pub async fn store_skill(&self, name: &str, description: &str) -> Result<StoreResult> {
        let skill = SkillMemory::new(name, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Skill(skill),
        });

        self.store_memory(memory).await
    }

    /// Store a pattern
    pub async fn store_pattern(&self, name: &str, description: &str, pattern_type: &str) -> Result<StoreResult> {
        let ptype = match pattern_type.to_lowercase().as_str() {
            "user_behavior" | "userbehavior" => PatternType::UserBehavior,
            "conversation" => PatternType::Conversation,
            "error" => PatternType::Error,
            "success" => PatternType::Success,
            "temporal" => PatternType::Temporal,
            "contextual" => PatternType::Contextual,
            _ => PatternType::Other,
        };

        let pattern = PatternMemory::new(name, description, ptype);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Pattern(pattern),
        });

        self.store_memory(memory).await
    }

    /// Store a case (experience for case-based reasoning)
    pub async fn store_case(
        &self,
        problem: &str,
        plan: &str,
        outcome: &str,
        reward: f64,
    ) -> Result<StoreResult> {
        let case = CaseMemory::new(problem, plan, outcome, reward);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov)
            .with_confidence(Confidence::new(reward.clamp(0.0, 1.0)));

        let memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Case(case),
        });

        self.store_memory(memory).await
    }

    /// Store a goal
    pub async fn store_goal(&self, description: &str) -> Result<StoreResult> {
        let goal = Goal::new(description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let memory = Memory::AgentState(AgentStateMemory {
            common,
            content: AgentStateContent::Goal(goal),
        });

        self.store_memory(memory).await
    }

    /// Store a task
    pub async fn store_task(&self, title: &str) -> Result<StoreResult> {
        let task = Task::new(title);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let memory = Memory::AgentState(AgentStateMemory {
            common,
            content: AgentStateContent::Task(task),
        });

        self.store_memory(memory).await
    }

    /// Store a fact with optional embedding and metadata
    pub async fn store_fact_with_embedding(
        &self,
        statement: &str,
        confidence: f64,
        embedding: Option<Embedding>,
        metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<StoreResult> {
        let mut memory = self.create_fact_memory(statement, confidence, metadata);
        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }
        self.store_memory(memory).await
    }

    /// Store a preference with optional embedding
    pub async fn store_preference_with_embedding(
        &self,
        holder: &str,
        subject: &str,
        preference: &str,
        strength: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let pref_strength = match strength.to_lowercase().as_str() {
            "weak" => PreferenceStrength::Weak,
            "moderate" => PreferenceStrength::Moderate,
            "strong" => PreferenceStrength::Strong,
            "absolute" => PreferenceStrength::Absolute,
            _ => PreferenceStrength::Moderate,
        };

        let pref = PreferenceMemory::new(holder, subject, preference).with_strength(pref_strength);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let mut memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Preference(pref),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store an event with optional embedding
    pub async fn store_event_with_embedding(
        &self,
        event_type: &str,
        description: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let event = EventMemory::new(event_type, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let mut memory = Memory::Episodic(EpisodicMemory {
            common,
            content: EpisodicContent::Event(event),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store an observation with optional embedding
    pub async fn store_observation_with_embedding(
        &self,
        content: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let observation = ObservationMemory::new(content);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.6));

        let mut memory = Memory::Episodic(EpisodicMemory {
            common,
            content: EpisodicContent::Observation(observation),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a concept with optional embedding
    pub async fn store_concept_with_embedding(
        &self,
        name: &str,
        definition: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let concept = ConceptMemory::new(name, definition);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let mut memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Concept(concept),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store an entity with optional embedding
    pub async fn store_entity_with_embedding(
        &self,
        name: &str,
        entity_type: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let etype = match entity_type.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "organization" => EntityType::Organization,
            "place" => EntityType::Place,
            "product" => EntityType::Product,
            "event" => EntityType::Event,
            "concept" => EntityType::Concept,
            "document" => EntityType::Document,
            "technology" => EntityType::Technology,
            _ => EntityType::Other,
        };

        let entity = EntityMemory::new(name, etype);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let mut memory = Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Entity(entity),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a workflow with optional embedding
    pub async fn store_workflow_with_embedding(
        &self,
        name: &str,
        description: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let workflow = WorkflowMemory::new(name, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let mut memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Workflow(workflow),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a skill with optional embedding
    pub async fn store_skill_with_embedding(
        &self,
        name: &str,
        description: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let skill = SkillMemory::new(name, description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let mut memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Skill(skill),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a pattern with optional embedding
    pub async fn store_pattern_with_embedding(
        &self,
        name: &str,
        description: &str,
        pattern_type: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let ptype = match pattern_type.to_lowercase().as_str() {
            "user_behavior" | "userbehavior" => PatternType::UserBehavior,
            "conversation" => PatternType::Conversation,
            "error" => PatternType::Error,
            "success" => PatternType::Success,
            "temporal" => PatternType::Temporal,
            "contextual" => PatternType::Contextual,
            _ => PatternType::Other,
        };

        let pattern = PatternMemory::new(name, description, ptype);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.7));

        let mut memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Pattern(pattern),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a case with optional embedding
    pub async fn store_case_with_embedding(
        &self,
        problem: &str,
        plan: &str,
        outcome: &str,
        reward: f64,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let case = CaseMemory::new(problem, plan, outcome, reward);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let mut memory = Memory::Procedural(ProceduralMemory {
            common,
            content: ProceduralContent::Case(case),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a goal with optional embedding
    pub async fn store_goal_with_embedding(
        &self,
        description: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let goal = Goal::new(description);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.9));

        let mut memory = Memory::AgentState(AgentStateMemory {
            common,
            content: AgentStateContent::Goal(goal),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Store a task with optional embedding
    pub async fn store_task_with_embedding(
        &self,
        title: &str,
        embedding: Option<Embedding>,
    ) -> Result<StoreResult> {
        let task = Task::new(title);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let common = MemoryCommon::new(self.agent_id, prov).with_confidence(Confidence::new(0.8));

        let mut memory = Memory::AgentState(AgentStateMemory {
            common,
            content: AgentStateContent::Task(task),
        });

        if let Some(emb) = embedding {
            memory.set_embedding(emb);
        }

        self.store_memory(memory).await
    }

    /// Search for memories with optional filters.
    ///
    /// Filters can narrow results by memory type, confidence, tags, or metadata.
    /// Intent detection is enabled for scoring benefits; gating is disabled
    /// when explicit filters are provided.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use membrain_ffi::MembrainClient;
    /// use membrain_ffi::SearchFiltersJson;
    ///
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// # let client = MembrainClient::new().await?;
    /// let filters = SearchFiltersJson {
    ///     memory_types: Some(vec!["semantic_fact".to_string()]),
    ///     min_confidence: Some(0.8),
    ///     tags: None,
    ///     agent_id: None,
    ///     metadata: None,
    ///     embedding: None,
    /// };
    /// let results = client.search_with_filters("dark mode", 10, Some(filters)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_with_filters(
        &self,
        query: &str,
        limit: usize,
        filters: Option<SearchFiltersJson>,
    ) -> Result<SearchResults> {
        // Intent detection stays ON for adaptive scoring, but gating is disabled
        // when explicit filters are provided (caller knows what they want).
        let mut request = RetrievalRequest::new(query)
            .with_limit(limit)
            .without_gating();

        if let Some(filters_json) = filters {
            let embedding = filters_json.embedding.clone().map(Embedding::new);
            request.filters = filters_json.into_retrieval_filters();

            if let Some(emb) = embedding {
                request = request.with_embedding(emb);
            }
        }

        self.execute_search(request).await
    }

    /// Search for memories matching a natural language query.
    ///
    /// Uses the full adaptive pipeline: intent detection, gating, temporal
    /// filtering, and adaptive scoring are all enabled.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use membrain_ffi::MembrainClient;
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// # let client = MembrainClient::new().await?;
    /// let results = client.search("programming languages", 5).await?;
    /// for memory in &results.memories {
    ///     println!("{} (score: {})", memory.content, memory.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search(&self, query: &str, limit: usize) -> Result<SearchResults> {
        let request = RetrievalRequest::new(query).with_limit(limit);
        self.execute_search(request).await
    }

    /// Shared search execution used by both `search()` and `search_with_filters()`.
    async fn execute_search(&self, request: RetrievalRequest) -> Result<SearchResults> {
        let result = self.retrieval_pipeline.retrieve(request).await?;

        Ok(SearchResults {
            memories: result
                .memories
                .into_iter()
                .map(|m| {
                    let created_at = m.memory.common().provenance.created_at.to_rfc3339();
                    SearchResult {
                        id: m.id.to_string(),
                        content: m.text_content,
                        score: m.score,
                        memory_type: m.memory.memory_type().to_string(),
                        created_at,
                    }
                })
                .collect(),
            was_gated: result.was_gated,
            duration_ms: result.duration_ms,
        })
    }

    /// Get a memory by its unique ID.
    ///
    /// Returns `None` if the memory does not exist.
    pub async fn get(&self, id: &str) -> Result<Option<MemoryInfo>> {
        let memory_id: MemoryId = id.parse().map_err(|e| membrain_core::error::Error::Deserialization(format!("{}", e)))?;

        let memory = self.storage.get(&memory_id).await?;

        Ok(memory.map(|m| MemoryInfo {
            id: m.id().to_string(),
            content: m.text_content(),
            memory_type: m.memory_type().to_string(),
            confidence: m.confidence().value(),
        }))
    }

    /// Delete a memory by its unique ID.
    ///
    /// Returns `true` if the memory was deleted.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let memory_id: MemoryId = id.parse().map_err(|e| membrain_core::error::Error::Deserialization(format!("{}", e)))?;

        self.storage.delete(&memory_id).await
    }

    /// Get memory count
    pub async fn count(&self) -> Result<usize> {
        self.storage.count(None).await
    }

    /// Get the graph bridge, if configured
    pub fn graph_bridge(&self) -> Option<&GraphBridge> {
        self.graph_bridge.as_deref()
    }

    /// Check vector backend health status
    pub async fn vector_backend_health(&self) -> Result<VectorBackendHealthResult> {
        let result = self.storage.health_check().await;

        match result {
            Ok(()) => Ok(VectorBackendHealthResult {
                status: "healthy".to_string(),
                backend: "membrain".to_string(),
            }),
            Err(e) => Ok(VectorBackendHealthResult {
                status: format!("unhealthy: {}", e),
                backend: "membrain".to_string(),
            }),
        }
    }

    /// Get vector backend capabilities and statistics
    pub async fn vector_backend_stats(&self) -> Result<VectorBackendStatsResult> {
        let stats = self.storage.stats().await?;

        Ok(VectorBackendStatsResult {
            backend: "membrain".to_string(),
            total_vectors: stats.embeddings_count,
            capabilities: VectorBackendCapabilities {
                supports_metadata_filtering: true,
                supports_hybrid_search: true,
                supports_batch_operations: true,
                max_dimension: 4096,
            },
        })
    }

    /// Get storage statistics including memory counts and sizes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use membrain_ffi::MembrainClient;
    /// # async fn example() -> membrain_core::error::Result<()> {
    /// # let client = MembrainClient::new().await?;
    /// let stats = client.stats().await?;
    /// println!("Total memories: {}", stats.total_memories);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stats(&self) -> Result<StorageStatsResult> {
        let stats = self.storage.stats().await?;

        let by_type: std::collections::HashMap<String, usize> = stats
            .by_type
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        Ok(StorageStatsResult {
            total_memories: stats.total_memories,
            by_type,
            storage_bytes: stats.storage_bytes,
            embeddings_count: stats.embeddings_count,
            avg_confidence: stats.avg_confidence,
            agent_count: stats.agent_count,
        })
    }

    fn try_create_embedding_provider(
        embedding_config: &membrain_core::traits::EmbeddingConfig,
    ) -> Option<Arc<dyn EmbeddingProvider>> {
        match OpenAiEmbeddingProvider::from_config(embedding_config) {
            Ok(provider) => Some(Arc::new(provider)),
            Err(error) => {
                tracing::warn!(?error, "Failed to create embedding provider, auto-embedding disabled");
                None
            }
        }
    }

    fn create_fact_memory(
        &self,
        statement: &str,
        confidence: f64,
        metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> Memory {
        let fact = FactMemory::new(statement);
        let prov = Provenance::new_direct(Source::user_input("api"), self.agent_id);
        let mut common = MemoryCommon::new(self.agent_id, prov)
            .with_confidence(Confidence::new(confidence));

        if let Some(meta) = metadata {
            for (key, value) in meta {
                common = common.with_metadata(key, value);
            }
        }

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(fact),
        })
    }

    async fn store_memory(&self, memory: Memory) -> Result<StoreResult> {
        let result = self.write_pipeline.process(memory).await?;

        match result {
            WriteResult::Stored { id, duration_ms } => Ok(StoreResult {
                success: true,
                id: Some(id.to_string()),
                merged_with: None,
                rejection_reason: None,
                duration_ms,
            }),
            WriteResult::Merged { original_id, merged_into } => Ok(StoreResult {
                success: true,
                id: Some(original_id.to_string()),
                merged_with: Some(merged_into.to_string()),
                rejection_reason: None,
                duration_ms: 0,
            }),
            WriteResult::Rejected(reason) => Ok(StoreResult {
                success: false,
                id: None,
                merged_with: None,
                rejection_reason: Some(reason.to_string()),
                duration_ms: 0,
            }),
        }
    }
}
