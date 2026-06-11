//! Memory types for the Membrain system
//!
//! The memory system supports four primary memory types:
//! - **Episodic**: Conversations, events, observations (high-detail, fast decay)
//! - **Semantic**: Facts, preferences, concepts, entities (stable knowledge)
//! - **Procedural**: Workflows, skills, patterns (how to do things)
//! - **AgentState**: Goals, tasks, working memory (current context)

mod agent_state;
mod episodic;
mod procedural;
mod semantic;

pub use agent_state::{
    AgentStateContent, AgentStateMemory, Goal, GoalStatus, Task, TaskStatus, WorkingMemoryItem,
    WorkingMemoryType,
};
pub use episodic::{
    ConversationMemory, EpisodicContent, EpisodicMemory, EventMemory, Message, MessageRole,
    ObservationMemory,
};
pub use procedural::{
    CaseMemory, PatternMemory, PatternType, ProceduralContent, ProceduralMemory, SkillMemory,
    StepDefinition, WorkflowMemory,
};
pub use semantic::{
    ConceptMemory, EntityMemory, EntityType, FactMemory, PreferenceMemory, PreferenceStrength,
    SemanticContent, SemanticMemory,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{AgentId, Confidence, Embedding, MemoryId, Provenance, Version};

/// The core memory type for Membrain.
///
/// Represents all categories of LLM memory, organized into four types
/// that mirror cognitive science models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Memory {
    /// Episodic memories: conversations, events, observations
    Episodic(EpisodicMemory),
    /// Semantic memories: facts, preferences, concepts, entities
    Semantic(SemanticMemory),
    /// Procedural memories: workflows, skills, patterns
    Procedural(ProceduralMemory),
    /// Agent state: goals, tasks, working memory
    AgentState(AgentStateMemory),
}

impl Memory {
    /// Get the common fields for this memory
    pub fn common(&self) -> &MemoryCommon {
        match self {
            Memory::Episodic(m) => &m.common,
            Memory::Semantic(m) => &m.common,
            Memory::Procedural(m) => &m.common,
            Memory::AgentState(m) => &m.common,
        }
    }

    /// Get mutable reference to common fields
    pub fn common_mut(&mut self) -> &mut MemoryCommon {
        match self {
            Memory::Episodic(m) => &mut m.common,
            Memory::Semantic(m) => &mut m.common,
            Memory::Procedural(m) => &mut m.common,
            Memory::AgentState(m) => &mut m.common,
        }
    }

    /// Get the memory ID
    pub fn id(&self) -> &MemoryId {
        &self.common().id
    }

    /// Get the memory type as a string
    pub fn memory_type(&self) -> MemoryType {
        match self {
            Memory::Episodic(e) => match e.content {
                episodic::EpisodicContent::Conversation(_) => MemoryType::EpisodicConversation,
                episodic::EpisodicContent::Event(_) => MemoryType::EpisodicEvent,
                episodic::EpisodicContent::Observation(_) => MemoryType::EpisodicObservation,
            },
            Memory::Semantic(s) => match s.content {
                semantic::SemanticContent::Fact(_) => MemoryType::SemanticFact,
                semantic::SemanticContent::Preference(_) => MemoryType::SemanticPreference,
                semantic::SemanticContent::Concept(_) => MemoryType::SemanticConcept,
                semantic::SemanticContent::Entity(_) => MemoryType::SemanticEntity,
            },
            Memory::Procedural(p) => match p.content {
                procedural::ProceduralContent::Workflow(_) => MemoryType::ProceduralWorkflow,
                procedural::ProceduralContent::Skill(_) => MemoryType::ProceduralSkill,
                procedural::ProceduralContent::Pattern(_) => MemoryType::ProceduralPattern,
                procedural::ProceduralContent::Case(_) => MemoryType::ProceduralCase,
            },
            Memory::AgentState(a) => match a.content {
                agent_state::AgentStateContent::Goal(_) => MemoryType::AgentStateGoal,
                agent_state::AgentStateContent::Task(_) => MemoryType::AgentStateTask,
                agent_state::AgentStateContent::WorkingMemory(_) => {
                    MemoryType::AgentStateWorkingMemory
                }
            },
        }
    }

    /// Get the confidence score
    pub fn confidence(&self) -> &Confidence {
        &self.common().confidence
    }

    /// Get the embedding if present
    pub fn embedding(&self) -> Option<&Embedding> {
        self.common().embedding.as_ref()
    }

    /// Set the embedding
    pub fn set_embedding(&mut self, embedding: Embedding) {
        self.common_mut().embedding = Some(embedding);
    }

    /// Get a text representation for embedding/indexing
    pub fn text_content(&self) -> String {
        match self {
            Memory::Episodic(e) => e.text_content(),
            Memory::Semantic(s) => s.text_content(),
            Memory::Procedural(p) => p.text_content(),
            Memory::AgentState(a) => a.text_content(),
        }
    }

    /// Set the primary text content of this memory.
    ///
    /// For semantic facts, this updates the statement. For other types, this
    /// sets the primary text field (description, content, etc.).
    pub fn set_text_content(&mut self, content: &str) {
        match self {
            Memory::Semantic(s) => match &mut s.content {
                SemanticContent::Fact(f) => f.statement = content.to_string(),
                SemanticContent::Preference(p) => p.preference = content.to_string(),
                SemanticContent::Concept(c) => c.definition = content.to_string(),
                SemanticContent::Entity(e) => e.description = Some(content.to_string()),
            },
            Memory::Episodic(e) => match &mut e.content {
                episodic::EpisodicContent::Conversation(c) => c.summary = Some(content.to_string()),
                episodic::EpisodicContent::Event(ev) => ev.description = content.to_string(),
                episodic::EpisodicContent::Observation(o) => o.content = content.to_string(),
            },
            Memory::Procedural(p) => match &mut p.content {
                procedural::ProceduralContent::Workflow(w) => w.description = content.to_string(),
                procedural::ProceduralContent::Skill(s) => s.description = content.to_string(),
                procedural::ProceduralContent::Pattern(pat) => {
                    pat.description = content.to_string()
                }
                procedural::ProceduralContent::Case(c) => c.outcome = content.to_string(),
            },
            Memory::AgentState(a) => match &mut a.content {
                agent_state::AgentStateContent::Goal(g) => g.description = content.to_string(),
                agent_state::AgentStateContent::Task(t) => {
                    t.description = Some(content.to_string())
                }
                agent_state::AgentStateContent::WorkingMemory(w) => w.content = content.to_string(),
            },
        }
    }

    /// Serialize to MessagePack bytes
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack bytes
    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

/// Common fields shared by all memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCommon {
    /// Unique identifier for this memory
    pub id: MemoryId,
    /// Version number for optimistic concurrency
    pub version: Version,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Confidence,
    /// Provenance information
    pub provenance: Provenance,
    /// Vector embedding for semantic search
    pub embedding: Option<Embedding>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Agent that owns this memory
    pub agent_id: AgentId,
    /// When this fact became true in the real world (event time start).
    /// None means "since forever / unknown start".
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// When this fact stopped being true in the real world (event time end).
    /// None means "still valid / currently true".
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
}

impl MemoryCommon {
    /// Create new memory common fields
    pub fn new(agent_id: AgentId, provenance: Provenance) -> Self {
        Self {
            id: MemoryId::new(),
            version: 1,
            confidence: Confidence::DEFAULT,
            provenance,
            embedding: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
            agent_id,
            valid_from: None,
            valid_until: None,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Set metadata value
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set when this fact became true in the real world
    pub fn with_valid_from(mut self, valid_from: chrono::DateTime<chrono::Utc>) -> Self {
        self.valid_from = Some(valid_from);
        self
    }

    /// Set when this fact stopped being true in the real world
    pub fn with_valid_until(mut self, valid_until: chrono::DateTime<chrono::Utc>) -> Self {
        self.valid_until = Some(valid_until);
        self
    }

    /// Mark this memory as no longer valid (sets valid_until to now)
    pub fn invalidate(&mut self) {
        self.valid_until = Some(chrono::Utc::now());
    }

    /// Check if this memory is currently valid (valid_until is None or in the future)
    pub fn is_valid(&self) -> bool {
        self.valid_until
            .is_none_or(|until| until > chrono::Utc::now())
    }

    /// Check if this memory was valid at a specific point in time
    pub fn was_valid_at(&self, at: chrono::DateTime<chrono::Utc>) -> bool {
        let after_start = self.valid_from.is_none_or(|from| at >= from);
        let before_end = self.valid_until.is_none_or(|until| at < until);
        after_start && before_end
    }

    /// Increment version
    pub fn increment_version(&mut self) {
        self.version += 1;
    }

    /// Record access
    pub fn record_access(&mut self) {
        self.provenance.record_access();
    }
}

/// Enum for memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    // Episodic
    EpisodicConversation,
    EpisodicEvent,
    EpisodicObservation,
    // Semantic
    SemanticFact,
    SemanticPreference,
    SemanticConcept,
    SemanticEntity,
    // Procedural
    ProceduralWorkflow,
    ProceduralSkill,
    ProceduralPattern,
    ProceduralCase,
    // Agent State
    AgentStateGoal,
    AgentStateTask,
    AgentStateWorkingMemory,
}

impl MemoryType {
    /// Get the broad category (Episodic, Semantic, Procedural, AgentState)
    pub fn category(&self) -> MemoryCategory {
        match self {
            MemoryType::EpisodicConversation
            | MemoryType::EpisodicEvent
            | MemoryType::EpisodicObservation => MemoryCategory::Episodic,
            MemoryType::SemanticFact
            | MemoryType::SemanticPreference
            | MemoryType::SemanticConcept
            | MemoryType::SemanticEntity => MemoryCategory::Semantic,
            MemoryType::ProceduralWorkflow
            | MemoryType::ProceduralSkill
            | MemoryType::ProceduralPattern
            | MemoryType::ProceduralCase => MemoryCategory::Procedural,
            MemoryType::AgentStateGoal
            | MemoryType::AgentStateTask
            | MemoryType::AgentStateWorkingMemory => MemoryCategory::AgentState,
        }
    }

    /// Get all types in this category
    pub fn all_in_category(category: MemoryCategory) -> Vec<MemoryType> {
        match category {
            MemoryCategory::Episodic => vec![
                MemoryType::EpisodicConversation,
                MemoryType::EpisodicEvent,
                MemoryType::EpisodicObservation,
            ],
            MemoryCategory::Semantic => vec![
                MemoryType::SemanticFact,
                MemoryType::SemanticPreference,
                MemoryType::SemanticConcept,
                MemoryType::SemanticEntity,
            ],
            MemoryCategory::Procedural => vec![
                MemoryType::ProceduralWorkflow,
                MemoryType::ProceduralSkill,
                MemoryType::ProceduralPattern,
                MemoryType::ProceduralCase,
            ],
            MemoryCategory::AgentState => vec![
                MemoryType::AgentStateGoal,
                MemoryType::AgentStateTask,
                MemoryType::AgentStateWorkingMemory,
            ],
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MemoryType::EpisodicConversation => "episodic_conversation",
            MemoryType::EpisodicEvent => "episodic_event",
            MemoryType::EpisodicObservation => "episodic_observation",
            MemoryType::SemanticFact => "semantic_fact",
            MemoryType::SemanticPreference => "semantic_preference",
            MemoryType::SemanticConcept => "semantic_concept",
            MemoryType::SemanticEntity => "semantic_entity",
            MemoryType::ProceduralWorkflow => "procedural_workflow",
            MemoryType::ProceduralSkill => "procedural_skill",
            MemoryType::ProceduralPattern => "procedural_pattern",
            MemoryType::ProceduralCase => "procedural_case",
            MemoryType::AgentStateGoal => "agent_state_goal",
            MemoryType::AgentStateTask => "agent_state_task",
            MemoryType::AgentStateWorkingMemory => "agent_state_working_memory",
        };
        write!(f, "{}", s)
    }
}

/// Broad memory categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Episodic,
    Semantic,
    Procedural,
    AgentState,
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MemoryCategory::Episodic => "episodic",
            MemoryCategory::Semantic => "semantic",
            MemoryCategory::Procedural => "procedural",
            MemoryCategory::AgentState => "agent_state",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Source;

    #[test]
    fn memory_serialization_roundtrip() {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov)
            .with_confidence(Confidence::new(0.9))
            .with_tag("test");

        let fact = FactMemory {
            statement: "The sky is blue".to_string(),
            subject: Some("sky".to_string()),
            predicate: Some("is".to_string()),
            object: Some("blue".to_string()),
        };

        let memory = Memory::Semantic(SemanticMemory {
            common,
            content: semantic::SemanticContent::Fact(fact),
        });

        let bytes = memory.to_msgpack().unwrap();
        let restored = Memory::from_msgpack(&bytes).unwrap();

        assert_eq!(memory.id(), restored.id());
        assert_eq!(memory.memory_type(), restored.memory_type());
    }

    #[test]
    fn memory_type_categories() {
        assert_eq!(
            MemoryType::EpisodicConversation.category(),
            MemoryCategory::Episodic
        );
        assert_eq!(
            MemoryType::SemanticFact.category(),
            MemoryCategory::Semantic
        );
        assert_eq!(
            MemoryType::ProceduralWorkflow.category(),
            MemoryCategory::Procedural
        );
        assert_eq!(
            MemoryType::AgentStateGoal.category(),
            MemoryCategory::AgentState
        );
    }
}
