//! JSON + MessagePack roundtrip tests for every `Memory` variant.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use membrain_core::memory::{
    AgentStateContent, AgentStateMemory, ConceptMemory, ConversationMemory, EntityMemory,
    EntityType, EpisodicContent, EpisodicMemory, EventMemory, FactMemory, Goal, Memory,
    MemoryCommon, Message, MessageRole, ObservationMemory, PreferenceMemory, PreferenceStrength,
    ProceduralContent, ProceduralMemory, SemanticContent, SemanticMemory, SkillMemory, Task,
    WorkflowMemory, WorkingMemoryItem, WorkingMemoryType,
};
use membrain_core::types::{AgentId, Confidence, Provenance, SessionId, Source};

fn common() -> MemoryCommon {
    let agent_id = AgentId::new();
    let provenance = Provenance::new_direct(Source::user_input("test"), agent_id);
    MemoryCommon::new(agent_id, provenance).with_confidence(Confidence::new(0.8))
}

fn roundtrip(memory: Memory) {
    let json = serde_json::to_string(&memory).expect("json serialize");
    let reparsed: Memory = serde_json::from_str(&json).expect("json parse");
    assert_eq!(memory.id(), reparsed.id());
    assert_eq!(memory.memory_type(), reparsed.memory_type());

    let bytes = memory.to_msgpack().expect("msgpack serialize");
    let reparsed = Memory::from_msgpack(&bytes).expect("msgpack parse");
    assert_eq!(memory.id(), reparsed.id());
    assert_eq!(memory.memory_type(), reparsed.memory_type());
    assert_eq!(memory.text_content(), reparsed.text_content());
}

#[test]
fn semantic_fact_roundtrip() {
    roundtrip(Memory::Semantic(SemanticMemory {
        common: common(),
        content: SemanticContent::Fact(FactMemory::new("the sky is blue")),
    }));
}

#[test]
fn semantic_preference_roundtrip() {
    roundtrip(Memory::Semantic(SemanticMemory {
        common: common(),
        content: SemanticContent::Preference(
            PreferenceMemory::new("Angela", "food", "pizza").with_strength(PreferenceStrength::Strong),
        ),
    }));
}

#[test]
fn semantic_concept_roundtrip() {
    roundtrip(Memory::Semantic(SemanticMemory {
        common: common(),
        content: SemanticContent::Concept(ConceptMemory::new(
            "HNSW",
            "Hierarchical Navigable Small World",
        )),
    }));
}

#[test]
fn semantic_entity_roundtrip() {
    roundtrip(Memory::Semantic(SemanticMemory {
        common: common(),
        content: SemanticContent::Entity(EntityMemory::new("Rust", EntityType::Other)),
    }));
}

#[test]
fn episodic_conversation_roundtrip() {
    let messages = vec![
        Message::new(MessageRole::User, "hello"),
        Message::new(MessageRole::Assistant, "hi there"),
    ];
    roundtrip(Memory::Episodic(EpisodicMemory {
        common: common(),
        content: EpisodicContent::Conversation(ConversationMemory::new(SessionId::new(), messages)),
    }));
}

#[test]
fn episodic_event_roundtrip() {
    roundtrip(Memory::Episodic(EpisodicMemory {
        common: common(),
        content: EpisodicContent::Event(EventMemory::new("login", "user logged in at 09:00")),
    }));
}

#[test]
fn episodic_observation_roundtrip() {
    roundtrip(Memory::Episodic(EpisodicMemory {
        common: common(),
        content: EpisodicContent::Observation(ObservationMemory::new("the room is quiet")),
    }));
}

#[test]
fn procedural_workflow_roundtrip() {
    roundtrip(Memory::Procedural(ProceduralMemory {
        common: common(),
        content: ProceduralContent::Workflow(WorkflowMemory::new(
            "deploy",
            "deploy service to production",
        )),
    }));
}

#[test]
fn procedural_skill_roundtrip() {
    roundtrip(Memory::Procedural(ProceduralMemory {
        common: common(),
        content: ProceduralContent::Skill(SkillMemory::new(
            "code-review",
            "review pull requests for style and correctness",
        )),
    }));
}

#[test]
fn agent_state_goal_roundtrip() {
    roundtrip(Memory::AgentState(AgentStateMemory {
        common: common(),
        content: AgentStateContent::Goal(Goal::new("finish refactor")),
    }));
}

#[test]
fn agent_state_task_roundtrip() {
    roundtrip(Memory::AgentState(AgentStateMemory {
        common: common(),
        content: AgentStateContent::Task(Task::new("write tests")),
    }));
}

#[test]
fn agent_state_working_memory_roundtrip() {
    roundtrip(Memory::AgentState(AgentStateMemory {
        common: common(),
        content: AgentStateContent::WorkingMemory(WorkingMemoryItem::new(
            WorkingMemoryType::TemporaryFact,
            "user asked about latency",
        )),
    }));
}

#[test]
fn memory_discriminator_matches_variant() {
    let fact = Memory::Semantic(SemanticMemory {
        common: common(),
        content: SemanticContent::Fact(FactMemory::new("x")),
    });
    let json = serde_json::to_value(&fact).expect("json");
    assert_eq!(json["type"], "Semantic");
}

#[test]
fn text_content_survives_msgpack() {
    let original = Memory::Episodic(EpisodicMemory {
        common: common(),
        content: EpisodicContent::Observation(ObservationMemory::new("payload")),
    });
    let bytes = original.to_msgpack().expect("msgpack");
    let restored = Memory::from_msgpack(&bytes).expect("msgpack parse");
    assert!(restored.text_content().contains("payload"));
}
