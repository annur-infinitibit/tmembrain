//! Test-only helpers shared across Membrain crates.
//!
//! This crate is `publish = false` and is intended to be pulled in as a
//! `[dev-dependencies]` entry. It provides small factory functions,
//! deterministic trait implementations, scripted HTTP servers, and JSON
//! fixtures used by the full Membrain test suite.
//!
//! Prefer these helpers over rolling your own. If you need a new factory,
//! add it here.

pub mod embedding;
pub mod extractor;
pub mod llm_fixtures;
pub mod storage;

#[cfg(feature = "conflict")]
pub mod conflict;

#[cfg(feature = "http-server")]
pub mod openai_mock_server;

pub use embedding::DeterministicEmbeddingProvider;
pub use extractor::DeterministicExtractor;
pub use storage::InMemoryStorageStub;

#[cfg(feature = "conflict")]
pub use conflict::FakeConflictResolver;

#[cfg(feature = "http-server")]
pub use openai_mock_server::{CapturedRequest, ScriptedOpenAiServer, ScriptedResponse};

use membrain_core::memory::{
    EpisodicContent, EpisodicMemory, EventMemory, FactMemory, Memory, MemoryCommon,
    ObservationMemory, SemanticContent, SemanticMemory,
};
use membrain_core::types::{AgentId, Confidence, Provenance, Source};

/// Build a `MemoryCommon` with a fresh agent, `user_input("test")` source,
/// and the given confidence.
pub fn common(confidence: f64) -> (AgentId, MemoryCommon) {
    let agent_id = AgentId::new();
    let provenance = Provenance::new_direct(Source::user_input("test"), agent_id);
    let common =
        MemoryCommon::new(agent_id, provenance).with_confidence(Confidence::new(confidence));
    (agent_id, common)
}

/// Semantic-fact memory with confidence 0.8 and the given text.
pub fn semantic_fact(text: &str) -> Memory {
    let (_agent, common) = common(0.8);
    Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(FactMemory::new(text)),
    })
}

/// Semantic-fact memory owned by the given agent.
pub fn semantic_fact_for(owner: AgentId, text: &str) -> Memory {
    let provenance = Provenance::new_direct(Source::user_input("test"), owner);
    let common = MemoryCommon::new(owner, provenance).with_confidence(Confidence::new(0.8));
    Memory::Semantic(SemanticMemory {
        common,
        content: SemanticContent::Fact(FactMemory::new(text)),
    })
}

/// Episodic observation memory with the given content.
pub fn episodic_observation(text: &str) -> Memory {
    let (_agent, common) = common(0.8);
    Memory::Episodic(EpisodicMemory {
        common,
        content: EpisodicContent::Observation(ObservationMemory::new(text)),
    })
}

/// Episodic event memory with the given event type and description.
pub fn episodic_event(event_type: &str, description: &str) -> Memory {
    let (_agent, common) = common(0.8);
    Memory::Episodic(EpisodicMemory {
        common,
        content: EpisodicContent::Event(EventMemory::new(event_type, description)),
    })
}
