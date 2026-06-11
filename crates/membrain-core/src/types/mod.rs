//! Core types for the Membrain memory system

mod confidence;
mod embedding;
mod ids;
mod provenance;

pub use confidence::Confidence;
pub use embedding::Embedding;
pub use ids::{AgentId, MemoryId, SessionId};
pub use provenance::{Derivation, DerivationType, Provenance, Source};

/// Version number for memory entries, used for optimistic concurrency
pub type Version = u64;
