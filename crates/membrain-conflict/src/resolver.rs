//! Core conflict resolution trait and types.
//!
//! When a new memory is stored, the conflict resolver compares it against
//! similar existing memories and classifies the relationship as ADD, UPDATE,
//! DELETE, or NOOP -- enabling automatic fact deduplication, contradiction
//! resolution, and memory merging.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::types::MemoryId;

/// The LLM's classification of how a new memory relates to existing ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ConflictDecision {
    /// New memory is genuinely new information. Store it as-is.
    Add,

    /// New memory augments or replaces an existing one. The target memory
    /// should be updated with the merged content.
    Update {
        /// ID of the existing memory to update
        target_id: MemoryId,
        /// Merged content that combines old and new information
        merged_content: String,
    },

    /// New memory contradicts an existing one. The old memory should be
    /// invalidated (valid_until set) and the new one stored.
    Delete {
        /// ID of the existing memory to invalidate
        target_id: MemoryId,
        /// Explanation of why the old memory is superseded
        reason: String,
    },

    /// New memory is already known or irrelevant. Discard it.
    Noop {
        /// Explanation of why the memory was discarded
        reason: String,
    },
}

/// Result of conflict resolution for one new memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionResult {
    /// The decision made by the resolver
    pub decision: ConflictDecision,
    /// Confidence in the decision (0.0-1.0)
    pub confidence: f64,
    /// Human-readable reasoning behind the decision
    pub reasoning: String,
}

/// Trait for LLM-based memory conflict resolution.
///
/// Given a new memory and a list of similar existing memories, the resolver
/// classifies the relationship and returns a decision on how to proceed.
#[async_trait]
pub trait ConflictResolver: Send + Sync {
    /// Resolve potential conflicts between a new memory and similar existing memories.
    ///
    /// # Arguments
    /// * `new_memory` - The memory being stored
    /// * `similar_memories` - Top-N most similar existing memories (pre-fetched by the pipeline)
    ///
    /// # Returns
    /// A resolution result containing the decision, confidence, and reasoning.
    async fn resolve(
        &self,
        new_memory: &Memory,
        similar_memories: &[Memory],
    ) -> Result<ConflictResolutionResult>;

    /// Name of this resolver implementation.
    fn name(&self) -> &str;

    /// Model identifier used by this resolver.
    fn model(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_decision_add_serializes() {
        let decision = ConflictDecision::Add;
        let json = serde_json::to_string(&decision).expect("should serialize");
        assert!(json.contains("\"decision\":\"add\""));
    }

    #[test]
    fn conflict_decision_update_serializes() {
        let decision = ConflictDecision::Update {
            target_id: MemoryId::new(),
            merged_content: "David likes basketball".to_string(),
        };
        let json = serde_json::to_string(&decision).expect("should serialize");
        assert!(json.contains("\"decision\":\"update\""));
        assert!(json.contains("basketball"));
    }

    #[test]
    fn conflict_decision_delete_serializes() {
        let decision = ConflictDecision::Delete {
            target_id: MemoryId::new(),
            reason: "Preference changed".to_string(),
        };
        let json = serde_json::to_string(&decision).expect("should serialize");
        assert!(json.contains("\"decision\":\"delete\""));
    }

    #[test]
    fn conflict_decision_noop_serializes() {
        let decision = ConflictDecision::Noop {
            reason: "Already known".to_string(),
        };
        let json = serde_json::to_string(&decision).expect("should serialize");
        assert!(json.contains("\"decision\":\"noop\""));
    }

    #[test]
    fn conflict_resolution_result_roundtrip() {
        let result = ConflictResolutionResult {
            decision: ConflictDecision::Add,
            confidence: 0.95,
            reasoning: "This is genuinely new information".to_string(),
        };
        let json = serde_json::to_string(&result).expect("should serialize");
        let parsed: ConflictResolutionResult =
            serde_json::from_str(&json).expect("should deserialize");
        assert!((parsed.confidence - 0.95).abs() < f64::EPSILON);
    }
}
