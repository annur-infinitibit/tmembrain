//! Write policy trait and common types

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::traits::MemoryStorage;
use membrain_core::types::MemoryId;

/// Result of a policy check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyResult {
    /// Memory should proceed to next check
    Pass {
        /// Score from this policy
        score: f64,
        /// Additional details
        details: Option<String>,
    },
    /// Memory should be rejected
    Reject {
        /// Reason for rejection
        reason: String,
        /// Score that caused rejection
        score: f64,
    },
    /// Memory should be merged with existing
    Merge {
        /// ID of memory to merge with
        merge_with: MemoryId,
        /// Similarity score
        similarity: f64,
    },
    /// Policy was skipped (disabled or exempt)
    Skipped {
        /// Reason for skipping
        reason: String,
    },
}

impl PolicyResult {
    /// Check if this is a pass result
    pub fn is_pass(&self) -> bool {
        matches!(self, PolicyResult::Pass { .. })
    }

    /// Check if this is a reject result
    pub fn is_reject(&self) -> bool {
        matches!(self, PolicyResult::Reject { .. })
    }

    /// Check if this is a merge result
    pub fn is_merge(&self) -> bool {
        matches!(self, PolicyResult::Merge { .. })
    }

    /// Get the score if available
    pub fn score(&self) -> Option<f64> {
        match self {
            PolicyResult::Pass { score, .. } => Some(*score),
            PolicyResult::Reject { score, .. } => Some(*score),
            PolicyResult::Merge { similarity, .. } => Some(*similarity),
            PolicyResult::Skipped { .. } => None,
        }
    }
}

/// Trait for write policies that evaluate memories before storage
#[async_trait]
pub trait WritePolicy: Send + Sync {
    /// Get the name of this policy
    fn name(&self) -> &str;

    /// Check if this policy is enabled
    fn is_enabled(&self) -> bool;

    /// Evaluate a memory against this policy
    async fn evaluate(&self, memory: &Memory, storage: &dyn MemoryStorage) -> Result<PolicyResult>;
}
