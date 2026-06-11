//! Memory sharing between agents

use serde::{Deserialize, Serialize};

use membrain_core::memory::MemoryType;
use membrain_core::types::{AgentId, MemoryId};

use crate::trust::TrustLevel;
use crate::visibility::MemoryVisibility;

/// Policy for sharing memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingPolicy {
    /// Whether sharing is enabled
    pub enabled: bool,
    /// Default visibility for new memories
    pub default_visibility: MemoryVisibility,
    /// Memory types that can be shared
    pub shareable_types: Vec<MemoryType>,
    /// Minimum trust level required to share
    pub min_trust_to_share: TrustLevel,
    /// Whether to require explicit approval
    pub require_approval: bool,
}

impl Default for SharingPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            default_visibility: MemoryVisibility::Private,
            shareable_types: vec![
                MemoryType::SemanticFact,
                MemoryType::SemanticPreference,
                MemoryType::SemanticEntity,
                MemoryType::ProceduralWorkflow,
                MemoryType::ProceduralSkill,
                MemoryType::ProceduralCase,
            ],
            min_trust_to_share: TrustLevel::Contribute,
            require_approval: true,
        }
    }
}

impl SharingPolicy {
    /// Check if a memory type is shareable
    pub fn is_shareable(&self, memory_type: MemoryType) -> bool {
        self.enabled && self.shareable_types.contains(&memory_type)
    }

    /// Check if an agent can share based on trust level
    pub fn can_share(&self, trust_level: TrustLevel) -> bool {
        self.enabled && trust_level >= self.min_trust_to_share
    }
}

/// Request to share a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareRequest {
    /// Memory to share
    pub memory_id: MemoryId,
    /// Agent making the request
    pub from_agent: AgentId,
    /// Target agent (or None for all trusted)
    pub to_agent: Option<AgentId>,
    /// Requested visibility
    pub visibility: MemoryVisibility,
    /// Optional message
    pub message: Option<String>,
    /// When the request was made
    pub requested_at: chrono::DateTime<chrono::Utc>,
    /// Status of the request
    pub status: ShareRequestStatus,
}

impl ShareRequest {
    /// Create a new share request
    pub fn new(memory_id: MemoryId, from_agent: AgentId, visibility: MemoryVisibility) -> Self {
        Self {
            memory_id,
            from_agent,
            to_agent: None,
            visibility,
            message: None,
            requested_at: chrono::Utc::now(),
            status: ShareRequestStatus::Pending,
        }
    }

    /// Set target agent
    pub fn with_target(mut self, agent: AgentId) -> Self {
        self.to_agent = Some(agent);
        self
    }

    /// Set message
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Approve the request
    pub fn approve(&mut self) {
        self.status = ShareRequestStatus::Approved;
    }

    /// Deny the request
    pub fn deny(&mut self, reason: Option<String>) {
        self.status = ShareRequestStatus::Denied { reason };
    }
}

/// Status of a share request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShareRequestStatus {
    /// Waiting for approval
    Pending,
    /// Approved and shared
    Approved,
    /// Request was denied
    Denied { reason: Option<String> },
    /// Request was cancelled
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharing_policy_default() {
        let policy = SharingPolicy::default();

        assert!(!policy.enabled);
        assert_eq!(policy.default_visibility, MemoryVisibility::Private);
    }

    #[test]
    fn test_is_shareable() {
        let policy = SharingPolicy {
            enabled: true,
            ..SharingPolicy::default()
        };

        assert!(policy.is_shareable(MemoryType::SemanticFact));
        assert!(!policy.is_shareable(MemoryType::AgentStateGoal));
    }

    #[test]
    fn test_can_share() {
        let policy = SharingPolicy {
            enabled: true,
            min_trust_to_share: TrustLevel::Contribute,
            ..SharingPolicy::default()
        };

        assert!(!policy.can_share(TrustLevel::ReadOnly));
        assert!(policy.can_share(TrustLevel::Contribute));
        assert!(policy.can_share(TrustLevel::Full));
    }

    #[test]
    fn test_share_request() {
        let memory_id = MemoryId::new();
        let agent = AgentId::new();

        let mut request = ShareRequest::new(memory_id, agent, MemoryVisibility::Shared)
            .with_message("Please review");

        assert!(matches!(request.status, ShareRequestStatus::Pending));

        request.approve();
        assert!(matches!(request.status, ShareRequestStatus::Approved));
    }
}
