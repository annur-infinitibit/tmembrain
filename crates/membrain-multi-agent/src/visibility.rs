//! Memory visibility rules for multi-agent systems

use serde::{Deserialize, Serialize};

use membrain_core::memory::Memory;
use membrain_core::types::AgentId;

use crate::trust::TrustManager;

/// Visibility level for a memory
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryVisibility {
    /// Only visible to the owning agent
    #[default]
    Private,
    /// Visible to agents with trust
    Shared,
    /// Visible to all agents
    Public,
}

impl std::fmt::Display for MemoryVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryVisibility::Private => write!(f, "private"),
            MemoryVisibility::Shared => write!(f, "shared"),
            MemoryVisibility::Public => write!(f, "public"),
        }
    }
}

/// Filter for checking memory visibility
pub struct VisibilityFilter<'a> {
    trust_manager: &'a TrustManager,
    viewing_agent: AgentId,
}

impl<'a> VisibilityFilter<'a> {
    /// Create a new visibility filter
    pub fn new(trust_manager: &'a TrustManager, viewing_agent: AgentId) -> Self {
        Self {
            trust_manager,
            viewing_agent,
        }
    }

    /// Check if a memory is visible to the viewing agent
    pub fn is_visible(&self, memory: &Memory, visibility: MemoryVisibility) -> bool {
        let owner = memory.common().agent_id;

        // Owner can always see their own memories
        if owner == self.viewing_agent {
            return true;
        }

        match visibility {
            MemoryVisibility::Private => false,
            MemoryVisibility::Public => true,
            MemoryVisibility::Shared => {
                // Check if viewing agent has trust from owner
                self.trust_manager.can_read(&self.viewing_agent)
            }
        }
    }

    /// Filter a list of memories by visibility
    pub fn filter_visible(&self, memories: Vec<(Memory, MemoryVisibility)>) -> Vec<Memory> {
        memories
            .into_iter()
            .filter(|(m, v)| self.is_visible(m, *v))
            .map(|(m, _)| m)
            .collect()
    }

    /// Get the effective visibility for an agent
    pub fn effective_visibility(&self, visibility: MemoryVisibility) -> EffectiveVisibility {
        match visibility {
            MemoryVisibility::Private => EffectiveVisibility::NotVisible,
            MemoryVisibility::Public => EffectiveVisibility::FullAccess,
            MemoryVisibility::Shared => {
                if self.trust_manager.can_delete(&self.viewing_agent) {
                    EffectiveVisibility::FullAccess
                } else if self.trust_manager.can_write(&self.viewing_agent) {
                    EffectiveVisibility::ReadWrite
                } else if self.trust_manager.can_read(&self.viewing_agent) {
                    EffectiveVisibility::ReadOnly
                } else {
                    EffectiveVisibility::NotVisible
                }
            }
        }
    }
}

/// Effective visibility after applying trust rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveVisibility {
    /// Memory is not visible
    NotVisible,
    /// Can only read
    ReadOnly,
    /// Can read and write
    ReadWrite,
    /// Full access including delete
    FullAccess,
}

impl EffectiveVisibility {
    /// Check if can read
    pub fn can_read(&self) -> bool {
        *self != EffectiveVisibility::NotVisible
    }

    /// Check if can write
    pub fn can_write(&self) -> bool {
        matches!(
            self,
            EffectiveVisibility::ReadWrite | EffectiveVisibility::FullAccess
        )
    }

    /// Check if can delete
    pub fn can_delete(&self) -> bool {
        *self == EffectiveVisibility::FullAccess
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::{AgentTrust, TrustLevel};
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{Confidence, Provenance, Source};

    fn create_memory(owner: AgentId) -> Memory {
        let prov = Provenance::new_direct(Source::user_input("test"), owner);
        let common = MemoryCommon::new(owner, prov).with_confidence(Confidence::new(0.8));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new("Test fact")),
        })
    }

    #[test]
    fn test_owner_sees_private() {
        let trust_manager = TrustManager::default();
        let owner = AgentId::new();
        let filter = VisibilityFilter::new(&trust_manager, owner);

        let memory = create_memory(owner);
        assert!(filter.is_visible(&memory, MemoryVisibility::Private));
    }

    #[test]
    fn test_other_cannot_see_private() {
        let trust_manager = TrustManager::default();
        let owner = AgentId::new();
        let other = AgentId::new();
        let filter = VisibilityFilter::new(&trust_manager, other);

        let memory = create_memory(owner);
        assert!(!filter.is_visible(&memory, MemoryVisibility::Private));
    }

    #[test]
    fn test_anyone_sees_public() {
        let trust_manager = TrustManager::default();
        let owner = AgentId::new();
        let other = AgentId::new();
        let filter = VisibilityFilter::new(&trust_manager, other);

        let memory = create_memory(owner);
        assert!(filter.is_visible(&memory, MemoryVisibility::Public));
    }

    #[test]
    fn test_shared_with_trust() {
        let mut trust_manager = TrustManager::default();
        let owner = AgentId::new();
        let trusted = AgentId::new();
        let untrusted = AgentId::new();

        trust_manager.set_trust(AgentTrust::new(trusted, TrustLevel::ReadOnly));

        let memory = create_memory(owner);

        let filter_trusted = VisibilityFilter::new(&trust_manager, trusted);
        assert!(filter_trusted.is_visible(&memory, MemoryVisibility::Shared));

        let filter_untrusted = VisibilityFilter::new(&trust_manager, untrusted);
        assert!(!filter_untrusted.is_visible(&memory, MemoryVisibility::Shared));
    }

    #[test]
    fn test_effective_visibility() {
        let mut trust_manager = TrustManager::default();
        let agent = AgentId::new();

        trust_manager.set_trust(AgentTrust::new(agent, TrustLevel::Contribute));

        let filter = VisibilityFilter::new(&trust_manager, agent);
        let effective = filter.effective_visibility(MemoryVisibility::Shared);

        assert_eq!(effective, EffectiveVisibility::ReadWrite);
        assert!(effective.can_read());
        assert!(effective.can_write());
        assert!(!effective.can_delete());
    }
}
