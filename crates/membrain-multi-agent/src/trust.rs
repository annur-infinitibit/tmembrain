//! Trust levels and management for multi-agent systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use membrain_core::types::AgentId;

/// Trust level for an agent
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum TrustLevel {
    /// No access to shared memories
    #[default]
    None = 0,
    /// Can read shared memories
    ReadOnly = 1,
    /// Can add memories to shared pool
    Contribute = 2,
    /// Full read/write/delete access
    Full = 3,
}

impl TrustLevel {
    /// Check if this level can read
    pub fn can_read(&self) -> bool {
        *self >= TrustLevel::ReadOnly
    }

    /// Check if this level can write
    pub fn can_write(&self) -> bool {
        *self >= TrustLevel::Contribute
    }

    /// Check if this level can delete
    pub fn can_delete(&self) -> bool {
        *self >= TrustLevel::Full
    }

    /// Check if this level can modify others' memories
    pub fn can_modify_others(&self) -> bool {
        *self >= TrustLevel::Full
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::None => write!(f, "none"),
            TrustLevel::ReadOnly => write!(f, "read_only"),
            TrustLevel::Contribute => write!(f, "contribute"),
            TrustLevel::Full => write!(f, "full"),
        }
    }
}

impl std::str::FromStr for TrustLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(TrustLevel::None),
            "read_only" | "readonly" => Ok(TrustLevel::ReadOnly),
            "contribute" => Ok(TrustLevel::Contribute),
            "full" => Ok(TrustLevel::Full),
            _ => Err(format!("Unknown trust level: {}", s)),
        }
    }
}

/// Trust information for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrust {
    /// The agent ID
    pub agent_id: AgentId,
    /// Trust level
    pub level: TrustLevel,
    /// When trust was established
    pub established_at: chrono::DateTime<chrono::Utc>,
    /// Who granted the trust
    pub granted_by: Option<AgentId>,
    /// Reason for trust level
    pub reason: Option<String>,
}

impl AgentTrust {
    /// Create new agent trust
    pub fn new(agent_id: AgentId, level: TrustLevel) -> Self {
        Self {
            agent_id,
            level,
            established_at: chrono::Utc::now(),
            granted_by: None,
            reason: None,
        }
    }

    /// Set who granted the trust
    pub fn with_grantor(mut self, grantor: AgentId) -> Self {
        self.granted_by = Some(grantor);
        self
    }

    /// Set reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Manager for agent trust relationships
pub struct TrustManager {
    /// Trust records by agent ID
    trusts: HashMap<AgentId, AgentTrust>,
    /// Default trust level for new agents
    default_level: TrustLevel,
}

impl TrustManager {
    /// Create a new trust manager
    pub fn new(default_level: TrustLevel) -> Self {
        Self {
            trusts: HashMap::new(),
            default_level,
        }
    }

    /// Get trust level for an agent
    pub fn get_trust(&self, agent_id: &AgentId) -> TrustLevel {
        self.trusts
            .get(agent_id)
            .map(|t| t.level)
            .unwrap_or(self.default_level)
    }

    /// Set trust level for an agent
    pub fn set_trust(&mut self, trust: AgentTrust) {
        self.trusts.insert(trust.agent_id, trust);
    }

    /// Remove trust for an agent (reverts to default)
    pub fn remove_trust(&mut self, agent_id: &AgentId) -> Option<AgentTrust> {
        self.trusts.remove(agent_id)
    }

    /// Check if an agent can read shared memories
    pub fn can_read(&self, agent_id: &AgentId) -> bool {
        self.get_trust(agent_id).can_read()
    }

    /// Check if an agent can write to shared memories
    pub fn can_write(&self, agent_id: &AgentId) -> bool {
        self.get_trust(agent_id).can_write()
    }

    /// Check if an agent can delete shared memories
    pub fn can_delete(&self, agent_id: &AgentId) -> bool {
        self.get_trust(agent_id).can_delete()
    }

    /// Get all agents with a specific trust level
    pub fn agents_with_level(&self, level: TrustLevel) -> Vec<AgentId> {
        self.trusts
            .iter()
            .filter(|(_, t)| t.level == level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all trusted agents (ReadOnly or higher)
    pub fn trusted_agents(&self) -> Vec<AgentId> {
        self.trusts
            .iter()
            .filter(|(_, t)| t.level.can_read())
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for TrustManager {
    fn default() -> Self {
        Self::new(TrustLevel::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::None < TrustLevel::ReadOnly);
        assert!(TrustLevel::ReadOnly < TrustLevel::Contribute);
        assert!(TrustLevel::Contribute < TrustLevel::Full);
    }

    #[test]
    fn test_trust_level_permissions() {
        assert!(!TrustLevel::None.can_read());
        assert!(TrustLevel::ReadOnly.can_read());
        assert!(!TrustLevel::ReadOnly.can_write());
        assert!(TrustLevel::Contribute.can_write());
        assert!(!TrustLevel::Contribute.can_delete());
        assert!(TrustLevel::Full.can_delete());
    }

    #[test]
    fn test_trust_manager() {
        let mut manager = TrustManager::new(TrustLevel::None);

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // Default is None
        assert_eq!(manager.get_trust(&agent1), TrustLevel::None);

        // Set trust
        manager.set_trust(AgentTrust::new(agent1, TrustLevel::ReadOnly));
        assert_eq!(manager.get_trust(&agent1), TrustLevel::ReadOnly);

        // agent2 still has default
        assert_eq!(manager.get_trust(&agent2), TrustLevel::None);

        // Check permissions
        assert!(manager.can_read(&agent1));
        assert!(!manager.can_read(&agent2));
    }

    #[test]
    fn test_trust_level_parsing() {
        assert_eq!(
            "read_only".parse::<TrustLevel>().unwrap(),
            TrustLevel::ReadOnly
        );
        assert_eq!("full".parse::<TrustLevel>().unwrap(), TrustLevel::Full);
        assert!("invalid".parse::<TrustLevel>().is_err());
    }
}
