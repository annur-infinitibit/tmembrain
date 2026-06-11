//! Budget policy - enforces memory limits

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::config::{BudgetConfig, BudgetExceededAction};
use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::traits::{MemoryStorage, SearchFilters};

use super::policy::{PolicyResult, WritePolicy};

/// Result of budget check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetResult {
    /// Current total memory count
    pub total_count: usize,
    /// Global limit
    pub global_limit: usize,
    /// Current count for this memory type
    pub type_count: usize,
    /// Limit for this memory type
    pub type_limit: Option<usize>,
    /// Whether budget allows storage
    pub within_budget: bool,
    /// Action to take if budget exceeded
    pub action: BudgetAction,
}

/// Action to take when budget is exceeded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetAction {
    /// Allow the memory
    Allow,
    /// Reject the memory
    Reject,
    /// Need to remove old memories first
    NeedEviction { count: usize },
}

/// Policy that enforces memory budgets
pub struct BudgetPolicy {
    config: BudgetConfig,
}

impl BudgetPolicy {
    /// Create a new budget policy
    pub fn new(config: BudgetConfig) -> Self {
        Self { config }
    }

    /// Check budget status
    pub async fn check_budget(
        &self,
        memory: &Memory,
        storage: &dyn MemoryStorage,
    ) -> Result<BudgetResult> {
        let memory_type = memory.memory_type();

        // Get current counts
        let total_count = storage.count(None).await?;

        let type_filters = SearchFilters::new().with_types(vec![memory_type]);
        let type_count = storage.count(Some(type_filters)).await?;

        // Check limits
        let global_exceeded = total_count >= self.config.global_max_memories;
        let type_limit = self.config.type_limits.get(&memory_type).copied();
        let type_exceeded = type_limit.map(|limit| type_count >= limit).unwrap_or(false);

        let within_budget = !global_exceeded && !type_exceeded;

        let action = if within_budget {
            BudgetAction::Allow
        } else {
            match self.config.on_exceeded {
                BudgetExceededAction::Reject => BudgetAction::Reject,
                BudgetExceededAction::RemoveOldest
                | BudgetExceededAction::RemoveLowestConfidence => {
                    BudgetAction::NeedEviction { count: 1 }
                }
                BudgetExceededAction::Consolidate => {
                    // For now, treat as rejection; consolidation handled elsewhere
                    BudgetAction::Reject
                }
            }
        };

        Ok(BudgetResult {
            total_count,
            global_limit: self.config.global_max_memories,
            type_count,
            type_limit,
            within_budget,
            action,
        })
    }
}

#[async_trait]
impl WritePolicy for BudgetPolicy {
    fn name(&self) -> &str {
        "budget"
    }

    fn is_enabled(&self) -> bool {
        true // Budget is always enabled
    }

    async fn evaluate(&self, memory: &Memory, storage: &dyn MemoryStorage) -> Result<PolicyResult> {
        let result = self.check_budget(memory, storage).await?;

        match result.action {
            BudgetAction::Allow => Ok(PolicyResult::Pass {
                score: 1.0,
                details: Some(format!(
                    "Budget OK: total {}/{}, type {}/{}",
                    result.total_count,
                    result.global_limit,
                    result.type_count,
                    result
                        .type_limit
                        .map(|l| l.to_string())
                        .unwrap_or_else(|| "∞".to_string())
                )),
            }),
            BudgetAction::Reject => Ok(PolicyResult::Reject {
                reason: if result.total_count >= result.global_limit {
                    format!(
                        "Global budget exceeded: {}/{}",
                        result.total_count, result.global_limit
                    )
                } else {
                    format!(
                        "Type budget exceeded for {:?}: {}/{}",
                        memory.memory_type(),
                        result.type_count,
                        result
                            .type_limit
                            .map(|l| l.to_string())
                            .unwrap_or_else(|| "∞".to_string())
                    )
                },
                score: 0.0,
            }),
            BudgetAction::NeedEviction { count } => {
                // For now, treat as rejection
                // In practice, the pipeline would handle eviction
                Ok(PolicyResult::Reject {
                    reason: format!("Budget exceeded, need to evict {} memories", count),
                    score: 0.0,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{
        FactMemory, MemoryCommon, MemoryType, SemanticContent, SemanticMemory,
    };
    use membrain_core::types::{AgentId, Confidence, Provenance, Source};
    use membrain_storage::InMemoryStorage;
    use std::collections::HashMap;

    fn create_memory() -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new("Test fact")),
        })
    }

    #[tokio::test]
    async fn test_within_budget() {
        let config = BudgetConfig {
            global_max_memories: 100,
            type_limits: HashMap::new(),
            on_exceeded: BudgetExceededAction::Reject,
        };

        let policy = BudgetPolicy::new(config);
        let storage = InMemoryStorage::new();
        let memory = create_memory();

        let result = policy.check_budget(&memory, &storage).await.unwrap();
        assert!(result.within_budget);
        assert!(matches!(result.action, BudgetAction::Allow));
    }

    #[tokio::test]
    async fn test_global_budget_exceeded() {
        let config = BudgetConfig {
            global_max_memories: 2,
            type_limits: HashMap::new(),
            on_exceeded: BudgetExceededAction::Reject,
        };

        let policy = BudgetPolicy::new(config);
        let storage = InMemoryStorage::new();

        // Fill up storage
        storage.store(create_memory()).await.unwrap();
        storage.store(create_memory()).await.unwrap();

        let memory = create_memory();
        let result = policy.check_budget(&memory, &storage).await.unwrap();
        assert!(!result.within_budget);
        assert!(matches!(result.action, BudgetAction::Reject));
    }

    #[tokio::test]
    async fn test_type_budget_exceeded() {
        let mut type_limits = HashMap::new();
        type_limits.insert(MemoryType::SemanticFact, 1);

        let config = BudgetConfig {
            global_max_memories: 100,
            type_limits,
            on_exceeded: BudgetExceededAction::Reject,
        };

        let policy = BudgetPolicy::new(config);
        let storage = InMemoryStorage::new();

        // Add one fact
        storage.store(create_memory()).await.unwrap();

        // Try to add another
        let memory = create_memory();
        let result = policy.check_budget(&memory, &storage).await.unwrap();
        assert!(!result.within_budget);
    }
}
