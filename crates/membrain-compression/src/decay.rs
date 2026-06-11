//! Decay policies for memory confidence

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use membrain_core::error::Result;
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::MemoryStorage;
use membrain_core::types::Confidence;

/// Decay policy configuration
#[derive(Debug, Clone)]
pub enum DecayPolicy {
    /// Decay based on time since last access
    AccessBased {
        /// Half-life for decay
        half_life: Duration,
    },
    /// Decay based on reinforcement patterns
    ReinforcementBased {
        /// Base decay rate per day
        decay_rate: f64,
    },
    /// Different decay rates per memory type
    TypeSpecific(HashMap<MemoryType, DecayConfig>),
    /// No decay
    None,
}

impl Default for DecayPolicy {
    fn default() -> Self {
        let mut type_configs = HashMap::new();

        // Episodic memories decay faster
        type_configs.insert(
            MemoryType::EpisodicConversation,
            DecayConfig {
                half_life: Duration::from_secs(7 * 24 * 3600), // 1 week
                min_confidence: 0.1,
            },
        );
        type_configs.insert(
            MemoryType::EpisodicEvent,
            DecayConfig {
                half_life: Duration::from_secs(14 * 24 * 3600), // 2 weeks
                min_confidence: 0.1,
            },
        );
        type_configs.insert(
            MemoryType::EpisodicObservation,
            DecayConfig {
                half_life: Duration::from_secs(7 * 24 * 3600), // 1 week
                min_confidence: 0.1,
            },
        );

        // Semantic memories decay slower
        type_configs.insert(
            MemoryType::SemanticFact,
            DecayConfig {
                half_life: Duration::from_secs(90 * 24 * 3600), // 90 days
                min_confidence: 0.2,
            },
        );
        type_configs.insert(
            MemoryType::SemanticPreference,
            DecayConfig {
                half_life: Duration::from_secs(30 * 24 * 3600), // 30 days
                min_confidence: 0.2,
            },
        );

        // Procedural memories are very stable
        type_configs.insert(
            MemoryType::ProceduralWorkflow,
            DecayConfig {
                half_life: Duration::from_secs(365 * 24 * 3600), // 1 year
                min_confidence: 0.3,
            },
        );
        type_configs.insert(
            MemoryType::ProceduralSkill,
            DecayConfig {
                half_life: Duration::from_secs(180 * 24 * 3600), // 6 months
                min_confidence: 0.2,
            },
        );
        type_configs.insert(
            MemoryType::ProceduralCase,
            DecayConfig {
                half_life: Duration::from_secs(180 * 24 * 3600), // 6 months
                min_confidence: 0.2,
            },
        );

        // Agent state doesn't decay (it's managed directly)
        type_configs.insert(
            MemoryType::AgentStateGoal,
            DecayConfig {
                half_life: Duration::from_secs(365 * 24 * 3600),
                min_confidence: 0.0,
            },
        );

        DecayPolicy::TypeSpecific(type_configs)
    }
}

/// Configuration for a specific memory type's decay
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Half-life for exponential decay
    pub half_life: Duration,
    /// Minimum confidence (won't decay below this)
    pub min_confidence: f64,
}

/// Engine for applying decay to memories
pub struct DecayEngine {
    policy: DecayPolicy,
    storage: Arc<dyn MemoryStorage>,
}

impl DecayEngine {
    /// Create a new decay engine
    pub fn new(storage: Arc<dyn MemoryStorage>, policy: DecayPolicy) -> Self {
        Self { policy, storage }
    }

    /// Calculate decayed confidence for a memory
    pub fn calculate_decay(&self, memory: &Memory) -> Confidence {
        let current = memory.confidence();
        let memory_type = memory.memory_type();
        let time_since_access = memory.common().provenance.time_since_access();

        match &self.policy {
            DecayPolicy::None => *current,

            DecayPolicy::AccessBased { half_life } => {
                let elapsed = Duration::from_secs(time_since_access.num_seconds().max(0) as u64);
                current.decay_exponential(elapsed, *half_life)
            }

            DecayPolicy::ReinforcementBased { decay_rate } => {
                let days = time_since_access.num_days() as f64;
                current.decay_linear(
                    Duration::from_secs((days * 24.0 * 3600.0) as u64),
                    *decay_rate,
                )
            }

            DecayPolicy::TypeSpecific(configs) => {
                if let Some(config) = configs.get(&memory_type) {
                    let elapsed =
                        Duration::from_secs(time_since_access.num_seconds().max(0) as u64);
                    let decayed = current.decay_exponential(elapsed, config.half_life);

                    // Don't go below minimum
                    if decayed.value() < config.min_confidence {
                        Confidence::new(config.min_confidence)
                    } else {
                        decayed
                    }
                } else {
                    // Default: no decay
                    *current
                }
            }
        }
    }

    /// Apply decay to a batch of memories
    pub async fn apply_decay(&self, batch_size: usize) -> Result<DecayResult> {
        let memories = self
            .storage
            .search(membrain_core::traits::SearchQuery::new().with_limit(batch_size))
            .await?;

        let mut processed = 0;
        let mut updated = 0;
        let mut deleted = 0;

        for result in memories {
            let mut memory = result.memory;
            let old_confidence = memory.confidence().value();
            let new_confidence = self.calculate_decay(&memory);

            // Skip if no significant change
            if (old_confidence - new_confidence.value()).abs() < 0.01 {
                continue;
            }

            processed += 1;

            // Check if should delete
            if new_confidence.value() < 0.05 {
                self.storage.delete(memory.id()).await?;
                deleted += 1;
            } else {
                // Update confidence
                let version = memory.common().version;
                memory.common_mut().confidence = new_confidence;
                memory.common_mut().increment_version();

                if self.storage.update(memory, version).await.is_ok() {
                    updated += 1;
                }
            }
        }

        Ok(DecayResult {
            processed,
            updated,
            deleted,
        })
    }
}

/// Result of decay operation
#[derive(Debug, Clone)]
pub struct DecayResult {
    /// Number of memories processed
    pub processed: usize,
    /// Number of memories updated
    pub updated: usize,
    /// Number of memories deleted
    pub deleted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{AgentId, Provenance, Source};
    use membrain_storage::InMemoryStorage;

    fn create_memory() -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(1.0));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new("Test fact")),
        })
    }

    #[test]
    fn test_access_based_decay() {
        let storage = Arc::new(InMemoryStorage::new());
        let policy = DecayPolicy::AccessBased {
            half_life: Duration::from_secs(3600), // 1 hour
        };
        let engine = DecayEngine::new(storage, policy);

        let memory = create_memory();
        let decayed = engine.calculate_decay(&memory);

        // Fresh memory should have minimal decay
        assert!(decayed.value() > 0.9);
    }

    #[test]
    fn test_type_specific_decay() {
        let storage = Arc::new(InMemoryStorage::new());
        let policy = DecayPolicy::default();
        let engine = DecayEngine::new(storage, policy);

        let memory = create_memory();
        let decayed = engine.calculate_decay(&memory);

        // Should use semantic fact config
        assert!(decayed.value() > 0.2); // Above min_confidence
    }

    #[test]
    fn test_no_decay_policy() {
        let storage = Arc::new(InMemoryStorage::new());
        let policy = DecayPolicy::None;
        let engine = DecayEngine::new(storage, policy);

        let memory = create_memory();
        let original = memory.confidence().value();
        let decayed = engine.calculate_decay(&memory);

        assert!((decayed.value() - original).abs() < 0.001);
    }
}
