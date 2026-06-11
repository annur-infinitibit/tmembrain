//! Distillation engine for extracting semantic memories from episodic

use std::sync::Arc;

use membrain_core::error::Result;
use membrain_core::memory::{
    FactMemory, Memory, MemoryCommon, MemoryType, PreferenceMemory, PreferenceStrength,
    SemanticContent, SemanticMemory,
};
use membrain_core::traits::{MemoryStorage, SearchFilters, SearchQuery};
use membrain_core::types::{Confidence, Derivation, DerivationType, Provenance, Source};

/// Engine for distilling episodic memories into semantic memories
pub struct DistillationEngine {
    storage: Arc<dyn MemoryStorage>,
    config: DistillationConfig,
}

/// Configuration for distillation
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Minimum age (hours) before distillation
    pub min_age_hours: u64,
    /// Minimum times mentioned before extracting as fact
    pub min_mentions: u32,
    /// Confidence threshold for extraction
    pub confidence_threshold: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_age_hours: 24,
            min_mentions: 2,
            confidence_threshold: 0.6,
        }
    }
}

impl DistillationEngine {
    /// Create a new distillation engine
    pub fn new(storage: Arc<dyn MemoryStorage>, config: DistillationConfig) -> Self {
        Self { storage, config }
    }

    /// Run distillation on eligible episodic memories
    pub async fn run(&self, batch_size: usize) -> Result<DistillationResult> {
        // Get episodic memories old enough for distillation
        let filters = SearchFilters::new().with_types(vec![
            MemoryType::EpisodicConversation,
            MemoryType::EpisodicObservation,
        ]);

        let query = SearchQuery::new()
            .with_limit(batch_size)
            .with_filters(filters);

        let memories = self.storage.search(query).await?;

        let mut processed = 0;
        let mut extracted = 0;

        for result in memories {
            let age_hours = result.memory.common().provenance.age().num_hours();
            if age_hours < self.config.min_age_hours as i64 {
                continue;
            }

            processed += 1;

            // Extract potential facts/preferences
            let extractions = self.extract_from_memory(&result.memory);

            for extraction in extractions {
                // Store the extracted memory
                if self.storage.store(extraction).await.is_ok() {
                    extracted += 1;
                }
            }
        }

        Ok(DistillationResult {
            processed,
            extracted,
        })
    }

    /// Extract semantic memories from a single episodic memory
    fn extract_from_memory(&self, memory: &Memory) -> Vec<Memory> {
        let text = memory.text_content();
        let mut extractions = Vec::new();

        // Try to extract preferences
        if let Some(pref) = self.extract_preference(&text, memory) {
            extractions.push(pref);
        }

        // Try to extract facts
        if let Some(fact) = self.extract_fact(&text, memory) {
            extractions.push(fact);
        }

        extractions
    }

    /// Try to extract a preference from text
    fn extract_preference(&self, text: &str, source: &Memory) -> Option<Memory> {
        let text_lower = text.to_lowercase();

        // Look for preference patterns
        let preference_patterns = [
            ("prefers", "prefer"),
            ("likes", "like"),
            ("wants", "want"),
            ("favorite", "favorite"),
            ("always uses", "use"),
            ("never uses", "avoid"),
        ];

        for (pattern, _verb) in preference_patterns {
            if text_lower.contains(pattern) {
                // Simple extraction - in production, use NLP
                let preference = PreferenceMemory::new("user", "setting", text.to_string())
                    .with_strength(PreferenceStrength::Moderate);

                let agent_id = source.common().agent_id;
                let derivation = Derivation::new(vec![*source.id()], DerivationType::Distillation);
                let prov = Provenance::new_derived(
                    Source::Consolidated {
                        source_count: 1,
                        method: "distillation".to_string(),
                    },
                    vec![derivation],
                    agent_id,
                );

                let common = MemoryCommon::new(agent_id, prov)
                    .with_confidence(Confidence::new(self.config.confidence_threshold));

                return Some(Memory::Semantic(SemanticMemory {
                    common,
                    content: SemanticContent::Preference(preference),
                }));
            }
        }

        None
    }

    /// Try to extract a fact from text
    fn extract_fact(&self, text: &str, source: &Memory) -> Option<Memory> {
        let text_lower = text.to_lowercase();

        // Look for factual assertion patterns
        let fact_patterns = [
            "is a",
            "are a",
            "consists of",
            "contains",
            "means",
            "defined as",
        ];

        for pattern in fact_patterns {
            if text_lower.contains(pattern) {
                let fact = FactMemory::new(text.to_string());

                let agent_id = source.common().agent_id;
                let derivation = Derivation::new(vec![*source.id()], DerivationType::Distillation);
                let prov = Provenance::new_derived(
                    Source::Consolidated {
                        source_count: 1,
                        method: "distillation".to_string(),
                    },
                    vec![derivation],
                    agent_id,
                );

                let common = MemoryCommon::new(agent_id, prov)
                    .with_confidence(Confidence::new(self.config.confidence_threshold));

                return Some(Memory::Semantic(SemanticMemory {
                    common,
                    content: SemanticContent::Fact(fact),
                }));
            }
        }

        None
    }
}

/// Result of distillation operation
#[derive(Debug, Clone)]
pub struct DistillationResult {
    /// Number of episodic memories processed
    pub processed: usize,
    /// Number of semantic memories extracted
    pub extracted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{ConversationMemory, EpisodicContent, EpisodicMemory, Message};
    use membrain_core::types::{AgentId, SessionId};
    use membrain_storage::InMemoryStorage;

    fn create_conversation(text: &str) -> Memory {
        let agent_id = AgentId::new();
        let session_id = SessionId::new();
        let prov = Provenance::new_direct(Source::user_input(text), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));

        let conversation = ConversationMemory::new(
            session_id,
            vec![Message::user(text), Message::assistant("Noted.")],
        );

        Memory::Episodic(EpisodicMemory {
            common,
            content: EpisodicContent::Conversation(conversation),
        })
    }

    #[test]
    fn test_extract_preference() {
        let storage = Arc::new(InMemoryStorage::new());
        let engine = DistillationEngine::new(storage, DistillationConfig::default());

        let memory = create_conversation("The user prefers dark mode for the IDE");
        let extractions = engine.extract_from_memory(&memory);

        assert!(!extractions.is_empty());
        assert!(matches!(
            extractions[0],
            Memory::Semantic(SemanticMemory {
                content: SemanticContent::Preference(_),
                ..
            })
        ));
    }

    #[test]
    fn test_extract_fact() {
        let storage = Arc::new(InMemoryStorage::new());
        let engine = DistillationEngine::new(storage, DistillationConfig::default());

        let memory = create_conversation("Rust is a systems programming language");
        let extractions = engine.extract_from_memory(&memory);

        assert!(!extractions.is_empty());
        assert!(matches!(
            extractions[0],
            Memory::Semantic(SemanticMemory {
                content: SemanticContent::Fact(_),
                ..
            })
        ));
    }

    #[test]
    fn test_no_extraction() {
        let storage = Arc::new(InMemoryStorage::new());
        let engine = DistillationEngine::new(storage, DistillationConfig::default());

        let memory = create_conversation("Hello, how are you?");
        let extractions = engine.extract_from_memory(&memory);

        assert!(extractions.is_empty());
    }
}
