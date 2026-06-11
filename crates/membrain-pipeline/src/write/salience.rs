//! Salience policy - evaluates content importance

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::config::SalienceConfig;
use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::traits::MemoryStorage;

use super::policy::{PolicyResult, WritePolicy};

/// Result of salience evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceResult {
    /// Overall salience score (0.0-1.0)
    pub score: f64,
    /// Content length score
    pub length_score: f64,
    /// Information density score
    pub density_score: f64,
    /// Specificity score
    pub specificity_score: f64,
    /// Whether threshold was met
    pub passed: bool,
}

/// Policy that evaluates content salience (importance)
pub struct SaliencePolicy {
    config: SalienceConfig,
}

impl SaliencePolicy {
    /// Create a new salience policy
    pub fn new(config: SalienceConfig) -> Self {
        Self { config }
    }

    /// Evaluate the salience of memory content
    pub fn evaluate_content(&self, memory: &Memory) -> SalienceResult {
        let text = memory.text_content();

        // Length score - not too short, not too long
        let length_score = self.score_length(&text);

        // Density score - information per character
        let density_score = self.score_density(&text);

        // Specificity score - how specific vs generic
        let specificity_score = self.score_specificity(&text);

        // Combine scores (weighted average)
        let score = length_score * 0.2 + density_score * 0.4 + specificity_score * 0.4;

        // Apply confidence boost
        let confidence_boost = memory.confidence().value() * 0.2;
        let final_score = (score + confidence_boost).min(1.0);

        SalienceResult {
            score: final_score,
            length_score,
            density_score,
            specificity_score,
            passed: final_score >= self.config.threshold,
        }
    }

    fn score_length(&self, text: &str) -> f64 {
        let len = text.len();

        // Optimal range: 50-500 characters
        if len < 10 {
            0.1
        } else if len < 50 {
            0.3 + (len as f64 / 50.0) * 0.3
        } else if len <= 500 {
            0.8 + (1.0 - ((len as f64 - 50.0) / 450.0).abs()) * 0.2
        } else if len <= 2000 {
            0.7 - ((len as f64 - 500.0) / 1500.0) * 0.3
        } else {
            0.4
        }
    }

    fn score_density(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Unique words ratio
        let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
        let uniqueness = unique.len() as f64 / words.len() as f64;

        // Average word length (longer words often carry more information)
        let avg_word_len: f64 =
            words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64;
        let word_len_score = (avg_word_len / 8.0).min(1.0); // Normalize to ~8 char avg

        // Numbers and special patterns often indicate specific information
        let has_numbers = text.chars().any(|c| c.is_numeric());
        let has_special = text.contains('@') || text.contains("://") || text.contains('#');
        let specificity_bonus =
            if has_numbers { 0.1 } else { 0.0 } + if has_special { 0.05 } else { 0.0 };

        (uniqueness * 0.5 + word_len_score * 0.3 + specificity_bonus).min(1.0)
    }

    fn score_specificity(&self, text: &str) -> f64 {
        let text_lower = text.to_lowercase();

        // Penalize very generic phrases
        let generic_phrases = [
            "i think",
            "maybe",
            "probably",
            "something",
            "stuff",
            "things",
            "etc",
            "and so on",
            "you know",
            "kind of",
            "sort of",
        ];

        let generic_count = generic_phrases
            .iter()
            .filter(|p| text_lower.contains(*p))
            .count();

        // Reward specific indicators
        let specific_indicators = [
            // Names (capitalized words in middle of text)
            text.split_whitespace()
                .skip(1)
                .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)),
            // Numbers
            text.chars().any(|c| c.is_numeric()),
            // Dates/times patterns
            text.contains('/') || text.contains(':'),
            // Technical terms (camelCase, snake_case)
            text.contains('_') || text.chars().filter(|c| c.is_uppercase()).count() > 2,
        ];

        let specific_count = specific_indicators.iter().filter(|&&b| b).count();

        let base_score = 0.5;
        let penalty = generic_count as f64 * 0.1;
        let bonus = specific_count as f64 * 0.15;

        (base_score - penalty + bonus).clamp(0.0, 1.0)
    }
}

#[async_trait]
impl WritePolicy for SaliencePolicy {
    fn name(&self) -> &str {
        "salience"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    async fn evaluate(
        &self,
        memory: &Memory,
        _storage: &dyn MemoryStorage,
    ) -> Result<PolicyResult> {
        if !self.config.enabled {
            return Ok(PolicyResult::Skipped {
                reason: "Salience policy disabled".to_string(),
            });
        }

        // Check if this memory type is exempt
        if self.config.exempt_types.contains(&memory.memory_type()) {
            return Ok(PolicyResult::Skipped {
                reason: format!("Memory type {:?} is exempt", memory.memory_type()),
            });
        }

        let result = self.evaluate_content(memory);

        if result.passed {
            Ok(PolicyResult::Pass {
                score: result.score,
                details: Some(format!(
                    "length={:.2}, density={:.2}, specificity={:.2}",
                    result.length_score, result.density_score, result.specificity_score
                )),
            })
        } else {
            Ok(PolicyResult::Reject {
                reason: format!(
                    "Salience score {:.2} below threshold {:.2}",
                    result.score, self.config.threshold
                ),
                score: result.score,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{AgentId, Confidence, Provenance, Source};

    fn create_memory(text: &str) -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new(text)),
        })
    }

    #[test]
    fn test_length_scoring() {
        let policy = SaliencePolicy::new(SalienceConfig::default());

        // Very short - low score
        assert!(policy.score_length("hi") < 0.3);

        // Medium length - good score
        let medium = "This is a medium length statement about something specific.";
        assert!(policy.score_length(medium) > 0.6);

        // Long but reasonable - still good
        let long = "This is a longer statement. ".repeat(10);
        assert!(policy.score_length(&long) > 0.5);
    }

    #[test]
    fn test_density_scoring() {
        let policy = SaliencePolicy::new(SalienceConfig::default());

        // Repetitive - low density
        let repetitive = "the the the the the";
        assert!(policy.score_density(repetitive) < 0.5);

        // Diverse vocabulary - higher density
        let diverse =
            "Machine learning algorithms process data efficiently using mathematical models";
        assert!(policy.score_density(diverse) > 0.5);
    }

    #[test]
    fn test_specificity_scoring() {
        let policy = SaliencePolicy::new(SalienceConfig::default());

        // Generic
        let generic = "I think maybe something happened or whatever";
        let generic_score = policy.score_specificity(generic);

        // Specific
        let specific = "John Smith's API endpoint at https://example.com returns 404";
        let specific_score = policy.score_specificity(specific);

        assert!(specific_score > generic_score);
    }

    #[test]
    fn test_overall_evaluation() {
        let config = SalienceConfig {
            enabled: true,
            threshold: 0.3,
            exempt_types: vec![],
        };
        let policy = SaliencePolicy::new(config);

        // Good content should pass
        let good_memory =
            create_memory("The user prefers dark mode for the IDE and uses vim keybindings");
        let result = policy.evaluate_content(&good_memory);
        assert!(result.passed);

        // Poor content should fail
        let poor_config = SalienceConfig {
            enabled: true,
            threshold: 0.8, // Very high threshold
            exempt_types: vec![],
        };
        let strict_policy = SaliencePolicy::new(poor_config);
        let poor_memory = create_memory("ok");
        let result = strict_policy.evaluate_content(&poor_memory);
        assert!(!result.passed);
    }
}
