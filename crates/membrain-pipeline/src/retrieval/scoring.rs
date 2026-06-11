//! Scoring strategies for retrieval results

use serde::{Deserialize, Serialize};

use membrain_core::traits::SearchResult;

use super::intent::IntentType;

/// Weights for different scoring factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    /// Weight for search relevance score
    pub relevance: f64,
    /// Weight for confidence score
    pub confidence: f64,
    /// Weight for recency
    pub recency: f64,
    /// Weight for access frequency
    pub frequency: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            relevance: 0.5,
            confidence: 0.3,
            recency: 0.15,
            frequency: 0.05,
        }
    }
}

impl ScoreWeights {
    /// Return intent-optimized weights.
    ///
    /// Different query intents benefit from different scoring profiles:
    /// - Fact/entity lookups prioritize relevance heavily
    /// - Conversation recall weighs recency more
    /// - Task status needs the most recent information
    pub fn for_intent(intent: IntentType) -> Self {
        match intent {
            IntentType::FactLookup => Self {
                relevance: 0.80,
                confidence: 0.10,
                recency: 0.05,
                frequency: 0.05,
            },
            IntentType::PreferenceLookup => Self {
                relevance: 0.65,
                confidence: 0.20,
                recency: 0.10,
                frequency: 0.05,
            },
            IntentType::ConversationRecall => Self {
                relevance: 0.50,
                confidence: 0.10,
                recency: 0.30,
                frequency: 0.10,
            },
            IntentType::EntityLookup => Self {
                relevance: 0.75,
                confidence: 0.15,
                recency: 0.05,
                frequency: 0.05,
            },
            IntentType::ProceduralLookup => Self {
                relevance: 0.70,
                confidence: 0.15,
                recency: 0.05,
                frequency: 0.10,
            },
            IntentType::TaskStatus => Self {
                relevance: 0.40,
                confidence: 0.10,
                recency: 0.40,
                frequency: 0.10,
            },
            IntentType::General | IntentType::NoRetrieval => Self::default(),
        }
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.relevance + self.confidence + self.recency + self.frequency;
        if sum > 0.0 {
            self.relevance /= sum;
            self.confidence /= sum;
            self.recency /= sum;
            self.frequency /= sum;
        }
    }
}

/// Strategy for scoring retrieval results
pub struct ScoringStrategy {
    weights: ScoreWeights,
    /// Max age in hours for recency calculation
    max_age_hours: f64,
    /// Max access count for frequency normalization
    max_access_count: u64,
}

impl ScoringStrategy {
    /// Create a new scoring strategy
    pub fn new(weights: ScoreWeights) -> Self {
        Self {
            weights,
            max_age_hours: 24.0 * 30.0, // 30 days
            max_access_count: 100,
        }
    }

    /// Score a search result
    pub fn score(&self, result: &SearchResult) -> f64 {
        let memory = &result.memory;
        let common = memory.common();

        // Relevance from search
        let relevance_score = result.score;

        // Confidence score
        let confidence_score = common.confidence.value();

        // Recency score
        let age_hours = common.provenance.age().num_hours() as f64;
        let recency_score = (1.0 - (age_hours / self.max_age_hours)).max(0.0);

        // Frequency score
        let access_count = common.provenance.access_count;
        let frequency_score = (access_count as f64 / self.max_access_count as f64).min(1.0);

        // Weighted combination
        relevance_score * self.weights.relevance
            + confidence_score * self.weights.confidence
            + recency_score * self.weights.recency
            + frequency_score * self.weights.frequency
    }

    /// Score and sort results
    pub fn rank(&self, results: Vec<SearchResult>) -> Vec<(SearchResult, f64)> {
        let mut scored: Vec<(SearchResult, f64)> = results
            .into_iter()
            .map(|r| {
                let score = self.score(&r);
                (r, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Set custom weights
    pub fn with_weights(mut self, weights: ScoreWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Get current weights
    pub fn weights(&self) -> &ScoreWeights {
        &self.weights
    }
}

impl Default for ScoringStrategy {
    fn default() -> Self {
        Self::new(ScoreWeights::default())
    }
}

/// Diversity-aware reranking
pub struct DiversityReranker {
    /// Similarity threshold for considering duplicates
    similarity_threshold: f64,
    /// Maximum items from same memory type
    max_per_type: usize,
}

impl DiversityReranker {
    /// Create a new diversity reranker
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.8,
            max_per_type: 100,
        }
    }

    /// Set the maximum number of results per memory type
    pub fn with_max_per_type(mut self, max_per_type: usize) -> Self {
        self.max_per_type = max_per_type;
        self
    }

    /// Set the similarity threshold for duplicate detection
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Return an intent-aware diversity reranker.
    ///
    /// The per-type cap is intentionally generous because users frequently
    /// store all data under a single memory type (e.g. `store_event`).
    /// The primary deduplication mechanism is `similarity_threshold`, not
    /// the per-type cap.
    pub fn for_intent(intent: IntentType) -> Self {
        match intent {
            IntentType::ConversationRecall => Self::new().with_similarity_threshold(0.85),
            IntentType::FactLookup | IntentType::EntityLookup | IntentType::PreferenceLookup => {
                Self::new().with_similarity_threshold(0.80)
            }
            IntentType::ProceduralLookup => Self::new().with_similarity_threshold(0.80),
            IntentType::TaskStatus => Self::new().with_similarity_threshold(0.75),
            IntentType::General | IntentType::NoRetrieval => Self::new(),
        }
    }

    /// Rerank results for diversity
    pub fn rerank(&self, results: Vec<(SearchResult, f64)>) -> Vec<(SearchResult, f64)> {
        use std::collections::HashMap;

        let mut type_counts: HashMap<membrain_core::memory::MemoryType, usize> = HashMap::new();
        let mut selected = Vec::new();
        let mut seen_content = Vec::new();

        for (result, score) in results {
            let memory_type = result.memory.memory_type();

            // Check type limit
            let count = type_counts.entry(memory_type).or_insert(0);
            if *count >= self.max_per_type {
                continue;
            }

            // Check similarity to already selected
            let content = result.memory.text_content();
            let too_similar = seen_content.iter().any(|prev: &String| {
                self.text_similarity(prev, &content) > self.similarity_threshold
            });

            if too_similar {
                continue;
            }

            // Add to selected
            *count += 1;
            seen_content.push(content);
            selected.push((result, score));
        }

        selected
    }

    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let lower_a = a.to_lowercase();
        let lower_b = b.to_lowercase();
        let words_a: std::collections::HashSet<&str> = lower_a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = lower_b.split_whitespace().collect();

        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        intersection as f64 / union as f64
    }
}

impl Default for DiversityReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_weights_normalize() {
        let mut weights = ScoreWeights {
            relevance: 2.0,
            confidence: 1.0,
            recency: 0.5,
            frequency: 0.5,
        };

        weights.normalize();

        let sum = weights.relevance + weights.confidence + weights.recency + weights.frequency;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_default_weights_normalized() {
        let weights = ScoreWeights::default();
        let sum = weights.relevance + weights.confidence + weights.recency + weights.frequency;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_intent_adaptive_scoring_weights_sum_to_one() {
        let intents = [
            IntentType::FactLookup,
            IntentType::PreferenceLookup,
            IntentType::ConversationRecall,
            IntentType::EntityLookup,
            IntentType::ProceduralLookup,
            IntentType::TaskStatus,
            IntentType::General,
        ];

        for intent in intents {
            let weights = ScoreWeights::for_intent(intent);
            let sum = weights.relevance + weights.confidence + weights.recency + weights.frequency;
            assert!(
                (sum - 1.0).abs() < 0.001,
                "Weights for {:?} sum to {}, expected 1.0",
                intent,
                sum,
            );
        }
    }

    #[test]
    fn test_intent_adaptive_scoring_fact_prioritizes_relevance() {
        let fact_weights = ScoreWeights::for_intent(IntentType::FactLookup);
        let default_weights = ScoreWeights::default();
        assert!(fact_weights.relevance > default_weights.relevance);
    }

    #[test]
    fn test_intent_adaptive_scoring_conversation_prioritizes_recency() {
        let conv_weights = ScoreWeights::for_intent(IntentType::ConversationRecall);
        let default_weights = ScoreWeights::default();
        assert!(conv_weights.recency > default_weights.recency);
    }

    #[test]
    fn test_diversity_reranker_intent_aware() {
        let default_reranker = DiversityReranker::new();
        let conv_reranker = DiversityReranker::for_intent(IntentType::ConversationRecall);
        let task_reranker = DiversityReranker::for_intent(IntentType::TaskStatus);

        // Default max_per_type is generous to avoid capping single-type stores
        assert_eq!(default_reranker.max_per_type, 100);
        // Intent-aware rerankers vary similarity thresholds instead
        assert!((conv_reranker.similarity_threshold - 0.85).abs() < f64::EPSILON);
        assert!((task_reranker.similarity_threshold - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversity_reranker_builders() {
        let reranker = DiversityReranker::new()
            .with_max_per_type(10)
            .with_similarity_threshold(0.9);

        assert_eq!(reranker.max_per_type, 10);
        assert!((reranker.similarity_threshold - 0.9).abs() < f64::EPSILON);
    }
}
