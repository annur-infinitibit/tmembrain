//! Novelty policy - evaluates information gain vs existing memories

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::config::NoveltyConfig;
use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::traits::{MemoryStorage, SearchFilters, SearchQuery};
use membrain_core::types::MemoryId;

use super::policy::{PolicyResult, WritePolicy};

/// Result of novelty evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyResult {
    /// Overall novelty score (0.0-1.0, higher = more novel)
    pub score: f64,
    /// Maximum similarity found with existing memories
    pub max_similarity: f64,
    /// IDs of most similar existing memories
    pub similar_memories: Vec<MemoryId>,
    /// Whether threshold was met
    pub passed: bool,
}

/// Policy that evaluates how novel/unique a memory is
pub struct NoveltyPolicy {
    config: NoveltyConfig,
}

impl NoveltyPolicy {
    /// Create a new novelty policy
    pub fn new(config: NoveltyConfig) -> Self {
        Self { config }
    }

    /// Calculate text similarity using Jaccard coefficient
    fn text_similarity(text1: &str, text2: &str) -> f64 {
        let lower1 = text1.to_lowercase();
        let lower2 = text2.to_lowercase();
        let words1: std::collections::HashSet<&str> = lower1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = lower2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f64 / union as f64
    }

    /// Calculate novelty score from similarities
    fn calculate_novelty(similarities: &[f64]) -> f64 {
        if similarities.is_empty() {
            return 1.0; // Completely novel
        }

        // Novelty is inverse of max similarity
        let max_sim = similarities.iter().cloned().fold(0.0, f64::max);
        1.0 - max_sim
    }
}

#[async_trait]
impl WritePolicy for NoveltyPolicy {
    fn name(&self) -> &str {
        "novelty"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    async fn evaluate(&self, memory: &Memory, storage: &dyn MemoryStorage) -> Result<PolicyResult> {
        if !self.config.enabled {
            return Ok(PolicyResult::Skipped {
                reason: "Novelty policy disabled".to_string(),
            });
        }

        let text = memory.text_content();

        // Search for similar memories.  Long or complex texts can contain
        // characters that the full-text query parser rejects.  When that
        // happens, treat the memory as fully novel rather than propagating
        // the error, since the write itself should still succeed.
        let filters = SearchFilters::new()
            .with_types(vec![memory.memory_type()])
            .exclude(vec![*memory.id()]);

        let query = SearchQuery::new()
            .with_query(&text)
            .with_limit(self.config.check_count)
            .with_filters(filters);

        let similar = match storage.search(query).await {
            Ok(results) => results,
            Err(_) => {
                tracing::debug!(
                    "Novelty text search failed for memory {}; treating as novel",
                    memory.id(),
                );
                Vec::new()
            }
        };

        // Calculate similarities. Prefer cosine over embeddings to catch
        // paraphrases that Jaccard would miss. Fall back to Jaccard when the
        // memory or candidate lacks an embedding, or when cosine fails
        // (dimension mismatch, zero-norm).
        let mut similarities = Vec::new();
        let mut similar_ids = Vec::new();

        for result in similar {
            let sim = match (memory.embedding(), result.memory.embedding()) {
                (Some(a), Some(b)) => match a.cosine_similarity(b) {
                    Ok(cos) => cos.max(0.0) as f64,
                    Err(_) => Self::text_similarity(&text, &result.memory.text_content()),
                },
                _ => Self::text_similarity(&text, &result.memory.text_content()),
            };
            if sim > 0.1 {
                similarities.push(sim);
                similar_ids.push(*result.memory.id());
            }
        }

        let max_similarity = similarities.iter().cloned().fold(0.0, f64::max);
        let novelty_score = Self::calculate_novelty(&similarities);

        let result = NoveltyResult {
            score: novelty_score,
            max_similarity,
            similar_memories: similar_ids,
            passed: novelty_score >= self.config.threshold,
        };

        if result.passed {
            Ok(PolicyResult::Pass {
                score: result.score,
                details: Some(format!(
                    "Novelty score: {:.2}, max similarity: {:.2}",
                    result.score, result.max_similarity
                )),
            })
        } else {
            Ok(PolicyResult::Reject {
                reason: format!(
                    "Novelty score {:.2} below threshold {:.2} (max similarity: {:.2})",
                    result.score, self.config.threshold, result.max_similarity
                ),
                score: result.score,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_similarity() {
        // Identical
        let sim = NoveltyPolicy::text_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.01);

        // Completely different
        let sim = NoveltyPolicy::text_similarity("hello world", "foo bar baz");
        assert!(sim < 0.1);

        // Partial overlap
        let sim = NoveltyPolicy::text_similarity("the quick brown fox", "the lazy brown dog");
        assert!(sim > 0.2 && sim < 0.8);
    }

    #[test]
    fn test_calculate_novelty() {
        // No similar items = completely novel
        let novelty = NoveltyPolicy::calculate_novelty(&[]);
        assert!((novelty - 1.0).abs() < 0.01);

        // Very similar item exists
        let novelty = NoveltyPolicy::calculate_novelty(&[0.9, 0.3, 0.2]);
        assert!((novelty - 0.1).abs() < 0.01);

        // Only mildly similar items
        let novelty = NoveltyPolicy::calculate_novelty(&[0.3, 0.2, 0.1]);
        assert!((novelty - 0.7).abs() < 0.01);
    }
}
