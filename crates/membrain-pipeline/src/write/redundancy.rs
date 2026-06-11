//! Redundancy policy - detects duplicate and near-duplicate memories

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use membrain_core::config::RedundancyConfig;
use membrain_core::error::Result;
use membrain_core::memory::Memory;
use membrain_core::traits::{MemoryStorage, SearchFilters, SearchQuery};
use membrain_core::types::MemoryId;

use super::policy::{PolicyResult, WritePolicy};

/// Result of redundancy check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyResult {
    /// Whether a redundant memory was found
    pub is_redundant: bool,
    /// ID of the redundant memory (if found)
    pub redundant_with: Option<MemoryId>,
    /// Similarity score with most similar memory
    pub similarity: f64,
    /// Recommendation
    pub recommendation: RedundancyRecommendation,
}

/// Recommendation for handling redundancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyRecommendation {
    /// Memory is unique, proceed
    Store,
    /// Memory is very similar, should merge
    Merge { target_id: MemoryId },
    /// Memory is duplicate, should reject
    Reject { duplicate_id: MemoryId },
    /// Memory adds new info, update existing
    Update { target_id: MemoryId },
}

/// Policy that detects redundant memories
pub struct RedundancyPolicy {
    config: RedundancyConfig,
}

impl RedundancyPolicy {
    /// Create a new redundancy policy
    pub fn new(config: RedundancyConfig) -> Self {
        Self { config }
    }

    /// Calculate detailed similarity between two texts
    fn calculate_similarity(text1: &str, text2: &str) -> f64 {
        // Normalize texts
        let norm1: String = text1
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        let norm2: String = text2
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        let words1: Vec<&str> = norm1.split_whitespace().collect();
        let words2: Vec<&str> = norm2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Jaccard similarity
        let set1: std::collections::HashSet<&str> = words1.iter().copied().collect();
        let set2: std::collections::HashSet<&str> = words2.iter().copied().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        let jaccard = intersection as f64 / union as f64;

        // Also consider ordering with longest common subsequence ratio
        let lcs_ratio = Self::lcs_ratio(&words1, &words2);

        // Combine both measures
        jaccard * 0.6 + lcs_ratio * 0.4
    }

    /// Calculate longest common subsequence ratio
    fn lcs_ratio(words1: &[&str], words2: &[&str]) -> f64 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let n = words1.len();
        let m = words2.len();
        let mut dp = vec![vec![0; m + 1]; n + 1];

        for i in 1..=n {
            for j in 1..=m {
                if words1[i - 1] == words2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        let lcs_len = dp[n][m];
        let max_len = n.max(m);
        lcs_len as f64 / max_len as f64
    }

    /// Determine if new memory adds information to existing
    fn adds_new_info(new_text: &str, existing_text: &str) -> bool {
        let new_lower = new_text.to_lowercase();
        let existing_lower = existing_text.to_lowercase();
        let new_words: std::collections::HashSet<&str> = new_lower.split_whitespace().collect();
        let existing_words: std::collections::HashSet<&str> =
            existing_lower.split_whitespace().collect();

        // Check if new memory has words not in existing
        let new_only: Vec<_> = new_words.difference(&existing_words).collect();

        // Consider it adds info if >20% of words are new
        let new_ratio = new_only.len() as f64 / new_words.len().max(1) as f64;
        new_ratio > 0.2
    }
}

#[async_trait]
impl WritePolicy for RedundancyPolicy {
    fn name(&self) -> &str {
        "redundancy"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    async fn evaluate(&self, memory: &Memory, storage: &dyn MemoryStorage) -> Result<PolicyResult> {
        if !self.config.enabled {
            return Ok(PolicyResult::Skipped {
                reason: "Redundancy policy disabled".to_string(),
            });
        }

        let text = memory.text_content();

        // Search for similar memories of same type
        let filters = SearchFilters::new()
            .with_types(vec![memory.memory_type()])
            .exclude(vec![*memory.id()]);

        let query = SearchQuery::new()
            .with_query(&text)
            .with_limit(10)
            .with_filters(filters);

        // Long or complex texts can trip the full-text query parser.
        // Treat parser failures as "no similar found" so the store succeeds.
        let similar = match storage.search(query).await {
            Ok(results) => results,
            Err(_) => {
                tracing::debug!(
                    "Redundancy text search failed for memory {}; treating as non-redundant",
                    memory.id(),
                );
                Vec::new()
            }
        };

        // Find most similar memory. Prefer cosine over embeddings (captures
        // paraphrases and synonyms). Fall back to Jaccard when either side
        // lacks an embedding or the cosine comparison fails (dimension
        // mismatch, zero-norm vector).
        let mut max_similarity = 0.0;
        let mut most_similar: Option<&Memory> = None;

        for result in &similar {
            let sim = match (memory.embedding(), result.memory.embedding()) {
                (Some(a), Some(b)) => match a.cosine_similarity(b) {
                    Ok(cos) => cos.max(0.0) as f64,
                    Err(_) => Self::calculate_similarity(&text, &result.memory.text_content()),
                },
                _ => Self::calculate_similarity(&text, &result.memory.text_content()),
            };
            if sim > max_similarity {
                max_similarity = sim;
                most_similar = Some(&result.memory);
            }
        }

        let (is_redundant, recommendation) = match most_similar {
            Some(existing) if max_similarity >= self.config.similarity_threshold => {
                let rec = if max_similarity > 0.98 {
                    RedundancyRecommendation::Reject {
                        duplicate_id: *existing.id(),
                    }
                } else if self.config.auto_merge {
                    RedundancyRecommendation::Merge {
                        target_id: *existing.id(),
                    }
                } else {
                    RedundancyRecommendation::Reject {
                        duplicate_id: *existing.id(),
                    }
                };
                (true, rec)
            }
            Some(existing) if max_similarity > 0.7 => {
                if Self::adds_new_info(&text, &existing.text_content()) {
                    (
                        false,
                        RedundancyRecommendation::Update {
                            target_id: *existing.id(),
                        },
                    )
                } else {
                    (false, RedundancyRecommendation::Store)
                }
            }
            _ => (false, RedundancyRecommendation::Store),
        };

        let result = RedundancyResult {
            is_redundant,
            redundant_with: most_similar.map(|m| *m.id()),
            similarity: max_similarity,
            recommendation: recommendation.clone(),
        };

        match result.recommendation {
            RedundancyRecommendation::Store | RedundancyRecommendation::Update { .. } => {
                Ok(PolicyResult::Pass {
                    score: 1.0 - result.similarity,
                    details: Some(format!(
                        "Max similarity: {:.2}, recommendation: {:?}",
                        result.similarity, result.recommendation
                    )),
                })
            }
            RedundancyRecommendation::Merge { target_id } => Ok(PolicyResult::Merge {
                merge_with: target_id,
                similarity: result.similarity,
            }),
            RedundancyRecommendation::Reject { duplicate_id } => Ok(PolicyResult::Reject {
                reason: format!(
                    "Redundant with memory {} (similarity: {:.2})",
                    duplicate_id, result.similarity
                ),
                score: result.similarity,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_similarity() {
        // Identical
        let sim = RedundancyPolicy::calculate_similarity("hello world", "hello world");
        assert!(sim > 0.95);

        // Very similar
        let sim =
            RedundancyPolicy::calculate_similarity("the quick brown fox", "the quick brown dog");
        assert!(sim > 0.5);

        // Different
        let sim = RedundancyPolicy::calculate_similarity("hello world", "foo bar baz qux");
        assert!(sim < 0.3);
    }

    #[test]
    fn test_lcs_ratio() {
        let words1: Vec<&str> = "the quick brown fox".split_whitespace().collect();
        let words2: Vec<&str> = "the quick brown dog".split_whitespace().collect();

        let ratio = RedundancyPolicy::lcs_ratio(&words1, &words2);
        assert!((ratio - 0.75).abs() < 0.01); // LCS = "the quick brown" = 3, max = 4
    }

    #[test]
    fn test_adds_new_info() {
        // Adds significant new info
        let adds = RedundancyPolicy::adds_new_info(
            "The user prefers dark mode and vim keybindings",
            "The user prefers dark mode",
        );
        assert!(adds);

        // Doesn't add much new info
        let adds = RedundancyPolicy::adds_new_info(
            "The user prefers dark mode",
            "The user really prefers dark mode a lot",
        );
        assert!(!adds);
    }
}
