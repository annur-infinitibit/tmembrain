//! Retrieval gating - determines if retrieval should happen

use serde::{Deserialize, Serialize};

use super::intent::{IntentType, QueryIntent};

/// Decision on whether to proceed with retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingDecision {
    /// Whether to proceed with retrieval
    pub should_retrieve: bool,
    /// Reason for the decision
    pub reason: String,
    /// Confidence in the decision
    pub confidence: f64,
}

impl GatingDecision {
    /// Create a decision to proceed with retrieval
    pub fn proceed(reason: impl Into<String>) -> Self {
        Self {
            should_retrieve: true,
            reason: reason.into(),
            confidence: 0.8,
        }
    }

    /// Create a decision to skip retrieval
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            should_retrieve: false,
            reason: reason.into(),
            confidence: 0.8,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Gating logic for retrieval decisions
pub struct RetrievalGating {
    /// Whether gating is enabled
    enabled: bool,
    /// Minimum query length to trigger retrieval
    min_query_length: usize,
    /// Keywords that always trigger retrieval
    always_retrieve_keywords: Vec<String>,
    /// Keywords that never trigger retrieval
    never_retrieve_keywords: Vec<String>,
}

impl RetrievalGating {
    /// Create a new retrieval gating instance
    pub fn new() -> Self {
        Self {
            enabled: true,
            min_query_length: 2,
            always_retrieve_keywords: vec![
                "remember".to_string(),
                "recall".to_string(),
                "what did".to_string(),
                "preference".to_string(),
                "history".to_string(),
            ],
            never_retrieve_keywords: vec![
                "hello".to_string(),
                "hi".to_string(),
                "thanks".to_string(),
                "bye".to_string(),
            ],
        }
    }

    /// Enable or disable gating
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Evaluate whether retrieval should happen
    pub fn evaluate(&self, query: &str, intent: &QueryIntent) -> GatingDecision {
        if !self.enabled {
            return GatingDecision::proceed("Gating disabled");
        }

        let query_lower = query.to_lowercase();

        // Check never-retrieve keywords
        for keyword in &self.never_retrieve_keywords {
            if query_lower.contains(keyword) && query.split_whitespace().count() <= 2 {
                return GatingDecision::skip(format!(
                    "Contains '{}' (no retrieval needed)",
                    keyword
                ));
            }
        }

        // Check always-retrieve keywords
        for keyword in &self.always_retrieve_keywords {
            if query_lower.contains(keyword) {
                return GatingDecision::proceed(format!(
                    "Contains '{}' (retrieval required)",
                    keyword
                ))
                .with_confidence(0.9);
            }
        }

        // Skip only when the query has no meaningful content. Short domain
        // abbreviations like "AI", "ML", "CV" are legitimate queries and must
        // not be gated off purely on byte length.
        if query.trim().is_empty() {
            return GatingDecision::skip("Query empty");
        }
        if query.trim().chars().count() < self.min_query_length {
            return GatingDecision::skip("Query too short");
        }

        // Use intent detection
        match intent.intent_type {
            IntentType::NoRetrieval => GatingDecision::skip("Intent detected as no-retrieval"),
            IntentType::FactLookup
            | IntentType::PreferenceLookup
            | IntentType::ConversationRecall
            | IntentType::ProceduralLookup
            | IntentType::EntityLookup
            | IntentType::TaskStatus => {
                GatingDecision::proceed(format!("Intent: {:?}", intent.intent_type))
                    .with_confidence(intent.confidence)
            }
            IntentType::General => {
                // General queries should always proceed -- the caller explicitly
                // invoked search(), so they expect results.  Intent detection
                // still benefits ranking through adaptive scoring weights.
                GatingDecision::proceed("General query").with_confidence(0.5)
            }
        }
    }
}

impl Default for RetrievalGating {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::IntentDetector;

    #[test]
    fn test_gating_skip_greeting() {
        let gating = RetrievalGating::new();
        let detector = IntentDetector::new();

        let intent = detector.detect("hello");
        let decision = gating.evaluate("hello", &intent);

        assert!(!decision.should_retrieve);
    }

    #[test]
    fn test_gating_proceed_fact() {
        let gating = RetrievalGating::new();
        let detector = IntentDetector::new();

        let query = "What is the user's preferred programming language?";
        let intent = detector.detect(query);
        let decision = gating.evaluate(query, &intent);

        assert!(decision.should_retrieve);
    }

    #[test]
    fn test_gating_always_retrieve_keyword() {
        let gating = RetrievalGating::new();
        let detector = IntentDetector::new();

        let query = "Do you remember what we discussed?";
        let intent = detector.detect(query);
        let decision = gating.evaluate(query, &intent);

        assert!(decision.should_retrieve);
        assert!(decision.confidence > 0.8);
    }

    #[test]
    fn test_gating_disabled() {
        let mut gating = RetrievalGating::new();
        gating.set_enabled(false);

        let detector = IntentDetector::new();
        let intent = detector.detect("hi");
        let decision = gating.evaluate("hi", &intent);

        assert!(decision.should_retrieve);
    }
}
