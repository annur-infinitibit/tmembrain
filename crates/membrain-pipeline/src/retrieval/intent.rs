//! Query intent detection for retrieval optimization

use serde::{Deserialize, Serialize};

use membrain_core::memory::MemoryType;

/// Detected intent of a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryIntent {
    /// Primary intent type
    pub intent_type: IntentType,
    /// Confidence in the detection
    pub confidence: f64,
    /// Suggested memory types to search
    pub suggested_types: Vec<MemoryType>,
    /// Whether this query likely needs memory retrieval
    pub needs_retrieval: bool,
    /// Keywords extracted from query
    pub keywords: Vec<String>,
    /// Time reference if detected
    pub time_reference: Option<TimeReference>,
}

/// Types of query intent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    /// Looking for specific facts
    FactLookup,
    /// Looking for preferences
    PreferenceLookup,
    /// Looking for past conversations
    ConversationRecall,
    /// Looking for how to do something
    ProceduralLookup,
    /// Looking for entity information
    EntityLookup,
    /// Looking for current tasks/goals
    TaskStatus,
    /// General/unclear intent
    General,
    /// No retrieval needed (greeting, simple query)
    NoRetrieval,
}

/// Time reference in a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeReference {
    /// Recent (last few hours/day)
    Recent,
    /// Specific date or range
    Specific(String),
    /// Historical (long ago)
    Historical,
    /// Current/ongoing
    Current,
}

/// Intent detector for analyzing queries
pub struct IntentDetector {
    /// Keywords indicating fact lookup
    fact_keywords: Vec<&'static str>,
    /// Keywords indicating preference lookup
    preference_keywords: Vec<&'static str>,
    /// Keywords indicating conversation recall
    conversation_keywords: Vec<&'static str>,
    /// Keywords indicating procedural lookup
    procedural_keywords: Vec<&'static str>,
    /// Keywords indicating no retrieval needed
    no_retrieval_keywords: Vec<&'static str>,
}

impl IntentDetector {
    /// Create a new intent detector
    pub fn new() -> Self {
        Self {
            fact_keywords: vec![
                "what is",
                "what are",
                "who is",
                "where is",
                "when did",
                "how many",
                "tell me about",
                "explain",
                "describe",
            ],
            preference_keywords: vec![
                "prefer",
                "like",
                "want",
                "favorite",
                "preferred",
                "usually",
                "always",
                "never",
                "settings",
                "configuration",
            ],
            conversation_keywords: vec![
                "we discussed",
                "we talked",
                "you said",
                "i said",
                "earlier",
                "last time",
                "remember when",
                "conversation",
                "mentioned",
            ],
            procedural_keywords: vec![
                "how do i",
                "how to",
                "steps to",
                "process for",
                "workflow",
                "procedure",
                "instructions",
                "guide me",
                "help me",
            ],
            no_retrieval_keywords: vec![
                "hello",
                "hi",
                "hey",
                "thanks",
                "thank you",
                "bye",
                "goodbye",
                "yes",
                "no",
                "ok",
                "okay",
                "sure",
            ],
        }
    }

    /// Detect the intent of a query
    pub fn detect(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        // Check for no-retrieval patterns first
        if self.is_no_retrieval(&query_lower) {
            return QueryIntent {
                intent_type: IntentType::NoRetrieval,
                confidence: 0.9,
                suggested_types: vec![],
                needs_retrieval: false,
                keywords: vec![],
                time_reference: None,
            };
        }

        // Detect intent type and collect scores
        let mut scores = vec![
            (
                IntentType::FactLookup,
                self.score_keywords(&query_lower, &self.fact_keywords),
            ),
            (
                IntentType::PreferenceLookup,
                self.score_keywords(&query_lower, &self.preference_keywords),
            ),
            (
                IntentType::ConversationRecall,
                self.score_keywords(&query_lower, &self.conversation_keywords),
            ),
            (
                IntentType::ProceduralLookup,
                self.score_keywords(&query_lower, &self.procedural_keywords),
            ),
        ];

        // Add entity detection
        if self.has_named_entity(&query_lower) {
            scores.push((IntentType::EntityLookup, 0.5));
        }

        // Add task detection
        if self.is_task_query(&query_lower) {
            scores.push((IntentType::TaskStatus, 0.7));
        }

        // Find best match
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (intent_type, confidence) = if scores[0].1 > 0.05 {
            // Even a single keyword match is significant
            scores[0]
        } else {
            (IntentType::General, 0.3)
        };

        // Determine suggested types
        let suggested_types = self.suggest_types(intent_type);

        // Extract keywords
        let keywords = self.extract_keywords(&query_lower);

        // Detect time reference
        let time_reference = self.detect_time_reference(&query_lower);

        QueryIntent {
            intent_type,
            confidence,
            suggested_types,
            needs_retrieval: intent_type != IntentType::NoRetrieval,
            keywords,
            time_reference,
        }
    }

    fn is_no_retrieval(&self, query: &str) -> bool {
        // Very short queries that are likely greetings
        if query.split_whitespace().count() <= 2 {
            for keyword in &self.no_retrieval_keywords {
                if query.contains(keyword) {
                    return true;
                }
            }
        }
        false
    }

    fn score_keywords(&self, query: &str, keywords: &[&str]) -> f64 {
        let matches: usize = keywords.iter().filter(|k| query.contains(*k)).count();
        (matches as f64 / keywords.len() as f64).min(1.0)
    }

    fn has_named_entity(&self, query: &str) -> bool {
        // Simple heuristic: capitalized words that aren't at start
        query
            .split_whitespace()
            .skip(1)
            .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
    }

    fn is_task_query(&self, query: &str) -> bool {
        let task_indicators = ["task", "todo", "goal", "current", "working on", "status"];
        task_indicators.iter().any(|t| query.contains(t))
    }

    fn suggest_types(&self, intent: IntentType) -> Vec<MemoryType> {
        match intent {
            IntentType::FactLookup => vec![
                MemoryType::SemanticFact,
                MemoryType::SemanticEntity,
                MemoryType::SemanticConcept,
                MemoryType::EpisodicConversation,
            ],
            IntentType::PreferenceLookup => vec![
                MemoryType::SemanticPreference,
                MemoryType::SemanticFact,
                MemoryType::EpisodicConversation,
            ],
            IntentType::ConversationRecall => {
                vec![MemoryType::EpisodicConversation, MemoryType::EpisodicEvent]
            }
            IntentType::ProceduralLookup => vec![
                MemoryType::ProceduralWorkflow,
                MemoryType::ProceduralSkill,
                MemoryType::ProceduralPattern,
                MemoryType::ProceduralCase,
            ],
            IntentType::EntityLookup => vec![
                MemoryType::SemanticEntity,
                MemoryType::SemanticFact,
                MemoryType::EpisodicConversation,
            ],
            IntentType::TaskStatus => vec![
                MemoryType::AgentStateTask,
                MemoryType::AgentStateGoal,
                MemoryType::AgentStateWorkingMemory,
            ],
            IntentType::General => vec![
                MemoryType::SemanticFact,
                MemoryType::SemanticPreference,
                MemoryType::EpisodicConversation,
                MemoryType::SemanticEntity,
            ],
            IntentType::NoRetrieval => vec![],
        }
    }

    fn extract_keywords(&self, query: &str) -> Vec<String> {
        let stopwords = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "i", "me", "my", "myself",
            "we", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
            "them", "their", "what", "which", "who", "this", "that", "these", "those", "am",
            "user's",
        ];

        query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 2)
            .filter(|w| !stopwords.contains(&w.to_lowercase().as_str()))
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn detect_time_reference(&self, query: &str) -> Option<TimeReference> {
        let recent_indicators = [
            "just",
            "recently",
            "today",
            "yesterday",
            "last hour",
            "earlier",
        ];
        let historical_indicators = ["long ago", "way back", "originally", "first", "initially"];
        let current_indicators = ["now", "current", "currently", "ongoing", "active"];

        if recent_indicators.iter().any(|t| query.contains(t)) {
            Some(TimeReference::Recent)
        } else if historical_indicators.iter().any(|t| query.contains(t)) {
            Some(TimeReference::Historical)
        } else if current_indicators.iter().any(|t| query.contains(t)) {
            Some(TimeReference::Current)
        } else {
            None
        }
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_lookup_detection() {
        let detector = IntentDetector::new();

        let intent = detector.detect("What is the capital of France?");
        assert_eq!(intent.intent_type, IntentType::FactLookup);
        assert!(intent.needs_retrieval);
    }

    #[test]
    fn test_preference_lookup_detection() {
        let detector = IntentDetector::new();

        let intent = detector.detect("What does the user prefer for the theme?");
        assert_eq!(intent.intent_type, IntentType::PreferenceLookup);
    }

    #[test]
    fn test_no_retrieval_detection() {
        let detector = IntentDetector::new();

        let intent = detector.detect("Hello!");
        assert_eq!(intent.intent_type, IntentType::NoRetrieval);
        assert!(!intent.needs_retrieval);
    }

    #[test]
    fn test_procedural_detection() {
        let detector = IntentDetector::new();

        let intent = detector.detect("How do I deploy the application?");
        assert_eq!(intent.intent_type, IntentType::ProceduralLookup);
        assert!(intent
            .suggested_types
            .contains(&MemoryType::ProceduralWorkflow));
    }

    #[test]
    fn test_keyword_extraction() {
        let detector = IntentDetector::new();

        let intent = detector.detect("What are the user's coding preferences?");
        assert!(intent.keywords.contains(&"coding".to_string()));
        assert!(intent.keywords.contains(&"preferences".to_string()));
    }

    #[test]
    fn test_time_reference_detection() {
        let detector = IntentDetector::new();

        let intent = detector.detect("What did we discuss yesterday?");
        assert!(matches!(intent.time_reference, Some(TimeReference::Recent)));

        let intent = detector.detect("What's the current status?");
        assert!(matches!(
            intent.time_reference,
            Some(TimeReference::Current)
        ));
    }
}
