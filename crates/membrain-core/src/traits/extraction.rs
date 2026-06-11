//! Memory extraction trait for deriving structured facts from raw memories

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// The type of fact extracted from a memory
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractedFactType {
    /// A concrete, verifiable statement
    Fact,
    /// A preference or opinion held by someone
    Preference,
    /// An event tied to a specific time
    TemporalEvent,
    /// A relationship between two entities
    Relationship,
}

/// A single fact extracted from a memory by an LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// The type of extracted fact
    #[serde(rename = "type")]
    pub fact_type: ExtractedFactType,
    /// The extracted fact as a standalone statement
    pub content: String,
    /// LLM-assigned confidence (0.0-1.0)
    pub confidence: f64,
}

/// Result of running memory extraction on a text
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    /// The extracted facts
    pub facts: Vec<ExtractedFact>,
}

/// Trait for LLM-based memory extraction.
///
/// Implementations call an LLM to extract structured facts, preferences,
/// temporal events, and relationships from raw text (typically conversation
/// messages or event descriptions).
#[async_trait]
pub trait MemoryExtractor: Send + Sync {
    /// Extract structured facts from the given text.
    ///
    /// Returns an empty result (not an error) when nothing worth extracting
    /// is found. Errors are reserved for infrastructure failures (network,
    /// auth, rate limits).
    async fn extract(&self, text: &str) -> Result<ExtractionResult>;

    /// Provider name (e.g. "openai")
    fn name(&self) -> &str;

    /// Model identifier (e.g. "gpt-4o-mini")
    fn model(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_fact_serde_roundtrip() {
        let fact = ExtractedFact {
            fact_type: ExtractedFactType::Preference,
            content: "Angela loves pizza".to_string(),
            confidence: 0.9,
        };

        let json = serde_json::to_string(&fact).unwrap();
        let deserialized: ExtractedFact = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.fact_type, ExtractedFactType::Preference);
        assert_eq!(deserialized.content, "Angela loves pizza");
        assert!((deserialized.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn extracted_fact_from_llm_json() {
        let llm_json = r#"{"type": "fact", "content": "Bob works at Google", "confidence": 0.85}"#;
        let fact: ExtractedFact = serde_json::from_str(llm_json).unwrap();
        assert_eq!(fact.fact_type, ExtractedFactType::Fact);
        assert_eq!(fact.content, "Bob works at Google");
    }

    #[test]
    fn extracted_fact_array_from_llm() {
        let llm_json = r#"[
            {"type": "fact", "content": "Angela loves pizza", "confidence": 0.9},
            {"type": "temporal_event", "content": "Angela visited Italy in 2021", "confidence": 0.7}
        ]"#;
        let facts: Vec<ExtractedFact> = serde_json::from_str(llm_json).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].fact_type, ExtractedFactType::Fact);
        assert_eq!(facts[1].fact_type, ExtractedFactType::TemporalEvent);
    }
}
