//! Contract tests for `MemoryExtractor` via `DeterministicExtractor`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use membrain_core::error::Error;
use membrain_core::traits::{ExtractedFact, ExtractedFactType, ExtractionResult, MemoryExtractor};
use membrain_test_utils::DeterministicExtractor;

#[tokio::test]
async fn extractor_splits_on_sentence_terminators() {
    let extractor = DeterministicExtractor::new();
    let result = extractor
        .extract("first fact. second fact! third fact?")
        .await
        .expect("extract");
    assert_eq!(result.facts.len(), 3);
    assert!(result.facts.iter().all(|fact| fact.fact_type == ExtractedFactType::Fact));
}

#[tokio::test]
async fn extractor_returns_canned_result_when_configured() {
    let canned = ExtractionResult {
        facts: vec![ExtractedFact {
            fact_type: ExtractedFactType::Preference,
            content: "likes pizza".to_string(),
            confidence: 0.95,
        }],
    };
    let extractor = DeterministicExtractor::with_result(canned.clone());
    let result = extractor.extract("ignored text").await.expect("extract");
    assert_eq!(result.facts.len(), 1);
    assert_eq!(result.facts[0].content, canned.facts[0].content);
}

#[tokio::test]
async fn extractor_failure_once_then_recovers() {
    let extractor = DeterministicExtractor::new();
    extractor.fail_with(Error::Internal("boom".to_string()));
    assert!(extractor.extract("first").await.is_err());
    assert!(extractor.extract("second. sentence.").await.is_ok());
}

#[tokio::test]
async fn extractor_reports_name_and_model() {
    let extractor = DeterministicExtractor::new();
    assert_eq!(extractor.name(), "deterministic-test");
    assert_eq!(extractor.model(), "deterministic-sentences");
}
