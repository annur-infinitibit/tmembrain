//! Deterministic memory extractor for tests.

use async_trait::async_trait;
use parking_lot::Mutex;

use membrain_core::error::{Error, Result};
use membrain_core::traits::{ExtractedFact, ExtractedFactType, ExtractionResult, MemoryExtractor};

/// Deterministic `MemoryExtractor` implementation.
///
/// Default: splits input on sentence terminators (`.`, `!`, `?`) and returns
/// each non-empty sentence as a `Fact` with confidence 0.8. Use `with_result`
/// to return a canned response, `fail_with` to inject an error once.
pub struct DeterministicExtractor {
    canned: Mutex<Option<ExtractionResult>>,
    failure: Mutex<Option<Error>>,
    name: String,
    model: String,
}

impl DeterministicExtractor {
    /// Default extractor using sentence splitting.
    pub fn new() -> Self {
        Self {
            canned: Mutex::new(None),
            failure: Mutex::new(None),
            name: "deterministic-test".to_string(),
            model: "deterministic-sentences".to_string(),
        }
    }

    /// Extractor that returns a fixed result, ignoring input.
    pub fn with_result(result: ExtractionResult) -> Self {
        let extractor = Self::new();
        *extractor.canned.lock() = Some(result);
        extractor
    }

    /// Queue an error to be returned on the next `extract` call.
    pub fn fail_with(&self, error: Error) {
        *self.failure.lock() = Some(error);
    }
}

impl Default for DeterministicExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryExtractor for DeterministicExtractor {
    async fn extract(&self, text: &str) -> Result<ExtractionResult> {
        if let Some(error) = self.failure.lock().take() {
            return Err(error);
        }
        let canned = self.canned.lock().clone();
        if let Some(result) = canned {
            return Ok(result);
        }
        let facts: Vec<ExtractedFact> = text
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|sentence| !sentence.is_empty())
            .map(|sentence| ExtractedFact {
                fact_type: ExtractedFactType::Fact,
                content: sentence.to_string(),
                confidence: 0.8,
            })
            .collect();
        Ok(ExtractionResult { facts })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }
}
