//! Fake conflict resolver for contract tests.
//!
//! Not a mock — a real `ConflictResolver` implementation with scripted
//! responses. Feature-gated behind `conflict` so crates that don't need
//! it don't pull `membrain-conflict` transitively.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;

use membrain_conflict::resolver::{ConflictDecision, ConflictResolutionResult, ConflictResolver};
use membrain_core::error::Result;
use membrain_core::memory::Memory;

/// Fake conflict resolver with programmable responses.
pub struct FakeConflictResolver {
    behavior: Behavior,
    calls: AtomicUsize,
    name: String,
    model: String,
}

enum Behavior {
    Single(ConflictResolutionResult),
    Scripted(Mutex<VecDeque<ConflictResolutionResult>>),
}

impl FakeConflictResolver {
    /// Always returns Add (genuinely new) with confidence 1.0.
    pub fn always_add() -> Self {
        Self::with_decision(ConflictDecision::Add, 1.0, "injected ADD")
    }

    /// Always returns Update.
    pub fn always_update(target_id: membrain_core::types::MemoryId, merged: impl Into<String>) -> Self {
        Self::with_decision(
            ConflictDecision::Update {
                target_id,
                merged_content: merged.into(),
            },
            0.9,
            "injected UPDATE",
        )
    }

    /// Always returns Delete.
    pub fn always_delete(
        target_id: membrain_core::types::MemoryId,
        reason: impl Into<String>,
    ) -> Self {
        Self::with_decision(
            ConflictDecision::Delete {
                target_id,
                reason: reason.into(),
            },
            0.9,
            "injected DELETE",
        )
    }

    /// Always returns Noop.
    pub fn always_noop(reason: impl Into<String>) -> Self {
        Self::with_decision(
            ConflictDecision::Noop {
                reason: reason.into(),
            },
            0.9,
            "injected NOOP",
        )
    }

    fn with_decision(decision: ConflictDecision, confidence: f64, reasoning: &str) -> Self {
        Self {
            behavior: Behavior::Single(ConflictResolutionResult {
                decision,
                confidence,
                reasoning: reasoning.to_string(),
            }),
            calls: AtomicUsize::new(0),
            name: "fake-resolver".to_string(),
            model: "fake-model".to_string(),
        }
    }

    /// Returns each scripted result in order; panics intentionally via
    /// returning `Err` if called more times than scripts provided.
    pub fn scripted(results: Vec<ConflictResolutionResult>) -> Self {
        Self {
            behavior: Behavior::Scripted(Mutex::new(results.into())),
            calls: AtomicUsize::new(0),
            name: "fake-resolver".to_string(),
            model: "fake-model".to_string(),
        }
    }

    /// Number of `resolve` calls observed.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ConflictResolver for FakeConflictResolver {
    async fn resolve(
        &self,
        _new_memory: &Memory,
        _similar_memories: &[Memory],
    ) -> Result<ConflictResolutionResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.behavior {
            Behavior::Single(result) => Ok(result.clone()),
            Behavior::Scripted(queue) => queue.lock().pop_front().ok_or_else(|| {
                membrain_core::error::Error::Internal(
                    "FakeConflictResolver scripted queue exhausted".to_string(),
                )
            }),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }
}
