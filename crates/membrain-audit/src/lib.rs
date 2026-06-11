#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Audit logging for Membrain memory operations.
//!
//! Records store, search, and delete operations for observability
//! and compliance in LLM memory systems.

pub mod decision_log;
pub mod metrics;

pub use decision_log::{AuditEntry, AuditEntryType, AuditLog, DecisionContext, DecisionOutcome};
pub use metrics::{Metrics, MetricsSnapshot, OperationMetrics};
