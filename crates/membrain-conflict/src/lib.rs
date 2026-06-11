#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! LLM-based conflict resolution for Membrain agent memory.
//!
//! This crate provides intelligent memory management capabilities:
//!
//! - **Conflict Resolution**: Classifies new memories as ADD/UPDATE/DELETE/NOOP
//!   against existing similar memories, enabling automatic deduplication and
//!   contradiction resolution.
//!
//! # Usage
//!
//! The primary entry point is the [`ConflictResolver`] trait, with
//! [`OpenAiConflictResolver`] as the default implementation.

pub mod openai_resolver;
pub mod resolver;

pub use openai_resolver::OpenAiConflictResolver;
pub use resolver::{ConflictDecision, ConflictResolutionResult, ConflictResolver};
