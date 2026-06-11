#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Background job processing for Membrain.
//!
//! Handles asynchronous tasks like memory consolidation,
//! embedding generation, and scheduled maintenance.

pub mod jobs;
pub mod scheduler;

pub use jobs::{Job, JobResult, JobStatus};
pub use scheduler::{JobHandle, JobScheduler};
