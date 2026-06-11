#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Multi-agent memory coordination for Membrain.
//!
//! Enables multiple LLM agents to share and synchronize memories
//! with conflict resolution and agent-scoped access control.

pub mod sharing;
pub mod trust;
pub mod visibility;

pub use sharing::{ShareRequest, SharingPolicy};
pub use trust::{AgentTrust, TrustLevel, TrustManager};
pub use visibility::{MemoryVisibility, VisibilityFilter};
