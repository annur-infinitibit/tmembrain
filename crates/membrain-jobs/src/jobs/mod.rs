//! Job definitions and implementations

mod consolidation;
mod decay;

pub use consolidation::ConsolidationJob;
pub use decay::DecayJob;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Trait for background jobs
#[async_trait]
pub trait Job: Send + Sync {
    /// Get the job name
    fn name(&self) -> &str;

    /// Get the interval between runs
    fn interval(&self) -> Duration;

    /// Execute the job
    async fn execute(&self) -> JobResult;

    /// Check if the job should run now
    fn should_run(&self, last_run: Option<DateTime<Utc>>) -> bool {
        match last_run {
            Some(last) => {
                let elapsed = Utc::now() - last;
                elapsed.num_seconds() >= self.interval().as_secs() as i64
            }
            None => true,
        }
    }
}

/// Result of a job execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Whether the job succeeded
    pub success: bool,
    /// Duration of execution
    pub duration_ms: u64,
    /// Items processed
    pub items_processed: usize,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional details
    pub details: Option<String>,
}

impl JobResult {
    /// Create a successful result
    pub fn success(items_processed: usize, duration_ms: u64) -> Self {
        Self {
            success: true,
            duration_ms,
            items_processed,
            error: None,
            details: None,
        }
    }

    /// Create a failed result
    pub fn failure(error: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            success: false,
            duration_ms,
            items_processed: 0,
            error: Some(error.into()),
            details: None,
        }
    }

    /// Add details
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Status of a job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is idle/waiting
    Idle,
    /// Job is currently running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job is disabled
    Disabled,
}
