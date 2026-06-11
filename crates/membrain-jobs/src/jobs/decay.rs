//! Memory decay job

use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};

use membrain_compression::{DecayEngine, DecayPolicy};
use membrain_core::config::JobScheduleConfig;
use membrain_core::traits::MemoryStorage;

use super::{Job, JobResult};

/// Job that applies decay to memories
pub struct DecayJob {
    config: JobScheduleConfig,
    decay_engine: DecayEngine,
}

impl DecayJob {
    /// Create a new decay job
    pub fn new(storage: Arc<dyn MemoryStorage>, config: JobScheduleConfig) -> Self {
        let decay_engine = DecayEngine::new(storage, DecayPolicy::default());

        Self {
            config,
            decay_engine,
        }
    }

    /// Create with custom decay policy
    pub fn with_policy(
        storage: Arc<dyn MemoryStorage>,
        config: JobScheduleConfig,
        policy: DecayPolicy,
    ) -> Self {
        let decay_engine = DecayEngine::new(storage, policy);

        Self {
            config,
            decay_engine,
        }
    }
}

#[async_trait]
impl Job for DecayJob {
    fn name(&self) -> &str {
        "decay"
    }

    fn interval(&self) -> Duration {
        self.config.interval()
    }

    async fn execute(&self) -> JobResult {
        let start = Instant::now();

        match self.decay_engine.apply_decay(self.config.batch_size).await {
            Ok(result) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                JobResult::success(result.processed, duration_ms).with_details(format!(
                    "Processed {} memories, updated {}, deleted {}",
                    result.processed, result.updated, result.deleted
                ))
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                JobResult::failure(e.to_string(), duration_ms)
            }
        }
    }
}
