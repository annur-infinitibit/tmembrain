//! Memory consolidation job

use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};

use membrain_compression::{DistillationConfig, DistillationEngine};
use membrain_core::config::JobScheduleConfig;
use membrain_core::traits::MemoryStorage;

use super::{Job, JobResult};

/// Job that consolidates and distills memories
pub struct ConsolidationJob {
    config: JobScheduleConfig,
    distillation_engine: DistillationEngine,
}

impl ConsolidationJob {
    /// Create a new consolidation job
    pub fn new(storage: Arc<dyn MemoryStorage>, config: JobScheduleConfig) -> Self {
        let distillation_engine = DistillationEngine::new(storage, DistillationConfig::default());

        Self {
            config,
            distillation_engine,
        }
    }
}

#[async_trait]
impl Job for ConsolidationJob {
    fn name(&self) -> &str {
        "consolidation"
    }

    fn interval(&self) -> Duration {
        self.config.interval()
    }

    async fn execute(&self) -> JobResult {
        let start = Instant::now();

        match self.distillation_engine.run(self.config.batch_size).await {
            Ok(result) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                JobResult::success(result.processed, duration_ms).with_details(format!(
                    "Processed {} memories, extracted {} semantic memories",
                    result.processed, result.extracted
                ))
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                JobResult::failure(e.to_string(), duration_ms)
            }
        }
    }
}
