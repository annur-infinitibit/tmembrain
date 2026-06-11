//! Job scheduler for running background tasks

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::jobs::{Job, JobResult, JobStatus};

/// Handle to control a running scheduler
pub struct JobHandle {
    shutdown_tx: mpsc::Sender<()>,
}

impl JobHandle {
    /// Signal the scheduler to stop
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }
}

/// State of a registered job
struct JobState {
    job: Arc<dyn Job>,
    status: JobStatus,
    last_run: Option<DateTime<Utc>>,
    last_result: Option<JobResult>,
    run_count: u64,
    failure_count: u64,
}

/// Background job scheduler
#[derive(Clone)]
pub struct JobScheduler {
    jobs: Arc<RwLock<HashMap<String, JobState>>>,
    max_concurrent: usize,
    check_interval: Duration,
}

impl JobScheduler {
    /// Create a new job scheduler
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent,
            check_interval: Duration::from_secs(10),
        }
    }

    /// Set the check interval
    pub fn with_check_interval(mut self, interval: Duration) -> Self {
        self.check_interval = interval;
        self
    }

    /// Register a job
    pub async fn register(&self, job: impl Job + 'static) {
        let name = job.name().to_string();
        let state = JobState {
            job: Arc::new(job),
            status: JobStatus::Idle,
            last_run: None,
            last_result: None,
            run_count: 0,
            failure_count: 0,
        };

        self.jobs.write().await.insert(name.clone(), state);
        info!(job = %name, "Registered job");
    }

    /// Start the scheduler
    pub fn start(self: Arc<Self>) -> JobHandle {
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        let scheduler = self.clone();
        tokio::spawn(async move {
            let mut check_interval = interval(scheduler.check_interval);

            loop {
                tokio::select! {
                    _ = check_interval.tick() => {
                        scheduler.check_and_run_jobs().await;
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Job scheduler shutting down");
                        break;
                    }
                }
            }
        });

        JobHandle { shutdown_tx }
    }

    async fn check_and_run_jobs(&self) {
        let jobs = self.jobs.read().await;
        let mut jobs_to_run = Vec::new();

        // Find jobs that should run
        for (name, state) in jobs.iter() {
            if state.status == JobStatus::Disabled || state.status == JobStatus::Running {
                continue;
            }

            if state.job.should_run(state.last_run) {
                jobs_to_run.push(name.clone());
            }
        }
        drop(jobs);

        // Limit concurrent jobs
        jobs_to_run.truncate(self.max_concurrent);

        // Run jobs concurrently
        let mut handles = Vec::new();
        for name in jobs_to_run {
            let scheduler = self.clone();
            let job_name = name.clone();

            let handle = tokio::spawn(async move {
                scheduler.run_job(&job_name).await;
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _ = handle.await;
        }
    }

    async fn run_job(&self, name: &str) {
        // Mark as running
        {
            let mut jobs = self.jobs.write().await;
            if let Some(state) = jobs.get_mut(name) {
                state.status = JobStatus::Running;
            }
        }

        // Get job reference
        let job = {
            let jobs = self.jobs.read().await;
            jobs.get(name).map(|s| s.job.clone())
        };

        let result = if let Some(job) = job {
            debug!(job = %name, "Starting job");
            job.execute().await
        } else {
            warn!(job = %name, "Job not found");
            return;
        };

        // Update state
        {
            let mut jobs = self.jobs.write().await;
            if let Some(state) = jobs.get_mut(name) {
                state.last_run = Some(Utc::now());
                state.run_count += 1;

                if result.success {
                    state.status = JobStatus::Completed;
                    info!(
                        job = %name,
                        items = result.items_processed,
                        duration_ms = result.duration_ms,
                        "Job completed successfully"
                    );
                } else {
                    state.status = JobStatus::Failed;
                    state.failure_count += 1;
                    error!(
                        job = %name,
                        error = ?result.error,
                        "Job failed"
                    );
                }

                state.last_result = Some(result);
            }
        }
    }

    /// Get job status
    pub async fn job_status(&self, name: &str) -> Option<JobStatus> {
        self.jobs.read().await.get(name).map(|s| s.status)
    }

    /// Get last job result
    pub async fn last_result(&self, name: &str) -> Option<JobResult> {
        self.jobs
            .read()
            .await
            .get(name)
            .and_then(|s| s.last_result.clone())
    }

    /// Disable a job
    pub async fn disable(&self, name: &str) {
        if let Some(state) = self.jobs.write().await.get_mut(name) {
            state.status = JobStatus::Disabled;
            info!(job = %name, "Job disabled");
        }
    }

    /// Enable a job
    pub async fn enable(&self, name: &str) {
        if let Some(state) = self.jobs.write().await.get_mut(name) {
            if state.status == JobStatus::Disabled {
                state.status = JobStatus::Idle;
                info!(job = %name, "Job enabled");
            }
        }
    }

    /// Trigger a job to run immediately
    pub async fn trigger(&self, name: &str) {
        self.run_job(name).await;
    }

    /// Get statistics for all jobs
    pub async fn stats(&self) -> HashMap<String, JobStats> {
        let jobs = self.jobs.read().await;
        jobs.iter()
            .map(|(name, state)| {
                (
                    name.clone(),
                    JobStats {
                        status: state.status,
                        run_count: state.run_count,
                        failure_count: state.failure_count,
                        last_run: state.last_run,
                    },
                )
            })
            .collect()
    }
}

/// Statistics for a job
#[derive(Debug, Clone)]
pub struct JobStats {
    /// Current status
    pub status: JobStatus,
    /// Total runs
    pub run_count: u64,
    /// Failed runs
    pub failure_count: u64,
    /// Last run time
    pub last_run: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobResult;

    struct TestJob {
        name: String,
        interval_secs: u64,
    }

    #[async_trait::async_trait]
    impl Job for TestJob {
        fn name(&self) -> &str {
            &self.name
        }

        fn interval(&self) -> Duration {
            Duration::from_secs(self.interval_secs)
        }

        async fn execute(&self) -> JobResult {
            JobResult::success(1, 10)
        }
    }

    #[tokio::test]
    async fn test_register_job() {
        let scheduler = JobScheduler::new(2);
        let job = TestJob {
            name: "test".to_string(),
            interval_secs: 60,
        };

        scheduler.register(job).await;

        let status = scheduler.job_status("test").await;
        assert_eq!(status, Some(JobStatus::Idle));
    }

    #[tokio::test]
    async fn test_trigger_job() {
        let scheduler = Arc::new(JobScheduler::new(2));
        let job = TestJob {
            name: "trigger_test".to_string(),
            interval_secs: 3600,
        };

        scheduler.register(job).await;
        scheduler.trigger("trigger_test").await;

        let status = scheduler.job_status("trigger_test").await;
        assert_eq!(status, Some(JobStatus::Completed));

        let result = scheduler.last_result("trigger_test").await;
        assert!(result.is_some());
        assert!(result.unwrap().success);
    }

    #[tokio::test]
    async fn test_disable_enable() {
        let scheduler = JobScheduler::new(2);
        let job = TestJob {
            name: "toggle".to_string(),
            interval_secs: 60,
        };

        scheduler.register(job).await;

        scheduler.disable("toggle").await;
        assert_eq!(
            scheduler.job_status("toggle").await,
            Some(JobStatus::Disabled)
        );

        scheduler.enable("toggle").await;
        assert_eq!(scheduler.job_status("toggle").await, Some(JobStatus::Idle));
    }
}
