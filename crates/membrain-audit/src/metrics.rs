//! Performance metrics for the Membrain memory system

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrics collector for performance tracking
pub struct Metrics {
    /// Operation counters
    counters: RwLock<HashMap<String, AtomicU64>>,
    /// Timing histograms (operation -> list of durations in microseconds)
    timings: RwLock<HashMap<String, Vec<u64>>>,
    /// Gauges (current values)
    gauges: RwLock<HashMap<String, f64>>,
    /// Start time
    start_time: Instant,
    /// Max timing samples per operation
    max_timing_samples: usize,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
            timings: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
            max_timing_samples: 1000,
        }
    }

    /// Increment a counter
    pub fn increment(&self, name: &str) {
        self.increment_by(name, 1);
    }

    /// Increment a counter by a specific amount
    pub fn increment_by(&self, name: &str, amount: u64) {
        let mut counters = self.counters.write();
        counters
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(amount, Ordering::Relaxed);
    }

    /// Get a counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .read()
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Record a timing
    pub fn record_timing(&self, name: &str, duration: Duration) {
        let micros = duration.as_micros() as u64;
        let mut timings = self.timings.write();
        let samples = timings.entry(name.to_string()).or_default();

        samples.push(micros);

        // Keep only recent samples
        if samples.len() > self.max_timing_samples {
            samples.remove(0);
        }
    }

    /// Time an operation and record it
    pub fn time<F, T>(&self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        self.record_timing(name, start.elapsed());
        result
    }

    /// Get timing statistics for an operation
    pub fn get_timing_stats(&self, name: &str) -> Option<TimingStats> {
        let timings = self.timings.read();
        let samples = timings.get(name)?;

        if samples.is_empty() {
            return None;
        }

        let count = samples.len();
        let sum: u64 = samples.iter().sum();
        let mean = sum as f64 / count as f64;

        let mut sorted = samples.clone();
        sorted.sort();

        let (&min, &max) = match (sorted.first(), sorted.last()) {
            (Some(first), Some(last)) => (first, last),
            _ => return None,
        };
        let p50 = sorted[count / 2];
        let p95 = sorted[(count as f64 * 0.95) as usize].min(max);
        let p99 = sorted[(count as f64 * 0.99) as usize].min(max);

        Some(TimingStats {
            count,
            min_us: min,
            max_us: max,
            mean_us: mean,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
        })
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.gauges.write().insert(name.to_string(), value);
    }

    /// Get a gauge value
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read().get(name).copied()
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters: HashMap<String, u64> = self
            .counters
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();

        let timings: HashMap<String, TimingStats> = self
            .timings
            .read()
            .keys()
            .filter_map(|k| self.get_timing_stats(k).map(|s| (k.clone(), s)))
            .collect();

        let gauges = self.gauges.read().clone();

        MetricsSnapshot {
            counters,
            timings,
            gauges,
            uptime_secs: self.uptime().as_secs(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.counters.write().clear();
        self.timings.write().clear();
        self.gauges.write().clear();
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for timing measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Number of samples
    pub count: usize,
    /// Minimum time in microseconds
    pub min_us: u64,
    /// Maximum time in microseconds
    pub max_us: u64,
    /// Mean time in microseconds
    pub mean_us: f64,
    /// 50th percentile (median) in microseconds
    pub p50_us: u64,
    /// 95th percentile in microseconds
    pub p95_us: u64,
    /// 99th percentile in microseconds
    pub p99_us: u64,
}

impl TimingStats {
    /// Get mean as Duration
    pub fn mean(&self) -> Duration {
        Duration::from_micros(self.mean_us as u64)
    }

    /// Get p50 as Duration
    pub fn p50(&self) -> Duration {
        Duration::from_micros(self.p50_us)
    }

    /// Get p95 as Duration
    pub fn p95(&self) -> Duration {
        Duration::from_micros(self.p95_us)
    }

    /// Get p99 as Duration
    pub fn p99(&self) -> Duration {
        Duration::from_micros(self.p99_us)
    }
}

/// Snapshot of all metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Counter values
    pub counters: HashMap<String, u64>,
    /// Timing statistics
    pub timings: HashMap<String, TimingStats>,
    /// Gauge values
    pub gauges: HashMap<String, f64>,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

impl MetricsSnapshot {
    /// Export to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Helper for tracking operation metrics
pub struct OperationMetrics {
    metrics: &'static Metrics,
    operation: &'static str,
    start: Instant,
}

impl OperationMetrics {
    /// Start tracking an operation
    pub fn start(metrics: &'static Metrics, operation: &'static str) -> Self {
        metrics.increment(&format!("{}.started", operation));
        Self {
            metrics,
            operation,
            start: Instant::now(),
        }
    }

    /// Mark operation as successful
    pub fn success(self) {
        self.metrics
            .increment(&format!("{}.success", self.operation));
        self.metrics
            .record_timing(self.operation, self.start.elapsed());
    }

    /// Mark operation as failed
    pub fn failure(self, _error: &str) {
        self.metrics
            .increment(&format!("{}.failure", self.operation));
        self.metrics
            .record_timing(self.operation, self.start.elapsed());
    }
}

/// Standard metric names
pub mod names {
    // Storage operations
    pub const STORAGE_STORE: &str = "storage.store";
    pub const STORAGE_GET: &str = "storage.get";
    pub const STORAGE_UPDATE: &str = "storage.update";
    pub const STORAGE_DELETE: &str = "storage.delete";
    pub const STORAGE_SEARCH: &str = "storage.search";

    // Pipeline operations
    pub const PIPELINE_WRITE: &str = "pipeline.write";
    pub const PIPELINE_RETRIEVE: &str = "pipeline.retrieve";

    // Policy checks
    pub const POLICY_SALIENCE: &str = "policy.salience";
    pub const POLICY_NOVELTY: &str = "policy.novelty";
    pub const POLICY_REDUNDANCY: &str = "policy.redundancy";
    pub const POLICY_BUDGET: &str = "policy.budget";

    // Outcomes
    pub const MEMORIES_STORED: &str = "memories.stored";
    pub const MEMORIES_REJECTED: &str = "memories.rejected";
    pub const MEMORIES_MERGED: &str = "memories.merged";

    // Embedding operations
    pub const EMBEDDING_GENERATE: &str = "embedding.generate";

    // Jobs
    pub const JOB_CONSOLIDATION: &str = "job.consolidation";
    pub const JOB_DECAY: &str = "job.decay";

    // Gauges
    pub const GAUGE_TOTAL_MEMORIES: &str = "gauge.total_memories";
    pub const GAUGE_EMBEDDING_QUEUE: &str = "gauge.embedding_queue";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counters() {
        let metrics = Metrics::new();

        metrics.increment("test.counter");
        assert_eq!(metrics.get_counter("test.counter"), 1);

        metrics.increment_by("test.counter", 5);
        assert_eq!(metrics.get_counter("test.counter"), 6);
    }

    #[test]
    fn test_timings() {
        let metrics = Metrics::new();

        metrics.record_timing("test.op", Duration::from_millis(10));
        metrics.record_timing("test.op", Duration::from_millis(20));
        metrics.record_timing("test.op", Duration::from_millis(15));

        let stats = metrics.get_timing_stats("test.op").unwrap();
        assert_eq!(stats.count, 3);
        assert!(stats.min_us >= 10000);
        assert!(stats.max_us >= 20000);
    }

    #[test]
    fn test_time_helper() {
        let metrics = Metrics::new();

        let result = metrics.time("test.timed", || {
            std::thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);

        let stats = metrics.get_timing_stats("test.timed").unwrap();
        assert!(stats.min_us >= 5000);
    }

    #[test]
    fn test_gauges() {
        let metrics = Metrics::new();

        metrics.set_gauge("test.gauge", 1.25);
        assert!((metrics.get_gauge("test.gauge").unwrap() - 1.25).abs() < 0.01);

        metrics.set_gauge("test.gauge", 2.71);
        assert!((metrics.get_gauge("test.gauge").unwrap() - 2.71).abs() < 0.01);
    }

    #[test]
    fn test_snapshot() {
        let metrics = Metrics::new();

        metrics.increment("counter1");
        metrics.set_gauge("gauge1", 100.0);
        metrics.record_timing("timing1", Duration::from_millis(5));

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.counters.get("counter1"), Some(&1));
        assert!(snapshot.gauges.contains_key("gauge1"));
        assert!(snapshot.timings.contains_key("timing1"));
    }

    #[test]
    fn test_reset() {
        let metrics = Metrics::new();

        metrics.increment("test");
        metrics.set_gauge("test", 1.0);

        metrics.reset();

        assert_eq!(metrics.get_counter("test"), 0);
        assert!(metrics.get_gauge("test").is_none());
    }
}
