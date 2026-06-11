//! Decision logging for audit trail

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

use membrain_core::config::AuditConfig;
use membrain_core::memory::MemoryType;
use membrain_core::types::{AgentId, Confidence, MemoryId};

/// Audit log for tracking all memory decisions
pub struct AuditLog {
    /// Log entries
    entries: Arc<RwLock<VecDeque<AuditEntry>>>,
    /// Configuration
    config: AuditConfig,
    /// Maximum entries to keep in memory
    max_entries: usize,
}

impl AuditLog {
    /// Create a new audit log
    pub fn new(config: AuditConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(VecDeque::new())),
            config,
            max_entries: 10000,
        }
    }

    /// Create with custom max entries
    pub fn with_max_entries(config: AuditConfig, max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(VecDeque::new())),
            config,
            max_entries,
        }
    }

    /// Log a storage decision
    pub fn log_storage(&self, entry: AuditEntry) {
        if !self.config.enabled || !self.config.log_storage {
            return;
        }
        self.add_entry(entry);
    }

    /// Log a retrieval decision
    pub fn log_retrieval(&self, entry: AuditEntry) {
        if !self.config.enabled || !self.config.log_retrieval {
            return;
        }
        self.add_entry(entry);
    }

    /// Log a rejection
    pub fn log_rejection(&self, entry: AuditEntry) {
        if !self.config.enabled || !self.config.log_rejections {
            return;
        }
        self.add_entry(entry);
    }

    /// Log any entry (respects enabled flag only)
    pub fn log(&self, entry: AuditEntry) {
        if !self.config.enabled {
            return;
        }
        self.add_entry(entry);
    }

    fn add_entry(&self, entry: AuditEntry) {
        let mut entries = self.entries.write();
        entries.push_back(entry);

        // Trim if over limit
        while entries.len() > self.max_entries {
            entries.pop_front();
        }
    }

    /// Get recent entries
    pub fn recent(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries.iter().rev().take(count).cloned().collect()
    }

    /// Get entries for a specific memory
    pub fn for_memory(&self, memory_id: &MemoryId) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries
            .iter()
            .filter(|e| e.memory_id.as_ref() == Some(memory_id))
            .cloned()
            .collect()
    }

    /// Get entries by type
    pub fn by_type(&self, entry_type: AuditEntryType) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries
            .iter()
            .filter(|e| e.entry_type == entry_type)
            .cloned()
            .collect()
    }

    /// Get entries in time range
    pub fn in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Get all entries
    pub fn all(&self) -> Vec<AuditEntry> {
        self.entries.read().iter().cloned().collect()
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Export entries to JSON
    pub fn export_json(&self) -> serde_json::Result<String> {
        let entries: Vec<AuditEntry> = self.entries.read().iter().cloned().collect();
        serde_json::to_string_pretty(&entries)
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

/// A single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique ID for this entry
    pub id: uuid::Uuid,
    /// When this occurred
    pub timestamp: DateTime<Utc>,
    /// Type of entry
    pub entry_type: AuditEntryType,
    /// Related memory ID (if applicable)
    pub memory_id: Option<MemoryId>,
    /// Memory type (if applicable)
    pub memory_type: Option<MemoryType>,
    /// Agent that triggered this
    pub agent_id: Option<AgentId>,
    /// The decision outcome
    pub outcome: DecisionOutcome,
    /// Context about the decision
    pub context: DecisionContext,
    /// Processing time in milliseconds
    pub duration_ms: Option<u64>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(entry_type: AuditEntryType, outcome: DecisionOutcome) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            entry_type,
            memory_id: None,
            memory_type: None,
            agent_id: None,
            outcome,
            context: DecisionContext::default(),
            duration_ms: None,
            metadata: None,
        }
    }

    /// Set memory ID
    pub fn with_memory_id(mut self, id: MemoryId) -> Self {
        self.memory_id = Some(id);
        self
    }

    /// Set memory type
    pub fn with_memory_type(mut self, mt: MemoryType) -> Self {
        self.memory_type = Some(mt);
        self
    }

    /// Set agent ID
    pub fn with_agent_id(mut self, id: AgentId) -> Self {
        self.agent_id = Some(id);
        self
    }

    /// Set context
    pub fn with_context(mut self, context: DecisionContext) -> Self {
        self.context = context;
        self
    }

    /// Set duration
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Create a storage decision entry
    pub fn storage_decision(
        memory_id: MemoryId,
        memory_type: MemoryType,
        outcome: DecisionOutcome,
    ) -> Self {
        Self::new(AuditEntryType::StorageDecision, outcome)
            .with_memory_id(memory_id)
            .with_memory_type(memory_type)
    }

    /// Create a retrieval decision entry
    pub fn retrieval_decision(query: impl Into<String>, results_count: usize) -> Self {
        Self::new(
            AuditEntryType::RetrievalDecision,
            DecisionOutcome::Retrieved {
                count: results_count,
            },
        )
        .with_context(DecisionContext {
            query: Some(query.into()),
            ..Default::default()
        })
    }

    /// Create a rejection entry
    pub fn rejection(memory_id: MemoryId, reason: impl Into<String>) -> Self {
        Self::new(
            AuditEntryType::Rejection,
            DecisionOutcome::Rejected {
                reason: reason.into(),
            },
        )
        .with_memory_id(memory_id)
    }

    /// Create a merge entry
    pub fn merge(original_id: MemoryId, merged_into: MemoryId) -> Self {
        Self::new(
            AuditEntryType::Merge,
            DecisionOutcome::Merged {
                source_id: original_id,
                target_id: merged_into,
            },
        )
        .with_memory_id(original_id)
    }
}

/// Types of audit entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEntryType {
    /// Decision to store a memory
    StorageDecision,
    /// Decision about retrieval
    RetrievalDecision,
    /// Memory was rejected
    Rejection,
    /// Memories were merged
    Merge,
    /// Consolidation occurred
    Consolidation,
    /// Decay was applied
    Decay,
    /// Contradiction was detected
    ContradictionDetected,
    /// Access was recorded
    Access,
    /// Memory was deleted
    Deletion,
    /// Configuration change
    ConfigChange,
}

/// Outcome of a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DecisionOutcome {
    /// Memory was stored
    Stored,
    /// Memory was rejected
    Rejected { reason: String },
    /// Memories were merged
    Merged {
        source_id: MemoryId,
        target_id: MemoryId,
    },
    /// Retrieval returned results
    Retrieved { count: usize },
    /// Retrieval was gated (skipped)
    Gated { reason: String },
    /// Consolidation completed
    Consolidated { source_count: usize },
    /// Decay was applied
    Decayed {
        old_confidence: f64,
        new_confidence: f64,
    },
    /// Contradiction found
    ContradictionFound {
        memory1: MemoryId,
        memory2: MemoryId,
    },
    /// Custom outcome
    Custom { description: String },
}

/// Context about a decision
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Query that triggered this (for retrieval)
    pub query: Option<String>,
    /// Salience score
    pub salience_score: Option<f64>,
    /// Novelty score
    pub novelty_score: Option<f64>,
    /// Similarity to existing memories
    pub max_similarity: Option<f64>,
    /// Similar memory IDs
    pub similar_memories: Vec<MemoryId>,
    /// Confidence at time of decision
    pub confidence: Option<Confidence>,
    /// Budget status
    pub budget_remaining: Option<usize>,
    /// Policy that made the decision
    pub policy: Option<String>,
    /// Additional notes
    pub notes: Option<String>,
}

impl DecisionContext {
    /// Create new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set salience score
    pub fn with_salience(mut self, score: f64) -> Self {
        self.salience_score = Some(score);
        self
    }

    /// Set novelty score
    pub fn with_novelty(mut self, score: f64) -> Self {
        self.novelty_score = Some(score);
        self
    }

    /// Set max similarity
    pub fn with_max_similarity(mut self, similarity: f64) -> Self {
        self.max_similarity = Some(similarity);
        self
    }

    /// Add similar memory
    pub fn with_similar_memory(mut self, id: MemoryId) -> Self {
        self.similar_memories.push(id);
        self
    }

    /// Set policy
    pub fn with_policy(mut self, policy: impl Into<String>) -> Self {
        self.policy = Some(policy.into());
        self
    }

    /// Set notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_basic() {
        let log = AuditLog::default();

        let entry = AuditEntry::new(AuditEntryType::StorageDecision, DecisionOutcome::Stored);
        log.log(entry);

        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_audit_log_disabled() {
        let config = AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let log = AuditLog::new(config);

        let entry = AuditEntry::new(AuditEntryType::StorageDecision, DecisionOutcome::Stored);
        log.log(entry);

        assert!(log.is_empty());
    }

    #[test]
    fn test_audit_entry_builders() {
        let memory_id = MemoryId::new();

        let entry = AuditEntry::storage_decision(
            memory_id,
            MemoryType::SemanticFact,
            DecisionOutcome::Stored,
        )
        .with_context(
            DecisionContext::new()
                .with_salience(0.8)
                .with_novelty(0.9)
                .with_policy("default"),
        )
        .with_duration_ms(15);

        assert_eq!(entry.memory_id, Some(memory_id));
        assert_eq!(entry.context.salience_score, Some(0.8));
        assert_eq!(entry.duration_ms, Some(15));
    }

    #[test]
    fn test_audit_log_filtering() {
        let log = AuditLog::default();
        let memory_id = MemoryId::new();

        log.log(
            AuditEntry::new(AuditEntryType::StorageDecision, DecisionOutcome::Stored)
                .with_memory_id(memory_id),
        );
        log.log(AuditEntry::new(
            AuditEntryType::RetrievalDecision,
            DecisionOutcome::Retrieved { count: 5 },
        ));

        let storage_entries = log.by_type(AuditEntryType::StorageDecision);
        assert_eq!(storage_entries.len(), 1);

        let memory_entries = log.for_memory(&memory_id);
        assert_eq!(memory_entries.len(), 1);
    }

    #[test]
    fn test_audit_log_max_entries() {
        let config = AuditConfig::default();
        let log = AuditLog::with_max_entries(config, 5);

        for _ in 0..10 {
            log.log(AuditEntry::new(
                AuditEntryType::StorageDecision,
                DecisionOutcome::Stored,
            ));
        }

        assert_eq!(log.len(), 5);
    }

    #[test]
    fn test_decision_context() {
        let ctx = DecisionContext::new()
            .with_salience(0.7)
            .with_novelty(0.8)
            .with_max_similarity(0.3)
            .with_policy("salience_check")
            .with_notes("Passed threshold");

        assert_eq!(ctx.salience_score, Some(0.7));
        assert_eq!(ctx.novelty_score, Some(0.8));
        assert_eq!(ctx.policy, Some("salience_check".to_string()));
    }

    #[test]
    fn test_export_json() {
        let log = AuditLog::default();

        log.log(AuditEntry::new(
            AuditEntryType::StorageDecision,
            DecisionOutcome::Stored,
        ));

        let json = log.export_json().unwrap();
        assert!(json.contains("StorageDecision") || json.contains("storage_decision"));
    }
}
