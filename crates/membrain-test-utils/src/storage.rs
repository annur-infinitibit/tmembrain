//! In-memory storage stub implementing `MemoryStorage` for contract tests.
//!
//! Not a mock — a real implementation backed by a `parking_lot::RwLock<HashMap>`.
//! Use `injected_error` to simulate failure, `count`/`transaction_count` for
//! assertions beyond the trait surface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use membrain_core::error::{Error, Result};
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::{
    MemoryStorage, SearchFilters, SearchQuery, SearchResult, StorageStats, Transaction,
};
use membrain_core::types::{AgentId, Embedding, MemoryId, Version};

type InnerMap = Arc<RwLock<HashMap<MemoryId, Memory>>>;

/// Real `MemoryStorage` impl backed by an in-memory map.
pub struct InMemoryStorageStub {
    memories: InnerMap,
    injected_errors: Mutex<Vec<Error>>,
    transactions: AtomicUsize,
}

impl InMemoryStorageStub {
    /// Empty stub.
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
            injected_errors: Mutex::new(Vec::new()),
            transactions: AtomicUsize::new(0),
        }
    }

    /// Pre-seed with a set of memories.
    pub fn with_memories(memories: Vec<Memory>) -> Self {
        let stub = Self::new();
        let mut map = stub.memories.write();
        for memory in memories {
            map.insert(*memory.id(), memory);
        }
        drop(map);
        stub
    }

    /// Queue an error to be returned by the next mutating call.
    pub fn injected_error(&self, error: Error) {
        self.injected_errors.lock().push(error);
    }

    fn pop_error(&self) -> Option<Error> {
        let mut errors = self.injected_errors.lock();
        if errors.is_empty() {
            None
        } else {
            Some(errors.remove(0))
        }
    }

    /// Current number of stored memories.
    pub fn count(&self) -> usize {
        self.memories.read().len()
    }

    /// Number of transactions opened via `begin_transaction`.
    pub fn transaction_count(&self) -> usize {
        self.transactions.load(Ordering::SeqCst)
    }
}

impl Default for InMemoryStorageStub {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStorage for InMemoryStorageStub {
    async fn store(&self, memory: Memory) -> Result<MemoryId> {
        if let Some(error) = self.pop_error() {
            return Err(error);
        }
        let id = *memory.id();
        self.memories.write().insert(id, memory);
        Ok(id)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        Ok(self.memories.read().get(id).cloned())
    }

    async fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        let map = self.memories.read();
        Ok(ids.iter().filter_map(|id| map.get(id).cloned()).collect())
    }

    async fn update(&self, memory: Memory, expected_version: Version) -> Result<()> {
        if let Some(error) = self.pop_error() {
            return Err(error);
        }
        let id = *memory.id();
        let mut map = self.memories.write();
        match map.get(&id) {
            Some(existing) if existing.common().version == expected_version => {
                map.insert(id, memory);
                Ok(())
            }
            Some(_) => Err(Error::WriteConflict(id)),
            None => Err(Error::MemoryNotFound(id)),
        }
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        if let Some(error) = self.pop_error() {
            return Err(error);
        }
        Ok(self.memories.write().remove(id).is_some())
    }

    async fn delete_many(&self, ids: &[MemoryId]) -> Result<usize> {
        if let Some(error) = self.pop_error() {
            return Err(error);
        }
        let mut map = self.memories.write();
        let mut removed = 0;
        for id in ids {
            if map.remove(id).is_some() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let map = self.memories.read();
        let results = map
            .values()
            .filter(|memory| matches_filters(memory, &query.filters))
            .filter(|memory| {
                query.query.as_deref().is_none_or(|needle| {
                    memory
                        .text_content()
                        .to_lowercase()
                        .contains(&needle.to_lowercase())
                })
            })
            .take(query.limit.max(1))
            .cloned()
            .map(|memory| {
                SearchResult::new(memory, 1.0, membrain_core::traits::MatchType::Exact)
            })
            .collect();
        Ok(results)
    }

    async fn vector_search(
        &self,
        _embedding: &Embedding,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let map = self.memories.read();
        let applied = filters.unwrap_or_default();
        let results = map
            .values()
            .filter(|memory| matches_filters(memory, &applied))
            .take(limit.max(1))
            .cloned()
            .map(|memory| {
                SearchResult::new(memory, 1.0, membrain_core::traits::MatchType::Vector)
            })
            .collect();
        Ok(results)
    }

    async fn text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let needle = query.to_lowercase();
        let map = self.memories.read();
        let results = map
            .values()
            .filter(|memory| memory.text_content().to_lowercase().contains(&needle))
            .take(limit.max(1))
            .cloned()
            .map(|memory| {
                SearchResult::new(memory, 1.0, membrain_core::traits::MatchType::Text)
            })
            .collect();
        Ok(results)
    }

    async fn count(&self, filters: Option<SearchFilters>) -> Result<usize> {
        let map = self.memories.read();
        let applied = filters.unwrap_or_default();
        Ok(map
            .values()
            .filter(|memory| matches_filters(memory, &applied))
            .count())
    }

    async fn exists(&self, id: &MemoryId) -> Result<bool> {
        Ok(self.memories.read().contains_key(id))
    }

    async fn get_by_agent(
        &self,
        agent_id: &AgentId,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let map = self.memories.read();
        let mut entries: Vec<Memory> = map
            .values()
            .filter(|memory| memory.common().agent_id == *agent_id)
            .cloned()
            .collect();
        entries.sort_by(|left, right| left.id().as_bytes().cmp(right.id().as_bytes()));
        Ok(entries.into_iter().skip(offset).take(limit.max(1)).collect())
    }

    async fn get_by_type(
        &self,
        memory_type: MemoryType,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let map = self.memories.read();
        let mut entries: Vec<Memory> = map
            .values()
            .filter(|memory| memory.memory_type() == memory_type)
            .cloned()
            .collect();
        entries.sort_by(|left, right| left.id().as_bytes().cmp(right.id().as_bytes()));
        Ok(entries.into_iter().skip(offset).take(limit.max(1)).collect())
    }

    async fn record_access(&self, id: &MemoryId) -> Result<()> {
        let mut map = self.memories.write();
        match map.get_mut(id) {
            Some(memory) => {
                memory.common_mut().record_access();
                Ok(())
            }
            None => Err(Error::MemoryNotFound(*id)),
        }
    }

    async fn stats(&self) -> Result<StorageStats> {
        let map = self.memories.read();
        let total = map.len();
        let mut by_type = HashMap::new();
        let mut agents = std::collections::HashSet::new();
        let mut embeddings_count = 0;
        let mut confidence_sum = 0.0_f64;
        for memory in map.values() {
            *by_type.entry(memory.memory_type()).or_insert(0) += 1;
            agents.insert(memory.common().agent_id);
            if memory.embedding().is_some() {
                embeddings_count += 1;
            }
            confidence_sum += memory.confidence().value();
        }
        let avg_confidence = if total == 0 {
            0.0
        } else {
            confidence_sum / total as f64
        };
        Ok(StorageStats {
            total_memories: total,
            by_type,
            storage_bytes: 0,
            embeddings_count,
            avg_confidence,
            agent_count: agents.len(),
        })
    }

    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.transactions.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(InMemoryTransaction {
            parent: Arc::clone(&self.memories),
            pending_inserts: Vec::new(),
            pending_updates: Vec::new(),
            pending_deletes: Vec::new(),
        }))
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

fn matches_filters(memory: &Memory, filters: &SearchFilters) -> bool {
    if let Some(ref types) = filters.memory_types {
        if !types.contains(&memory.memory_type()) {
            return false;
        }
    }
    if let Some(ref confidence) = filters.min_confidence {
        if memory.confidence().value() < confidence.value() {
            return false;
        }
    }
    if let Some(ref agents) = filters.agent_ids {
        if !agents.contains(&memory.common().agent_id) {
            return false;
        }
    }
    if let Some(ref excluded) = filters.exclude_ids {
        if excluded.contains(memory.id()) {
            return false;
        }
    }
    true
}

struct InMemoryTransaction {
    parent: InnerMap,
    pending_inserts: Vec<Memory>,
    pending_updates: Vec<(Memory, Version)>,
    pending_deletes: Vec<MemoryId>,
}

#[async_trait]
impl Transaction for InMemoryTransaction {
    async fn store(&mut self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        self.pending_inserts.push(memory);
        Ok(id)
    }

    async fn update(&mut self, memory: Memory, expected_version: Version) -> Result<()> {
        self.pending_updates.push((memory, expected_version));
        Ok(())
    }

    async fn delete(&mut self, id: &MemoryId) -> Result<bool> {
        self.pending_deletes.push(*id);
        Ok(true)
    }

    async fn commit(self: Box<Self>) -> Result<()> {
        let mut map = self.parent.write();
        for memory in self.pending_inserts {
            map.insert(*memory.id(), memory);
        }
        for (memory, expected_version) in self.pending_updates {
            let id = *memory.id();
            match map.get(&id) {
                Some(existing) if existing.common().version == expected_version => {
                    map.insert(id, memory);
                }
                Some(_) => return Err(Error::WriteConflict(id)),
                None => return Err(Error::MemoryNotFound(id)),
            }
        }
        for id in self.pending_deletes {
            map.remove(&id);
        }
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}
