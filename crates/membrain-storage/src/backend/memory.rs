//! In-memory storage backend for testing and development

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use membrain_core::error::{Error, Result};
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::{
    MatchType, MemoryStorage, SearchFilters, SearchQuery, SearchResult, StorageStats, Transaction,
};
use membrain_core::types::{AgentId, Embedding, MemoryId, Version};

/// In-memory storage backend
pub struct InMemoryStorage {
    memories: Arc<RwLock<HashMap<MemoryId, Memory>>>,
}

impl InMemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the number of stored memories
    pub fn len(&self) -> usize {
        self.memories.read().len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.memories.read().is_empty()
    }

    /// Clear all memories
    pub fn clear(&self) {
        self.memories.write().clear();
    }

    fn matches_filters(memory: &Memory, filters: &SearchFilters) -> bool {
        let common = memory.common();

        // Check memory type filter
        if let Some(ref types) = filters.memory_types {
            if !types.contains(&memory.memory_type()) {
                return false;
            }
        }

        // Check min confidence
        if let Some(ref min_conf) = filters.min_confidence {
            if common.confidence.value() < min_conf.value() {
                return false;
            }
        }

        // Check agent IDs
        if let Some(ref agents) = filters.agent_ids {
            if !agents.contains(&common.agent_id) {
                return false;
            }
        }

        // Check tags (any match)
        if let Some(ref tags) = filters.tags {
            if !tags.iter().any(|t| common.tags.contains(t)) {
                return false;
            }
        }

        // Check required tags (all must match)
        if let Some(ref required) = filters.required_tags {
            if !required.iter().all(|t| common.tags.contains(t)) {
                return false;
            }
        }

        // Check created time range
        if let Some(ref after) = filters.created_after {
            if common.provenance.created_at < *after {
                return false;
            }
        }

        if let Some(ref before) = filters.created_before {
            if common.provenance.created_at > *before {
                return false;
            }
        }

        // Check accessed time
        if let Some(ref after) = filters.accessed_after {
            if common.provenance.last_accessed_at < *after {
                return false;
            }
        }

        // Check excluded IDs
        if let Some(ref excluded) = filters.exclude_ids {
            if excluded.contains(&common.id) {
                return false;
            }
        }

        // Check metadata (all key-value pairs must match exactly)
        if let Some(ref filter_metadata) = filters.metadata {
            for (key, expected_value) in filter_metadata {
                match common.metadata.get(key) {
                    Some(actual_value) if actual_value == expected_value => {}
                    _ => return false,
                }
            }
        }

        // Check bi-temporal validity: valid_at filters for event-time validity
        if let Some(ref at) = filters.valid_at {
            let after_start = common.valid_from.is_none_or(|from| *at >= from);
            let before_end = common.valid_until.is_none_or(|until| *at < until);
            if !after_start || !before_end {
                return false;
            }
        }

        // Check known_at: filter for system-time knowledge
        if let Some(ref at) = filters.known_at {
            if common.provenance.created_at > *at {
                return false;
            }
        }

        // Exclude invalidated memories (valid_until is set and in the past)
        if filters.exclude_invalidated == Some(true) && !common.is_valid() {
            return false;
        }

        true
    }

    fn calculate_text_score(memory: &Memory, query: &str) -> f64 {
        let text = memory.text_content().to_lowercase();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let matches: usize = query_words.iter().filter(|w| text.contains(*w)).count();
        matches as f64 / query_words.len() as f64
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStorage for InMemoryStorage {
    async fn store(&self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        self.memories.write().insert(id, memory);
        Ok(id)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        Ok(self.memories.read().get(id).cloned())
    }

    async fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        let memories = self.memories.read();
        Ok(ids
            .iter()
            .filter_map(|id| memories.get(id).cloned())
            .collect())
    }

    async fn update(&self, memory: Memory, expected_version: Version) -> Result<()> {
        let mut memories = self.memories.write();
        let id = *memory.id();

        if let Some(existing) = memories.get(&id) {
            if existing.common().version != expected_version {
                return Err(Error::WriteConflict(id));
            }
            memories.insert(id, memory);
            Ok(())
        } else {
            Err(Error::MemoryNotFound(id))
        }
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        Ok(self.memories.write().remove(id).is_some())
    }

    async fn delete_many(&self, ids: &[MemoryId]) -> Result<usize> {
        let mut memories = self.memories.write();
        let mut count = 0;
        for id in ids {
            if memories.remove(id).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let memories = self.memories.read();
        let mut results: Vec<SearchResult> = Vec::new();

        for memory in memories.values() {
            if !Self::matches_filters(memory, &query.filters) {
                continue;
            }

            let mut score = 0.0;
            let mut match_type = MatchType::Exact;

            // Text search scoring
            if let Some(ref q) = query.query {
                let text_score = Self::calculate_text_score(memory, q);
                if text_score > 0.0 {
                    score = text_score;
                    match_type = MatchType::Text;
                }
            }

            // Vector search scoring
            if let (Some(ref query_emb), Some(mem_emb)) = (&query.embedding, memory.embedding()) {
                if let Ok(sim) = query_emb.cosine_similarity(mem_emb) {
                    let vector_score = ((sim + 1.0) / 2.0) as f64; // Normalize to 0-1 and convert to f64
                    if vector_score > score {
                        score = vector_score;
                        match_type = MatchType::Vector;
                    } else if score > 0.0 {
                        score = (score + vector_score) / 2.0;
                        match_type = MatchType::Hybrid;
                    }
                }
            }

            // If no query, include all matching filters with base score
            if query.query.is_none() && query.embedding.is_none() {
                score = memory.common().confidence.value();
            }

            if let Some(min_score) = query.min_score {
                if score < min_score {
                    continue;
                }
            }

            if score > 0.0 || (query.query.is_none() && query.embedding.is_none()) {
                results.push(SearchResult::new(memory.clone(), score, match_type));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply offset and limit
        let results: Vec<SearchResult> = results
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();

        Ok(results)
    }

    async fn vector_search(
        &self,
        embedding: &Embedding,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let query = SearchQuery::new()
            .with_embedding(embedding.clone())
            .with_limit(limit)
            .with_filters(filters.unwrap_or_default());

        self.search(query).await
    }

    async fn text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let search_query = SearchQuery::new().with_query(query).with_limit(limit);

        self.search(search_query).await
    }

    async fn count(&self, filters: Option<SearchFilters>) -> Result<usize> {
        let memories = self.memories.read();

        if let Some(ref f) = filters {
            Ok(memories
                .values()
                .filter(|m| Self::matches_filters(m, f))
                .count())
        } else {
            Ok(memories.len())
        }
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
        let memories = self.memories.read();
        Ok(memories
            .values()
            .filter(|m| &m.common().agent_id == agent_id)
            .skip(offset)
            .take(limit)
            .cloned()
            .collect())
    }

    async fn get_by_type(
        &self,
        memory_type: MemoryType,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let memories = self.memories.read();
        Ok(memories
            .values()
            .filter(|m| m.memory_type() == memory_type)
            .skip(offset)
            .take(limit)
            .cloned()
            .collect())
    }

    async fn record_access(&self, id: &MemoryId) -> Result<()> {
        let mut memories = self.memories.write();
        if let Some(memory) = memories.get_mut(id) {
            memory.common_mut().record_access();
            Ok(())
        } else {
            Err(Error::MemoryNotFound(*id))
        }
    }

    async fn stats(&self) -> Result<StorageStats> {
        let memories = self.memories.read();
        let mut by_type: HashMap<MemoryType, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        let mut embeddings_count = 0;
        let mut agents: std::collections::HashSet<AgentId> = std::collections::HashSet::new();

        for memory in memories.values() {
            *by_type.entry(memory.memory_type()).or_insert(0) += 1;
            total_confidence += memory.common().confidence.value();
            if memory.embedding().is_some() {
                embeddings_count += 1;
            }
            agents.insert(memory.common().agent_id);
        }

        let total = memories.len();
        let avg_confidence = if total > 0 {
            total_confidence / total as f64
        } else {
            0.0
        };

        Ok(StorageStats {
            total_memories: total,
            by_type,
            storage_bytes: 0, // Not tracked in memory
            embeddings_count,
            avg_confidence,
            agent_count: agents.len(),
        })
    }

    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        Ok(Box::new(InMemoryTransaction::new(self.memories.clone())))
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// In-memory transaction (uses copy-on-write semantics)
struct InMemoryTransaction {
    memories: Arc<RwLock<HashMap<MemoryId, Memory>>>,
    pending_stores: Vec<Memory>,
    pending_updates: Vec<(Memory, Version)>,
    pending_deletes: Vec<MemoryId>,
}

impl InMemoryTransaction {
    fn new(memories: Arc<RwLock<HashMap<MemoryId, Memory>>>) -> Self {
        Self {
            memories,
            pending_stores: Vec::new(),
            pending_updates: Vec::new(),
            pending_deletes: Vec::new(),
        }
    }
}

#[async_trait]
impl Transaction for InMemoryTransaction {
    async fn store(&mut self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        self.pending_stores.push(memory);
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
        let mut memories = self.memories.write();

        // Verify versions before committing
        for (memory, expected_version) in &self.pending_updates {
            let id = memory.id();
            if let Some(existing) = memories.get(id) {
                if existing.common().version != *expected_version {
                    return Err(Error::WriteConflict(*id));
                }
            } else {
                return Err(Error::MemoryNotFound(*id));
            }
        }

        // Apply stores
        for memory in self.pending_stores {
            memories.insert(*memory.id(), memory);
        }

        // Apply updates
        for (memory, _) in self.pending_updates {
            memories.insert(*memory.id(), memory);
        }

        // Apply deletes
        for id in self.pending_deletes {
            memories.remove(&id);
        }

        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<()> {
        // Nothing to do - pending changes are discarded
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{Confidence, Provenance, Source};

    fn create_test_memory(statement: &str) -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new(statement)),
        })
    }

    #[tokio::test]
    async fn test_store_and_get() {
        let storage = InMemoryStorage::new();
        let memory = create_test_memory("Test fact");
        let id = *memory.id();

        storage.store(memory.clone()).await.unwrap();

        let retrieved = storage.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.id(), &id);
    }

    #[tokio::test]
    async fn test_delete() {
        let storage = InMemoryStorage::new();
        let memory = create_test_memory("To delete");
        let id = *memory.id();

        storage.store(memory).await.unwrap();
        assert!(storage.exists(&id).await.unwrap());

        let deleted = storage.delete(&id).await.unwrap();
        assert!(deleted);
        assert!(!storage.exists(&id).await.unwrap());
    }

    #[tokio::test]
    async fn test_text_search() {
        let storage = InMemoryStorage::new();

        storage
            .store(create_test_memory("The sky is blue"))
            .await
            .unwrap();
        storage
            .store(create_test_memory("Grass is green"))
            .await
            .unwrap();
        storage
            .store(create_test_memory("The ocean is blue"))
            .await
            .unwrap();

        let results = storage.text_search("blue", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_count() {
        let storage = InMemoryStorage::new();

        storage.store(create_test_memory("One")).await.unwrap();
        storage.store(create_test_memory("Two")).await.unwrap();
        storage.store(create_test_memory("Three")).await.unwrap();

        let count = storage.count(None).await.unwrap();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_transaction() {
        let storage = InMemoryStorage::new();

        let mut tx = storage.begin_transaction().await.unwrap();

        let mem1 = create_test_memory("Transaction test 1");
        let mem2 = create_test_memory("Transaction test 2");

        tx.store(mem1).await.unwrap();
        tx.store(mem2).await.unwrap();

        // Before commit, storage should be empty
        assert_eq!(storage.count(None).await.unwrap(), 0);

        tx.commit().await.unwrap();

        // After commit, both should be stored
        assert_eq!(storage.count(None).await.unwrap(), 2);
    }

    fn create_test_memory_with_metadata(
        statement: &str,
        metadata: std::collections::HashMap<String, serde_json::Value>,
    ) -> Memory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let mut common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.8));
        common.metadata = metadata;

        Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new(statement)),
        })
    }

    #[tokio::test]
    async fn test_metadata_filtering() {
        let storage = InMemoryStorage::new();

        let mut meta1 = std::collections::HashMap::new();
        meta1.insert("source".to_string(), serde_json::json!("arxiv"));
        meta1.insert("year".to_string(), serde_json::json!(2024));
        storage
            .store(create_test_memory_with_metadata("Paper about LLMs", meta1))
            .await
            .unwrap();

        let mut meta2 = std::collections::HashMap::new();
        meta2.insert("source".to_string(), serde_json::json!("arxiv"));
        meta2.insert("year".to_string(), serde_json::json!(2023));
        storage
            .store(create_test_memory_with_metadata(
                "Paper about transformers",
                meta2,
            ))
            .await
            .unwrap();

        let mut meta3 = std::collections::HashMap::new();
        meta3.insert("source".to_string(), serde_json::json!("blog"));
        storage
            .store(create_test_memory_with_metadata(
                "Blog post about Rust",
                meta3,
            ))
            .await
            .unwrap();

        // Filter by source=arxiv should return 2 results
        use membrain_core::traits::SearchFilters;
        let filters =
            SearchFilters::new().with_metadata_entry("source", serde_json::json!("arxiv"));
        let count = storage.count(Some(filters)).await.unwrap();
        assert_eq!(count, 2);

        // Filter by source=arxiv AND year=2024 should return 1 result
        let filters = SearchFilters::new()
            .with_metadata_entry("source", serde_json::json!("arxiv"))
            .with_metadata_entry("year", serde_json::json!(2024));
        let count = storage.count(Some(filters)).await.unwrap();
        assert_eq!(count, 1);

        // Filter by source=blog should return 1 result
        let filters = SearchFilters::new().with_metadata_entry("source", serde_json::json!("blog"));
        let count = storage.count(Some(filters)).await.unwrap();
        assert_eq!(count, 1);

        // Filter by nonexistent key should return 0 results
        let filters =
            SearchFilters::new().with_metadata_entry("nonexistent", serde_json::json!("value"));
        let count = storage.count(Some(filters)).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_transaction_rollback() {
        let storage = InMemoryStorage::new();

        let mut tx = storage.begin_transaction().await.unwrap();
        tx.store(create_test_memory("Will be rolled back"))
            .await
            .unwrap();
        tx.rollback().await.unwrap();

        assert_eq!(storage.count(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let storage = InMemoryStorage::new();

        storage.store(create_test_memory("Fact 1")).await.unwrap();
        storage.store(create_test_memory("Fact 2")).await.unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.total_memories, 2);
        assert!(stats.by_type.contains_key(&MemoryType::SemanticFact));
    }
}
