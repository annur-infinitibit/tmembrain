//! MemscaleDB storage backend with vector indexing and full-text search.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use membrain_core::config::{HybridFusionMethod, RetrievalConfig};
use membrain_core::error::{Error, Result};
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::{
    MatchType, MemoryStorage, SearchFilters, SearchMode, SearchQuery, SearchResult, StorageStats,
    Transaction,
};
use membrain_core::types::{AgentId, Embedding, MemoryId, Version};
use memscaledb::storage::metadata::MemoryRecord;
use memscaledb::{MemscaleStorage, MemscaleStorageConfig, VectorId};

/// Minimum fused score threshold for RRF results. Results below this are noise.
const MIN_SCORE_THRESHOLD_RRF: f64 = 0.01;

/// Minimum fused score threshold for weighted average results. Results below this are noise.
const MIN_SCORE_THRESHOLD_WEIGHTED: f64 = 0.1;

/// Configuration for hybrid search fusion behavior.
#[derive(Debug, Clone)]
struct HybridSearchConfig {
    vector_weight: f64,
    text_weight: f64,
    fusion_method: HybridFusionMethod,
    rrf_k: f64,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            text_weight: 0.3,
            fusion_method: HybridFusionMethod::ReciprocalRankFusion,
            rrf_k: 60.0,
        }
    }
}

impl From<&RetrievalConfig> for HybridSearchConfig {
    fn from(config: &RetrievalConfig) -> Self {
        Self {
            vector_weight: config.vector_weight,
            text_weight: config.text_weight,
            fusion_method: config.fusion_method,
            rrf_k: config.rrf_k,
        }
    }
}

/// MemscaleDB storage backend.
pub struct MemscaleDbStorage {
    storage: MemscaleStorage,
    hybrid_config: HybridSearchConfig,
}

impl MemscaleDbStorage {
    /// Create a new MemscaleDB storage backend.
    pub async fn new(path: impl Into<String>) -> Result<Self> {
        let path_str = path.into();
        let config = MemscaleStorageConfig::new(path_str);
        let dimension = 1536;

        let storage = MemscaleStorage::new(config, dimension)
            .map_err(|e| Error::Storage(format!("Failed to create MemscaleDB: {}", e)))?;

        Ok(Self {
            storage,
            hybrid_config: HybridSearchConfig::default(),
        })
    }

    /// Create with custom configuration and dimension.
    pub async fn with_config(config: MemscaleStorageConfig, dimension: usize) -> Result<Self> {
        let storage = MemscaleStorage::new(config, dimension)
            .map_err(|e| Error::Storage(format!("Failed to create MemscaleDB: {}", e)))?;
        Ok(Self {
            storage,
            hybrid_config: HybridSearchConfig::default(),
        })
    }

    /// Create with custom MemscaleDB config, dimension, and retrieval config for hybrid search.
    pub async fn with_retrieval_config(
        memscale_config: MemscaleStorageConfig,
        dimension: usize,
        retrieval_config: &RetrievalConfig,
    ) -> Result<Self> {
        let storage = MemscaleStorage::new(memscale_config, dimension)
            .map_err(|e| Error::Storage(format!("Failed to create MemscaleDB: {}", e)))?;
        Ok(Self {
            storage,
            hybrid_config: HybridSearchConfig::from(retrieval_config),
        })
    }

    /// Convert Memory to MemoryRecord.
    fn memory_to_record(memory: &Memory) -> MemoryRecord {
        let common = memory.common();
        MemoryRecord {
            memory_type: memory.memory_type().to_string(),
            version: common.version,
            confidence: common.confidence.value(),
            agent_id: *common.agent_id.as_uuid(),
            content: memory.to_msgpack().unwrap_or_default(),
            text_content: memory.text_content(),
            tags: common.tags.clone(),
            created_at: common.provenance.created_at.timestamp_millis(),
            modified_at: common.provenance.modified_at.timestamp_millis(),
            last_accessed_at: common.provenance.last_accessed_at.timestamp_millis(),
            access_count: common.provenance.access_count,
            has_embedding: common.embedding.is_some(),
            metadata: common.metadata.clone(),
        }
    }

    /// Convert MemoryRecord back to Memory.
    fn record_to_memory(record: &MemoryRecord) -> Result<Memory> {
        let mut memory = Memory::from_msgpack(&record.content)
            .map_err(|e| Error::Deserialization(format!("Failed to deserialize memory: {}", e)))?;

        let common = memory.common_mut();
        common.provenance.access_count = record.access_count;
        common.provenance.last_accessed_at =
            chrono::DateTime::from_timestamp_millis(record.last_accessed_at)
                .unwrap_or_else(chrono::Utc::now);

        Ok(memory)
    }

    /// Perform hybrid search combining vector and text.
    fn hybrid_search(
        &self,
        query_embedding: Option<&Embedding>,
        query_text: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(Uuid, f64)>> {
        let fetch_limit = limit;

        let mut vector_results: Vec<(Uuid, f64)> = Vec::new();
        let mut text_results: Vec<(Uuid, f64)> = Vec::new();

        if let Some(embedding) = query_embedding {
            let index = self.storage.vector_index.read();
            let results = index
                .search(embedding.values(), fetch_limit)
                .map_err(|e| Error::Storage(format!("Vector search failed: {}", e)))?;

            for result in results {
                let uuid = *result.id.as_uuid();
                vector_results.push((uuid, result.score as f64));
            }
        }

        if let Some(text) = query_text {
            let results = self
                .storage
                .fulltext
                .search(text, fetch_limit)
                .map_err(|e| Error::Storage(format!("Text search failed: {}", e)))?;

            for result in results {
                text_results.push((result.memory_id, result.score));
            }
        }

        let config = &self.hybrid_config;
        let (combined, min_score_threshold) = match config.fusion_method {
            HybridFusionMethod::ReciprocalRankFusion => (
                fuse_rrf(&vector_results, &text_results, config.rrf_k),
                MIN_SCORE_THRESHOLD_RRF,
            ),
            HybridFusionMethod::WeightedAverage => (
                fuse_weighted_average(
                    &vector_results,
                    &text_results,
                    config.vector_weight,
                    config.text_weight,
                ),
                MIN_SCORE_THRESHOLD_WEIGHTED,
            ),
        };

        let mut results: Vec<_> = combined
            .into_iter()
            .filter(|(_, score)| *score >= min_score_threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }
}

/// Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i + 1)) across result lists.
/// Rank is 0-based, so the formula is 1/(k + 0 + 1) for the top result.
///
/// Raw RRF scores are already bounded (max ~2/(k+1) when a doc appears first in
/// both lists) and directly comparable across queries, so no normalization is applied.
/// Dividing by the max score would compress the quality gap between great and mediocre
/// matches, making it impossible to filter noise downstream.
fn fuse_rrf(
    vector_results: &[(Uuid, f64)],
    text_results: &[(Uuid, f64)],
    k: f64,
) -> HashMap<Uuid, f64> {
    let mut scores: HashMap<Uuid, f64> = HashMap::new();

    for (rank, (id, _score)) in vector_results.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
    }

    for (rank, (id, _score)) in text_results.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
    }

    scores
}

/// Weighted average fusion: combined = v_score * vector_weight + t_score * text_weight.
///
/// BM25 text scores are unbounded (often 5-30+) while vector similarity scores are
/// [0,1]. To make the weighted average meaningful, BM25 scores are normalized to [0,1]
/// by dividing each by the maximum BM25 score in the result set before combining.
fn fuse_weighted_average(
    vector_results: &[(Uuid, f64)],
    text_results: &[(Uuid, f64)],
    vector_weight: f64,
    text_weight: f64,
) -> HashMap<Uuid, f64> {
    let mut vector_scores: HashMap<Uuid, f64> = HashMap::new();
    let mut text_scores: HashMap<Uuid, f64> = HashMap::new();

    for (id, score) in vector_results {
        vector_scores.insert(*id, *score);
    }

    // Normalize BM25 text scores to [0,1] by dividing by the max score
    let max_text_score = text_results
        .iter()
        .map(|(_, score)| *score)
        .fold(0.0_f64, f64::max);

    for (id, score) in text_results {
        let normalized_score = if max_text_score > 0.0 {
            *score / max_text_score
        } else {
            0.0
        };
        text_scores.insert(*id, normalized_score);
    }

    let all_ids: std::collections::HashSet<_> = vector_scores
        .keys()
        .chain(text_scores.keys())
        .copied()
        .collect();

    let mut combined: HashMap<Uuid, f64> = HashMap::new();
    for id in all_ids {
        let v_score = vector_scores.get(&id).copied().unwrap_or(0.0);
        let t_score = text_scores.get(&id).copied().unwrap_or(0.0);
        combined.insert(id, (v_score * vector_weight) + (t_score * text_weight));
    }

    combined
}

#[async_trait]
impl MemoryStorage for MemscaleDbStorage {
    async fn store(&self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        let uuid = *id.as_uuid();
        let record = Self::memory_to_record(&memory);
        let embedding = memory.embedding().map(|e| e.values());

        self.storage
            .metadata
            .store(&uuid, &record)
            .map_err(|e| Error::Storage(format!("Failed to store metadata: {}", e)))?;

        if let Some(emb) = embedding {
            let vector_id = VectorId::from_uuid(uuid);
            let mut index = self.storage.vector_index.write();
            index
                .add(vector_id, emb)
                .map_err(|e| Error::Storage(format!("Failed to add vector: {}", e)))?;
        }

        self.storage
            .fulltext
            .index_memory(
                &uuid,
                &record.text_content,
                &record.tags,
                &record.memory_type,
                record.confidence,
                record.created_at,
            )
            .map_err(|e| Error::Storage(format!("Failed to index fulltext: {}", e)))?;

        Ok(id)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let uuid = id.as_uuid();
        let record_opt = self
            .storage
            .metadata
            .get(uuid)
            .map_err(|e| Error::Storage(format!("Failed to get memory: {}", e)))?;

        if let Some(record) = record_opt {
            let memory = Self::record_to_memory(&record)?;
            Ok(Some(memory))
        } else {
            Ok(None)
        }
    }

    async fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        let uuids: Vec<Uuid> = ids.iter().map(|id| *id.as_uuid()).collect();
        let records = self
            .storage
            .metadata
            .get_many(&uuids)
            .map_err(|e| Error::Storage(format!("Failed to get many: {}", e)))?;

        let mut memories = Vec::new();
        for (_id, record) in records {
            let memory = Self::record_to_memory(&record)?;
            memories.push(memory);
        }

        Ok(memories)
    }

    async fn update(&self, memory: Memory, expected_version: Version) -> Result<()> {
        let id = *memory.id();
        let uuid = *id.as_uuid();
        let mut record = Self::memory_to_record(&memory);
        record.version = expected_version + 1;

        let embedding = memory.embedding().map(|e| e.values());

        self.storage
            .metadata
            .update(&uuid, &record, expected_version)
            .map_err(|e| Error::Storage(format!("Failed to update: {}", e)))?;

        if let Some(emb) = embedding {
            let vector_id = VectorId::from_uuid(uuid);
            let mut index = self.storage.vector_index.write();
            index.remove(&vector_id).ok();
            index
                .add(vector_id, emb)
                .map_err(|e| Error::Storage(format!("Failed to update vector: {}", e)))?;
        }

        self.storage
            .fulltext
            .index_memory(
                &uuid,
                &record.text_content,
                &record.tags,
                &record.memory_type,
                record.confidence,
                record.created_at,
            )
            .map_err(|e| Error::Storage(format!("Failed to update fulltext: {}", e)))?;

        Ok(())
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        let uuid = id.as_uuid();

        let record_opt = self
            .storage
            .metadata
            .get(uuid)
            .map_err(|e| Error::Storage(format!("Failed to get memory: {}", e)))?;

        if let Some(record) = record_opt {
            self.storage
                .metadata
                .delete(uuid)
                .map_err(|e| Error::Storage(format!("Failed to delete: {}", e)))?;

            if record.has_embedding {
                let vector_id = VectorId::from_uuid(*uuid);
                let mut index = self.storage.vector_index.write();
                index.remove(&vector_id).ok();
            }

            self.storage
                .fulltext
                .remove_memory(uuid)
                .map_err(|e| Error::Storage(format!("Failed to remove from fulltext: {}", e)))?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn delete_many(&self, ids: &[MemoryId]) -> Result<usize> {
        let mut count = 0;
        for id in ids {
            if self.delete(id).await? {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        match query.mode {
            SearchMode::Vector => {
                if let Some(ref embedding) = query.embedding {
                    self.vector_search(embedding, query.limit, Some(query.filters))
                        .await
                } else {
                    Ok(Vec::new())
                }
            }
            SearchMode::Text => {
                if let Some(ref text) = query.query {
                    self.text_search(text, query.limit).await
                } else {
                    Ok(Vec::new())
                }
            }
            SearchMode::Hybrid => {
                let results = self.hybrid_search(
                    query.embedding.as_ref(),
                    query.query.as_deref(),
                    query.limit,
                )?;

                let mut search_results = Vec::new();
                for (uuid, score) in results {
                    let memory_id = MemoryId::from_uuid(uuid);
                    if let Some(memory) = self.get(&memory_id).await? {
                        // Apply filters if provided
                        if !self.matches_filters(&memory, &query.filters) {
                            continue;
                        }

                        search_results.push(SearchResult::new(memory, score, MatchType::Hybrid));

                        if search_results.len() >= query.limit {
                            break;
                        }
                    }
                }

                Ok(search_results)
            }
        }
    }

    async fn vector_search(
        &self,
        embedding: &Embedding,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let candidate_ids = self.prefilter_candidate_ids(filters.as_ref())?;
        if candidate_ids.as_ref().is_some_and(|s| s.is_empty()) {
            return Ok(Vec::new());
        }

        let over_fetch = limit.saturating_mul(2).max(limit);
        let results = {
            let index = self.storage.vector_index.read();
            match candidate_ids.as_ref() {
                Some(set) => {
                    let closure = |vid: &VectorId| set.contains(vid.as_uuid());
                    index
                        .search_with_filter(embedding.values(), over_fetch, &closure)
                        .map_err(|e| Error::Storage(format!("Vector search failed: {}", e)))?
                }
                None => index
                    .search(embedding.values(), over_fetch)
                    .map_err(|e| Error::Storage(format!("Vector search failed: {}", e)))?,
            }
        };

        let mut search_results = Vec::new();
        for result in results {
            let uuid = *result.id.as_uuid();
            let memory_id = MemoryId::from_uuid(uuid);

            if let Some(memory) = self.get(&memory_id).await? {
                if let Some(ref filt) = filters {
                    if !self.matches_filters(&memory, filt) {
                        continue;
                    }
                }

                search_results.push(SearchResult::new(
                    memory,
                    result.score as f64,
                    MatchType::Vector,
                ));

                if search_results.len() >= limit {
                    break;
                }
            }
        }

        Ok(search_results)
    }

    async fn text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let results = self
            .storage
            .fulltext
            .search(query, limit)
            .map_err(|e| Error::Storage(format!("Text search failed: {}", e)))?;

        let mut search_results = Vec::new();
        for result in results {
            let memory_id = MemoryId::from_uuid(result.memory_id);
            if let Some(memory) = self.get(&memory_id).await? {
                search_results.push(SearchResult::new(memory, result.score, MatchType::Text));
            }
        }

        Ok(search_results)
    }

    async fn count(&self, filters: Option<SearchFilters>) -> Result<usize> {
        let Some(filt) = filters else {
            return self
                .storage
                .metadata
                .count()
                .map_err(|e| Error::Storage(format!("Count failed: {}", e)));
        };

        let candidate_ids = self.prefilter_candidate_ids(Some(&filt))?;
        let ids: Vec<Uuid> = match candidate_ids {
            Some(set) => set.into_iter().collect(),
            None => self
                .storage
                .metadata
                .all_ids()
                .map_err(|e| Error::Storage(format!("Failed to get all ids: {}", e)))?,
        };

        let mut count = 0;
        for uuid in ids {
            let memory_id = MemoryId::from_uuid(uuid);
            if let Some(memory) = self.get(&memory_id).await? {
                if self.matches_filters(&memory, &filt) {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    async fn exists(&self, id: &MemoryId) -> Result<bool> {
        let uuid = id.as_uuid();
        self.storage
            .metadata
            .exists(uuid)
            .map_err(|e| Error::Storage(format!("Exists check failed: {}", e)))
    }

    async fn get_by_agent(
        &self,
        agent_id: &AgentId,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let uuid = agent_id.as_uuid();
        let records = self
            .storage
            .metadata
            .get_by_agent(uuid, limit, offset)
            .map_err(|e| Error::Storage(format!("Get by agent failed: {}", e)))?;

        let mut memories = Vec::new();
        for (_id, record) in records {
            let memory = Self::record_to_memory(&record)?;
            memories.push(memory);
        }

        Ok(memories)
    }

    async fn get_by_type(
        &self,
        memory_type: MemoryType,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let type_str = memory_type.to_string();
        let records = self
            .storage
            .metadata
            .get_by_type(&type_str, limit, offset)
            .map_err(|e| Error::Storage(format!("Get by type failed: {}", e)))?;

        let mut memories = Vec::new();
        for (_id, record) in records {
            let memory = Self::record_to_memory(&record)?;
            memories.push(memory);
        }

        Ok(memories)
    }

    async fn record_access(&self, id: &MemoryId) -> Result<()> {
        let uuid = id.as_uuid();
        self.storage
            .metadata
            .record_access(uuid)
            .map_err(|e| Error::Storage(format!("Record access failed: {}", e)))
    }

    async fn stats(&self) -> Result<StorageStats> {
        let total_memories = self
            .storage
            .metadata
            .count()
            .map_err(|e| Error::Storage(format!("Count failed: {}", e)))?;

        let index = self.storage.vector_index.read();
        let embeddings_count = index.len();

        Ok(StorageStats {
            total_memories,
            by_type: HashMap::new(),
            storage_bytes: 0,
            embeddings_count,
            avg_confidence: 0.0,
            agent_count: 0,
        })
    }

    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        Err(Error::Internal(
            "Transactions not yet implemented for MemscaleDB".to_string(),
        ))
    }

    async fn health_check(&self) -> Result<()> {
        let count = self
            .storage
            .metadata
            .count()
            .map_err(|e| Error::Storage(format!("Health check failed: {}", e)))?;
        tracing::debug!("MemscaleDB health check: {} memories", count);
        Ok(())
    }
}

impl MemscaleDbStorage {
    /// Pre-filter candidate memory IDs by any indexed-metadata constraints
    /// present in `filters`. Returns `Ok(None)` when no indexed-key constraint
    /// is present — callers should fall back to full-set strategies.
    fn prefilter_candidate_ids(
        &self,
        filters: Option<&SearchFilters>,
    ) -> Result<Option<HashSet<Uuid>>> {
        let Some(filter_meta) = filters.and_then(|f| f.metadata.as_ref()) else {
            return Ok(None);
        };
        if filter_meta.is_empty() {
            return Ok(None);
        }
        self.storage
            .metadata
            .ids_matching_metadata(filter_meta)
            .map_err(|e| Error::Storage(format!("Metadata pre-filter failed: {}", e)))
    }

    /// Check if a memory matches the given filters.
    fn matches_filters(&self, memory: &Memory, filters: &SearchFilters) -> bool {
        let common = memory.common();

        if let Some(ref types) = filters.memory_types {
            if !types.contains(&memory.memory_type()) {
                return false;
            }
        }

        if let Some(min_conf) = filters.min_confidence {
            if common.confidence.value() < min_conf.value() {
                return false;
            }
        }

        if let Some(ref agents) = filters.agent_ids {
            if !agents.contains(&common.agent_id) {
                return false;
            }
        }

        if let Some(ref required_tags) = filters.required_tags {
            for tag in required_tags {
                if !common.tags.contains(tag) {
                    return false;
                }
            }
        }

        if let Some(ref tags) = filters.tags {
            let has_any = tags.iter().any(|tag| common.tags.contains(tag));
            if !has_any {
                return false;
            }
        }

        if let Some(created_after) = filters.created_after {
            if common.provenance.created_at < created_after {
                return false;
            }
        }

        if let Some(created_before) = filters.created_before {
            if common.provenance.created_at > created_before {
                return false;
            }
        }

        if let Some(ref exclude_ids) = filters.exclude_ids {
            if exclude_ids.contains(memory.id()) {
                return false;
            }
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_overlapping_results() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        // Both lists rank A first, B second
        let vector = vec![(id_a, 0.9), (id_b, 0.7), (id_c, 0.5)];
        let text = vec![(id_a, 5.0), (id_b, 3.0)];

        let k = 60.0;
        let scores = fuse_rrf(&vector, &text, k);

        // A appears at rank 0 in both lists: 2 * 1/(60+0+1) = 2/61
        let expected_a = 2.0 / 61.0;
        assert!((scores[&id_a] - expected_a).abs() < 1e-10);

        // Relative ordering must be preserved: A > B > C
        assert!(scores[&id_a] > scores[&id_b]);
        assert!(scores[&id_b] > scores[&id_c]);

        // C only appears in vector at rank 2: 1/(60+2+1) = 1/63
        let expected_c = 1.0 / 63.0;
        assert!((scores[&id_c] - expected_c).abs() < 1e-10);
    }

    #[test]
    fn rrf_disjoint_results() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        let vector = vec![(id_a, 0.9)];
        let text = vec![(id_b, 5.0)];

        let k = 60.0;
        let scores = fuse_rrf(&vector, &text, k);

        // Both appear at rank 0 in their respective lists with equal raw RRF scores.
        let expected = 1.0 / 61.0;
        assert!((scores[&id_a] - expected).abs() < 1e-10);
        assert!((scores[&id_b] - expected).abs() < 1e-10);
    }

    #[test]
    fn rrf_quality_gap_preserved() {
        // A document appearing in both lists should have a meaningful score gap
        // compared to one appearing in only one list.
        let id_both = Uuid::new_v4();
        let id_single = Uuid::new_v4();

        let k = 60.0;
        let vector = vec![(id_both, 0.9), (id_single, 0.3)];
        let text = vec![(id_both, 5.0)];

        let scores = fuse_rrf(&vector, &text, k);

        // id_both: 1/61 + 1/61 = 2/61 ~ 0.0328
        // id_single: 1/62 ~ 0.0161
        // The ratio should be roughly 2:1, preserving quality gap
        assert!(scores[&id_both] > 1.5 * scores[&id_single]);
    }

    #[test]
    fn weighted_average_normalizes_bm25() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        let vector = vec![(id_a, 0.8), (id_b, 0.4)];
        // BM25 scores are unbounded; max is 20.0 so text scores become 1.0 and 0.5
        let text = vec![(id_a, 20.0), (id_b, 10.0)];

        let scores = fuse_weighted_average(&vector, &text, 0.7, 0.3);

        // A: 0.8 * 0.7 + (20/20) * 0.3 = 0.56 + 0.3 = 0.86
        assert!((scores[&id_a] - 0.86).abs() < 1e-10);

        // B: 0.4 * 0.7 + (10/20) * 0.3 = 0.28 + 0.15 = 0.43
        assert!((scores[&id_b] - 0.43).abs() < 1e-10);
    }

    #[test]
    fn weighted_average_single_text_result() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        let vector = vec![(id_a, 0.8), (id_b, 0.4)];
        // Single text result: normalized to 1.0
        let text = vec![(id_a, 2.0)];

        let scores = fuse_weighted_average(&vector, &text, 0.7, 0.3);

        // A: 0.8 * 0.7 + (2.0/2.0) * 0.3 = 0.56 + 0.3 = 0.86
        assert!((scores[&id_a] - 0.86).abs() < 1e-10);

        // B: 0.4 * 0.7 + 0.0 * 0.3 = 0.28
        assert!((scores[&id_b] - 0.28).abs() < 1e-10);
    }

    #[test]
    fn weighted_average_empty_lists() {
        let scores = fuse_weighted_average(&[], &[], 0.7, 0.3);
        assert!(scores.is_empty());
    }

    #[test]
    fn weighted_average_no_text_results() {
        let id_a = Uuid::new_v4();
        let vector = vec![(id_a, 0.8)];
        let text: Vec<(Uuid, f64)> = vec![];

        let scores = fuse_weighted_average(&vector, &text, 0.7, 0.3);

        // A: 0.8 * 0.7 + 0.0 * 0.3 = 0.56
        assert!((scores[&id_a] - 0.56).abs() < 1e-10);
    }

    #[test]
    fn min_score_threshold_constants_are_sane() {
        // RRF threshold should be well below the score of a single top-ranked result
        // with k=60: 1/(60+0+1) ~ 0.0164
        const _: () = assert!(MIN_SCORE_THRESHOLD_RRF < 1.0 / 61.0);
        const _: () = assert!(MIN_SCORE_THRESHOLD_RRF > 0.0);

        // Weighted average threshold should be below a moderately-relevant result
        const _: () = assert!(MIN_SCORE_THRESHOLD_WEIGHTED < 0.5);
        const _: () = assert!(MIN_SCORE_THRESHOLD_WEIGHTED > 0.0);
    }
}
