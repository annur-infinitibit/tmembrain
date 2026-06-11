//! SQLite storage backend with FTS5 full-text search

use async_trait::async_trait;
use rusqlite::params;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio_rusqlite::Connection;

use membrain_core::error::{Error, Result};
use membrain_core::memory::{Memory, MemoryType};
use membrain_core::traits::{
    MatchType, MemoryStorage, SearchFilters, SearchQuery, SearchResult, StorageStats, Transaction,
};
use membrain_core::types::{AgentId, Embedding, MemoryId, Version};

/// SQLite storage backend
pub struct SqliteStorage {
    conn: Arc<Connection>,
}

impl SqliteStorage {
    /// Create a new SQLite storage
    pub async fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)
            .await
            .map_err(|e| Error::DatabaseConnection(e.to_string()))?;

        let storage = Self {
            conn: Arc::new(conn),
        };

        storage.initialize().await?;
        Ok(storage)
    }

    /// Create an in-memory SQLite database
    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open(":memory:")
            .await
            .map_err(|e| Error::DatabaseConnection(e.to_string()))?;

        let storage = Self {
            conn: Arc::new(conn),
        };

        storage.initialize().await?;
        Ok(storage)
    }

    async fn initialize(&self) -> Result<()> {
        self.conn
            .call(|conn| {
                // Enable WAL mode for better concurrent access
                conn.execute_batch("PRAGMA journal_mode=WAL;")?;
                conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
                conn.execute_batch("PRAGMA foreign_keys=ON;")?;

                // Create main memories table
                conn.execute(
                    r#"
                    CREATE TABLE IF NOT EXISTS memories (
                        id BLOB PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        confidence REAL NOT NULL,
                        agent_id BLOB NOT NULL,
                        content BLOB NOT NULL,
                        text_content TEXT NOT NULL,
                        embedding BLOB,
                        tags TEXT,
                        created_at TEXT NOT NULL,
                        modified_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0
                    )
                    "#,
                    [],
                )?;

                // Create indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id)",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence)",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed_at)",
                    [],
                )?;

                // Create FTS5 virtual table for full-text search
                // Using external content table pointing to text_content column
                conn.execute(
                    r#"
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        content_text,
                        tags
                    )
                    "#,
                    [],
                )?;

                // Create triggers to keep FTS in sync
                conn.execute_batch(
                    r#"
                    CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memories BEGIN
                        INSERT INTO memory_fts(memory_id, content_text, tags)
                        VALUES (new.id, new.text_content, new.tags);
                    END;

                    CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON memories BEGIN
                        DELETE FROM memory_fts WHERE memory_id = old.id;
                    END;

                    CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE ON memories BEGIN
                        UPDATE memory_fts SET content_text = new.text_content, tags = new.tags
                        WHERE memory_id = new.id;
                    END;
                    "#,
                )?;

                Ok(())
            })
            .await
            .map_err(|e| Error::SchemaMigration(e.to_string()))?;

        Ok(())
    }

    fn memory_type_to_string(mt: MemoryType) -> String {
        mt.to_string()
    }

    fn string_to_memory_type(s: &str) -> Option<MemoryType> {
        match s {
            "episodic_conversation" => Some(MemoryType::EpisodicConversation),
            "episodic_event" => Some(MemoryType::EpisodicEvent),
            "episodic_observation" => Some(MemoryType::EpisodicObservation),
            "semantic_fact" => Some(MemoryType::SemanticFact),
            "semantic_preference" => Some(MemoryType::SemanticPreference),
            "semantic_concept" => Some(MemoryType::SemanticConcept),
            "semantic_entity" => Some(MemoryType::SemanticEntity),
            "procedural_workflow" => Some(MemoryType::ProceduralWorkflow),
            "procedural_skill" => Some(MemoryType::ProceduralSkill),
            "procedural_pattern" => Some(MemoryType::ProceduralPattern),
            "procedural_case" => Some(MemoryType::ProceduralCase),
            "agent_state_goal" => Some(MemoryType::AgentStateGoal),
            "agent_state_task" => Some(MemoryType::AgentStateTask),
            "agent_state_working_memory" => Some(MemoryType::AgentStateWorkingMemory),
            _ => None,
        }
    }
}

#[async_trait]
impl MemoryStorage for SqliteStorage {
    async fn store(&self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        let id_bytes = id.as_bytes().to_vec();
        let memory_type = Self::memory_type_to_string(memory.memory_type());
        let common = memory.common();
        let confidence = common.confidence.value();
        let agent_id_bytes = common.agent_id.as_bytes().to_vec();
        let content = memory
            .to_msgpack()
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let text_content = memory.text_content();
        let embedding_bytes = common.embedding.as_ref().map(|e| e.to_bytes());
        let tags = common.tags.join(",");
        let created_at = common.provenance.created_at.to_rfc3339();
        let modified_at = common.provenance.modified_at.to_rfc3339();
        let last_accessed = common.provenance.last_accessed_at.to_rfc3339();
        let access_count = common.provenance.access_count as i64;
        let version = common.version as i64;

        self.conn
            .call(move |conn| {
                conn.execute(
                    r#"
                    INSERT INTO memories (
                        id, memory_type, version, confidence, agent_id, content, text_content,
                        embedding, tags, created_at, modified_at, last_accessed_at, access_count
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                    "#,
                    params![
                        id_bytes,
                        memory_type,
                        version,
                        confidence,
                        agent_id_bytes,
                        content,      // Binary msgpack content
                        text_content, // Text content for FTS
                        embedding_bytes,
                        tags,
                        created_at,
                        modified_at,
                        last_accessed,
                        access_count
                    ],
                )?;

                Ok(())
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(id)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let id_bytes = id.as_bytes().to_vec();

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare("SELECT content FROM memories WHERE id = ?1")?;
                let mut rows = stmt.query(params![id_bytes])?;

                if let Some(row) = rows.next()? {
                    let content: Vec<u8> = row.get(0)?;
                    let memory = Memory::from_msgpack(&content)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                    Ok(Some(memory))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn get_many(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_bytes: Vec<Vec<u8>> = ids.iter().map(|id| id.as_bytes().to_vec()).collect();

        self.conn
            .call(move |conn| {
                let placeholders: String = (0..id_bytes.len())
                    .map(|i| format!("?{}", i + 1))
                    .collect::<Vec<_>>()
                    .join(",");

                let sql = format!(
                    "SELECT content FROM memories WHERE id IN ({})",
                    placeholders
                );
                let mut stmt = conn.prepare(&sql)?;

                let params: Vec<&dyn rusqlite::ToSql> =
                    id_bytes.iter().map(|b| b as &dyn rusqlite::ToSql).collect();

                let mut rows = stmt.query(params.as_slice())?;
                let mut memories = Vec::new();

                while let Some(row) = rows.next()? {
                    let content: Vec<u8> = row.get(0)?;
                    if let Ok(memory) = Memory::from_msgpack(&content) {
                        memories.push(memory);
                    }
                }

                Ok(memories)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn update(&self, memory: Memory, expected_version: Version) -> Result<()> {
        let id = *memory.id();
        let id_bytes = id.as_bytes().to_vec();
        let memory_type = Self::memory_type_to_string(memory.memory_type());
        let common = memory.common();
        let new_version = (expected_version + 1) as i64;
        let expected_version = expected_version as i64;
        let confidence = common.confidence.value();
        let content = memory
            .to_msgpack()
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let text_content = memory.text_content();
        let embedding_bytes = common.embedding.as_ref().map(|e| e.to_bytes());
        let tags = common.tags.join(",");
        let modified_at = common.provenance.modified_at.to_rfc3339();
        let last_accessed = common.provenance.last_accessed_at.to_rfc3339();

        let rows_affected = self
            .conn
            .call(move |conn| {
                // First check version
                let current_version: Option<i64> = conn
                    .query_row(
                        "SELECT version FROM memories WHERE id = ?1",
                        params![&id_bytes],
                        |row| row.get(0),
                    )
                    .ok();

                if current_version.is_none() {
                    return Ok(0);
                }

                if current_version != Some(expected_version) {
                    return Ok(-1); // Version conflict
                }

                let affected = conn.execute(
                    r#"
                    UPDATE memories SET
                        memory_type = ?1,
                        version = ?2,
                        confidence = ?3,
                        content = ?4,
                        embedding = ?5,
                        tags = ?6,
                        modified_at = ?7,
                        last_accessed_at = ?8
                    WHERE id = ?9 AND version = ?10
                    "#,
                    params![
                        memory_type,
                        new_version,
                        confidence,
                        content,
                        embedding_bytes,
                        tags,
                        modified_at,
                        last_accessed,
                        id_bytes,
                        expected_version
                    ],
                )?;

                // Update FTS with text content
                if affected > 0 {
                    conn.execute(
                        "UPDATE memories SET content = ?1 WHERE id = ?2",
                        params![text_content, id_bytes],
                    )?;
                    conn.execute(
                        "UPDATE memories SET content = ?1 WHERE id = ?2",
                        params![content, id_bytes],
                    )?;
                }

                Ok(affected as i64)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        if rows_affected == 0 {
            return Err(Error::MemoryNotFound(id));
        }
        if rows_affected < 0 {
            return Err(Error::WriteConflict(id));
        }

        Ok(())
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        let id_bytes = id.as_bytes().to_vec();

        let affected = self
            .conn
            .call(move |conn| {
                let affected =
                    conn.execute("DELETE FROM memories WHERE id = ?1", params![id_bytes])?;
                Ok(affected)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(affected > 0)
    }

    async fn delete_many(&self, ids: &[MemoryId]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        let id_bytes: Vec<Vec<u8>> = ids.iter().map(|id| id.as_bytes().to_vec()).collect();

        self.conn
            .call(move |conn| {
                let placeholders: String = (0..id_bytes.len())
                    .map(|i| format!("?{}", i + 1))
                    .collect::<Vec<_>>()
                    .join(",");

                let sql = format!("DELETE FROM memories WHERE id IN ({})", placeholders);
                let params: Vec<&dyn rusqlite::ToSql> =
                    id_bytes.iter().map(|b| b as &dyn rusqlite::ToSql).collect();

                let affected = conn.execute(&sql, params.as_slice())?;
                Ok(affected)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let query_text = query.query.clone();
        let limit = query.limit as i64;
        let offset = query.offset as i64;
        let filters = query.filters.clone();
        let min_confidence = filters.min_confidence.as_ref().map(|c| c.value());
        let memory_types: Option<Vec<String>> = filters.memory_types.as_ref().map(|types| {
            types
                .iter()
                .map(|t| Self::memory_type_to_string(*t))
                .collect()
        });

        self.conn
            .call(move |conn| {
                let mut conditions = Vec::new();
                let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
                let mut param_idx = 1;

                // Build WHERE clause
                if let Some(conf) = min_confidence {
                    conditions.push(format!("confidence >= ?{}", param_idx));
                    params.push(Box::new(conf));
                    param_idx += 1;
                }

                if let Some(ref types) = memory_types {
                    let placeholders: String = types
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("?{}", param_idx + i))
                        .collect::<Vec<_>>()
                        .join(",");
                    conditions.push(format!("memory_type IN ({})", placeholders));
                    for t in types {
                        params.push(Box::new(t.clone()));
                        param_idx += 1;
                    }
                }

                let where_clause = if conditions.is_empty() {
                    String::new()
                } else {
                    format!("WHERE {}", conditions.join(" AND "))
                };

                let sql = if let Some(ref q) = query_text {
                    // Multi-word queries use OR logic for flexible matching
                    let words: Vec<&str> = q.split_whitespace().collect();
                    let fts_query = if words.len() > 1 {
                        words
                            .iter()
                            .map(|w| format!("\"{}\"", w.replace('"', "\"\"")))
                            .collect::<Vec<_>>()
                            .join(" OR ")
                    } else {
                        format!("\"{}\"", q.replace('"', "\"\""))
                    };
                    params.push(Box::new(fts_query));
                    format!(
                        r#"
                        SELECT m.content, bm25(memory_fts) as score
                        FROM memories m
                        JOIN memory_fts ON memory_fts.memory_id = m.id
                        WHERE memory_fts MATCH ?{}
                        {}
                        ORDER BY score
                        LIMIT ?{} OFFSET ?{}
                        "#,
                        param_idx,
                        if conditions.is_empty() {
                            String::new()
                        } else {
                            format!("AND {}", conditions.join(" AND "))
                        },
                        param_idx + 1,
                        param_idx + 2
                    )
                } else {
                    format!(
                        r#"
                        SELECT content, confidence as score
                        FROM memories
                        {}
                        ORDER BY last_accessed_at DESC
                        LIMIT ?{} OFFSET ?{}
                        "#,
                        where_clause,
                        param_idx,
                        param_idx + 1
                    )
                };

                params.push(Box::new(limit));
                params.push(Box::new(offset));

                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();

                let mut stmt = conn.prepare(&sql)?;
                let mut rows = stmt.query(params_refs.as_slice())?;

                let mut results = Vec::new();
                while let Some(row) = rows.next()? {
                    let content: Vec<u8> = row.get(0)?;
                    let score: f64 = row.get(1)?;

                    if let Ok(memory) = Memory::from_msgpack(&content) {
                        // Post-deserialization metadata filtering
                        if let Some(ref filter_metadata) = filters.metadata {
                            let memory_metadata = &memory.common().metadata;
                            let all_match = filter_metadata.iter().all(|(key, expected)| {
                                memory_metadata
                                    .get(key)
                                    .is_some_and(|actual| actual == expected)
                            });
                            if !all_match {
                                continue;
                            }
                        }

                        let match_type = if query_text.is_some() {
                            MatchType::Text
                        } else {
                            MatchType::Exact
                        };
                        // Normalize BM25 score (it's negative, more negative = better match)
                        let normalized_score = if query_text.is_some() {
                            (1.0 / (1.0 - score)).min(1.0)
                        } else {
                            score
                        };
                        results.push(SearchResult::new(memory, normalized_score, match_type));
                    }
                }

                Ok(results)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn vector_search(
        &self,
        _embedding: &Embedding,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        // For now, fall back to regular search
        // Full vector search would require loading embeddings and computing similarity
        let query = SearchQuery::new()
            .with_limit(limit)
            .with_filters(filters.unwrap_or_default());

        self.search(query).await
    }

    async fn text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let search_query = SearchQuery::new().with_query(query).with_limit(limit);

        self.search(search_query).await
    }

    async fn count(&self, filters: Option<SearchFilters>) -> Result<usize> {
        let min_confidence = filters
            .as_ref()
            .and_then(|f| f.min_confidence.as_ref())
            .map(|c| c.value());

        let memory_types: Option<Vec<String>> = filters
            .as_ref()
            .and_then(|f| f.memory_types.as_ref())
            .map(|types| {
                types
                    .iter()
                    .map(|t| Self::memory_type_to_string(*t))
                    .collect()
            });

        self.conn
            .call(move |conn| {
                let mut conditions = Vec::new();
                let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
                let mut param_idx = 1;

                if let Some(conf) = min_confidence {
                    conditions.push(format!("confidence >= ?{}", param_idx));
                    params.push(Box::new(conf));
                    param_idx += 1;
                }

                if let Some(ref types) = memory_types {
                    let placeholders: String = types
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("?{}", param_idx + i))
                        .collect::<Vec<_>>()
                        .join(",");
                    conditions.push(format!("memory_type IN ({})", placeholders));
                    for t in types {
                        params.push(Box::new(t.clone()));
                    }
                }

                let where_clause = if conditions.is_empty() {
                    String::new()
                } else {
                    format!("WHERE {}", conditions.join(" AND "))
                };

                let sql = format!("SELECT COUNT(*) FROM memories {}", where_clause);
                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();

                let count: i64 = conn.query_row(&sql, params_refs.as_slice(), |row| row.get(0))?;
                Ok(count as usize)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn exists(&self, id: &MemoryId) -> Result<bool> {
        let id_bytes = id.as_bytes().to_vec();

        self.conn
            .call(move |conn| {
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = ?1",
                    params![id_bytes],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn get_by_agent(
        &self,
        agent_id: &AgentId,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let agent_bytes = agent_id.as_bytes().to_vec();
        let limit = limit as i64;
        let offset = offset as i64;

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    r#"
                    SELECT content FROM memories
                    WHERE agent_id = ?1
                    ORDER BY created_at DESC
                    LIMIT ?2 OFFSET ?3
                    "#,
                )?;

                let mut rows = stmt.query(params![agent_bytes, limit, offset])?;
                let mut memories = Vec::new();

                while let Some(row) = rows.next()? {
                    let content: Vec<u8> = row.get(0)?;
                    if let Ok(memory) = Memory::from_msgpack(&content) {
                        memories.push(memory);
                    }
                }

                Ok(memories)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn get_by_type(
        &self,
        memory_type: MemoryType,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Memory>> {
        let type_str = Self::memory_type_to_string(memory_type);
        let limit = limit as i64;
        let offset = offset as i64;

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    r#"
                    SELECT content FROM memories
                    WHERE memory_type = ?1
                    ORDER BY created_at DESC
                    LIMIT ?2 OFFSET ?3
                    "#,
                )?;

                let mut rows = stmt.query(params![type_str, limit, offset])?;
                let mut memories = Vec::new();

                while let Some(row) = rows.next()? {
                    let content: Vec<u8> = row.get(0)?;
                    if let Ok(memory) = Memory::from_msgpack(&content) {
                        memories.push(memory);
                    }
                }

                Ok(memories)
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn record_access(&self, id: &MemoryId) -> Result<()> {
        let id_bytes = id.as_bytes().to_vec();
        let now = chrono::Utc::now().to_rfc3339();

        self.conn
            .call(move |conn| {
                conn.execute(
                    r#"
                    UPDATE memories
                    SET last_accessed_at = ?1, access_count = access_count + 1
                    WHERE id = ?2
                    "#,
                    params![now, id_bytes],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn stats(&self) -> Result<StorageStats> {
        self.conn
            .call(|conn| {
                let total: i64 =
                    conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

                let mut by_type: HashMap<MemoryType, usize> = HashMap::new();
                let mut stmt =
                    conn.prepare("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")?;
                let mut rows = stmt.query([])?;

                while let Some(row) = rows.next()? {
                    let type_str: String = row.get(0)?;
                    let count: i64 = row.get(1)?;
                    if let Some(mt) = SqliteStorage::string_to_memory_type(&type_str) {
                        by_type.insert(mt, count as usize);
                    }
                }

                let embeddings_count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL",
                    [],
                    |row| row.get(0),
                )?;

                let avg_confidence: f64 = conn
                    .query_row("SELECT AVG(confidence) FROM memories", [], |row| row.get(0))
                    .unwrap_or(0.0);

                let agent_count: i64 = conn.query_row(
                    "SELECT COUNT(DISTINCT agent_id) FROM memories",
                    [],
                    |row| row.get(0),
                )?;

                // Get database file size
                let storage_bytes: i64 = conn
                    .query_row("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()", [], |row| row.get(0))
                    .unwrap_or(0);

                Ok(StorageStats {
                    total_memories: total as usize,
                    by_type,
                    storage_bytes: storage_bytes as u64,
                    embeddings_count: embeddings_count as usize,
                    avg_confidence,
                    agent_count: agent_count as usize,
                })
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }

    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        // SQLite transactions are handled differently with tokio-rusqlite
        // For now, return a simple wrapper
        Ok(Box::new(SqliteTransaction {
            conn: self.conn.clone(),
            operations: Vec::new(),
        }))
    }

    async fn health_check(&self) -> Result<()> {
        self.conn
            .call(|conn| {
                conn.execute("SELECT 1", [])?;
                Ok(())
            })
            .await
            .map_err(|e| Error::Storage(e.to_string()))
    }
}

/// SQLite transaction implementation
struct SqliteTransaction {
    conn: Arc<Connection>,
    operations: Vec<TransactionOp>,
}

enum TransactionOp {
    Store(Memory),
    Update(Memory, Version),
    Delete(MemoryId),
}

#[async_trait]
impl Transaction for SqliteTransaction {
    async fn store(&mut self, memory: Memory) -> Result<MemoryId> {
        let id = *memory.id();
        self.operations.push(TransactionOp::Store(memory));
        Ok(id)
    }

    async fn update(&mut self, memory: Memory, expected_version: Version) -> Result<()> {
        self.operations
            .push(TransactionOp::Update(memory, expected_version));
        Ok(())
    }

    async fn delete(&mut self, id: &MemoryId) -> Result<bool> {
        self.operations.push(TransactionOp::Delete(*id));
        Ok(true)
    }

    async fn commit(self: Box<Self>) -> Result<()> {
        let operations = self.operations;

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                for op in operations {
                    match op {
                        TransactionOp::Store(memory) => {
                            let id_bytes = memory.id().as_bytes().to_vec();
                            let memory_type = SqliteStorage::memory_type_to_string(memory.memory_type());
                            let common = memory.common();
                            let confidence = common.confidence.value();
                            let agent_id_bytes = common.agent_id.as_bytes().to_vec();
                            let content = memory
                                .to_msgpack()
                                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                            let text_content = memory.text_content();
                            let embedding_bytes = common.embedding.as_ref().map(|e| e.to_bytes());
                            let tags = common.tags.join(",");
                            let created_at = common.provenance.created_at.to_rfc3339();
                            let modified_at = common.provenance.modified_at.to_rfc3339();
                            let last_accessed = common.provenance.last_accessed_at.to_rfc3339();

                            tx.execute(
                                r#"
                                INSERT INTO memories (
                                    id, memory_type, version, confidence, agent_id, content,
                                    embedding, tags, created_at, modified_at, last_accessed_at, access_count
                                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 0)
                                "#,
                                params![
                                    id_bytes,
                                    memory_type,
                                    common.version as i64,
                                    confidence,
                                    agent_id_bytes,
                                    text_content,
                                    embedding_bytes,
                                    tags,
                                    created_at,
                                    modified_at,
                                    last_accessed
                                ],
                            )?;

                            tx.execute(
                                "UPDATE memories SET content = ?1 WHERE id = ?2",
                                params![content, id_bytes],
                            )?;
                        }
                        TransactionOp::Update(memory, expected_version) => {
                            let id_bytes = memory.id().as_bytes().to_vec();
                            let content = memory
                                .to_msgpack()
                                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                            let affected = tx.execute(
                                r#"
                                UPDATE memories SET
                                    content = ?1,
                                    version = version + 1,
                                    modified_at = ?2
                                WHERE id = ?3 AND version = ?4
                                "#,
                                params![
                                    content,
                                    chrono::Utc::now().to_rfc3339(),
                                    id_bytes,
                                    expected_version as i64
                                ],
                            )?;

                            if affected == 0 {
                                return Err(tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows));
                            }
                        }
                        TransactionOp::Delete(id) => {
                            let id_bytes = id.as_bytes().to_vec();
                            tx.execute("DELETE FROM memories WHERE id = ?1", params![id_bytes])?;
                        }
                    }
                }

                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(|e| Error::TransactionFailed(e.to_string()))
    }

    async fn rollback(self: Box<Self>) -> Result<()> {
        // Operations are just discarded
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
    async fn test_sqlite_store_and_get() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let memory = create_test_memory("Test fact");
        let id = *memory.id();

        storage.store(memory).await.unwrap();

        let retrieved = storage.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.id(), &id);
    }

    #[tokio::test]
    async fn test_sqlite_delete() {
        let storage = SqliteStorage::in_memory().await.unwrap();
        let memory = create_test_memory("To delete");
        let id = *memory.id();

        storage.store(memory).await.unwrap();
        assert!(storage.exists(&id).await.unwrap());

        storage.delete(&id).await.unwrap();
        assert!(!storage.exists(&id).await.unwrap());
    }

    #[tokio::test]
    async fn test_sqlite_count() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        storage.store(create_test_memory("One")).await.unwrap();
        storage.store(create_test_memory("Two")).await.unwrap();

        let count = storage.count(None).await.unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_sqlite_text_search() {
        let storage = SqliteStorage::in_memory().await.unwrap();

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
    async fn test_sqlite_stats() {
        let storage = SqliteStorage::in_memory().await.unwrap();

        storage.store(create_test_memory("Fact 1")).await.unwrap();
        storage.store(create_test_memory("Fact 2")).await.unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.total_memories, 2);
    }
}
