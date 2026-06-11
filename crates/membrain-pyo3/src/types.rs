//! Python-friendly type wrappers for Membrain result types.
//!
//! PyO3 cannot directly expose Rust structs from other crates as Python classes
//! unless those crates also depend on PyO3. These thin wrappers bridge
//! `membrain_ffi` types to `#[pyclass]` types with identical field names.

use pyo3::prelude::*;

use membrain_ffi::{
    MemoryInfo, SearchResult, SearchResults, StoreResult,
};

// ---------------------------------------------------------------------------
// StoreResult
// ---------------------------------------------------------------------------

/// Result of a store operation.
#[pyclass(name = "StoreResult")]
#[derive(Clone)]
pub struct PyStoreResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub id: Option<String>,
    #[pyo3(get)]
    pub merged_with: Option<String>,
    #[pyo3(get)]
    pub rejection_reason: Option<String>,
    #[pyo3(get)]
    pub duration_ms: u64,
}

impl From<StoreResult> for PyStoreResult {
    fn from(r: StoreResult) -> Self {
        Self {
            success: r.success,
            id: r.id,
            merged_with: r.merged_with,
            rejection_reason: r.rejection_reason,
            duration_ms: r.duration_ms,
        }
    }
}

#[pymethods]
impl PyStoreResult {
    fn __repr__(&self) -> String {
        format!(
            "StoreResult(success={}, id={:?}, duration_ms={})",
            self.success, self.id, self.duration_ms
        )
    }
}

// ---------------------------------------------------------------------------
// MemoryEntry (individual search result)
// ---------------------------------------------------------------------------

/// A single memory from a search result.
#[pyclass(name = "MemoryEntry")]
#[derive(Clone)]
pub struct PyMemoryEntry {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub memory_type: String,
    #[pyo3(get)]
    pub created_at: String,
}

impl From<SearchResult> for PyMemoryEntry {
    fn from(r: SearchResult) -> Self {
        Self {
            id: r.id,
            content: r.content,
            score: r.score,
            memory_type: r.memory_type,
            created_at: r.created_at,
        }
    }
}

#[pymethods]
impl PyMemoryEntry {
    fn __repr__(&self) -> String {
        format!(
            "MemoryEntry(id={:?}, score={:.4}, type={:?})",
            self.id, self.score, self.memory_type
        )
    }
}

// ---------------------------------------------------------------------------
// SearchResults
// ---------------------------------------------------------------------------

/// Results from a search query.
#[pyclass(name = "SearchResults")]
#[derive(Clone)]
pub struct PySearchResults {
    #[pyo3(get)]
    pub memories: Vec<PyMemoryEntry>,
    #[pyo3(get)]
    pub was_gated: bool,
    #[pyo3(get)]
    pub duration_ms: u64,
}

impl From<SearchResults> for PySearchResults {
    fn from(r: SearchResults) -> Self {
        Self {
            memories: r.memories.into_iter().map(PyMemoryEntry::from).collect(),
            was_gated: r.was_gated,
            duration_ms: r.duration_ms,
        }
    }
}

#[pymethods]
impl PySearchResults {
    fn __repr__(&self) -> String {
        format!(
            "SearchResults(count={}, was_gated={}, duration_ms={})",
            self.memories.len(),
            self.was_gated,
            self.duration_ms
        )
    }

    fn __len__(&self) -> usize {
        self.memories.len()
    }
}

// ---------------------------------------------------------------------------
// MemoryInfo (single memory lookup)
// ---------------------------------------------------------------------------

/// Information about a single memory retrieved by ID.
#[pyclass(name = "MemoryInfo")]
#[derive(Clone)]
pub struct PyMemoryInfo {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub memory_type: String,
    #[pyo3(get)]
    pub confidence: f64,
}

impl From<MemoryInfo> for PyMemoryInfo {
    fn from(m: MemoryInfo) -> Self {
        Self {
            id: m.id,
            content: m.content,
            memory_type: m.memory_type,
            confidence: m.confidence,
        }
    }
}

#[pymethods]
impl PyMemoryInfo {
    fn __repr__(&self) -> String {
        format!(
            "MemoryInfo(id={:?}, type={:?}, confidence={:.2})",
            self.id, self.memory_type, self.confidence
        )
    }
}

// ---------------------------------------------------------------------------
// Case types (for CBR)
// ---------------------------------------------------------------------------

/// A single case entry from case-based reasoning search.
#[pyclass(name = "CaseEntry")]
#[derive(Clone)]
pub struct PyCaseEntry {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub problem: String,
    #[pyo3(get)]
    pub plan: String,
    #[pyo3(get)]
    pub outcome: String,
    #[pyo3(get)]
    pub reward: f64,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl PyCaseEntry {
    fn __repr__(&self) -> String {
        format!(
            "CaseEntry(id={:?}, reward={:.2}, score={:.4})",
            self.id, self.reward, self.score
        )
    }
}

/// Results from a case-based reasoning search.
#[pyclass(name = "CaseSearchResults")]
#[derive(Clone)]
pub struct PyCaseSearchResults {
    #[pyo3(get)]
    pub positive_cases: Vec<PyCaseEntry>,
    #[pyo3(get)]
    pub negative_cases: Vec<PyCaseEntry>,
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl PyCaseSearchResults {
    fn __repr__(&self) -> String {
        format!(
            "CaseSearchResults(positive={}, negative={}, duration_ms={})",
            self.positive_cases.len(),
            self.negative_cases.len(),
            self.duration_ms
        )
    }
}
