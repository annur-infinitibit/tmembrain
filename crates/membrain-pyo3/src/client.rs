//! PyO3 client wrapper for `membrain_ffi::MembrainClient`.
//!
//! Every method that performs I/O returns a Python coroutine via
//! `pyo3_async_runtimes::tokio::future_into_py`. The coroutine runs on
//! the shared Tokio runtime managed by `pyo3-async-runtimes`.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use membrain_core::config::Config;
use membrain_core::types::Embedding;
use membrain_ffi::MembrainClient as RustClient;

use crate::types::*;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

/// Convert a `membrain_core::error::Error` into a `PyErr`.
fn to_py_err(e: membrain_core::error::Error) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e.to_string())
}

// ---------------------------------------------------------------------------
// PyMembrainClient
// ---------------------------------------------------------------------------

/// The main Membrain client. Every method that touches storage is async
/// (returns a Python coroutine).
///
/// Usage from Python::
///
///     from membrain._native import MembrainClient
///
///     client = MembrainClient()
///     result = await client.store_fact("User prefers dark mode", 0.9)
///     results = await client.search("dark mode", limit=10)
///     client.close()
#[pyclass(name = "MembrainClient")]
pub struct PyMembrainClient {
    inner: Option<Arc<RustClient>>,
}

#[pymethods]
impl PyMembrainClient {
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    /// Create a new MembrainClient.
    ///
    /// Args:
    ///     config: Optional dict of configuration settings.
    ///     lib_path: Ignored (kept for API compatibility with old ctypes client).
    #[new]
    #[pyo3(signature = (config=None, *, lib_path=None))]
    fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyDict>>,
        lib_path: Option<&str>,
    ) -> PyResult<Self> {
        // lib_path is kept for backward compatibility — ignored in PyO3 impl
        let _ = lib_path;

        let rust_config = if let Some(cfg) = config {
            let json_str: String = py
                .import("json")?
                .call_method1("dumps", (cfg,))?
                .extract()?;
            serde_json::from_str::<Config>(&json_str).map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("invalid config: {}", e))
            })?
        } else {
            Config::default()
        };

        // Use the pyo3-async-runtimes Tokio runtime to initialize the client.
        // The constructor is synchronous (#[new] cannot return a coroutine),
        // but this is acceptable because construction is a one-time setup call.
        let client = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(RustClient::with_config(rust_config))
            .map_err(to_py_err)?;

        Ok(Self {
            inner: Some(Arc::new(client)),
        })
    }

    // ------------------------------------------------------------------
    // Store methods — each returns a Python coroutine
    // ------------------------------------------------------------------

    /// Store a fact in LLM memory.
    ///
    /// Args:
    ///     statement: The fact to store.
    ///     confidence: Confidence score (0.0-1.0).
    ///     embedding: Optional pre-computed embedding vector.
    ///     metadata: Optional metadata key-value pairs.
    ///
    /// Returns:
    ///     StoreResult with success status and memory ID.
    #[pyo3(signature = (statement, confidence=0.9, embedding=None, metadata=None))]
    fn store_fact<'py>(
        &self,
        py: Python<'py>,
        statement: String,
        confidence: f64,
        embedding: Option<Vec<f32>>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);
        let meta = metadata.map(|m| {
            m.into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect()
        });

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_fact_with_embedding(&statement, confidence, emb, meta)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a preference.
    #[pyo3(signature = (holder, subject, preference, strength=None, embedding=None))]
    fn store_preference<'py>(
        &self,
        py: Python<'py>,
        holder: String,
        subject: String,
        preference: String,
        strength: Option<String>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let strength = strength.unwrap_or_else(|| "moderate".to_string());
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_preference_with_embedding(&holder, &subject, &preference, &strength, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store an event.
    #[pyo3(signature = (event_type, description, embedding=None))]
    fn store_event<'py>(
        &self,
        py: Python<'py>,
        event_type: String,
        description: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_event_with_embedding(&event_type, &description, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store an observation.
    #[pyo3(signature = (content, embedding=None))]
    fn store_observation<'py>(
        &self,
        py: Python<'py>,
        content: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_observation_with_embedding(&content, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a concept.
    #[pyo3(signature = (name, definition, embedding=None))]
    fn store_concept<'py>(
        &self,
        py: Python<'py>,
        name: String,
        definition: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_concept_with_embedding(&name, &definition, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store an entity.
    #[pyo3(signature = (name, entity_type, embedding=None))]
    fn store_entity<'py>(
        &self,
        py: Python<'py>,
        name: String,
        entity_type: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_entity_with_embedding(&name, &entity_type, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a workflow.
    #[pyo3(signature = (name, description, embedding=None))]
    fn store_workflow<'py>(
        &self,
        py: Python<'py>,
        name: String,
        description: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_workflow_with_embedding(&name, &description, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a skill.
    #[pyo3(signature = (name, description, embedding=None))]
    fn store_skill<'py>(
        &self,
        py: Python<'py>,
        name: String,
        description: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_skill_with_embedding(&name, &description, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a pattern.
    #[pyo3(signature = (name, description, pattern_type, embedding=None))]
    fn store_pattern<'py>(
        &self,
        py: Python<'py>,
        name: String,
        description: String,
        pattern_type: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_pattern_with_embedding(&name, &description, &pattern_type, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a case (experience for case-based reasoning).
    #[pyo3(signature = (problem, plan, outcome, reward=None, embedding=None))]
    fn store_case<'py>(
        &self,
        py: Python<'py>,
        problem: String,
        plan: String,
        outcome: String,
        reward: Option<f64>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let reward = reward.unwrap_or(1.0);
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_case_with_embedding(&problem, &plan, &outcome, reward, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a goal.
    #[pyo3(signature = (description, embedding=None))]
    fn store_goal<'py>(
        &self,
        py: Python<'py>,
        description: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_goal_with_embedding(&description, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    /// Store a task.
    #[pyo3(signature = (title, embedding=None))]
    fn store_task<'py>(
        &self,
        py: Python<'py>,
        title: String,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .store_task_with_embedding(&title, emb)
                .await
                .map_err(to_py_err)?;
            Ok(PyStoreResult::from(result))
        })
    }

    // ------------------------------------------------------------------
    // Search methods
    // ------------------------------------------------------------------

    /// Search for memories matching a natural language query.
    ///
    /// Args:
    ///     query: The search query.
    ///     limit: Maximum number of results (default: 10).
    ///     filters: Optional filter criteria (passed from Python pre-serialized as a JSON string).
    ///     embedding: Optional pre-computed query embedding.
    ///
    /// Returns:
    ///     SearchResults with matching memories.
    #[pyo3(signature = (query, limit=10, filters=None, embedding=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
        filters: Option<String>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;
        let emb = embedding.map(Embedding::new);

        // Parse filters JSON string into SearchFiltersJson if provided
        let parsed_filters = if let Some(ref f) = filters {
            Some(
                serde_json::from_str::<membrain_ffi::SearchFiltersJson>(f).map_err(|e| {
                    PyErr::new::<PyRuntimeError, _>(format!("invalid filters JSON: {}", e))
                })?,
            )
        } else {
            None
        };

        // If we have an embedding but parsed_filters doesn't have one, inject it
        let search_filters = match (parsed_filters, emb) {
            (Some(mut f), Some(e)) => {
                if f.embedding.is_none() {
                    f.embedding = Some(e.into_values());
                }
                Some(f)
            }
            (Some(f), None) => Some(f),
            (None, Some(e)) => Some(membrain_ffi::SearchFiltersJson {
                memory_types: None,
                min_confidence: None,
                tags: None,
                agent_id: None,
                metadata: None,
                embedding: Some(e.into_values()),
            }),
            (None, None) => None,
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = if search_filters.is_some() {
                inner
                    .search_with_filters(&query, limit, search_filters)
                    .await
                    .map_err(to_py_err)?
            } else {
                inner.search(&query, limit).await.map_err(to_py_err)?
            };
            Ok(PySearchResults::from(result))
        })
    }

    // ------------------------------------------------------------------
    // Get / Delete / Count
    // ------------------------------------------------------------------

    /// Get a memory by its unique ID.
    ///
    /// Returns MemoryInfo or None if not found.
    fn get<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.get(&id).await.map_err(to_py_err)?;
            Ok(result.map(PyMemoryInfo::from))
        })
    }

    /// Delete a memory by its unique ID.
    ///
    /// Returns True if the memory was deleted.
    fn delete<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.delete(&id).await.map_err(to_py_err)
        })
    }

    /// Get the total number of memories stored.
    fn count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.count().await.map_err(to_py_err)
        })
    }

    // ------------------------------------------------------------------
    // Stats and health
    // ------------------------------------------------------------------

    /// Get storage statistics as a JSON string.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let s = inner.stats().await.map_err(to_py_err)?;
            serde_json::to_string(&s).map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("serialization error: {}", e))
            })
        })
    }

    /// Check vector backend health status as a JSON string.
    fn vector_backend_health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let h = inner.vector_backend_health().await.map_err(to_py_err)?;
            serde_json::to_string(&h).map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("serialization error: {}", e))
            })
        })
    }

    /// Get vector backend capabilities and stats as a JSON string.
    fn vector_backend_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let s = inner.vector_backend_stats().await.map_err(to_py_err)?;
            serde_json::to_string(&s).map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("serialization error: {}", e))
            })
        })
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Close the client and release resources.
    /// Synchronous — no thread draining needed with true async.
    fn close(&mut self) {
        // Drop the Arc<RustClient>. If this is the last reference,
        // the Rust client and all its resources (storage, pipelines) are dropped.
        self.inner = None;
    }

    /// Context manager support: `with MembrainClient() as client:`
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) {
        self.close();
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "MembrainClient(open)".to_string()
        } else {
            "MembrainClient(closed)".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl PyMembrainClient {
    /// Get the inner client, or raise an error if the client has been closed.
    fn get_inner(&self) -> PyResult<Arc<RustClient>> {
        self.inner.clone().ok_or_else(|| {
            PyErr::new::<PyRuntimeError, _>("client is closed")
        })
    }
}
