//! PyO3 graph client wrapper.
//!
//! Graph queries are heavily CPU-bound and use `spawn_blocking` to avoid
//! blocking the Tokio executor thread pool. Data modification and serialization 
//! APIs (`add_node`, `save`, etc.) execute synchronously on the caller's thread.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use membrain_graph::{GraphBridge, GraphConfig, GraphQueryResult, MemoryGraph};


// ---------------------------------------------------------------------------
// PyMembrainGraph
// ---------------------------------------------------------------------------

/// Memory graph for graph-augmented retrieval.
///
/// Graph operations are CPU-bound. Heavy queries run on the Tokio blocking
/// thread pool via `spawn_blocking` so they do not block the async executor.
#[pyclass(name = "MembrainGraph")]
pub struct PyMembrainGraph {
    inner: Option<Arc<GraphBridge>>,
}

#[pymethods]
impl PyMembrainGraph {
    #[new]
    #[pyo3(signature = (embedding_dim=None))]
    fn new(embedding_dim: Option<usize>) -> PyResult<Self> {
        let config = GraphConfig {
            embedding_dim: embedding_dim.unwrap_or(384),
            ..Default::default()
        };
        let graph = Arc::new(MemoryGraph::new(config));
        let bridge = Arc::new(GraphBridge::new(graph));
        Ok(Self {
            inner: Some(bridge),
        })
    }

    /// Query the graph with a query embedding.
    ///
    /// Returns results as a Python coroutine (runs on blocking thread pool).
    #[pyo3(signature = (embedding, max_hops=2, top_k=10))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        embedding: Vec<f32>,
        max_hops: usize,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get_inner()?;

        // Graph query involves BFS/DFS traversal — CPU-bound.
        // Run in thread pool so we do not block the async executor.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = tokio::task::spawn_blocking(move || {
                use membrain_graph::GraphAugmentedRetrieval;
                let emb = membrain_core::types::Embedding::new(embedding);
                inner.graph_query(&emb, max_hops, top_k)
            })
            .await
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("graph query join error: {}", e))
            })?
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("graph query error: {}", e))
            })?;

            Ok(convert_graph_result(result))
        })
    }

    /// Add a node to the graph.
    #[pyo3(signature = (memory_id, embedding, confidence=0.5))]
    fn add_node(&self, memory_id: String, embedding: Vec<f32>, confidence: f64) -> PyResult<()> {
        let inner = self.get_inner()?;
        let id_parsed = memory_id.parse().map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("invalid memory ID: {}", e))
        })?;
        let emb = membrain_core::types::Embedding::new(embedding);
        let conf = membrain_core::types::Confidence::new(confidence);
        inner.graph().add_node(id_parsed, &emb, conf).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("failed to add node: {}", e))
        })
    }

    /// Remove a node from the graph.
    fn remove_node(&self, memory_id: String) -> PyResult<()> {
        let inner = self.get_inner()?;
        let id_parsed = memory_id.parse().map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("invalid memory ID: {}", e))
        })?;
        inner.graph().remove_node(&id_parsed).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("failed to remove node: {}", e))
        })
    }

    /// Prune weak edges.
    fn prune(&self) -> PyResult<String> {
        let inner = self.get_inner()?;
        let result = membrain_graph::pruning::prune(inner.graph());
        Ok(serde_json::to_string(&result).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("failed to serialize prune result: {}", e))
        })?)
    }

    /// Save graph to base64 string.
    fn save(&self) -> PyResult<String> {
        use base64::Engine;
        let inner = self.get_inner()?;
        let bytes = membrain_graph::persistence::save_to_bytes(inner.graph()).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("failed to save graph: {}", e))
        })?;
        Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
    }

    /// Load graph from base64 string.
    #[classmethod]
    #[pyo3(signature = (data, lib_path=None))]
    fn load(_cls: &Bound<'_, pyo3::types::PyType>, data: String, lib_path: Option<&str>) -> PyResult<Self> {
        use base64::Engine;
        let _ = lib_path; // ignored for compatibility
        let bytes = base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("invalid base64 data: {}", e))
        })?;
        let graph = membrain_graph::persistence::load_from_bytes(&bytes).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("failed to load graph: {}", e))
        })?;
        let bridge = Arc::new(GraphBridge::new(Arc::new(graph)));
        Ok(Self {
            inner: Some(bridge),
        })
    }

    /// Get the number of nodes in the graph.
    fn node_count(&self) -> PyResult<usize> {
        let inner = self.get_inner()?;
        Ok(inner.graph().node_count())
    }

    /// Get the number of edges in the graph.
    fn edge_count(&self) -> PyResult<usize> {
        let inner = self.get_inner()?;
        Ok(inner.graph().edge_count())
    }

    /// Close the graph and release resources.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __repr__(&self) -> String {
        if let Some(ref inner) = self.inner {
            format!(
                "MembrainGraph(nodes={}, edges={})",
                inner.graph().node_count(),
                inner.graph().edge_count()
            )
        } else {
            "MembrainGraph(closed)".to_string()
        }
    }
}

impl PyMembrainGraph {
    fn get_inner(&self) -> PyResult<Arc<GraphBridge>> {
        self.inner.clone().ok_or_else(|| {
            PyErr::new::<PyRuntimeError, _>("graph is closed")
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a Rust GraphQueryResult into a Python-friendly dict-like string.
/// For Phase 2 we return a simple representation; the full Python layer
/// will add richer conversion in Phase 3.
fn convert_graph_result(result: GraphQueryResult) -> String {
    serde_json::json!({
        "nodes": result.nodes.iter().map(|n| {
            serde_json::json!({
                "memory_id": n.memory_id.to_string(),
                "score": n.score,
                "hop_distance": n.hop_distance,
            })
        }).collect::<Vec<_>>(),
        "hops_performed": result.hops_performed,
        "nodes_visited": result.nodes_visited,
        "traversed_edges": result.traversed_edges.iter().map(|s| {
            serde_json::json!({
                "from": s.from.to_string(),
                "to": s.to.to_string(),
                "edge_weight": s.edge_weight,
                "attention_score": s.attention_score,
                "hop": s.hop,
            })
        }).collect::<Vec<_>>(),
    })
    .to_string()
}
