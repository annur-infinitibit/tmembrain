#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]
//! Graph-based memory retrieval for Membrain.
//!
//! Provides attention-weighted graph traversal over memory relationships,
//! enabling multi-hop reasoning for LLM context enrichment.
//!
//! # Examples
//!
//! **Rust:**
//! ```
//! use membrain_graph::{GraphConfig, MemoryGraph};
//!
//! let graph = MemoryGraph::new(GraphConfig::default());
//! ```
//!
//! **Python:**
//! ```python
//! from membrain import MembrainGraph
//!
//! graph = MembrainGraph()
//! ```
//!
//! **JavaScript:**
//! ```javascript
//! const { MembrainGraph } = require("membrain");
//!
//! const graph = new MembrainGraph();
//! ```

pub mod attention;
pub mod bridge;
pub mod config;
pub mod edge;
pub mod error;
pub mod graph;
pub mod gru;
pub mod message_passing;
pub mod node;
pub mod persistence;
pub mod pruning;
pub mod query;
pub mod tensor;

pub use attention::MultiHeadAttention;
pub use bridge::{GraphAugmentedRetrieval, GraphBridge};
pub use config::{
    AggregationMethod, EdgeConfig, GraphConfig, GruConfig, MessagePassingConfig, PruningConfig,
};
pub use edge::{EdgeId, GraphEdge, RelationType};
pub use error::{GraphError, Result};
pub use graph::MemoryGraph;
pub use gru::GruCell;
pub use message_passing::graph_query;
pub use node::GraphNode;
pub use persistence::{load_from_bytes, save_to_bytes, GraphSnapshot};
pub use pruning::{prune, PruningResult};
pub use query::{GraphQueryResult, ScoredNode, TraversalStep};
pub use tensor::{cosine_sim, sigmoid, softmax, tanh_vec, xavier_init, Matrix, Vector};
