use membrain_core::MemoryId;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(MemoryId),

    #[error("Edge not found: {0} -> {1}")]
    EdgeNotFound(MemoryId, MemoryId),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Graph full: max {max_nodes} nodes")]
    GraphFull { max_nodes: usize },

    #[error("Core error: {0}")]
    Core(#[from] membrain_core::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<rmp_serde::encode::Error> for GraphError {
    fn from(err: rmp_serde::encode::Error) -> Self {
        GraphError::Serialization(err.to_string())
    }
}

impl From<rmp_serde::decode::Error> for GraphError {
    fn from(err: rmp_serde::decode::Error) -> Self {
        GraphError::Serialization(err.to_string())
    }
}
