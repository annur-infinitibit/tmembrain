use std::collections::HashMap;

use membrain_core::MemoryId;
use serde::{Deserialize, Serialize};

use crate::attention::MultiHeadAttention;
use crate::config::GraphConfig;
use crate::edge::{EdgeId, GraphEdge};
use crate::error::{GraphError, Result};
use crate::graph::MemoryGraph;
use crate::gru::GruCell;
use crate::node::GraphNode;
use crate::tensor::Matrix;

#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub config: GraphConfig,
    pub gru: GruCell,
    pub attention: MultiHeadAttention,
    pub input_projection: Vec<Vec<f32>>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

impl GraphSnapshot {
    pub fn from_graph(graph: &MemoryGraph) -> Self {
        let nodes: Vec<GraphNode> = graph.nodes.read().values().cloned().collect();
        let edges: Vec<GraphEdge> = graph.edges.read().values().cloned().collect();

        // Convert Matrix to nested Vec for serialization
        let proj = &graph.input_projection;
        let input_projection: Vec<Vec<f32>> =
            proj.rows().into_iter().map(|row| row.to_vec()).collect();

        Self {
            config: graph.config.clone(),
            gru: graph.gru.clone(),
            attention: graph.attention.clone(),
            input_projection,
            nodes,
            edges,
        }
    }

    pub fn into_graph(self) -> MemoryGraph {
        let h_dim = self.config.hidden_dim;
        let e_dim = self.config.embedding_dim;

        // Reconstruct input_projection Matrix
        let input_projection = if self.input_projection.is_empty() {
            Matrix::zeros((h_dim, e_dim))
        } else {
            let rows = self.input_projection.len();
            let cols = self.input_projection[0].len();
            let flat: Vec<f32> = self.input_projection.into_iter().flatten().collect();
            Matrix::from_shape_vec((rows, cols), flat)
                .unwrap_or_else(|_| Matrix::zeros((h_dim, e_dim)))
        };

        // Build adjacency lists from edges
        let mut adjacency_out: HashMap<MemoryId, Vec<MemoryId>> = HashMap::new();
        let mut adjacency_in: HashMap<MemoryId, Vec<MemoryId>> = HashMap::new();
        let mut edge_map: HashMap<EdgeId, GraphEdge> = HashMap::new();

        for edge in self.edges {
            adjacency_out
                .entry(edge.id.source)
                .or_default()
                .push(edge.id.target);
            adjacency_in
                .entry(edge.id.target)
                .or_default()
                .push(edge.id.source);
            edge_map.insert(edge.id, edge);
        }

        let mut node_map: HashMap<MemoryId, GraphNode> = HashMap::new();
        for node in self.nodes {
            // Ensure adjacency entries exist for all nodes
            adjacency_out.entry(node.memory_id).or_default();
            adjacency_in.entry(node.memory_id).or_default();
            node_map.insert(node.memory_id, node);
        }

        MemoryGraph {
            config: self.config,
            gru: self.gru,
            attention: self.attention,
            input_projection,
            nodes: parking_lot::RwLock::new(node_map),
            edges: parking_lot::RwLock::new(edge_map),
            adjacency_out: parking_lot::RwLock::new(adjacency_out),
            adjacency_in: parking_lot::RwLock::new(adjacency_in),
            update_counter: parking_lot::RwLock::new(0),
        }
    }
}

pub fn save_to_bytes(graph: &MemoryGraph) -> Result<Vec<u8>> {
    let snapshot = GraphSnapshot::from_graph(graph);
    let bytes = rmp_serde::to_vec(&snapshot)?;
    Ok(bytes)
}

pub fn load_from_bytes(bytes: &[u8]) -> Result<MemoryGraph> {
    let snapshot: GraphSnapshot =
        rmp_serde::from_slice(bytes).map_err(|e| GraphError::Serialization(e.to_string()))?;
    Ok(snapshot.into_graph())
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::{Confidence, Embedding};

    fn test_config() -> GraphConfig {
        GraphConfig {
            hidden_dim: 8,
            embedding_dim: 16,
            max_nodes: 100,
            seed: Some(42),
            ..Default::default()
        }
    }

    #[test]
    fn test_roundtrip() {
        let config = test_config();
        let graph = MemoryGraph::new(config);

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        graph
            .add_node(id1, &Embedding::new(vec![1.0; 16]), Confidence::default())
            .unwrap();
        graph
            .add_node(id2, &Embedding::new(vec![0.5; 16]), Confidence::default())
            .unwrap();

        let bytes = save_to_bytes(&graph).unwrap();
        let restored = load_from_bytes(&bytes).unwrap();

        assert_eq!(restored.node_count(), graph.node_count());
        assert_eq!(restored.edge_count(), graph.edge_count());
        assert!(restored.has_node(&id1));
        assert!(restored.has_node(&id2));
    }
}
