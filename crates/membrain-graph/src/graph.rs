use std::collections::HashMap;

use membrain_core::{Confidence, Embedding, MemoryId};
use parking_lot::RwLock;
use rand::SeedableRng;

use crate::attention::MultiHeadAttention;
use crate::config::GraphConfig;
use crate::edge::{EdgeId, GraphEdge, RelationType};
use crate::error::{GraphError, Result};
use crate::gru::GruCell;
use crate::node::GraphNode;
use crate::tensor::{cosine_sim, xavier_init, Matrix, Vector};

pub struct MemoryGraph {
    pub(crate) config: GraphConfig,
    pub(crate) gru: GruCell,
    pub(crate) attention: MultiHeadAttention,
    pub(crate) input_projection: Matrix,
    pub(crate) nodes: RwLock<HashMap<MemoryId, GraphNode>>,
    pub(crate) edges: RwLock<HashMap<EdgeId, GraphEdge>>,
    pub(crate) adjacency_out: RwLock<HashMap<MemoryId, Vec<MemoryId>>>,
    pub(crate) adjacency_in: RwLock<HashMap<MemoryId, Vec<MemoryId>>>,
    pub(crate) update_counter: RwLock<usize>,
}

impl MemoryGraph {
    pub fn new(config: GraphConfig) -> Self {
        let mut rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let h_dim = config.hidden_dim;
        let e_dim = config.embedding_dim;

        let gru = GruCell::new(h_dim, &config.gru, &mut rng);
        let attention =
            MultiHeadAttention::new(h_dim, config.message_passing.num_attention_heads, &mut rng);
        let input_projection = xavier_init(h_dim, e_dim, &mut rng);

        Self {
            config,
            gru,
            attention,
            input_projection,
            nodes: RwLock::new(HashMap::new()),
            edges: RwLock::new(HashMap::new()),
            adjacency_out: RwLock::new(HashMap::new()),
            adjacency_in: RwLock::new(HashMap::new()),
            update_counter: RwLock::new(0),
        }
    }

    /// Add a node to the graph from a memory embedding.
    ///
    /// Projects the embedding to hidden_dim, creates the node, and discovers
    /// edges to similar existing nodes.
    pub fn add_node(
        &self,
        memory_id: MemoryId,
        embedding: &Embedding,
        confidence: Confidence,
    ) -> Result<()> {
        {
            let nodes = self.nodes.read();
            if nodes.len() >= self.config.max_nodes {
                return Err(GraphError::GraphFull {
                    max_nodes: self.config.max_nodes,
                });
            }
            if nodes.contains_key(&memory_id) {
                return Ok(());
            }
        }

        // Project embedding to hidden dim
        let emb_values = embedding.values();
        if emb_values.len() != self.config.embedding_dim {
            return Err(GraphError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: emb_values.len(),
            });
        }

        let emb_vec = Vector::from_vec(emb_values.to_vec());
        let projected = self.input_projection.dot(&emb_vec);
        let hidden_state = projected.clone();

        let node = GraphNode::new(memory_id, hidden_state, projected.clone(), confidence);

        // Discover edges to similar existing nodes
        let edges_to_add = {
            let nodes = self.nodes.read();
            let mut similarities: Vec<(MemoryId, f32)> = Vec::new();

            for (existing_id, existing_node) in nodes.iter() {
                let existing_proj = existing_node.projected_embedding_vec();
                let sim = cosine_sim(&projected, &existing_proj);
                if sim >= self.config.edge.creation_similarity_threshold {
                    similarities.push((*existing_id, sim));
                }
            }

            // Sort by similarity descending, take top max_edges_per_node
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            similarities.truncate(self.config.edge.max_edges_per_node);
            similarities
        };

        // Insert node
        self.nodes.write().insert(memory_id, node);
        self.adjacency_out.write().entry(memory_id).or_default();
        self.adjacency_in.write().entry(memory_id).or_default();

        // Create bidirectional edges
        for (target_id, sim) in edges_to_add {
            let weight = sim * 0.5;
            let relation = self.infer_relation(memory_id, target_id);

            self.add_edge(memory_id, target_id, weight, relation);
            self.add_edge(target_id, memory_id, weight, relation);
        }

        Ok(())
    }

    fn add_edge(&self, source: MemoryId, target: MemoryId, weight: f32, relation: RelationType) {
        let edge_id = EdgeId::new(source, target);
        let edge = GraphEdge::new(source, target, weight, relation);

        self.edges.write().insert(edge_id, edge);
        self.adjacency_out
            .write()
            .entry(source)
            .or_default()
            .push(target);
        self.adjacency_in
            .write()
            .entry(target)
            .or_default()
            .push(source);
    }

    fn infer_relation(&self, new_id: MemoryId, existing_id: MemoryId) -> RelationType {
        // Use UUID v7 timestamps to determine temporal ordering
        let new_ts = new_id.timestamp_millis().unwrap_or(0);
        let existing_ts = existing_id.timestamp_millis().unwrap_or(0);

        if new_ts.abs_diff(existing_ts) < 1000 {
            // Within 1 second — likely semantic relation
            RelationType::Semantic
        } else {
            RelationType::Temporal
        }
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&self, memory_id: &MemoryId) -> Result<()> {
        // Remove outgoing edges
        if let Some(targets) = self.adjacency_out.write().remove(memory_id) {
            for target in &targets {
                let edge_id = EdgeId::new(*memory_id, *target);
                self.edges.write().remove(&edge_id);
                // Remove from target's incoming adjacency
                if let Some(incoming) = self.adjacency_in.write().get_mut(target) {
                    incoming.retain(|id| id != memory_id);
                }
            }
        }

        // Remove incoming edges
        if let Some(sources) = self.adjacency_in.write().remove(memory_id) {
            for source in &sources {
                let edge_id = EdgeId::new(*source, *memory_id);
                self.edges.write().remove(&edge_id);
                // Remove from source's outgoing adjacency
                if let Some(outgoing) = self.adjacency_out.write().get_mut(source) {
                    outgoing.retain(|id| id != memory_id);
                }
            }
        }

        self.nodes
            .write()
            .remove(memory_id)
            .ok_or(GraphError::NodeNotFound(*memory_id))?;

        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.read().len()
    }

    pub fn has_node(&self, id: &MemoryId) -> bool {
        self.nodes.read().contains_key(id)
    }

    pub fn neighbors_out(&self, id: &MemoryId) -> Vec<MemoryId> {
        self.adjacency_out
            .read()
            .get(id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn neighbors_in(&self, id: &MemoryId) -> Vec<MemoryId> {
        self.adjacency_in
            .read()
            .get(id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn increment_update_counter(&self) -> usize {
        let mut counter = self.update_counter.write();
        *counter += 1;
        *counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;

    fn test_config() -> GraphConfig {
        GraphConfig {
            hidden_dim: 8,
            embedding_dim: 16,
            max_nodes: 100,
            seed: Some(42),
            ..Default::default()
        }
    }

    fn make_embedding(dim: usize, val: f32) -> Embedding {
        Embedding::new(vec![val; dim])
    }

    #[test]
    fn test_add_and_count() {
        let graph = MemoryGraph::new(test_config());
        let id = MemoryId::new();
        graph
            .add_node(id, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();
        assert_eq!(graph.node_count(), 1);
        assert!(graph.has_node(&id));
    }

    #[test]
    fn test_remove_node() {
        let graph = MemoryGraph::new(test_config());
        let id = MemoryId::new();
        graph
            .add_node(id, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();
        graph.remove_node(&id).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert!(!graph.has_node(&id));
    }

    #[test]
    fn test_edge_discovery() {
        let mut config = test_config();
        config.edge.creation_similarity_threshold = 0.0; // Any similarity creates edges

        let graph = MemoryGraph::new(config);
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        graph
            .add_node(id1, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();
        graph
            .add_node(id2, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();

        // Should have bidirectional edges
        assert!(graph.edge_count() >= 2);
    }

    #[test]
    fn test_remove_cleans_edges() {
        let mut config = test_config();
        config.edge.creation_similarity_threshold = 0.0;

        let graph = MemoryGraph::new(config);
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        graph
            .add_node(id1, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();
        graph
            .add_node(id2, &make_embedding(16, 1.0), Confidence::default())
            .unwrap();

        let edges_before = graph.edge_count();
        assert!(edges_before > 0);

        graph.remove_node(&id2).unwrap();
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let graph = MemoryGraph::new(test_config());
        let result = graph.add_node(
            MemoryId::new(),
            &make_embedding(32, 1.0), // Wrong dim
            Confidence::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_full() {
        let mut config = test_config();
        config.max_nodes = 2;
        let graph = MemoryGraph::new(config);

        graph
            .add_node(
                MemoryId::new(),
                &make_embedding(16, 1.0),
                Confidence::default(),
            )
            .unwrap();
        graph
            .add_node(
                MemoryId::new(),
                &make_embedding(16, 0.5),
                Confidence::default(),
            )
            .unwrap();

        let result = graph.add_node(
            MemoryId::new(),
            &make_embedding(16, 0.3),
            Confidence::default(),
        );
        assert!(matches!(result, Err(GraphError::GraphFull { .. })));
    }
}
