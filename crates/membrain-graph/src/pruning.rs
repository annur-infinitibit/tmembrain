use chrono::Utc;
use membrain_core::MemoryId;
use serde::{Deserialize, Serialize};

use crate::edge::EdgeId;
use crate::graph::MemoryGraph;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningResult {
    pub edges_removed: usize,
    pub nodes_removed: usize,
    pub edges_remaining: usize,
    pub nodes_remaining: usize,
}

/// Run pruning on the graph, enforcing sparsity constraints.
pub fn prune(graph: &MemoryGraph) -> PruningResult {
    let config = &graph.config;
    let mut edges_removed = 0;
    let mut nodes_removed = 0;

    // 1. Remove edges with effective weight below min_weight (after temporal decay)
    {
        let now = Utc::now();
        let half_life = config.edge.weight_half_life_secs;
        let min_weight = config.edge.min_weight;

        let to_remove: Vec<EdgeId> = {
            let edges = graph.edges.read();
            edges
                .iter()
                .filter(|(_, edge)| {
                    let elapsed_secs = (now - edge.last_updated_at).num_seconds().max(0) as f64;
                    let effective_weight = if half_life > 0 {
                        let decay =
                            (-(std::f64::consts::LN_2) * elapsed_secs / half_life as f64).exp();
                        edge.weight * decay as f32
                    } else {
                        edge.weight
                    };
                    effective_weight < min_weight
                })
                .map(|(id, _)| *id)
                .collect()
        };

        for edge_id in &to_remove {
            remove_edge(graph, edge_id);
            edges_removed += 1;
        }
    }

    // 2. Fan-out limit: per node, keep only top max_edges_per_node by weight
    {
        let max_per_node = config.edge.max_edges_per_node;
        let node_ids: Vec<MemoryId> = graph.nodes.read().keys().copied().collect();

        for node_id in &node_ids {
            let outgoing = graph.neighbors_out(node_id);
            if outgoing.len() <= max_per_node {
                continue;
            }

            // Collect edges with weights
            let mut edge_weights: Vec<(EdgeId, f32)> = {
                let edges = graph.edges.read();
                outgoing
                    .iter()
                    .filter_map(|target| {
                        let eid = EdgeId::new(*node_id, *target);
                        edges.get(&eid).map(|e| (eid, e.weight))
                    })
                    .collect()
            };

            // Sort descending by weight
            edge_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove excess edges (those beyond max_per_node)
            for (eid, _) in edge_weights.iter().skip(max_per_node) {
                remove_edge(graph, eid);
                edges_removed += 1;
            }
        }
    }

    // 3. Global budget: if total edges > max_total_edges, remove weakest
    {
        let max_total = config.edge.max_total_edges;
        let current_count = graph.edges.read().len();

        if current_count > max_total {
            let excess = current_count - max_total;
            let mut all_edges: Vec<(EdgeId, f32)> = {
                let edges = graph.edges.read();
                edges.iter().map(|(id, e)| (*id, e.weight)).collect()
            };

            // Sort ascending by weight (weakest first)
            all_edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for (eid, _) in all_edges.iter().take(excess) {
                remove_edge(graph, eid);
                edges_removed += 1;
            }
        }
    }

    // 4. Optionally remove isolated nodes (0 edges, 0 activations)
    if config.pruning.prune_isolated_nodes {
        let node_ids: Vec<MemoryId> = graph.nodes.read().keys().copied().collect();

        for node_id in &node_ids {
            let has_edges = {
                let adj_out = graph.adjacency_out.read();
                let adj_in = graph.adjacency_in.read();
                let out_count = adj_out.get(node_id).map(|v| v.len()).unwrap_or(0);
                let in_count = adj_in.get(node_id).map(|v| v.len()).unwrap_or(0);
                out_count > 0 || in_count > 0
            };

            if !has_edges {
                let zero_activations = {
                    let nodes = graph.nodes.read();
                    nodes
                        .get(node_id)
                        .map(|n| n.activation_count == 0)
                        .unwrap_or(false)
                };

                if zero_activations {
                    let _ = graph.remove_node(node_id);
                    nodes_removed += 1;
                }
            }
        }
    }

    // 5. State normalization: clamp node hidden state L2 norm to max 10.0
    {
        let mut nodes = graph.nodes.write();
        for node in nodes.values_mut() {
            let h = node.hidden_state_vec();
            let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 10.0 {
                let clamped = &h * (10.0 / norm);
                node.set_hidden_state(&clamped);
            }
        }
    }

    PruningResult {
        edges_removed,
        nodes_removed,
        edges_remaining: graph.edge_count(),
        nodes_remaining: graph.node_count(),
    }
}

fn remove_edge(graph: &MemoryGraph, edge_id: &EdgeId) {
    graph.edges.write().remove(edge_id);
    if let Some(outgoing) = graph.adjacency_out.write().get_mut(&edge_id.source) {
        outgoing.retain(|id| *id != edge_id.target);
    }
    if let Some(incoming) = graph.adjacency_in.write().get_mut(&edge_id.target) {
        incoming.retain(|id| *id != edge_id.source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;
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
    fn test_prune_no_crash_on_empty() {
        let graph = MemoryGraph::new(test_config());
        let result = prune(&graph);
        assert_eq!(result.edges_removed, 0);
        assert_eq!(result.nodes_removed, 0);
    }

    #[test]
    fn test_prune_removes_isolated_nodes() {
        let mut config = test_config();
        config.pruning.prune_isolated_nodes = true;
        config.edge.creation_similarity_threshold = 1.0; // No auto-edges

        let graph = MemoryGraph::new(config);
        let id = MemoryId::new();
        graph
            .add_node(id, &Embedding::new(vec![1.0; 16]), Confidence::default())
            .unwrap();

        assert_eq!(graph.node_count(), 1);
        let result = prune(&graph);
        assert_eq!(result.nodes_removed, 1);
        assert_eq!(graph.node_count(), 0);
    }
}
