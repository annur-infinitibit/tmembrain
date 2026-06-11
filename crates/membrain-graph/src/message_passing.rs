use std::collections::{HashMap, HashSet};

use membrain_core::MemoryId;

use crate::edge::EdgeId;
use crate::graph::MemoryGraph;
use crate::pruning::prune;
use crate::query::{GraphQueryResult, ScoredNode, TraversalStep};
use crate::tensor::{cosine_sim, Vector};

/// Multi-hop query engine with GRU state updates and edge evolution.
pub fn graph_query(
    graph: &MemoryGraph,
    query_embedding: &[f32],
    max_hops: Option<usize>,
    top_k: usize,
) -> GraphQueryResult {
    let max_hops = max_hops.unwrap_or(graph.config.message_passing.max_hops);
    let damping = graph.config.message_passing.damping;

    // 1. Project query
    let q_vec = Vector::from_vec(query_embedding.to_vec());
    let q_projected = graph.input_projection.dot(&q_vec);

    // 2. Seed selection: cosine similarity against all node projected embeddings
    let seed_nodes: Vec<(MemoryId, f32)> = {
        let nodes = graph.nodes.read();
        let mut scores: Vec<(MemoryId, f32)> = nodes
            .iter()
            .map(|(id, node)| {
                let proj = node.projected_embedding_vec();
                let sim = cosine_sim(&q_projected, &proj);
                (*id, sim)
            })
            .filter(|(_, sim)| *sim > 0.3)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k * 2);
        scores
    };

    let mut best_scores: HashMap<MemoryId, (f32, usize)> = HashMap::new(); // (score, hop_distance)
    let mut traversal_steps: Vec<TraversalStep> = Vec::new();
    let mut visited: HashSet<MemoryId> = HashSet::new();

    // Seeds are at hop 0
    for (id, score) in &seed_nodes {
        best_scores.insert(*id, (*score, 0));
        visited.insert(*id);
    }

    // 3. Multi-hop traversal
    let mut frontier: Vec<(MemoryId, f32)> = seed_nodes.clone();

    for hop in 1..=max_hops {
        let mut next_frontier: Vec<(MemoryId, f32)> = Vec::new();

        for (src_id, src_score) in &frontier {
            let neighbors = graph.neighbors_out(src_id);

            for target_id in neighbors {
                let edge_id = EdgeId::new(*src_id, target_id);

                // Get edge weight
                let edge_weight = {
                    let edges = graph.edges.read();
                    match edges.get(&edge_id) {
                        Some(e) => e.weight,
                        None => continue,
                    }
                };

                // Compute attention score
                let attn_score = {
                    let nodes = graph.nodes.read();
                    match nodes.get(&target_id) {
                        Some(target_node) => {
                            let target_h = target_node.hidden_state_vec();
                            graph.attention.compute_attention(&q_projected, &target_h)
                        }
                        None => continue,
                    }
                };

                let propagated_score =
                    src_score * edge_weight * attn_score * damping.powi(hop as i32);

                traversal_steps.push(TraversalStep {
                    from: *src_id,
                    to: target_id,
                    edge_weight,
                    attention_score: attn_score,
                    hop,
                });

                // Evolve edge weight
                {
                    let nodes = graph.nodes.read();
                    let src_conf = nodes
                        .get(src_id)
                        .map(|n| n.confidence.value() as f32)
                        .unwrap_or(0.5);
                    let tgt_conf = nodes
                        .get(&target_id)
                        .map(|n| n.confidence.value() as f32)
                        .unwrap_or(0.5);
                    drop(nodes);

                    let mut edges = graph.edges.write();
                    if let Some(edge) = edges.get_mut(&edge_id) {
                        edge.evolve(
                            attn_score,
                            src_conf,
                            tgt_conf,
                            graph.config.edge.weight_half_life_secs,
                            graph.config.gru.learning_rate,
                            graph.config.pruning.l1_lambda,
                        );
                    }
                }

                // Update best score
                let entry = best_scores.entry(target_id).or_insert((0.0, hop));
                if propagated_score > entry.0 {
                    *entry = (propagated_score, hop);
                }

                if !visited.contains(&target_id) {
                    visited.insert(target_id);
                    next_frontier.push((target_id, propagated_score));
                }
            }
        }

        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    // 4. GRU state update for visited nodes
    {
        let visited_ids: Vec<MemoryId> = visited.iter().copied().collect();
        for node_id in &visited_ids {
            // Aggregate incoming neighbor messages
            let incoming = graph.neighbors_in(node_id);
            let neighbor_data: Vec<(Vector, f32)> = {
                let nodes = graph.nodes.read();
                let edges = graph.edges.read();
                incoming
                    .iter()
                    .filter_map(|src_id| {
                        let edge_id = EdgeId::new(*src_id, *node_id);
                        let weight = edges.get(&edge_id).map(|e| e.weight)?;
                        let h = nodes.get(src_id)?.hidden_state_vec();
                        Some((h, weight))
                    })
                    .collect()
            };

            let (input, h_prev) = {
                let nodes = graph.nodes.read();
                match nodes.get(node_id) {
                    Some(node) => (node.projected_embedding_vec(), node.hidden_state_vec()),
                    None => continue,
                }
            };

            let neighbor_msg = graph.attention.aggregate_messages(&h_prev, &neighbor_data);
            let h_new = graph.gru.forward(&h_prev, &input, &neighbor_msg);

            // Clamp hidden state norm to prevent divergence
            let norm: f32 = h_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            let h_clamped = if norm > 10.0 {
                &h_new * (10.0 / norm)
            } else {
                h_new
            };

            let mut nodes = graph.nodes.write();
            if let Some(node) = nodes.get_mut(node_id) {
                node.set_hidden_state(&h_clamped);
                node.record_activation();
            }
        }
    }

    // 5. Build result: top-k by score
    let mut scored: Vec<(MemoryId, f32, usize)> = best_scores
        .into_iter()
        .map(|(id, (score, hop))| (id, score, hop))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    let nodes_lock = graph.nodes.read();
    let result_nodes: Vec<ScoredNode> = scored
        .into_iter()
        .filter_map(|(id, score, hop)| {
            let node = nodes_lock.get(&id)?;
            Some(ScoredNode {
                memory_id: id,
                score,
                hidden_state: node.hidden_state_vec(),
                hop_distance: hop,
            })
        })
        .collect();
    drop(nodes_lock);

    let nodes_visited = visited.len();

    // 6. Trigger pruning if needed
    let counter = graph.increment_update_counter();
    if counter.is_multiple_of(graph.config.pruning.pruning_interval) {
        let _ = prune(graph);
    }

    GraphQueryResult {
        nodes: result_nodes,
        traversed_edges: traversal_steps,
        hops_performed: max_hops,
        nodes_visited,
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
    fn test_query_empty_graph() {
        let graph = MemoryGraph::new(test_config());
        let query = vec![1.0_f32; 16];
        let result = graph_query(&graph, &query, Some(2), 5);
        assert!(result.nodes.is_empty());
    }

    #[test]
    fn test_query_single_node() {
        let graph = MemoryGraph::new(test_config());
        let id = MemoryId::new();
        graph
            .add_node(id, &Embedding::new(vec![1.0; 16]), Confidence::default())
            .unwrap();

        let result = graph_query(&graph, &[1.0; 16], Some(1), 5);
        // Should find the seed node
        assert!(!result.nodes.is_empty());
        assert_eq!(result.nodes[0].memory_id, id);
    }

    #[test]
    fn test_multi_hop_discovery() {
        let mut config = test_config();
        config.edge.creation_similarity_threshold = 0.0; // Force edge creation

        let graph = MemoryGraph::new(config);

        // Create a chain: A -> B -> C
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        graph
            .add_node(a, &Embedding::new(vec![1.0; 16]), Confidence::default())
            .unwrap();
        graph
            .add_node(b, &Embedding::new(vec![0.9; 16]), Confidence::default())
            .unwrap();
        graph
            .add_node(c, &Embedding::new(vec![0.8; 16]), Confidence::default())
            .unwrap();

        // Query near A with enough hops to reach C
        let result = graph_query(&graph, &[1.0; 16], Some(3), 10);
        let found_ids: Vec<MemoryId> = result.nodes.iter().map(|n| n.memory_id).collect();
        assert!(found_ids.contains(&a));
    }
}
