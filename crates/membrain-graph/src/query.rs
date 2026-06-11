use membrain_core::MemoryId;
use serde::{Deserialize, Serialize};

use crate::tensor::Vector;

#[derive(Debug, Clone)]
pub struct GraphQueryResult {
    pub nodes: Vec<ScoredNode>,
    pub traversed_edges: Vec<TraversalStep>,
    pub hops_performed: usize,
    pub nodes_visited: usize,
}

#[derive(Debug, Clone)]
pub struct ScoredNode {
    pub memory_id: MemoryId,
    pub score: f32,
    pub hidden_state: Vector,
    pub hop_distance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalStep {
    pub from: MemoryId,
    pub to: MemoryId,
    pub edge_weight: f32,
    pub attention_score: f32,
    pub hop: usize,
}
