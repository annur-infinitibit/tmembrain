use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphConfig {
    pub hidden_dim: usize,
    pub embedding_dim: usize,
    pub max_nodes: usize,
    pub seed: Option<u64>,
    pub gru: GruConfig,
    pub edge: EdgeConfig,
    pub pruning: PruningConfig,
    pub message_passing: MessagePassingConfig,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            embedding_dim: 768,
            max_nodes: 100_000,
            seed: None,
            gru: GruConfig::default(),
            edge: EdgeConfig::default(),
            pruning: PruningConfig::default(),
            message_passing: MessagePassingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GruConfig {
    pub learning_rate: f32,
    pub bias_init_scale: f32,
}

impl Default for GruConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            bias_init_scale: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeConfig {
    pub min_weight: f32,
    pub max_edges_per_node: usize,
    pub weight_half_life_secs: u64,
    pub creation_similarity_threshold: f32,
    pub max_total_edges: usize,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.05,
            max_edges_per_node: 32,
            weight_half_life_secs: 604_800, // 7 days
            creation_similarity_threshold: 0.7,
            max_total_edges: 100_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PruningConfig {
    pub pruning_interval: usize,
    pub target_sparsity_ratio: f32,
    pub l1_lambda: f32,
    pub prune_isolated_nodes: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            pruning_interval: 100,
            target_sparsity_ratio: 8.0,
            l1_lambda: 0.001,
            prune_isolated_nodes: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MessagePassingConfig {
    pub max_hops: usize,
    pub num_attention_heads: usize,
    pub damping: f32,
    pub aggregation: AggregationMethod,
}

impl Default for MessagePassingConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            num_attention_heads: 4,
            damping: 0.85,
            aggregation: AggregationMethod::AttentionWeighted,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum AggregationMethod {
    #[default]
    AttentionWeighted,
    Mean,
    Max,
}
