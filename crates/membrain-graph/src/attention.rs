use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tensor::{sigmoid, xavier_init, Matrix, Vector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    w_query: Vec<Matrix>,
    w_key: Vec<Matrix>,
    w_value: Vec<Matrix>,
    w_out: Matrix,
}

impl MultiHeadAttention {
    pub fn new(h_dim: usize, num_heads: usize, rng: &mut impl Rng) -> Self {
        assert!(
            h_dim.is_multiple_of(num_heads),
            "h_dim ({}) must be divisible by num_heads ({})",
            h_dim,
            num_heads
        );
        let head_dim = h_dim / num_heads;

        let mut w_query = Vec::with_capacity(num_heads);
        let mut w_key = Vec::with_capacity(num_heads);
        let mut w_value = Vec::with_capacity(num_heads);

        for _ in 0..num_heads {
            w_query.push(xavier_init(head_dim, h_dim, rng));
            w_key.push(xavier_init(head_dim, h_dim, rng));
            w_value.push(xavier_init(head_dim, h_dim, rng));
        }

        let w_out = xavier_init(h_dim, h_dim, rng);

        Self {
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            w_out,
        }
    }

    /// Compute attention score between a query vector and a target hidden state.
    ///
    /// Returns a scalar in [0, 1] (sigmoid of averaged scaled dot-product across heads).
    pub fn compute_attention(&self, query: &Vector, target_h: &Vector) -> f32 {
        let scale = (self.head_dim as f32).sqrt();
        let mut total_score = 0.0;

        for head in 0..self.num_heads {
            let q = self.w_query[head].dot(query);
            let k = self.w_key[head].dot(target_h);
            let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            total_score += score / scale;
        }

        let avg_score = total_score / self.num_heads as f32;
        // Sigmoid to [0, 1]
        let result = sigmoid(&Vector::from_vec(vec![avg_score]));
        result[0]
    }

    /// Aggregate neighbor messages using multi-head attention.
    ///
    /// `neighbors` is a slice of (hidden_state, edge_weight) pairs.
    /// Returns a weighted-summed and projected message vector.
    pub fn aggregate_messages(&self, query: &Vector, neighbors: &[(Vector, f32)]) -> Vector {
        if neighbors.is_empty() {
            return Vector::zeros(query.len());
        }

        let h_dim = query.len();
        // Compute attention-weighted sum of value projections
        let mut attention_weights = Vec::with_capacity(neighbors.len());
        let mut values_per_head: Vec<Vec<Vector>> = vec![Vec::new(); self.num_heads];

        let scale = (self.head_dim as f32).sqrt();

        for (neighbor_h, edge_weight) in neighbors {
            let mut head_scores = Vec::with_capacity(self.num_heads);

            for (head, head_values) in values_per_head.iter_mut().enumerate() {
                let q = self.w_query[head].dot(query);
                let k = self.w_key[head].dot(neighbor_h);
                let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                head_scores.push(score / scale);

                let v = self.w_value[head].dot(neighbor_h);
                head_values.push(v);
            }

            let avg_score: f32 = head_scores.iter().sum::<f32>() / self.num_heads as f32;
            attention_weights.push(avg_score * edge_weight);
        }

        // Softmax over attention weights
        let max_w = attention_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_weights: Vec<f32> = attention_weights
            .iter()
            .map(|w| (w - max_w).exp())
            .collect();
        let sum_exp: f32 = exp_weights.iter().sum();
        let norm_weights: Vec<f32> = if sum_exp > 0.0 {
            exp_weights.iter().map(|w| w / sum_exp).collect()
        } else {
            vec![1.0 / neighbors.len() as f32; neighbors.len()]
        };

        // Weighted sum of values per head, then concatenate
        let mut concat_values = Vector::zeros(h_dim);
        for (head, head_values) in values_per_head.iter().enumerate() {
            let mut head_out = Vector::zeros(self.head_dim);
            for (i, weight) in norm_weights.iter().enumerate() {
                head_out = &head_out + &(head_values[i].clone() * *weight);
            }
            let offset = head * self.head_dim;
            for (j, val) in head_out.iter().enumerate() {
                concat_values[offset + j] = *val;
            }
        }

        // Output projection
        self.w_out.dot(&concat_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_score_range() {
        let mut rng = rand::thread_rng();
        let attn = MultiHeadAttention::new(8, 2, &mut rng);

        let query = Vector::from_vec(vec![1.0; 8]);
        let target = Vector::from_vec(vec![0.5; 8]);

        let score = attn.compute_attention(&query, &target);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_aggregate_empty() {
        let mut rng = rand::thread_rng();
        let attn = MultiHeadAttention::new(8, 2, &mut rng);
        let query = Vector::from_vec(vec![1.0; 8]);

        let result = attn.aggregate_messages(&query, &[]);
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_aggregate_output_dim() {
        let mut rng = rand::thread_rng();
        let attn = MultiHeadAttention::new(8, 2, &mut rng);
        let query = Vector::from_vec(vec![1.0; 8]);
        let neighbors = vec![
            (Vector::from_vec(vec![0.5; 8]), 0.8),
            (Vector::from_vec(vec![0.3; 8]), 0.6),
        ];

        let result = attn.aggregate_messages(&query, &neighbors);
        assert_eq!(result.len(), 8);
    }
}
