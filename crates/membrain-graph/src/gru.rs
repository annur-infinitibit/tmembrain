use ndarray::concatenate;
use ndarray::Axis;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::config::GruConfig;
use crate::tensor::{sigmoid, tanh_vec, xavier_init, Matrix, Vector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GruCell {
    h_dim: usize,
    learning_rate: f32,
    // Reset gate weights
    w_r: Matrix,
    b_r: Vector,
    // Update gate weights
    w_z: Matrix,
    b_z: Vector,
    // Candidate state weights
    w_n: Matrix,
    b_n: Vector,
}

impl GruCell {
    pub fn new(h_dim: usize, config: &GruConfig, rng: &mut impl Rng) -> Self {
        let input_dim = h_dim;
        let concat_dim = input_dim + h_dim;

        Self {
            h_dim,
            learning_rate: config.learning_rate,
            w_r: xavier_init(h_dim, concat_dim, rng),
            b_r: Vector::from_elem(h_dim, config.bias_init_scale),
            w_z: xavier_init(h_dim, concat_dim, rng),
            b_z: Vector::from_elem(h_dim, config.bias_init_scale),
            w_n: xavier_init(h_dim, concat_dim, rng),
            b_n: Vector::from_elem(h_dim, config.bias_init_scale),
        }
    }

    pub fn h_dim(&self) -> usize {
        self.h_dim
    }

    /// GRU forward pass.
    ///
    /// - `h_prev`: previous hidden state (h_dim,)
    /// - `input`: projected embedding input (h_dim,)
    /// - `neighbor_msg`: aggregated neighbor message (h_dim,)
    ///
    /// Returns new hidden state (h_dim,).
    pub fn forward(&self, h_prev: &Vector, input: &Vector, neighbor_msg: &Vector) -> Vector {
        // x = input + learning_rate * neighbor_msg
        let x = input + &(self.learning_rate * neighbor_msg);

        // concat = [x ; h_prev]
        let concat = concatenate![Axis(0), x, h_prev.clone()];

        // Reset gate: r = σ(W_r · concat + b_r)
        let r = sigmoid(&(self.w_r.dot(&concat) + &self.b_r));

        // Update gate: z = σ(W_z · concat + b_z)
        let z = sigmoid(&(self.w_z.dot(&concat) + &self.b_z));

        // Candidate: n = tanh(W_n · [x ; r⊙h_prev] + b_n)
        let r_h = &r * h_prev;
        let concat_n = concatenate![Axis(0), x, r_h];
        let n = tanh_vec(&(self.w_n.dot(&concat_n) + &self.b_n));

        // h_new = (1-z)⊙n + z⊙h_prev
        let ones = Vector::ones(self.h_dim);
        (&ones - &z) * &n + &z * h_prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GruConfig;

    #[test]
    fn test_output_dimensions() {
        let mut rng = rand::thread_rng();
        let gru = GruCell::new(8, &GruConfig::default(), &mut rng);

        let h = Vector::zeros(8);
        let input = Vector::zeros(8);
        let msg = Vector::zeros(8);

        let h_new = gru.forward(&h, &input, &msg);
        assert_eq!(h_new.len(), 8);
    }

    #[test]
    fn test_zero_input_stability() {
        let mut rng = rand::thread_rng();
        let gru = GruCell::new(8, &GruConfig::default(), &mut rng);

        let mut h = Vector::zeros(8);
        let input = Vector::zeros(8);
        let msg = Vector::zeros(8);

        // Run 100 steps with zero input; hidden state should remain bounded
        for _ in 0..100 {
            h = gru.forward(&h, &input, &msg);
            let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm < 100.0, "hidden state diverged: norm={}", norm);
        }
    }

    #[test]
    fn test_update_gate_passthrough() {
        // When update gate is saturated to 1, output should be close to h_prev
        let config = GruConfig {
            learning_rate: 0.0,
            bias_init_scale: 0.01,
        };
        let mut rng = rand::thread_rng();
        let mut gru = GruCell::new(4, &config, &mut rng);

        // Force update gate bias to large positive value → z ≈ 1
        gru.b_z = Vector::from_elem(4, 10.0);
        gru.w_z = Matrix::zeros((4, 8));

        let h_prev = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let input = Vector::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let msg = Vector::zeros(4);

        let h_new = gru.forward(&h_prev, &input, &msg);
        // z ≈ 1 → h_new ≈ h_prev
        for (a, b) in h_new.iter().zip(h_prev.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "Expected h_new ≈ h_prev, got {} vs {}",
                a,
                b
            );
        }
    }
}
