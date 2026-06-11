use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::Rng;

pub type Vector = Array1<f32>;
pub type Matrix = Array2<f32>;

pub fn sigmoid(v: &Vector) -> Vector {
    v.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

pub fn tanh_vec(v: &Vector) -> Vector {
    v.mapv(|x| x.tanh())
}

pub fn softmax(v: &Vector) -> Vector {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = v.mapv(|x| (x - max).exp());
    let sum = exps.sum();
    if sum == 0.0 {
        Vector::from_elem(v.len(), 1.0 / v.len() as f32)
    } else {
        exps / sum
    }
}

pub fn xavier_init(rows: usize, cols: usize, rng: &mut impl Rng) -> Matrix {
    let std_dev = (2.0 / (rows + cols) as f64).sqrt() as f32;
    let normal = rand::distributions::Uniform::new(-std_dev, std_dev);
    Matrix::from_shape_fn((rows, cols), |_| normal.sample(rng))
}

pub fn cosine_sim(a: &Vector, b: &Vector) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let v = array![0.0_f32];
        let result = sigmoid(&v);
        assert!((result[0] - 0.5).abs() < 1e-6);

        let v2 = array![100.0_f32];
        let result2 = sigmoid(&v2);
        assert!((result2[0] - 1.0).abs() < 1e-4);

        let v3 = array![-100.0_f32];
        let result3 = sigmoid(&v3);
        assert!(result3[0].abs() < 1e-4);
    }

    #[test]
    fn test_tanh_vec() {
        let v = array![0.0_f32];
        let result = tanh_vec(&v);
        assert!(result[0].abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let v = array![1.0_f32, 2.0, 3.0];
        let result = softmax(&v);
        assert!((result.sum() - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_cosine_sim() {
        let a = array![1.0_f32, 0.0];
        let b = array![0.0_f32, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);

        let c = array![1.0_f32, 0.0];
        assert!((cosine_sim(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_xavier_init_shape() {
        let mut rng = rand::thread_rng();
        let m = xavier_init(3, 4, &mut rng);
        assert_eq!(m.shape(), &[3, 4]);
    }
}
