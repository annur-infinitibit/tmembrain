//! Vector embedding wrapper with dimension validation

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A vector embedding with dimension validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The vector values
    values: Vec<f32>,
    /// Expected dimension (for validation)
    dimension: usize,
}

impl Embedding {
    /// Common embedding dimensions
    pub const OPENAI_ADA_002: usize = 1536;
    pub const OPENAI_SMALL: usize = 1536;
    pub const OPENAI_LARGE: usize = 3072;
    pub const COHERE_MULTILINGUAL: usize = 768;
    pub const BGE_BASE: usize = 768;
    pub const BGE_LARGE: usize = 1024;

    /// Create a new embedding, inferring dimension from values
    pub fn new(values: Vec<f32>) -> Self {
        let dimension = values.len();
        Self { values, dimension }
    }

    /// Create a new embedding with explicit dimension validation
    pub fn with_dimension(values: Vec<f32>, expected_dimension: usize) -> Result<Self> {
        if values.len() != expected_dimension {
            return Err(Error::EmbeddingDimensionMismatch {
                expected: expected_dimension,
                actual: values.len(),
            });
        }
        Ok(Self {
            values,
            dimension: expected_dimension,
        })
    }

    /// Get the dimension of this embedding
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the values as a slice
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Convert to owned vector
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f32> {
        if self.dimension != other.dimension {
            return Err(Error::EmbeddingDimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot / (norm_a * norm_b))
    }

    /// Compute Euclidean distance with another embedding
    pub fn euclidean_distance(&self, other: &Embedding) -> Result<f32> {
        if self.dimension != other.dimension {
            return Err(Error::EmbeddingDimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let sum_sq: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        Ok(sum_sq.sqrt())
    }

    /// Compute dot product with another embedding
    pub fn dot_product(&self, other: &Embedding) -> Result<f32> {
        if self.dimension != other.dimension {
            return Err(Error::EmbeddingDimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        Ok(self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    /// Normalize the embedding to unit length
    pub fn normalize(&self) -> Self {
        let norm: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return self.clone();
        }

        Self {
            values: self.values.iter().map(|x| x / norm).collect(),
            dimension: self.dimension,
        }
    }

    /// Compute the L2 norm (magnitude) of the embedding
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Convert to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(4) {
            return Err(Error::InvalidEmbeddingBytes);
        }

        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self::new(values))
    }

    /// Create a zero embedding of the given dimension
    pub fn zeros(dimension: usize) -> Self {
        Self {
            values: vec![0.0; dimension],
            dimension,
        }
    }

    /// Average multiple embeddings
    pub fn average(embeddings: &[Embedding]) -> Result<Self> {
        if embeddings.is_empty() {
            return Err(Error::EmptyEmbeddingList);
        }

        let dimension = embeddings[0].dimension;
        for emb in embeddings.iter().skip(1) {
            if emb.dimension != dimension {
                return Err(Error::EmbeddingDimensionMismatch {
                    expected: dimension,
                    actual: emb.dimension,
                });
            }
        }

        let count = embeddings.len() as f32;
        let mut avg = vec![0.0f32; dimension];

        for emb in embeddings {
            for (i, v) in emb.values.iter().enumerate() {
                avg[i] += v / count;
            }
        }

        Ok(Self {
            values: avg,
            dimension,
        })
    }
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        if self.dimension != other.dimension {
            return false;
        }
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(a, b)| (a - b).abs() < f32::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_creation() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dimension(), 3);
        assert_eq!(emb.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn embedding_dimension_validation() {
        let result = Embedding::with_dimension(vec![1.0, 2.0], 3);
        assert!(result.is_err());

        let result = Embedding::with_dimension(vec![1.0, 2.0, 3.0], 3);
        assert!(result.is_ok());
    }

    #[test]
    fn cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);
        let c = Embedding::new(vec![1.0, 0.0]);

        // Orthogonal vectors
        let sim_ab = a.cosine_similarity(&b).unwrap();
        assert!((sim_ab - 0.0).abs() < 0.001);

        // Identical vectors
        let sim_ac = a.cosine_similarity(&c).unwrap();
        assert!((sim_ac - 1.0).abs() < 0.001);
    }

    #[test]
    fn euclidean_distance() {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![3.0, 4.0]);

        let dist = a.euclidean_distance(&b).unwrap();
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn normalize() {
        let emb = Embedding::new(vec![3.0, 4.0]);
        let norm = emb.normalize();

        assert!((norm.norm() - 1.0).abs() < 0.001);
        assert!((norm.values()[0] - 0.6).abs() < 0.001);
        assert!((norm.values()[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn bytes_roundtrip() {
        let emb = Embedding::new(vec![1.5, -2.5, 3.25]);
        let bytes = emb.to_bytes();
        let restored = Embedding::from_bytes(&bytes).unwrap();

        assert_eq!(emb.dimension(), restored.dimension());
        for (a, b) in emb.values().iter().zip(restored.values().iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn average_embeddings() {
        let embs = vec![
            Embedding::new(vec![1.0, 2.0]),
            Embedding::new(vec![3.0, 4.0]),
        ];

        let avg = Embedding::average(&embs).unwrap();
        assert_eq!(avg.values(), &[2.0, 3.0]);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let a = Embedding::new(vec![1.0, 2.0]);
        let b = Embedding::new(vec![1.0, 2.0, 3.0]);

        assert!(a.cosine_similarity(&b).is_err());
        assert!(a.euclidean_distance(&b).is_err());
        assert!(a.dot_product(&b).is_err());
    }
}
