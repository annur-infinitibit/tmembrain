use chrono::{DateTime, Utc};
use membrain_core::MemoryId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId {
    pub source: MemoryId,
    pub target: MemoryId,
}

impl EdgeId {
    pub fn new(source: MemoryId, target: MemoryId) -> Self {
        Self { source, target }
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.source, self.target)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Temporal,
    Causal,
    Semantic,
    Derived,
    Contradicts,
    Supports,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub weight: f32,
    pub relation: RelationType,
    #[serde(skip)]
    pub last_attention_score: f32,
    pub created_at: DateTime<Utc>,
    pub last_updated_at: DateTime<Utc>,
    pub reinforcement_count: u64,
}

impl GraphEdge {
    pub fn new(source: MemoryId, target: MemoryId, weight: f32, relation: RelationType) -> Self {
        let now = Utc::now();
        Self {
            id: EdgeId::new(source, target),
            weight: weight.clamp(0.0, 1.0),
            relation,
            last_attention_score: 0.0,
            created_at: now,
            last_updated_at: now,
            reinforcement_count: 0,
        }
    }

    /// Evolve edge weight during traversal.
    ///
    /// 1. Temporal decay: w *= exp(-ln2 * elapsed / half_life)
    /// 2. Attention reinforcement: w += attn * sqrt(src_conf * tgt_conf) * lr * (1-w)
    /// 3. Clamp to [0, 1]
    /// 4. L1 penalty: w = max(0, w - λ)
    pub fn evolve(
        &mut self,
        attention_score: f32,
        src_confidence: f32,
        tgt_confidence: f32,
        half_life_secs: u64,
        learning_rate: f32,
        l1_lambda: f32,
    ) {
        let now = Utc::now();
        let elapsed_secs = (now - self.last_updated_at).num_seconds().max(0) as f64;

        // Temporal decay
        if half_life_secs > 0 {
            let decay = (-(std::f64::consts::LN_2) * elapsed_secs / half_life_secs as f64).exp();
            self.weight *= decay as f32;
        }

        // Attention reinforcement
        let conf_factor = (src_confidence * tgt_confidence).sqrt();
        self.weight += attention_score * conf_factor * learning_rate * (1.0 - self.weight);

        // Clamp
        self.weight = self.weight.clamp(0.0, 1.0);

        // L1 penalty
        self.weight = (self.weight - l1_lambda).max(0.0);

        self.last_attention_score = attention_score;
        self.last_updated_at = now;
        self.reinforcement_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::MemoryId;

    #[test]
    fn test_weight_clamping() {
        let edge = GraphEdge::new(
            MemoryId::new(),
            MemoryId::new(),
            1.5,
            RelationType::Semantic,
        );
        assert!(edge.weight <= 1.0);

        let edge2 = GraphEdge::new(
            MemoryId::new(),
            MemoryId::new(),
            -0.5,
            RelationType::Temporal,
        );
        assert!(edge2.weight >= 0.0);
    }

    #[test]
    fn test_evolve_reinforcement_increases_weight() {
        let mut edge = GraphEdge::new(
            MemoryId::new(),
            MemoryId::new(),
            0.3,
            RelationType::Semantic,
        );
        let initial = edge.weight;

        // High attention, high confidence, no decay (very large half_life), no L1
        edge.evolve(0.8, 0.9, 0.9, u64::MAX, 0.5, 0.0);
        assert!(edge.weight > initial);
    }

    #[test]
    fn test_weight_invariant_after_evolve() {
        let mut edge = GraphEdge::new(MemoryId::new(), MemoryId::new(), 0.5, RelationType::Causal);
        for _ in 0..50 {
            edge.evolve(0.5, 0.5, 0.5, 3600, 0.1, 0.001);
            assert!(edge.weight >= 0.0 && edge.weight <= 1.0);
        }
    }
}
