use chrono::{DateTime, Utc};
use membrain_core::{Confidence, MemoryId};
use serde::{Deserialize, Serialize};

use crate::tensor::Vector;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub memory_id: MemoryId,
    pub hidden_state: Vec<f32>,
    pub projected_embedding: Vec<f32>,
    pub confidence: Confidence,
    pub created_at: DateTime<Utc>,
    pub last_updated_at: DateTime<Utc>,
    pub activation_count: u64,
}

impl GraphNode {
    pub fn new(
        memory_id: MemoryId,
        hidden_state: Vector,
        projected_embedding: Vector,
        confidence: Confidence,
    ) -> Self {
        let now = Utc::now();
        Self {
            memory_id,
            hidden_state: hidden_state.to_vec(),
            projected_embedding: projected_embedding.to_vec(),
            confidence,
            created_at: now,
            last_updated_at: now,
            activation_count: 0,
        }
    }

    pub fn hidden_state_vec(&self) -> Vector {
        Vector::from_vec(self.hidden_state.clone())
    }

    pub fn projected_embedding_vec(&self) -> Vector {
        Vector::from_vec(self.projected_embedding.clone())
    }

    pub fn set_hidden_state(&mut self, h: &Vector) {
        self.hidden_state = h.to_vec();
        self.last_updated_at = Utc::now();
    }

    pub fn record_activation(&mut self) {
        self.activation_count += 1;
        self.last_updated_at = Utc::now();
    }
}
