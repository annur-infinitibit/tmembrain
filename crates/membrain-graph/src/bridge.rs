use std::sync::Arc;

use membrain_core::{Embedding, Memory, MemoryId};

use crate::edge::EdgeId;
use crate::error::Result;
use crate::graph::MemoryGraph;
use crate::message_passing::graph_query;
use crate::query::GraphQueryResult;

pub trait GraphAugmentedRetrieval: Send + Sync {
    fn graph_query(
        &self,
        query_embedding: &Embedding,
        max_hops: usize,
        top_k: usize,
    ) -> Result<GraphQueryResult>;

    fn on_memory_stored(&self, memory: &Memory) -> Result<()>;
    fn on_memory_deleted(&self, id: &MemoryId) -> Result<()>;
    fn on_memory_retrieved(&self, ids: &[MemoryId]) -> Result<()>;
}

pub struct GraphBridge {
    graph: Arc<MemoryGraph>,
}

impl GraphBridge {
    pub fn new(graph: Arc<MemoryGraph>) -> Self {
        Self { graph }
    }

    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }
}

impl GraphAugmentedRetrieval for GraphBridge {
    fn graph_query(
        &self,
        query_embedding: &Embedding,
        max_hops: usize,
        top_k: usize,
    ) -> Result<GraphQueryResult> {
        Ok(graph_query(
            &self.graph,
            query_embedding.values(),
            Some(max_hops),
            top_k,
        ))
    }

    fn on_memory_stored(&self, memory: &Memory) -> Result<()> {
        if let Some(embedding) = memory.embedding() {
            self.graph
                .add_node(*memory.id(), embedding, *memory.confidence())?;
        }
        Ok(())
    }

    fn on_memory_deleted(&self, id: &MemoryId) -> Result<()> {
        // Ignore NodeNotFound — memory may not have had an embedding
        match self.graph.remove_node(id) {
            Ok(()) => Ok(()),
            Err(crate::error::GraphError::NodeNotFound(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn on_memory_retrieved(&self, ids: &[MemoryId]) -> Result<()> {
        // Reinforce edges between co-retrieved memories
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let edge_id_fwd = EdgeId::new(ids[i], ids[j]);
                let edge_id_rev = EdgeId::new(ids[j], ids[i]);

                let mut edges = self.graph.edges.write();
                if let Some(edge) = edges.get_mut(&edge_id_fwd) {
                    edge.weight = (edge.weight + 0.05).min(1.0);
                    edge.reinforcement_count += 1;
                }
                if let Some(edge) = edges.get_mut(&edge_id_rev) {
                    edge.weight = (edge.weight + 0.05).min(1.0);
                    edge.reinforcement_count += 1;
                }
            }
        }
        Ok(())
    }
}
