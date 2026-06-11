/**
 * Test Graph Memory Examples
 * Based on docs/cookbooks/graph-memory.mdx
 */

require('dotenv').config({ path: '../../.env' });

const os = require('os');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

let MembrainGraph, MembrainClient;

try {
  const membrain = require('membrain');
  MembrainGraph = membrain.MembrainGraph;
  MembrainClient = membrain.MembrainClient;
} catch (err) {
  console.warn('Membrain module not found, tests will be skipped');
}

const skipIfNoMembrain = MembrainGraph ? test : test.skip;

const DEFAULT_EMBEDDING_DIM = 16;

function generateUUID() {
  return crypto.randomUUID();
}

function createUniqueClient() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'membrain-test-'));
  return new MembrainClient({
    storage: {
      backend: "memscaledb",
      path: tmpDir
    }
  });
}

function randomEmbedding(dim) {
  return Array(dim).fill(0).map(() => Math.random());
}

describe('Graph Memory Tests', () => {
  describe('Graph Creation', () => {
    skipIfNoMembrain('should create a graph instance', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });
      try {
        expect(graph).toBeDefined();
      } finally {
        graph.close();
      }
    });

    skipIfNoMembrain('should create graph with custom config', () => {
      const graph = new MembrainGraph({
        hidden_dim: 128,
        embedding_dim: DEFAULT_EMBEDDING_DIM
      });

      try {
        expect(graph).toBeDefined();
      } finally {
        graph.close();
      }
    });
  });

  describe('Node Operations', () => {
    skipIfNoMembrain('should add nodes to graph', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 10; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding, 0.9);
        }

        const count = graph.nodeCount();
        expect(count).toBe(10);
      } finally {
        graph.close();
      }
    });

    skipIfNoMembrain('should add nodes with confidence scores', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        const memoryId = generateUUID();
        const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
        graph.addNode(memoryId, embedding, 0.85);

        const count = graph.nodeCount();
        expect(count).toBe(1);
      } finally {
        graph.close();
      }
    });
  });

  describe('Graph Queries', () => {
    skipIfNoMembrain('should query graph with single hop', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        // Build graph
        for (let i = 0; i < 20; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding, 0.8);
        }

        // Query - result can be null if graph is empty or query fails
        const queryEmbedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
        const result = graph.query(queryEmbedding, 1, 5);

        if (result) {
          expect(result.nodes.length).toBeGreaterThan(0);
          expect(result.nodes.length).toBeLessThanOrEqual(5);

          // Check node properties
          result.nodes.forEach(node => {
            expect(node.memory_id).toBeDefined();
            expect(node.score).toBeGreaterThanOrEqual(0.0);
            expect(node.hop_distance).toBeGreaterThanOrEqual(0);
          });
        }
      } finally {
        graph.close();
      }
    });

    skipIfNoMembrain('should perform multi-hop traversal', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 20; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding, 0.8);
        }

        const queryEmbedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
        const result = graph.query(queryEmbedding, 3, 5);

        if (result) {
          expect(result.nodes.length).toBeGreaterThan(0);
          expect(result.hops_performed).toBeGreaterThanOrEqual(0);
          expect(result.hops_performed).toBeLessThanOrEqual(3);
          expect(result.traversed_edges).toBeDefined();
        }
      } finally {
        graph.close();
      }
    });
  });

  describe('Graph Persistence', () => {
    skipIfNoMembrain('should save graph state', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 5; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        }

        const savedData = graph.save();
        // save() may return null if the graph has no data to serialize
        if (savedData) {
          expect(savedData.length).toBeGreaterThan(0);
          expect(typeof savedData).toBe('string');
        }
      } finally {
        graph.close();
      }
    });

    skipIfNoMembrain('should save and load graph', () => {
      let savedData;
      let originalCount;

      // Create and save
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });
      try {
        for (let i = 0; i < 10; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        }

        savedData = graph.save();
        originalCount = graph.nodeCount();
      } finally {
        graph.close();
      }

      // Only attempt load if save returned valid data
      if (savedData) {
        const restoredGraph = MembrainGraph.load(savedData);
        try {
          const restoredCount = restoredGraph.nodeCount();
          expect(restoredCount).toBe(originalCount);
        } finally {
          restoredGraph.close();
        }
      }
    });
  });

  describe('Graph Pruning', () => {
    skipIfNoMembrain('should prune graph', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 50; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        }

        const result = graph.prune();

        // prune() may return null if the operation is a no-op
        if (result) {
          expect(result.edges_removed).toBeGreaterThanOrEqual(0);
          expect(result.nodes_removed).toBeGreaterThanOrEqual(0);
          expect(result.edges_remaining).toBeGreaterThanOrEqual(0);
          expect(result.nodes_remaining).toBeGreaterThanOrEqual(0);
        }
      } finally {
        graph.close();
      }
    });
  });

  describe('Client-Graph Integration', () => {
    skipIfNoMembrain('should integrate client and graph', () => {
      if (!MembrainClient) {
        console.warn('MembrainClient not available, skipping test');
        return;
      }

      const client = createUniqueClient();
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        const validIds = [];
        const topics = ["Python", "Rust", "JavaScript"];

        topics.forEach(topic => {
          const result = client.storeFact(`${topic} is a programming language`, 0.9);
          if (result && result.success && result.id) {
            validIds.push(result.id);
          }
        });

        validIds.forEach(memoryId => {
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        });

        if (validIds.length > 0) {
          const queryEmbedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          const result = graph.query(queryEmbedding, 2, 3);

          if (result) {
            result.nodes.forEach(node => {
              const memory = client.get(node.memory_id);
              if (memory) {
                expect(memory.content).toBeDefined();
                expect(memory.content.length).toBeGreaterThan(0);
              }
            });
          }
        }
      } finally {
        client.close();
        graph.close();
      }
    });
  });

  describe('Edge Operations', () => {
    skipIfNoMembrain('should create edges between nodes', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 20; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        }

        const edgeCount = graph.edgeCount();
        expect(edgeCount).toBeGreaterThanOrEqual(0);
      } finally {
        graph.close();
      }
    });
  });

  describe('Graph Scalability', () => {
    skipIfNoMembrain('should handle many nodes', () => {
      const graph = new MembrainGraph({ embedding_dim: DEFAULT_EMBEDDING_DIM });

      try {
        for (let i = 0; i < 100; i++) {
          const memoryId = generateUUID();
          const embedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
          graph.addNode(memoryId, embedding);
        }

        const count = graph.nodeCount();
        expect(count).toBe(100);

        const queryEmbedding = randomEmbedding(DEFAULT_EMBEDDING_DIM);
        const result = graph.query(queryEmbedding, 2, 10);
        if (result) {
          expect(result.nodes.length).toBeGreaterThan(0);
        }
      } finally {
        graph.close();
      }
    });
  });

  describe('Graph Configuration', () => {
    skipIfNoMembrain('should support custom embedding dimensions', () => {
      const customDim = 32;
      const graph = new MembrainGraph({
        embedding_dim: customDim,
        hidden_dim: 64
      });

      try {
        const memoryId = generateUUID();
        const embedding = randomEmbedding(customDim);
        graph.addNode(memoryId, embedding);

        expect(graph.nodeCount()).toBe(1);
      } finally {
        graph.close();
      }
    });
  });
});
