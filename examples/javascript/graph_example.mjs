/** MembrainGraph -- knowledge graph with multi-hop traversal and persistence. */

import { MembrainClient, MembrainGraph } from "membrain";

const DIMENSION = 128;

function embed(seed) {
  const out = [];
  let state = seed * 1000 + 1;
  for (let i = 0; i < DIMENSION; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out.push((state / 0x7fffffff) * 2 - 1);
  }
  return out;
}

const client = new MembrainClient();
const graph = new MembrainGraph();

try {
  const topics = [
    ["Python is a high-level programming language", 0.95],
    ["Rust provides memory safety without garbage collection", 0.95],
    ["HNSW is a graph-based ANN algorithm", 0.9],
    ["Vector databases store high-dimensional embeddings", 0.9],
    ["RAG combines retrieval with generation", 0.85],
  ];

  for (let i = 0; i < topics.length; i++) {
    const [statement, confidence] = topics[i];
    const result = client.storeFact(statement, confidence);
    if (result.success && result.id) {
      graph.addNode(result.id, embed(i), confidence);
    }
  }

  console.log(`Graph: ${graph.nodeCount()} nodes, ${graph.edgeCount()} edges`);

  const query = embed(2.5);
  const result = graph.query(query, 3, 3);
  console.log(`\nMulti-hop query: ${result.hops_performed} hops, ${result.nodes_visited} visited`);
  for (const node of result.nodes) {
    const memory = client.get(node.memory_id);
    console.log(`  ${memory.content.slice(0, 60)}... (score=${node.score.toFixed(3)}, hop=${node.hop_distance})`);
  }

  const serialized = graph.save();
  console.log(`\nSerialized: ${serialized.length} bytes`);

  const restored = MembrainGraph.load(serialized);
  console.log(`Restored: ${restored.nodeCount()} nodes, ${restored.edgeCount()} edges`);
  restored.close();
} finally {
  graph.close();
  client.close();
}
