/** Reranker -- re-score search results using LLM-based relevance.
 *
 * Requires: OPENAI_API_KEY environment variable.
 */

import { MembrainClient, OpenAIReranker } from "membrain";

if (!process.env.OPENAI_API_KEY) {
  console.log("Set OPENAI_API_KEY to run this example.");
  process.exit(0);
}

const client = new MembrainClient();

try {
  for (const [statement, confidence] of [
    ["HNSW creates a multi-layered graph for ANN search", 0.95],
    ["Cosine similarity measures the angle between two vectors", 0.9],
    ["IVF partitions the vector space using k-means clustering", 0.85],
    ["LSH uses random projections for fast approximate search", 0.85],
    ["Product quantization compresses vectors by splitting into subspaces", 0.8],
  ]) {
    client.storeFact(statement, confidence);
  }

  const query = "How does vector search work?";
  const results = client.search(query, 5);

  console.log(`Initial search for '${query}':`);
  for (const m of results.memories) {
    console.log(`  [${m.score.toFixed(3)}] ${m.content.slice(0, 70)}`);
  }

  const reranker = new OpenAIReranker({ apiKey: process.env.OPENAI_API_KEY, topK: 5 });
  const reranked = await reranker.rerank(query, results, 3);

  console.log(`\nReranked (model=${reranked.model}, ${reranked.duration_ms}ms):`);
  for (const m of reranked.memories) {
    console.log(`  [${m.relevance_score.toFixed(3)}] ${m.content.slice(0, 70)}`);
  }
} finally {
  client.close();
}
