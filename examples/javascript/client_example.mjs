/** MembrainClient -- store and retrieve LLM memories with case-based reasoning. */

import { MembrainClient } from "membrain";

const client = new MembrainClient();

try {
  client.storeFact("Python was created by Guido van Rossum in 1991", 0.95);
  client.storePreference("user", "languages", "prefers strongly-typed languages", "strong");
  client.storeEntity("membrain", "software_project");
  client.storeConcept("Vector Similarity Search", "Finding closest vectors in high-dimensional space");

  client.storeCase(
    "User asked to summarize a 10-page document",
    "Split into sections, summarize each, then combine",
    "User satisfied with hierarchical summary",
    1.0,
  );

  console.log(`Stored ${client.count()} memories`);

  const results = client.search("programming languages", 5);
  console.log(`\nSearch results (${results.duration_ms}ms):`);
  for (const m of results.memories) {
    console.log(`  [${m.memory_type}] ${m.content.slice(0, 80)} (score=${m.score.toFixed(3)})`);
  }

  const filtered = client.search("vector search", 5, { memory_types: ["fact", "concept"] });
  console.log(`\nFiltered (facts+concepts): ${filtered.memories.length} results`);

  const cases = client.searchCases("summarize a long document", 3, 0.5);
  console.log(`\nCases: ${cases.positive_cases.length} positive, ${cases.negative_cases.length} negative`);
} finally {
  client.close();
}
