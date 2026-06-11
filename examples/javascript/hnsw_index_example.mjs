/** HNSW index -- hierarchical navigable small world graph with persistence. */

import { randomUUID } from "node:crypto";
import { MembrainIndex } from "membrain";

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

const config = { dimension: DIMENSION, m: 16, ef_construction: 200, ef_search: 100, distance_metric: "Cosine" };
const index = new MembrainIndex(DIMENSION, config);
let serialized;

try {
  for (let i = 0; i < 200; i++) index.add(randomUUID(), embed(i));
  console.log(`Indexed ${index.len()} vectors`);

  const results = index.search(embed(0.5), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }

  serialized = index.save();
  console.log(`\nSerialized: ${serialized.length} bytes`);
} finally {
  index.close();
}

const restored = MembrainIndex.load(serialized);
try {
  const results = restored.search(embed(0.5), 3);
  console.log(`Restored: ${restored.len()} vectors, top score=${results[0].score.toFixed(4)}`);
} finally {
  restored.close();
}
