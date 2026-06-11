/** Vamana index -- DiskANN-style graph. Requires training data. */

import { randomUUID } from "node:crypto";
import { MembrainVamanaIndex } from "membrain";

const DIMENSION = 128;
const NUM_TRAINING = 500;

function embed(seed) {
  const out = [];
  let state = seed * 1000 + 1;
  for (let i = 0; i < DIMENSION; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out.push((state / 0x7fffffff) * 2 - 1);
  }
  return out;
}

const ids = Array.from({ length: NUM_TRAINING }, () => randomUUID());
const vectors = Array.from({ length: NUM_TRAINING }, (_, i) => embed(i));
const config = { dimension: DIMENSION, max_degree: 64, alpha: 1.2, search_list_size: 100, distance_metric: "Cosine" };

const index = MembrainVamanaIndex.build(ids, vectors, config);

try {
  console.log(`Built Vamana index: ${index.len()} vectors`);

  const results = index.search(embed(0.5), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }
} finally {
  index.close();
}
