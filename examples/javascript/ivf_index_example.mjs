/** IVF index -- inverted file with k-means clustering. Requires training data. */

import { randomUUID } from "node:crypto";
import { MembrainIvfIndex } from "membrain";

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
const config = { dimension: DIMENSION, num_cells: 16, nprobe: 4, distance_metric: "Cosine" };

const index = MembrainIvfIndex.build(ids, vectors, config);

try {
  console.log(`Built IVF index: ${index.len()} vectors`);

  const newId = randomUUID();
  index.add(newId, embed(999));
  console.log(`After add: ${index.len()} vectors`);

  const results = index.search(embed(0.5), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }

  index.remove(newId);
  console.log(`After remove: ${index.len()} vectors`);
} finally {
  index.close();
}
