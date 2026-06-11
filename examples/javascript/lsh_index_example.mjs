/** LSH index -- locality-sensitive hashing for ultra-fast approximate search. */

import { randomUUID } from "node:crypto";
import { MembrainLshIndex } from "membrain";

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

const config = { dimension: DIMENSION, num_hyperplanes: 12, num_tables: 6, distance_metric: "Cosine" };
const index = new MembrainLshIndex(DIMENSION, config);

try {
  for (let i = 0; i < 200; i++) index.add(randomUUID(), embed(i));
  console.log(`Indexed ${index.len()} vectors`);

  const results = index.search(embed(0), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }
} finally {
  index.close();
}
