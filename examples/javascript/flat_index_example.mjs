/** Flat index -- brute-force exact search with 100% recall. */

import { randomUUID } from "node:crypto";
import { MembrainFlatIndex } from "membrain";

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

const index = new MembrainFlatIndex(DIMENSION);

try {
  const ids = Array.from({ length: 50 }, () => randomUUID());
  ids.forEach((id, i) => index.add(id, embed(i)));

  console.log(`Indexed ${index.len()} vectors (dimension=${index.dimension()})`);

  const results = index.search(embed(0), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }

  const filtered = index.searchWithFilter(embed(0), 3, ids.slice(0, 10));
  console.log(`\nFiltered to 10 IDs: ${filtered.length} results`);

  index.remove(ids[0]);
  console.log(`After remove: ${index.len()} vectors`);
} finally {
  index.close();
}
