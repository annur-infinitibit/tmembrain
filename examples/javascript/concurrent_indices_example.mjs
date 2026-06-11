/** Concurrent indices -- thread-safe vector indices with internal locking. */

import { randomUUID } from "node:crypto";
import { ConcurrentFlatIndex, ConcurrentIvfIndex } from "membrain";

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

function demoConcurrentFlat() {
  console.log("=== ConcurrentFlatIndex ===");
  const index = new ConcurrentFlatIndex(DIMENSION);

  try {
    for (let i = 0; i < 200; i++) index.add(`vec-${i}`, embed(i));
    console.log(`  Indexed ${index.len()} vectors`);

    const results = index.search(embed(0), 3);
    console.log(`  Top score: ${results[0].score.toFixed(4)}`);

    const clone = index.clone();
    console.log(`  Clone size: ${clone.len()}`);
    clone.close();

    index.remove("vec-0");
    console.log(`  After remove: ${index.len()}`);
  } finally {
    index.close();
  }
}

function demoConcurrentIvf() {
  console.log("\n=== ConcurrentIvfIndex ===");
  const ids = Array.from({ length: 200 }, () => randomUUID());
  const vectors = Array.from({ length: 200 }, (_, i) => embed(i));

  const index = ConcurrentIvfIndex.build(ids, vectors, DIMENSION, { num_clusters: 8, nprobe: 3 });

  try {
    console.log(`  Built: ${index.len()} vectors`);

    for (let i = 0; i < 20; i++) index.add(`new-${i}`, embed(1000 + i));
    console.log(`  After adds: ${index.len()} vectors`);

    const results = index.search(embed(0), 3);
    console.log(`  Top score: ${results[0].score.toFixed(4)}`);
  } finally {
    index.close();
  }
}

demoConcurrentFlat();
demoConcurrentIvf();
