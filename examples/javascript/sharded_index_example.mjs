/** ShardedIndex -- HNSW shards with k-means routing and rebalancing. */

import { randomUUID } from "node:crypto";
import { MembrainShardedIndex } from "membrain";

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

const ids = Array.from({ length: 1000 }, () => randomUUID());
const vectors = Array.from({ length: 1000 }, (_, i) => embed(i));

const config = {
  dimension: DIMENSION,
  num_shards: 4,
  nprobe: 2,
  overlap_factor: 1.5,
  hnsw_config: { m: 16, ef_construction: 100, ef_search: 50 },
};

const index = MembrainShardedIndex.build(ids, vectors, config);

try {
  const info = index.info();
  console.log(`Built: ${index.len()} vectors across ${info.num_shards} shards`);
  for (const shard of info.shards) {
    console.log(`  Shard ${shard.shard_index}: ${shard.count} vectors`);
  }

  const results = index.search(embed(0.5), 5);
  for (const r of results) {
    console.log(`  id=${r.id.slice(0, 8)}... score=${r.score.toFixed(4)}`);
  }

  for (let i = 0; i < 50; i++) index.add(randomUUID(), embed(1000 + i));

  index.rebalance();
  console.log(`\nAfter rebalance: ${index.len()} vectors, stddev=${index.info().size_stddev.toFixed(2)}`);
} finally {
  index.close();
}
