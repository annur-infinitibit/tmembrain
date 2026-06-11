"""ShardedIndex -- HNSW shards with k-means routing and rebalancing."""

import random
import uuid

from membrain import MembrainShardedIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    ids = [str(uuid.uuid4()) for _ in range(1000)]
    vectors = [embed(seed=float(i)) for i in range(1000)]

    config = {
        "num_shards": 4,
        "nprobe": 2,
        "overlap_factor": 1.5,
        "hnsw_config": {"m": 16, "ef_construction": 100, "ef_search": 50},
    }

    with MembrainShardedIndex.build(
        dimension=DIMENSION, ids=ids, vectors=vectors, config=config,
    ) as index:
        info = index.info()
        print(f"Built: {len(index)} vectors across {info.num_shards} shards")
        for shard in info.shards:
            print(f"  Shard {shard.shard_index}: {shard.count} vectors")

        results = index.search(embed(seed=0.5), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")

        for i in range(50):
            index.add(str(uuid.uuid4()), embed(seed=float(1000 + i)))

        index.rebalance()
        print(f"\nAfter rebalance: {len(index)} vectors, stddev={index.info().size_stddev:.2f}")


if __name__ == "__main__":
    main()
