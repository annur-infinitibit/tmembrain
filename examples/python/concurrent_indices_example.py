"""Concurrent indices -- thread-safe vector indices with internal locking.

All concurrent indices release Python's GIL during Rust operations,
enabling true multi-threaded parallelism.
"""

import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from membrain import ConcurrentFlatIndex, ConcurrentIvfIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def demo_flat() -> None:
    """ConcurrentFlatIndex with multi-threaded writes and reads."""
    print("=== ConcurrentFlatIndex ===")

    with ConcurrentFlatIndex(dimension=DIMENSION) as index:

        def insert_batch(thread_id: int) -> int:
            for i in range(50):
                index.add(f"t{thread_id}-{i}", embed(seed=float(thread_id * 100 + i)))
            return 50

        with ThreadPoolExecutor(max_workers=4) as pool:
            total = sum(f.result() for f in as_completed(
                [pool.submit(insert_batch, t) for t in range(4)]
            ))

        print(f"  Inserted {total} vectors from 4 threads, size={len(index)}")

        results = index.search(embed(seed=0.0), k=3)
        print(f"  Top score: {results[0].score:.4f}")

        clone = index.clone()
        print(f"  Clone size: {len(clone)}")
        clone.close()


def demo_ivf() -> None:
    """ConcurrentIvfIndex -- build-type index with concurrent adds."""
    print("\n=== ConcurrentIvfIndex ===")

    ids = [str(uuid.uuid4()) for _ in range(200)]
    vectors = [embed(seed=float(i)) for i in range(200)]

    with ConcurrentIvfIndex.build(
        ids=ids, vectors=vectors, dimension=DIMENSION, config={"num_clusters": 8, "nprobe": 3},
    ) as index:
        print(f"  Built: {len(index)} vectors")

        with ThreadPoolExecutor(max_workers=4) as pool:
            for t in range(4):
                pool.submit(lambda tid=t: [
                    index.add(f"new-{tid}-{i}", embed(seed=float(1000 + tid * 100 + i)))
                    for i in range(20)
                ])

        print(f"  After concurrent adds: {len(index)} vectors")

        results = index.search(embed(seed=0.0), k=3)
        print(f"  Top score: {results[0].score:.4f}")


def main() -> None:
    demo_flat()
    demo_ivf()


if __name__ == "__main__":
    main()
