"""HNSW index -- hierarchical navigable small world graph with persistence."""

import random
import uuid

from membrain import MembrainIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    config = {"m": 16, "ef_construction": 200, "ef_search": 100, "distance_metric": "Cosine"}

    with MembrainIndex(dimension=DIMENSION, config=config) as index:
        for i in range(200):
            index.add(str(uuid.uuid4()), embed(seed=float(i)))

        print(f"Indexed {len(index)} vectors")

        results = index.search(embed(seed=0.5), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")

        serialized = index.save()
        print(f"\nSerialized: {len(serialized)} bytes")

    restored = MembrainIndex.load(serialized)
    results_after = restored.search(embed(seed=0.5), k=3)
    print(f"Restored: {len(restored)} vectors, top score={results_after[0].score:.4f}")
    restored.close()


if __name__ == "__main__":
    main()
