"""Vamana index -- DiskANN-style graph. Requires training data."""

import random
import uuid

from membrain import MembrainVamanaIndex

DIMENSION = 128
NUM_TRAINING = 500


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    ids = [str(uuid.uuid4()) for _ in range(NUM_TRAINING)]
    vectors = [embed(seed=float(i)) for i in range(NUM_TRAINING)]

    config = {"max_degree": 64, "alpha": 1.2, "search_list_size": 100, "distance_metric": "Cosine"}

    with MembrainVamanaIndex.build(
        dimension=DIMENSION, ids=ids, vectors=vectors, config=config,
    ) as index:
        print(f"Built Vamana index: {len(index)} vectors")

        results = index.search(embed(seed=0.5), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")


if __name__ == "__main__":
    main()
