"""IVF index -- inverted file with k-means clustering. Requires training data."""

import random
import uuid

from membrain import MembrainIvfIndex

DIMENSION = 128
NUM_TRAINING = 500


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    ids = [str(uuid.uuid4()) for _ in range(NUM_TRAINING)]
    vectors = [embed(seed=float(i)) for i in range(NUM_TRAINING)]

    config = {"num_cells": 16, "nprobe": 4, "distance_metric": "Cosine"}

    with MembrainIvfIndex.build(
        dimension=DIMENSION, ids=ids, vectors=vectors, config=config,
    ) as index:
        print(f"Built IVF index: {len(index)} vectors")

        new_id = str(uuid.uuid4())
        index.add(new_id, embed(seed=999.0))
        print(f"After add: {len(index)} vectors")

        results = index.search(embed(seed=0.5), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")

        index.remove(new_id)
        print(f"After remove: {len(index)} vectors")


if __name__ == "__main__":
    main()
