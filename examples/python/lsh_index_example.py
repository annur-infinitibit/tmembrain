"""LSH index -- locality-sensitive hashing for ultra-fast approximate search."""

import random
import uuid

from membrain import MembrainLshIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    config = {"num_hyperplanes": 12, "num_tables": 6, "distance_metric": "Cosine"}

    with MembrainLshIndex(dimension=DIMENSION, config=config) as index:
        for i in range(200):
            index.add(str(uuid.uuid4()), embed(seed=float(i)))

        print(f"Indexed {len(index)} vectors")

        results = index.search(embed(seed=0.0), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")


if __name__ == "__main__":
    main()
