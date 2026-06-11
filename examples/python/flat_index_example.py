"""Flat index -- brute-force exact search with 100% recall."""

import random
import uuid

from membrain import MembrainFlatIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        ids = [str(uuid.uuid4()) for _ in range(50)]
        for i, vector_id in enumerate(ids):
            index.add(vector_id, embed(seed=float(i)))

        print(f"Indexed {len(index)} vectors (dimension={index.dimension})")

        results = index.search(embed(seed=0.0), k=5)
        for result in results:
            print(f"  id={result.id[:8]}... score={result.score:.4f}")

        filtered = index.search_with_filter(embed(seed=0.0), k=3, allowed_ids=ids[:10])
        print(f"\nFiltered to 10 IDs: {len(filtered)} results")

        index.remove(ids[0])
        print(f"After remove: {len(index)} vectors")


if __name__ == "__main__":
    main()
