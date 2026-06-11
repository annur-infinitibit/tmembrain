"""Tests for MembrainFlatIndex."""

import uuid

import pytest

from membrain import MembrainFlatIndex


DIMENSION = 32


def random_vector(dimension: int = DIMENSION) -> list[float]:
    import random
    return [random.random() for _ in range(dimension)]


def test_create_and_close():
    index = MembrainFlatIndex(dimension=DIMENSION)
    assert len(index) == 0
    assert index.dimension == DIMENSION
    index.close()


def test_context_manager():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        assert len(index) == 0


def test_add_and_search():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        vector = random_vector()
        id1 = str(uuid.uuid4())
        index.add(id1, vector)
        assert len(index) == 1

        results = index.search(vector, k=1)
        assert len(results) == 1
        assert results[0].id == id1
        assert results[0].distance < 1e-5


def test_remove():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        index.add(id1, random_vector())
        index.add(id2, random_vector())
        assert len(index) == 2

        index.remove(id1)
        assert len(index) == 1


def test_search_with_filter():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        ids = [str(uuid.uuid4()) for _ in range(10)]
        for i in ids:
            index.add(i, random_vector())

        allowed = ids[:5]
        results = index.search_with_filter(random_vector(), k=3, allowed_ids=allowed)
        for r in results:
            assert r.id in allowed


def test_batch_search():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        for _ in range(20):
            index.add(str(uuid.uuid4()), random_vector())

        queries = [random_vector() for _ in range(3)]
        batch_results = index.batch_search(queries, k=5)
        assert len(batch_results) == 3
        for results in batch_results:
            assert len(results) <= 5


def test_metrics():
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        index.add(str(uuid.uuid4()), random_vector())
        index.search(random_vector(), k=1)

        m = index.metrics()
        assert m.inserts >= 1
        assert m.searches >= 1


def test_exact_recall():
    """Flat index should achieve 100% recall."""
    with MembrainFlatIndex(dimension=DIMENSION) as index:
        pairs = []
        for _ in range(50):
            vid = str(uuid.uuid4())
            vec = random_vector()
            index.add(vid, vec)
            pairs.append((vid, vec))

        for vid, vec in pairs:
            results = index.search(vec, k=1)
            assert results[0].id == vid


def test_config_distance_metric():
    with MembrainFlatIndex(
        dimension=DIMENSION,
        config={"distance_metric": "Euclidean"},
    ) as index:
        vid = str(uuid.uuid4())
        vec = random_vector()
        index.add(vid, vec)
        results = index.search(vec, k=1)
        assert results[0].id == vid
