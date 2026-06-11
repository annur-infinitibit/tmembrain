"""Tests for MembrainLshIndex."""

import uuid

import pytest

from membrain import MembrainLshIndex


DIMENSION = 32


def random_vector(dimension: int = DIMENSION) -> list[float]:
    import random
    return [random.random() for _ in range(dimension)]


def test_create_and_close():
    index = MembrainLshIndex(dimension=DIMENSION)
    assert len(index) == 0
    assert index.dimension == DIMENSION
    index.close()


def test_context_manager():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        assert len(index) == 0


def test_add_and_search():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        ids = []
        vectors = []
        for _ in range(100):
            vid = str(uuid.uuid4())
            vec = random_vector()
            index.add(vid, vec)
            ids.append(vid)
            vectors.append(vec)

        assert len(index) == 100

        results = index.search(vectors[0], k=5)
        assert len(results) <= 5
        # The exact vector should be in results (high probability with LSH)
        result_ids = [r.id for r in results]
        assert ids[0] in result_ids


def test_remove():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        index.add(id1, random_vector())
        index.add(id2, random_vector())
        assert len(index) == 2

        index.remove(id1)
        assert len(index) == 1


def test_search_with_filter():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        ids = []
        for _ in range(50):
            vid = str(uuid.uuid4())
            index.add(vid, random_vector())
            ids.append(vid)

        allowed = ids[:10]
        results = index.search_with_filter(random_vector(), k=5, allowed_ids=allowed)
        for r in results:
            assert r.id in allowed


def test_batch_search():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        for _ in range(50):
            index.add(str(uuid.uuid4()), random_vector())

        queries = [random_vector() for _ in range(3)]
        batch_results = index.batch_search(queries, k=5)
        assert len(batch_results) == 3


def test_metrics():
    with MembrainLshIndex(dimension=DIMENSION) as index:
        index.add(str(uuid.uuid4()), random_vector())
        index.search(random_vector(), k=1)

        m = index.metrics()
        assert m.inserts >= 1
        assert m.searches >= 1


def test_config_with_options():
    with MembrainLshIndex(
        dimension=DIMENSION,
        config={"num_hyperplanes": 8, "num_tables": 4, "seed": 42},
    ) as index:
        vid = str(uuid.uuid4())
        vec = random_vector()
        index.add(vid, vec)
        results = index.search(vec, k=1)
        assert len(results) == 1
