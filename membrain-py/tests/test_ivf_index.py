"""Tests for MembrainIvfIndex."""

import uuid

import pytest

from membrain import MembrainIvfIndex


DIMENSION = 32


def random_vector(dimension: int = DIMENSION) -> list[float]:
    import random
    return [random.random() for _ in range(dimension)]


def _build_index(count: int = 200, **config_overrides):
    ids = [str(uuid.uuid4()) for _ in range(count)]
    vectors = [random_vector() for _ in range(count)]
    config = {"num_cells": 8, "nprobe": 4, **config_overrides}
    return MembrainIvfIndex.build(
        dimension=DIMENSION, ids=ids, vectors=vectors, config=config,
    ), ids, vectors


def test_build_and_search():
    index, ids, vectors = _build_index()
    with index:
        assert len(index) == 200
        assert index.dimension == DIMENSION

        results = index.search(vectors[0], k=5)
        assert len(results) == 5
        assert results[0].id == ids[0]


def test_add_after_build():
    index, _, _ = _build_index(count=100)
    with index:
        new_id = str(uuid.uuid4())
        new_vec = random_vector()
        index.add(new_id, new_vec)
        assert len(index) == 101


def test_remove():
    index, ids, _ = _build_index(count=100)
    with index:
        index.remove(ids[0])
        assert len(index) == 99


def test_search_with_filter():
    index, ids, _ = _build_index()
    with index:
        allowed = ids[:20]
        results = index.search_with_filter(random_vector(), k=5, allowed_ids=allowed)
        for r in results:
            assert r.id in allowed


def test_batch_search():
    index, _, _ = _build_index()
    with index:
        queries = [random_vector() for _ in range(3)]
        batch_results = index.batch_search(queries, k=5)
        assert len(batch_results) == 3
        for results in batch_results:
            assert len(results) <= 5


def test_metrics():
    index, _, vectors = _build_index()
    with index:
        index.search(vectors[0], k=1)
        m = index.metrics()
        assert m.searches >= 1
        assert m.distance_computations > 0


def test_context_manager():
    index, _, _ = _build_index(count=50)
    with index:
        assert len(index) == 50
