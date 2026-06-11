"""
Test documentation examples from docs/integrations/memscaledb-concurrent.mdx

Exercises every Python code snippet from the Concurrent Indices documentation
page to ensure docs stay in sync with the implementation.
"""

import threading
import uuid

import pytest

from membrain import (
    ConcurrentFlatIndex,
    ConcurrentHnswIndex,
    ConcurrentIvfIndex,
)

pytestmark = pytest.mark.docs

DIMENSION = 128


def _uuid() -> str:
    return str(uuid.uuid4())


# -- docs/integrations/memscaledb-concurrent.mdx: Quick Start ----------------


class TestConcurrentQuickStart:
    """Mirrors the 'Quick Start' code block."""

    def test_hnsw_quick_start(self):
        index = ConcurrentHnswIndex(dimension=DIMENSION)

        def worker(thread_id):
            vec_id = _uuid()
            embedding = [float(thread_id) / float(DIMENSION)] * DIMENSION
            index.add(vec_id, embedding)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = index.search([0.5] * DIMENSION, k=10)
        assert len(results) > 0
        assert len(index) == 8

        index.close()


# -- docs/integrations/memscaledb-concurrent.mdx: Handle Cloning -------------


class TestConcurrentHandleCloning:
    """Mirrors the 'Handle Cloning' code block."""

    def test_flat_clone(self):
        index = ConcurrentFlatIndex(dimension=DIMENSION)
        index.add(_uuid(), [0.1] * DIMENSION)

        clone = index.clone()
        clone.add(_uuid(), [0.2] * DIMENSION)

        assert len(index) == 2
        assert len(clone) == 2

        clone.close()
        index.close()


# -- docs/integrations/memscaledb-concurrent.mdx: Build-Style Indices --------


class TestConcurrentBuildStyle:
    """Mirrors the 'Build-Style Indices' code block (ConcurrentIvfIndex)."""

    def test_concurrent_ivf_build_and_add(self):
        ids = [_uuid() for _ in range(1000)]
        vectors = [[float(i % 128) / 128.0] * DIMENSION for i in range(1000)]

        index = ConcurrentIvfIndex.build(ids, vectors, dimension=DIMENSION)

        def add_more():
            index.add(_uuid(), [0.5] * DIMENSION)

        threads = [threading.Thread(target=add_more) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = index.search([0.5] * DIMENSION, k=10)
        assert len(results) > 0

        index.close()


# -- docs/integrations/memscaledb-concurrent.mdx: Configuration --------------


class TestConcurrentConfiguration:
    """Mirrors the 'Configuration' code block."""

    def test_hnsw_with_config(self):
        index = ConcurrentHnswIndex(dimension=DIMENSION, config={
            "m": 16,
            "ef_construction": 200,
            "ef_search": 100,
            "distance_metric": "Cosine",
        })

        index.add(_uuid(), [0.5] * DIMENSION)
        results = index.search([0.5] * DIMENSION, k=10)
        assert len(results) >= 1

        index.close()
