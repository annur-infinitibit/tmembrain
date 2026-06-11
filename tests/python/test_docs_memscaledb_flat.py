"""
Test documentation examples from docs/integrations/memscaledb-flat.mdx

Exercises every Python code snippet from the Flat index documentation page
to ensure docs stay in sync with the implementation.
"""

import threading
import uuid

import pytest

from membrain import ConcurrentFlatIndex, MembrainFlatIndex

pytestmark = pytest.mark.docs

DIMENSION = 128


def _vector(seed: float, dimension: int = DIMENSION) -> list[float]:
    """Generate a deterministic vector from a seed value."""
    return [seed + (i * 0.001) for i in range(dimension)]


def _uuid() -> str:
    """Generate a fresh UUID string."""
    return str(uuid.uuid4())


# -- docs/integrations/memscaledb-flat.mdx: Usage ----------------------------


class TestFlatUsage:
    """Mirrors the 'Usage' code block."""

    def test_basic_usage(self):
        index = MembrainFlatIndex(dimension=DIMENSION)

        embedding_1 = _vector(0.1)
        embedding_2 = _vector(0.2)

        id1 = _uuid()
        id2 = _uuid()
        index.add(id1, embedding_1)
        index.add(id2, embedding_2)

        query_embedding = _vector(0.1)
        results = index.search(query_embedding, k=5)
        for r in results:
            assert r.id in (id1, id2)
            assert isinstance(r.distance, float)

        index.close()


# -- docs/integrations/memscaledb-flat.mdx: Custom Configuration -------------


class TestFlatCustomConfig:
    """Mirrors the 'Custom Configuration' code block."""

    def test_euclidean_config(self):
        index = MembrainFlatIndex(config={
            "dimension": DIMENSION,
            "distance_metric": "Euclidean",
            "cache_config": {
                "capacity": 2048,
                "enabled": True,
            },
        })

        vid = _uuid()
        index.add(vid, _vector(0.5))
        results = index.search(_vector(0.5), k=1)
        assert len(results) == 1
        assert results[0].id == vid

        index.close()


# -- docs/integrations/memscaledb-flat.mdx: Filtered Search ------------------


class TestFlatFilteredSearch:
    """Mirrors the 'Filtered Search' code block."""

    def test_search_with_filter(self):
        with MembrainFlatIndex(dimension=DIMENSION) as index:
            ids = [_uuid() for _ in range(10)]
            for i, vector_id in enumerate(ids):
                index.add(vector_id, _vector(float(i) / 10.0))

            query_embedding = _vector(0.5)
            allowed = ids[:3]
            results = index.search_with_filter(
                query_embedding, k=5,
                allowed_ids=allowed,
            )
            for r in results:
                assert r.id in allowed


# -- docs/integrations/memscaledb-flat.mdx: Batch Search ---------------------


class TestFlatBatchSearch:
    """Mirrors the 'Batch Search' code block."""

    def test_batch_search(self):
        with MembrainFlatIndex(dimension=DIMENSION) as index:
            for i in range(50):
                index.add(_uuid(), _vector(i * 0.02))

            query1 = _vector(0.1)
            query2 = _vector(0.5)
            query3 = _vector(0.9)
            queries = [query1, query2, query3]
            batch_results = index.batch_search(queries, k=5)

            assert len(batch_results) == 3
            for results in batch_results:
                assert len(results) <= 5


# -- docs/integrations/memscaledb-flat.mdx: Observability --------------------


class TestFlatObservability:
    """Mirrors the 'Observability' code block."""

    def test_metrics(self):
        with MembrainFlatIndex(dimension=DIMENSION) as index:
            index.add(_uuid(), _vector(0.1))
            index.search(_vector(0.1), k=1)

            metrics = index.metrics()
            assert metrics.searches >= 1
            assert metrics.inserts >= 1
            assert isinstance(metrics.distance_computations, int)


# -- docs/integrations/memscaledb-flat.mdx: Concurrent Access ----------------


class TestFlatConcurrentAccess:
    """Mirrors the 'Concurrent Access' code block."""

    def test_concurrent_flat_index(self):
        index = ConcurrentFlatIndex(dimension=DIMENSION)

        def worker():
            index.add(str(uuid.uuid4()), [0.5] * DIMENSION)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = index.search([0.5] * DIMENSION, k=5)
        assert len(results) <= 5
        assert len(index) == 8

        index.close()
