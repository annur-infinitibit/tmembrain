"""
Test documentation examples from docs/integrations/memscaledb-lsh.mdx

Exercises every Python code snippet from the LSH index documentation page
to ensure docs stay in sync with the implementation.
"""

import threading
import uuid

import pytest

from membrain import ConcurrentLshIndex, MembrainLshIndex

pytestmark = pytest.mark.docs

DIMENSION = 128


def _vector(seed: float, dimension: int = DIMENSION) -> list[float]:
    """Generate a deterministic vector from a seed value."""
    return [seed + (i * 0.001) for i in range(dimension)]


def _uuid() -> str:
    return str(uuid.uuid4())


# -- docs/integrations/memscaledb-lsh.mdx: Usage -----------------------------


class TestLshUsage:
    """Mirrors the 'Usage' code block."""

    def test_basic_usage(self):
        index = MembrainLshIndex(dimension=DIMENSION)

        documents = [
            (_uuid(), _vector(i * 0.1))
            for i in range(20)
        ]

        for doc_id, embedding in documents:
            index.add(doc_id, embedding)

        query_embedding = _vector(0.0)
        results = index.search(query_embedding, k=10)
        for r in results:
            assert isinstance(r.id, str)
            assert isinstance(r.distance, float)

        index.close()


# -- docs/integrations/memscaledb-lsh.mdx: Custom Configuration --------------


class TestLshCustomConfig:
    """Mirrors the 'Custom Configuration' code block."""

    def test_custom_config(self):
        index = MembrainLshIndex(config={
            "dimension": DIMENSION,
            "num_hyperplanes": 24,
            "num_tables": 12,
            "distance_metric": "Cosine",
            "seed": 42,
        })

        index.add(_uuid(), _vector(0.5))
        results = index.search(_vector(0.5), k=1)
        assert len(results) >= 1

        index.close()


# -- docs/integrations/memscaledb-lsh.mdx: Filtered Search -------------------


class TestLshFilteredSearch:
    """Mirrors the 'Filtered Search' code block."""

    def test_search_with_filter(self):
        with MembrainLshIndex(dimension=DIMENSION) as index:
            ids = [_uuid() for _ in range(20)]
            for i, vector_id in enumerate(ids):
                index.add(vector_id, _vector(float(i) / 20.0))

            query_embedding = _vector(0.5)
            allowed = ids[:3]
            results = index.search_with_filter(
                query_embedding, k=10,
                allowed_ids=allowed,
            )
            for r in results:
                assert r.id in allowed


# -- docs/integrations/memscaledb-lsh.mdx: Batch Search ----------------------


class TestLshBatchSearch:
    """Mirrors the 'Batch Search' code block."""

    def test_batch_search(self):
        with MembrainLshIndex(dimension=DIMENSION) as index:
            for i in range(50):
                index.add(_uuid(), _vector(i * 0.02))

            query1 = _vector(0.1)
            query2 = _vector(0.5)
            query3 = _vector(0.9)
            queries = [query1, query2, query3]
            batch_results = index.batch_search(queries, k=10)

            assert len(batch_results) == 3
            for results in batch_results:
                assert len(results) <= 10


# -- docs/integrations/memscaledb-lsh.mdx: Observability ---------------------


class TestLshObservability:
    """Mirrors the 'Observability' code block."""

    def test_metrics(self):
        with MembrainLshIndex(dimension=DIMENSION) as index:
            index.add(_uuid(), _vector(0.1))
            index.search(_vector(0.1), k=1)

            metrics = index.metrics()
            assert metrics.searches >= 1
            assert metrics.inserts >= 1


# -- docs/integrations/memscaledb-lsh.mdx: Concurrent Access -----------------


class TestLshConcurrentAccess:
    """Mirrors the 'Concurrent Access' code block."""

    def test_concurrent_lsh_index(self):
        index = ConcurrentLshIndex(dimension=DIMENSION)

        def worker():
            index.add(str(uuid.uuid4()), [0.5] * DIMENSION)
            index.search([0.5] * DIMENSION, k=10)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        index.close()
