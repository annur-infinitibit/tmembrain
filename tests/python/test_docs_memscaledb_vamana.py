"""
Test documentation examples from docs/integrations/memscaledb-vamana.mdx

Exercises every Python code snippet from the Vamana index documentation page
to ensure docs stay in sync with the implementation.
"""

import threading
import uuid

import pytest

from membrain import ConcurrentVamanaIndex, MembrainVamanaIndex

pytestmark = pytest.mark.docs

DIMENSION = 64
NUM_VECTORS = 200


def _uuid() -> str:
    return str(uuid.uuid4())


def _make_training_data(
    count: int = NUM_VECTORS,
    dimension: int = DIMENSION,
) -> tuple[list[str], list[list[float]]]:
    """Generate deterministic training data for Vamana build."""
    ids = [_uuid() for _ in range(count)]
    vectors = [
        [float(i % dimension) / dimension + (j * 0.001) for j in range(dimension)]
        for i in range(count)
    ]
    return ids, vectors


# -- docs/integrations/memscaledb-vamana.mdx: Usage --------------------------


class TestVamanaUsage:
    """Mirrors the 'Usage' code block."""

    def test_build_search_add(self):
        ids, vectors = _make_training_data()

        index = MembrainVamanaIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={
                "max_degree": 32,
                "alpha": 1.2,
                "search_list_size": 64,
            },
        )

        query_embedding = vectors[0]
        results = index.search(query_embedding, k=10)
        assert len(results) > 0
        for r in results:
            assert isinstance(r.id, str)
            assert isinstance(r.distance, float)

        # Add more vectors after build
        new_embedding = [0.5] * DIMENSION
        index.add(_uuid(), new_embedding)
        assert len(index) == NUM_VECTORS + 1

        index.close()


# -- docs/integrations/memscaledb-vamana.mdx: Filtered Search ----------------


class TestVamanaFilteredSearch:
    """Mirrors the 'Filtered Search' code block."""

    def test_search_with_filter(self):
        ids, vectors = _make_training_data()

        with MembrainVamanaIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={
                "max_degree": 32,
                "search_list_size": 128,
            },
        ) as index:
            allowed = [ids[0], ids[2], ids[4]]
            query_embedding = vectors[0]
            results = index.search_with_filter(
                query_embedding, k=10,
                allowed_ids=allowed,
            )
            for r in results:
                assert r.id in allowed


# -- docs/integrations/memscaledb-vamana.mdx: Batch Search -------------------


class TestVamanaBatchSearch:
    """Mirrors the 'Batch Search' code block."""

    def test_batch_search(self):
        ids, vectors = _make_training_data()

        with MembrainVamanaIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"max_degree": 32},
        ) as index:
            query1 = vectors[0]
            query2 = vectors[10]
            query3 = vectors[20]
            queries = [query1, query2, query3]
            batch_results = index.batch_search(queries, k=10)

            assert len(batch_results) == 3
            for results in batch_results:
                assert len(results) <= 10


# -- docs/integrations/memscaledb-vamana.mdx: Observability ------------------


class TestVamanaObservability:
    """Mirrors the 'Observability' code block."""

    def test_metrics(self):
        ids, vectors = _make_training_data()

        with MembrainVamanaIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"max_degree": 32},
        ) as index:
            index.search(vectors[0], k=5)

            metrics = index.metrics()
            assert metrics.searches >= 1
            assert isinstance(metrics.distance_computations, int)


# -- docs/integrations/memscaledb-vamana.mdx: Concurrent Access --------------


class TestVamanaConcurrentAccess:
    """Mirrors the 'Concurrent Access' code block."""

    def test_concurrent_vamana_index(self):
        ids = [str(uuid.uuid4()) for _ in range(1000)]
        vectors = [[float(i % 64) / 64.0] * 64 for i in range(1000)]

        index = ConcurrentVamanaIndex.build(ids, vectors, dimension=64)

        def search_worker():
            results = index.search([0.5] * 64, k=10)
            assert len(results) > 0

        threads = [threading.Thread(target=search_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        index.close()
