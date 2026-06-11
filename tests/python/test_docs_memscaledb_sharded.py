"""
Test documentation examples from docs/integrations/memscaledb-sharded.mdx

Exercises every Python code snippet from the Sharded index documentation page
to ensure docs stay in sync with the implementation.
"""

import uuid

import pytest

from membrain import MembrainShardedIndex

pytestmark = pytest.mark.docs

DIMENSION = 64
NUM_VECTORS = 500


def _uuid() -> str:
    return str(uuid.uuid4())


def _make_training_data(
    count: int = NUM_VECTORS,
    dimension: int = DIMENSION,
) -> tuple[list[str], list[list[float]]]:
    """Generate deterministic training data for sharded build."""
    ids = [_uuid() for _ in range(count)]
    vectors = [
        [float(i % dimension) / dimension + (j * 0.001) for j in range(dimension)]
        for i in range(count)
    ]
    return ids, vectors


# -- docs/integrations/memscaledb-sharded.mdx: Usage -------------------------


class TestShardedUsage:
    """Mirrors the 'Usage' code block."""

    def test_build_search_save_load(self):
        ids, vectors = _make_training_data()

        index = MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={
                "num_shards": 4,
                "nprobe": 3,
                "overlap_factor": 1,
            },
        )

        query_vector = vectors[0]
        results = index.search(query_vector, k=10)
        assert len(results) > 0
        for result in results:
            assert isinstance(result.id, str)
            assert isinstance(result.distance, float)

        # Save and restore
        data = index.save()
        assert isinstance(data, str)
        assert len(data) > 0

        restored = MembrainShardedIndex.load(data)
        restored_results = restored.search(query_vector, k=10)
        assert len(restored_results) > 0

        restored.close()
        index.close()


# -- docs/integrations/memscaledb-sharded.mdx: Adding and Removing Vectors ---


class TestShardedAddRemove:
    """Mirrors the 'Adding and Removing Vectors' code block."""

    def test_add_and_remove(self):
        ids, vectors = _make_training_data()

        with MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 3},
        ) as index:
            initial_len = len(index)

            new_id = _uuid()
            index.add(new_id, [0.1 + (i * 0.001) for i in range(DIMENSION)])
            assert len(index) == initial_len + 1

            index.remove(new_id)
            assert len(index) == initial_len


# -- docs/integrations/memscaledb-sharded.mdx: Filtered Search ---------------


class TestShardedFilteredSearch:
    """Mirrors the 'Filtered Search' code block."""

    def test_search_with_filter(self):
        ids, vectors = _make_training_data()

        with MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 4},
        ) as index:
            query_vector = vectors[0]
            allowed = [ids[0], ids[2], ids[4]]
            results = index.search_with_filter(
                query_vector,
                k=10,
                allowed_ids=allowed,
            )
            for r in results:
                assert r.id in allowed


# -- docs/integrations/memscaledb-sharded.mdx: Batch Search ------------------


class TestShardedBatchSearch:
    """Mirrors the 'Batch Search' code block."""

    def test_batch_search(self):
        ids, vectors = _make_training_data()

        with MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 3},
        ) as index:
            query1 = vectors[0]
            query2 = vectors[50]
            query3 = vectors[100]
            queries = [query1, query2, query3]
            batch_results = index.batch_search(queries, k=10)

            assert len(batch_results) == 3
            for results in batch_results:
                assert len(results) <= 10


# -- docs/integrations/memscaledb-sharded.mdx: Rebalancing -------------------


class TestShardedRebalancing:
    """Mirrors the 'Rebalancing' code block."""

    def test_rebalance(self):
        ids, vectors = _make_training_data()

        with MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 3},
        ) as index:
            # Add some extra vectors to unbalance shards
            for i in range(50):
                index.add(_uuid(), [0.5 + (i * 0.001)] * DIMENSION)

            index.rebalance()

            info = index.info()
            assert info.num_shards == 4
            assert isinstance(info.size_stddev, float)


# -- docs/integrations/memscaledb-sharded.mdx: Monitoring --------------------


class TestShardedMonitoring:
    """Mirrors the 'Monitoring' code block."""

    def test_info_and_metrics(self):
        ids, vectors = _make_training_data()

        with MembrainShardedIndex.build(
            dimension=DIMENSION,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 3},
        ) as index:
            # Per-shard statistics
            info = index.info()
            assert info.num_shards == 4
            for shard in info.shards:
                assert isinstance(shard.shard_index, int)
                assert isinstance(shard.count, int)

            # Performance metrics
            index.search(vectors[0], k=5)
            metrics = index.metrics()
            assert metrics.searches >= 1
