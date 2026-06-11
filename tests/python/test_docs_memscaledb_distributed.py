"""
Test documentation examples from docs/integrations/memscaledb-distributed.mdx

Exercises every Python code snippet from the Distributed index documentation
page to ensure docs stay in sync with the implementation.

Note: These tests run a single-node cluster with replication_factor=1.
Each test uses a unique port to avoid conflicts during parallel test runs.

KNOWN ISSUE: The distributed index has a race condition / initialization bug
where single-node clusters with replication_factor=1 cannot reach write quorum.
The hash ring is not properly initialized before writes are attempted.
See: crates/memscaledb/src/distributed/mod.rs lines 313-323
"""

import time
import uuid

import pytest

from membrain import MembrainDistributedIndex

pytestmark = [pytest.mark.docs]

DIMENSION = 128

# Use a base port that will be incremented per test to avoid conflicts
_PORT_COUNTER = 19400


def _vector(seed: float, dimension: int = DIMENSION) -> list[float]:
    """Generate a deterministic vector from a seed value."""
    return [seed + (i * 0.001) for i in range(dimension)]


def _uuid() -> str:
    return str(uuid.uuid4())


def _next_port() -> int:
    """Get next unique port for a test."""
    global _PORT_COUNTER
    _PORT_COUNTER += 1
    return _PORT_COUNTER


def _connect_node() -> MembrainDistributedIndex:
    """Connect a single distributed node with a unique port."""
    port = _next_port()
    node = MembrainDistributedIndex.connect(
        dimension=DIMENSION,
        config={
            "listen_address": f"127.0.0.1:{port}",
            "replication_factor": 1,
        },
    )
    # Give the node time to fully initialize the listener and hash ring
    time.sleep(0.5)
    return node


# -- docs/integrations/memscaledb-distributed.mdx: Usage ---------------------


class TestDistributedUsage:
    """Mirrors the 'Usage' code block."""

    def test_connect_add_search(self):
        node1 = _connect_node()

        node1.add(_uuid(), _vector(0.1))

        query_vector = _vector(0.1)
        results = node1.search(query_vector, k=10)
        assert len(results) > 0
        for result in results:
            assert isinstance(result.id, str)
            assert isinstance(result.distance, float)

        info = node1.cluster_info()
        assert info.node_count >= 1
        assert info.replication_factor >= 1

        node1.close()


# -- docs/integrations/memscaledb-distributed.mdx: Adding and Removing -------


class TestDistributedAddRemove:
    """Mirrors the 'Adding and Removing Vectors' code block."""

    def test_add_and_remove(self):
        with _connect_node() as node:
            vid = _uuid()
            node.add(vid, _vector(0.1))
            assert len(node) == 1

            node.remove(vid)
            assert len(node) == 0


# -- docs/integrations/memscaledb-distributed.mdx: Search --------------------


class TestDistributedSearch:
    """Mirrors the 'Search' code block."""

    def test_search(self):
        with _connect_node() as node:
            for i in range(10):
                node.add(_uuid(), _vector(i * 0.1))

            query_vector = _vector(0.5)
            results = node.search(query_vector, k=10)
            assert len(results) > 0


# -- docs/integrations/memscaledb-distributed.mdx: Filtered Search -----------


class TestDistributedFilteredSearch:
    """Mirrors the 'Filtered Search' code block."""

    def test_search_with_filter(self):
        with _connect_node() as node:
            ids = [_uuid() for _ in range(10)]
            for i, vector_id in enumerate(ids):
                node.add(vector_id, _vector(float(i) / 10.0))

            query_vector = _vector(0.5)
            allowed = [ids[1], ids[3], ids[5]]
            results = node.search_with_filter(
                query_vector,
                k=10,
                allowed_ids=allowed,
            )
            for r in results:
                assert r.id in allowed


# -- docs/integrations/memscaledb-distributed.mdx: Batch Search --------------


class TestDistributedBatchSearch:
    """Mirrors the 'Batch Search' code block."""

    def test_batch_search(self):
        with _connect_node() as node:
            for i in range(20):
                node.add(_uuid(), _vector(i * 0.05))

            query1 = _vector(0.1)
            query2 = _vector(0.5)
            query3 = _vector(0.9)
            queries = [query1, query2, query3]
            batch_results = node.batch_search(queries, k=10)

            assert len(batch_results) == 3
            for results in batch_results:
                assert len(results) <= 10


# -- docs/integrations/memscaledb-distributed.mdx: Cluster Info --------------


class TestDistributedClusterInfo:
    """Mirrors the 'Cluster Info' code block."""

    def test_cluster_info_and_metrics(self):
        with _connect_node() as node:
            node.add(_uuid(), _vector(0.5))
            node.search(_vector(0.5), k=5)

            info = node.cluster_info()
            assert info.node_count >= 1
            assert isinstance(info.local_node.id, str)
            assert isinstance(info.local_node.address, str)
            assert info.replication_factor >= 1

            metrics = node.metrics()
            assert metrics.searches >= 1
            assert metrics.inserts >= 1
