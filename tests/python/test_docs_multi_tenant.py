"""
Test documentation examples from docs/memscaledb/multi-tenant.mdx

Exercises every Python code snippet from the multi-tenant documentation
page to ensure docs stay in sync with the implementation.
"""

import threading
import uuid

import pytest

from membrain import MultiTenantIndex

pytestmark = pytest.mark.docs

DIMENSION = 128


def _vector(seed: float, dimension: int = DIMENSION) -> list[float]:
    """Generate a deterministic vector from a seed value."""
    return [seed + (i * 0.001) for i in range(dimension)]


def _uuid() -> str:
    return str(uuid.uuid4())


# -- docs/memscaledb/multi-tenant.mdx: Quick Start ---------------------------


class TestMultiTenantQuickStart:
    """Mirrors the 'Quick Start' code block."""

    def test_basic_isolation(self):
        index = MultiTenantIndex(dimension=DIMENSION, index_type="hnsw")

        index.create_tenant("user-alice")
        index.create_tenant("user-bob")

        alice_vec = _uuid()
        index.add("user-alice", alice_vec, _vector(0.1))

        results = index.search("user-alice", _vector(0.1), k=10)
        assert len(results) == 1

        # Bob's namespace is empty -- complete isolation
        results = index.search("user-bob", _vector(0.1), k=10)
        assert len(results) == 0

        index.close()

    def test_context_manager(self):
        with MultiTenantIndex(dimension=DIMENSION, index_type="flat") as index:
            index.create_tenant("user-alice")
            index.add("user-alice", _uuid(), _vector(0.5))
            results = index.search("user-alice", _vector(0.5), k=5)
            assert len(results) == 1


# -- docs/memscaledb/multi-tenant.mdx: Tenant Lifecycle ---------------------


class TestTenantLifecycle:
    """Mirrors the 'Tenant Lifecycle' code block."""

    def test_create_check_list_delete(self):
        index = MultiTenantIndex(dimension=DIMENSION, index_type="flat")

        index.create_tenant("team-a")
        index.create_tenant("team-b")

        assert index.has_tenant("team-a")
        assert not index.has_tenant("team-c")

        tenants = index.list_tenants()
        assert "team-a" in tenants
        assert "team-b" in tenants
        assert tenants == sorted(tenants)  # alphabetically sorted

        count = index.tenant_count()
        assert count == 2

        found = index.delete_tenant("team-a")
        assert found is True

        found = index.delete_tenant("team-a")
        assert found is False  # already deleted

        assert not index.has_tenant("team-a")
        assert index.tenant_count() == 1

        index.close()

    def test_create_duplicate_tenant_raises(self):
        with MultiTenantIndex(dimension=DIMENSION) as index:
            index.create_tenant("dup")
            with pytest.raises(Exception):
                index.create_tenant("dup")

    def test_max_tenants_limit(self):
        with MultiTenantIndex(dimension=DIMENSION, max_tenants=2) as index:
            index.create_tenant("t1")
            index.create_tenant("t2")
            with pytest.raises(Exception):
                index.create_tenant("t3")  # exceeds limit


# -- docs/memscaledb/multi-tenant.mdx: Per-Tenant Operations ----------------


class TestPerTenantOperations:
    """Mirrors the 'Per-Tenant Operations' code block."""

    def test_add_len_search_remove(self):
        index = MultiTenantIndex(dimension=3, index_type="flat")
        index.create_tenant("alice")

        vec_id = _uuid()
        index.add("alice", vec_id, [1.0, 0.0, 0.0])

        count = index.tenant_len("alice")
        assert count == 1

        results = index.search("alice", [1.0, 0.1, 0.0], k=5)
        assert len(results) == 1
        assert results[0].id == vec_id
        assert isinstance(results[0].score, float)
        assert isinstance(results[0].distance, float)

        found = index.remove("alice", vec_id)
        assert found is True

        assert index.tenant_len("alice") == 0

        index.close()

    def test_remove_nonexistent_returns_false(self):
        with MultiTenantIndex(dimension=DIMENSION) as index:
            index.create_tenant("alice")
            found = index.remove("alice", _uuid())
            assert found is False

    def test_dimension_property(self):
        with MultiTenantIndex(dimension=DIMENSION) as index:
            assert index.dimension == DIMENSION

    def test_different_index_types(self):
        for index_type in ("flat", "hnsw", "lsh"):
            with MultiTenantIndex(dimension=DIMENSION, index_type=index_type) as index:
                index.create_tenant("tenant")
                index.add("tenant", _uuid(), _vector(0.5))
                results = index.search("tenant", _vector(0.5), k=5)
                assert len(results) >= 1


# -- docs/memscaledb/multi-tenant.mdx: Concurrency Model --------------------


class TestConcurrencyModel:
    """Mirrors the 'Concurrency Model' code block."""

    def test_parallel_tenants_no_cross_contamination(self):
        index = MultiTenantIndex(dimension=DIMENSION, index_type="hnsw")

        tenant_ids = [f"tenant-{i}" for i in range(4)]
        for tenant_id in tenant_ids:
            index.create_tenant(tenant_id)

        results_per_tenant = {}
        errors = []

        def worker(tenant_id: str) -> None:
            try:
                for _ in range(10):
                    index.add(tenant_id, _uuid(), _vector(0.5))
                results = index.search(tenant_id, _vector(0.5), k=10)
                results_per_tenant[tenant_id] = len(results)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(tenant_id,))
            for tenant_id in tenant_ids
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"Thread errors: {errors}"

        for tenant_id in tenant_ids:
            assert results_per_tenant[tenant_id] > 0

        # Verify isolation: each tenant's len == 10 (no cross-contamination)
        for tenant_id in tenant_ids:
            assert index.tenant_len(tenant_id) == 10

        index.close()


# -- docs/memscaledb/multi-tenant.mdx: LLM Memory Example ------------------


class TestLlmMemoryExample:
    """Mirrors the 'LLM Memory Example' code block."""

    def test_per_user_memory(self):
        memory = MultiTenantIndex(dimension=DIMENSION, index_type="hnsw")

        def on_user_signup(user_id: str) -> None:
            memory.create_tenant(user_id)

        def store_memory(
            user_id: str, embedding: list[float],
        ) -> str:
            memory_id = _uuid()
            memory.add(user_id, memory_id, embedding)
            return memory_id

        def recall_memories(
            user_id: str, query_embedding: list[float], k: int = 5,
        ) -> list:
            return memory.search(user_id, query_embedding, k=k)

        def on_user_delete(user_id: str) -> None:
            memory.delete_tenant(user_id)

        # Simulate two users
        on_user_signup("user-1")
        on_user_signup("user-2")

        mem_id = store_memory("user-1", _vector(0.1))
        store_memory("user-1", _vector(0.2))
        store_memory("user-2", _vector(0.9))

        results = recall_memories("user-1", _vector(0.1), k=5)
        assert len(results) == 2
        assert any(r.id == mem_id for r in results)

        # user-2 has their own separate memory
        results = recall_memories("user-2", _vector(0.9), k=5)
        assert len(results) == 1

        on_user_delete("user-1")
        assert not memory.has_tenant("user-1")
        assert memory.has_tenant("user-2")

        memory.close()


# -- docs/memscaledb/multi-tenant.mdx: Configuration Reference --------------


class TestConfiguration:
    """Covers the index_config parameter from the Configuration Reference."""

    def test_hnsw_with_index_config(self):
        index = MultiTenantIndex(
            dimension=DIMENSION,
            index_type="hnsw",
            max_tenants=1000,
            index_config={"m": 16, "ef_construction": 200},
        )

        index.create_tenant("tenant-1")
        index.add("tenant-1", _uuid(), _vector(0.5))
        results = index.search("tenant-1", _vector(0.5), k=5)
        assert len(results) == 1
        assert index.dimension == DIMENSION

        index.close()
