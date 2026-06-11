"""Tests for MultiTenantIndex."""

import threading
import uuid

import pytest

from membrain import MultiTenantIndex


def test_create_and_list_tenants():
    """Test creating tenants and listing them."""
    index = MultiTenantIndex(dimension=32, index_type="flat")
    index.create_tenant("alice")
    index.create_tenant("bob")

    assert index.tenant_count() == 2
    assert index.has_tenant("alice")
    assert index.has_tenant("bob")
    assert not index.has_tenant("charlie")

    tenants = index.list_tenants()
    assert tenants == ["alice", "bob"]

    index.close()


def test_delete_tenant():
    """Test deleting a tenant."""
    index = MultiTenantIndex(dimension=32, index_type="flat")
    index.create_tenant("alice")
    assert index.has_tenant("alice")

    found = index.delete_tenant("alice")
    assert found is True
    assert not index.has_tenant("alice")

    found = index.delete_tenant("alice")
    assert found is False

    index.close()


def test_add_search_per_tenant():
    """Test add and search within a single tenant."""
    index = MultiTenantIndex(dimension=3, index_type="flat")
    index.create_tenant("alice")

    vec_id = str(uuid.uuid4())
    index.add("alice", vec_id, [1.0, 0.0, 0.0])
    assert index.tenant_len("alice") == 1

    results = index.search("alice", [1.0, 0.1, 0.0], k=1)
    assert len(results) == 1
    assert results[0].id == vec_id

    index.close()


def test_remove_from_tenant():
    """Test removing a vector from a tenant."""
    index = MultiTenantIndex(dimension=3, index_type="flat")
    index.create_tenant("alice")

    vec_id = str(uuid.uuid4())
    index.add("alice", vec_id, [1.0, 0.0, 0.0])
    assert index.tenant_len("alice") == 1

    found = index.remove("alice", vec_id)
    assert found is True
    assert index.tenant_len("alice") == 0

    index.close()


def test_cross_tenant_isolation():
    """Test that tenants are isolated from each other."""
    index = MultiTenantIndex(dimension=3, index_type="flat")
    index.create_tenant("alice")
    index.create_tenant("bob")

    vec_id = str(uuid.uuid4())
    index.add("alice", vec_id, [1.0, 0.0, 0.0])

    # Alice can find it
    results = index.search("alice", [1.0, 0.0, 0.0], k=1)
    assert len(results) == 1

    # Bob cannot
    results = index.search("bob", [1.0, 0.0, 0.0], k=1)
    assert len(results) == 0

    index.close()


def test_max_tenants():
    """Test max_tenants limit enforcement."""
    index = MultiTenantIndex(dimension=32, index_type="flat", max_tenants=2)
    index.create_tenant("a")
    index.create_tenant("b")

    with pytest.raises(Exception):
        index.create_tenant("c")

    index.close()


def test_multi_threaded_different_tenants():
    """Test concurrent operations across different tenants."""
    index = MultiTenantIndex(dimension=16, index_type="flat")
    num_tenants = 4
    vectors_per_tenant = 25

    for i in range(num_tenants):
        index.create_tenant(f"tenant-{i}")

    def insert_vectors(tenant_id):
        for _ in range(vectors_per_tenant):
            vec_id = str(uuid.uuid4())
            index.add(tenant_id, vec_id, [1.0] * 16)

    threads = [
        threading.Thread(target=insert_vectors, args=(f"tenant-{i}",))
        for i in range(num_tenants)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(num_tenants):
        assert index.tenant_len(f"tenant-{i}") == vectors_per_tenant

    index.close()


def test_context_manager():
    """Test using the index as a context manager."""
    with MultiTenantIndex(dimension=3, index_type="flat") as index:
        index.create_tenant("alice")
        index.add("alice", str(uuid.uuid4()), [1.0, 0.0, 0.0])
        assert index.tenant_len("alice") == 1


def test_hnsw_index_type():
    """Test using HNSW as the per-tenant index type."""
    index = MultiTenantIndex(dimension=8, index_type="hnsw")
    index.create_tenant("alice")

    vec_id = str(uuid.uuid4())
    index.add("alice", vec_id, [0.5] * 8)

    results = index.search("alice", [0.5] * 8, k=1)
    assert len(results) == 1
    assert results[0].id == vec_id

    index.close()


def test_dimension_property():
    """Test that dimension is reported correctly."""
    index = MultiTenantIndex(dimension=768, index_type="flat")
    assert index.dimension == 768
    index.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
