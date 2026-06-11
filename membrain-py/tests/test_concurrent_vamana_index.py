"""Tests for ConcurrentVamanaIndex."""

import threading
import uuid
import pytest

from membrain import ConcurrentVamanaIndex


def test_concurrent_vamana_index_build():
    """Test building a concurrent Vamana index."""
    ids = [str(uuid.uuid4()) for _ in range(100)]
    vectors = [[float(i % 3)] * 3 for i in range(100)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=3)
    assert index.dimension == 3
    assert len(index) == 100
    index.close()


def test_concurrent_vamana_index_search():
    """Test searching in a concurrent Vamana index."""
    ids = [str(uuid.uuid4()) for _ in range(100)]
    vectors = [[float(i % 3)] * 3 for i in range(100)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=3)

    results = index.search([1.0, 0.1, 0.0], k=10)
    assert len(results) <= 10

    index.close()


def test_concurrent_vamana_index_add_and_remove():
    """Test adding and removing vectors."""
    ids = [str(uuid.uuid4()) for _ in range(50)]
    vectors = [[float(i % 3)] * 3 for i in range(50)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=3)
    initial_len = len(index)

    # Add new vector
    new_id = str(uuid.uuid4())
    index.add(new_id, [1.0, 2.0, 3.0])
    assert len(index) == initial_len + 1

    # Remove vector
    found = index.remove(new_id)
    assert found is True
    assert len(index) == initial_len

    # Try removing non-existent
    found = index.remove(new_id)
    assert found is False

    index.close()


def test_concurrent_vamana_index_clone():
    """Test cloning the index handle."""
    ids = [str(uuid.uuid4()) for _ in range(50)]
    vectors = [[float(i % 3)] * 3 for i in range(50)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=3)

    # Clone the handle
    clone = index.clone()

    # Both should see the same data
    assert len(index) == 50
    assert len(clone) == 50

    # Add from clone
    new_id = str(uuid.uuid4())
    clone.add(new_id, [1.0, 2.0, 3.0])

    # Both should see the update
    assert len(index) == 51
    assert len(clone) == 51

    # Clean up both handles
    clone.close()
    index.close()


def test_concurrent_vamana_index_multi_threaded_search():
    """Test concurrent searches from multiple threads."""
    ids = [str(uuid.uuid4()) for _ in range(100)]
    vectors = [[float(i % 16)] * 16 for i in range(100)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=16)

    search_results = []
    lock = threading.Lock()

    def search_vectors():
        query = [1.0] * 16
        results = index.search(query, k=10)
        with lock:
            search_results.append(len(results))

    threads = [threading.Thread(target=search_vectors) for _ in range(8)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All searches should succeed
    assert len(search_results) == 8
    for count in search_results:
        assert count <= 10

    index.close()


def test_concurrent_vamana_index_mixed_operations():
    """Test mixed concurrent operations (add, remove, search)."""
    ids = [str(uuid.uuid4()) for _ in range(50)]
    vectors = [[float(i % 8)] * 8 for i in range(50)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=8)

    vec_ids = ids.copy()
    results = {"inserts": 0, "removes": 0, "searches": 0}
    lock = threading.Lock()

    def insert_vectors():
        for i in range(10):
            index.add(str(uuid.uuid4()), [1.0] * 8)
        with lock:
            results["inserts"] += 10

    def remove_vectors():
        for i in range(5):
            if i < len(vec_ids):
                index.remove(vec_ids[i])
        with lock:
            results["removes"] += 5

    def search_vectors():
        for _ in range(10):
            index.search([1.0] * 8, k=5)
        with lock:
            results["searches"] += 10

    threads = []
    threads.extend([threading.Thread(target=insert_vectors) for _ in range(2)])
    threads.extend([threading.Thread(target=remove_vectors) for _ in range(2)])
    threads.extend([threading.Thread(target=search_vectors) for _ in range(4)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify operations completed
    assert results["inserts"] == 20
    assert results["removes"] == 10
    assert results["searches"] == 40

    # Index should have vectors
    assert len(index) > 0

    index.close()


def test_concurrent_vamana_index_context_manager():
    """Test using the index as a context manager."""
    ids = [str(uuid.uuid4()) for _ in range(50)]
    vectors = [[float(i % 3)] * 3 for i in range(50)]

    with ConcurrentVamanaIndex.build(ids, vectors, dimension=3) as index:
        assert len(index) == 50
    # Index should be closed after exiting context


def test_concurrent_vamana_index_with_config():
    """Test creating index with custom configuration."""
    config = {
        "max_degree": 32,
        "distance_metric": "Euclidean",
    }

    ids = [str(uuid.uuid4()) for _ in range(100)]
    vectors = [[float(i % 16)] * 16 for i in range(100)]

    index = ConcurrentVamanaIndex.build(ids, vectors, dimension=16, config=config)
    assert index.dimension == 16

    results = index.search([1.0] * 16, k=5)
    assert len(results) <= 5

    index.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
