"""Tests for ConcurrentLshIndex."""

import threading
import uuid
import pytest

from membrain import ConcurrentLshIndex


def test_concurrent_lsh_index_create():
    """Test creating a concurrent LSH index."""
    index = ConcurrentLshIndex(dimension=128)
    assert index.dimension == 128
    assert len(index) == 0
    index.close()


def test_concurrent_lsh_index_add_and_search():
    """Test adding vectors and searching."""
    index = ConcurrentLshIndex(dimension=3)

    # Add vectors
    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())
    vec3_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])
    index.add(vec2_id, [0.0, 1.0, 0.0])
    index.add(vec3_id, [0.0, 0.0, 1.0])

    assert len(index) == 3

    # Search
    results = index.search([1.0, 0.1, 0.0], k=2)
    assert len(results) <= 2

    index.close()


def test_concurrent_lsh_index_remove():
    """Test removing vectors."""
    index = ConcurrentLshIndex(dimension=3)

    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])
    index.add(vec2_id, [0.0, 1.0, 0.0])

    assert len(index) == 2

    # Remove
    found = index.remove(vec1_id)
    assert found is True
    assert len(index) == 1

    # Try removing non-existent
    found = index.remove(vec1_id)
    assert found is False

    index.close()


def test_concurrent_lsh_index_clone():
    """Test cloning the index handle."""
    index = ConcurrentLshIndex(dimension=3)
    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])

    # Clone the handle
    clone = index.clone()

    # Both should see the same data
    assert len(index) == 1
    assert len(clone) == 1

    # Add from clone
    clone.add(vec2_id, [0.0, 1.0, 0.0])

    # Both should see the update
    assert len(index) == 2
    assert len(clone) == 2

    # Clean up both handles
    clone.close()
    index.close()


def test_concurrent_lsh_index_multi_threaded_insert():
    """Test concurrent inserts from multiple threads."""
    index = ConcurrentLshIndex(dimension=32)
    num_threads = 4
    vectors_per_thread = 25

    def insert_vectors(thread_id):
        for i in range(vectors_per_thread):
            vec_id = str(uuid.uuid4())
            vector = [float(thread_id + i)] * 32
            index.add(vec_id, vector)

    threads = [
        threading.Thread(target=insert_vectors, args=(i,))
        for i in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All inserts should succeed
    assert len(index) == num_threads * vectors_per_thread

    index.close()


def test_concurrent_lsh_index_multi_threaded_search():
    """Test concurrent searches from multiple threads."""
    index = ConcurrentLshIndex(dimension=16)

    # Pre-populate
    for i in range(100):
        index.add(str(uuid.uuid4()), [float(i % 16)] * 16)

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


def test_concurrent_lsh_index_mixed_operations():
    """Test mixed concurrent operations (add, remove, search)."""
    index = ConcurrentLshIndex(dimension=8)

    # Pre-populate
    vec_ids = []
    for i in range(50):
        vec_id = str(uuid.uuid4())
        index.add(vec_id, [float(i % 8)] * 8)
        vec_ids.append(vec_id)

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


def test_concurrent_lsh_index_context_manager():
    """Test using the index as a context manager."""
    with ConcurrentLshIndex(dimension=3) as index:
        index.add(str(uuid.uuid4()), [1.0, 0.0, 0.0])
        assert len(index) == 1
    # Index should be closed after exiting context


def test_concurrent_lsh_index_with_config():
    """Test creating index with custom configuration."""
    config = {
        "distance_metric": "Euclidean",
    }

    index = ConcurrentLshIndex(dimension=16, config=config)
    assert index.dimension == 16

    index.add(str(uuid.uuid4()), [1.0] * 16)
    results = index.search([1.0] * 16, k=1)
    assert len(results) <= 1

    index.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
