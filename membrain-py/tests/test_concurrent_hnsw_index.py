"""Tests for ConcurrentHnswIndex."""

import threading
import uuid
import pytest

from membrain import ConcurrentHnswIndex


def test_concurrent_hnsw_index_create():
    """Test creating a concurrent HNSW index."""
    index = ConcurrentHnswIndex(dimension=128)
    assert index.dimension == 128
    assert len(index) == 0
    index.close()


def test_concurrent_hnsw_index_add_and_search():
    """Test adding vectors and searching."""
    index = ConcurrentHnswIndex(dimension=3)

    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())
    vec3_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])
    index.add(vec2_id, [0.0, 1.0, 0.0])
    index.add(vec3_id, [0.0, 0.0, 1.0])

    assert len(index) == 3

    results = index.search([1.0, 0.1, 0.0], k=2)
    assert len(results) == 2
    assert results[0].id == vec1_id

    index.close()


def test_concurrent_hnsw_index_remove():
    """Test removing vectors."""
    index = ConcurrentHnswIndex(dimension=3)

    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])
    index.add(vec2_id, [0.0, 1.0, 0.0])

    assert len(index) == 2

    found = index.remove(vec1_id)
    assert found is True
    assert len(index) == 1

    found = index.remove(vec1_id)
    assert found is False

    index.close()


def test_concurrent_hnsw_index_clone():
    """Test cloning the index handle."""
    index = ConcurrentHnswIndex(dimension=3)
    vec1_id = str(uuid.uuid4())
    vec2_id = str(uuid.uuid4())

    index.add(vec1_id, [1.0, 0.0, 0.0])

    clone = index.clone()

    assert len(index) == 1
    assert len(clone) == 1

    clone.add(vec2_id, [0.0, 1.0, 0.0])

    assert len(index) == 2
    assert len(clone) == 2

    clone.close()
    index.close()


def test_concurrent_hnsw_index_multi_threaded_insert():
    """Test concurrent inserts from multiple threads."""
    index = ConcurrentHnswIndex(dimension=32)
    num_threads = 4
    vectors_per_thread = 25

    def insert_vectors(thread_id):
        for i in range(vectors_per_thread):
            vec_id = str(uuid.uuid4())
            vector = [float(thread_id + i) / 100.0] * 32
            index.add(vec_id, vector)

    threads = [
        threading.Thread(target=insert_vectors, args=(i,))
        for i in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(index) == num_threads * vectors_per_thread

    index.close()


def test_concurrent_hnsw_index_multi_threaded_search():
    """Test concurrent searches from multiple threads."""
    index = ConcurrentHnswIndex(dimension=16)

    for i in range(100):
        index.add(str(uuid.uuid4()), [float(i % 16) / 16.0] * 16)

    search_results = []
    lock = threading.Lock()

    def search_vectors():
        query = [0.5] * 16
        results = index.search(query, k=10)
        with lock:
            search_results.append(len(results))

    threads = [threading.Thread(target=search_vectors) for _ in range(8)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(search_results) == 8
    for count in search_results:
        assert count == 10

    index.close()


def test_concurrent_hnsw_index_mixed_operations():
    """Test mixed concurrent operations (add, remove, search)."""
    index = ConcurrentHnswIndex(dimension=8)

    vec_ids = []
    for i in range(100):
        vec_id = str(uuid.uuid4())
        index.add(vec_id, [float(i % 8) / 8.0] * 8)
        vec_ids.append(vec_id)

    results = {"inserts": 0, "removes": 0, "searches": 0}
    lock = threading.Lock()

    def insert_vectors():
        for i in range(10):
            index.add(str(uuid.uuid4()), [0.5] * 8)
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
            index.search([0.5] * 8, k=5)
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

    assert results["inserts"] == 20
    assert results["removes"] == 10
    assert results["searches"] == 40

    assert len(index) > 0

    index.close()


def test_concurrent_hnsw_index_context_manager():
    """Test using the index as a context manager."""
    with ConcurrentHnswIndex(dimension=3) as index:
        index.add(str(uuid.uuid4()), [1.0, 0.0, 0.0])
        assert len(index) == 1


def test_concurrent_hnsw_index_with_config():
    """Test creating index with custom configuration."""
    config = {
        "m": 8,
        "ef_construction": 50,
        "ef_search": 50,
        "distance_metric": "Euclidean",
    }

    index = ConcurrentHnswIndex(dimension=16, config=config)
    assert index.dimension == 16

    for _ in range(50):
        index.add(str(uuid.uuid4()), [0.5] * 16)

    results = index.search([0.5] * 16, k=5)
    assert len(results) == 5

    index.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
