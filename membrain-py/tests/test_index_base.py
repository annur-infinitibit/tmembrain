"""Unit tests for `MembrainIndex` (HNSW default)."""

from __future__ import annotations

import uuid

import pytest

from membrain import MembrainError, MembrainIndex


def uid() -> str:
    return str(uuid.uuid4())


def sample_vector(index: int, dimension: int) -> list[float]:
    return [(slot + index) * 0.01 for slot in range(dimension)]


def test_index_default_dimension() -> None:
    index = MembrainIndex(dimension=16)
    try:
        assert index.dimension == 16
        assert len(index) == 0
    finally:
        index.close()


def test_index_with_config_respects_dimension() -> None:
    index = MembrainIndex(config={"dimension": 12, "m": 8, "ef_construction": 50, "ef_search": 20})
    try:
        assert index.dimension == 12
    finally:
        index.close()


def test_index_invalid_config_raises() -> None:
    with pytest.raises(MembrainError):
        MembrainIndex(config={"dimension": "not-int"})


def test_index_add_wrong_dim_raises() -> None:
    index = MembrainIndex(dimension=8)
    try:
        with pytest.raises(MembrainError):
            index.add(uid(), [0.0] * 4)
    finally:
        index.close()


def test_index_search_empty_returns_empty() -> None:
    index = MembrainIndex(dimension=8)
    try:
        hits = index.search([0.0] * 8, k=5)
        assert hits == []
    finally:
        index.close()


def test_index_search_k_larger_than_size() -> None:
    index = MembrainIndex(dimension=8)
    try:
        for i in range(3):
            index.add(uid(), sample_vector(i, 8))
        hits = index.search(sample_vector(0, 8), k=10)
        assert len(hits) <= 3
    finally:
        index.close()


def test_index_batch_search_returns_array_per_query() -> None:
    index = MembrainIndex(dimension=8)
    try:
        for i in range(4):
            index.add(uid(), sample_vector(i, 8))
        queries = [sample_vector(0, 8), sample_vector(2, 8)]
        results = index.batch_search(queries, k=2)
        assert len(results) == 2
    finally:
        index.close()


def test_index_remove_bogus_uuid_raises() -> None:
    index = MembrainIndex(dimension=8)
    try:
        with pytest.raises(MembrainError):
            index.remove("not-a-uuid")
    finally:
        index.close()


def test_index_metrics_returns_object() -> None:
    index = MembrainIndex(dimension=8)
    try:
        metrics = index.metrics()
        assert metrics is not None
    finally:
        index.close()


def test_index_close_then_use_raises() -> None:
    index = MembrainIndex(dimension=8)
    index.close()
    with pytest.raises(MembrainError):
        index.add(uid(), [0.0] * 8)


def test_index_len_after_add() -> None:
    index = MembrainIndex(dimension=8)
    try:
        for i in range(3):
            index.add(uid(), sample_vector(i, 8))
        assert len(index) == 3
    finally:
        index.close()


def test_index_search_with_filter_respects_allowed_ids() -> None:
    index = MembrainIndex(dimension=8)
    try:
        ids = [uid() for _ in range(3)]
        for pos, memory_id in enumerate(ids):
            index.add(memory_id, sample_vector(pos, 8))
        hits = index.search_with_filter(sample_vector(0, 8), k=3, allowed_ids=[ids[1]])
        assert all(hit.id == ids[1] for hit in hits)
    finally:
        index.close()
