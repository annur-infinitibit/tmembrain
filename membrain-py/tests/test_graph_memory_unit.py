"""Unit tests for the MembrainGraph Python wrapper.

Requires the Rust FFI shared library (libmembrain_ffi.so/.dylib/.dll).
"""

from __future__ import annotations

import uuid

import pytest

from membrain import MembrainError, MembrainGraph


def uid() -> str:
    return str(uuid.uuid4())


def test_graph_new_without_config_initialises() -> None:
    graph = MembrainGraph()
    try:
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
    finally:
        graph.close()


def test_graph_new_with_config_initialises() -> None:
    graph = MembrainGraph(config={"embedding_dim": 32})
    try:
        assert graph.node_count() == 0
    finally:
        graph.close()


def test_graph_new_with_invalid_config_raises() -> None:
    with pytest.raises(MembrainError):
        MembrainGraph(config={"embedding_dim": "not-an-int"})


def test_graph_node_count_after_add_and_remove() -> None:
    graph = MembrainGraph(config={"embedding_dim": 16})
    memory_id = uid()
    try:
        graph.add_node(memory_id=memory_id, embedding=[0.1] * 16, confidence=0.8)
        assert graph.node_count() == 1
        graph.remove_node(memory_id)
        assert graph.node_count() == 0
    finally:
        graph.close()


def test_graph_remove_missing_node_raises() -> None:
    graph = MembrainGraph(config={"embedding_dim": 16})
    try:
        with pytest.raises(MembrainError):
            graph.remove_node(uid())
    finally:
        graph.close()


def test_graph_query_returns_result() -> None:
    graph = MembrainGraph(config={"embedding_dim": 16})
    try:
        graph.add_node(memory_id=uid(), embedding=[0.1] * 16, confidence=0.9)
        graph.add_node(memory_id=uid(), embedding=[0.2] * 16, confidence=0.7)
        result = graph.query(embedding=[0.1] * 16, max_hops=1, top_k=2)
        assert result is not None
    finally:
        graph.close()


def test_graph_save_load_roundtrip() -> None:
    graph = MembrainGraph(config={"embedding_dim": 16})
    try:
        graph.add_node(memory_id=uid(), embedding=[0.3] * 16, confidence=0.5)
        data = graph.save()
        assert isinstance(data, str)
        assert data
    finally:
        graph.close()

    restored = MembrainGraph.load(data)
    try:
        assert restored.node_count() == 1
    finally:
        restored.close()


def test_graph_close_then_use_raises() -> None:
    graph = MembrainGraph(config={"embedding_dim": 8})
    graph.close()
    with pytest.raises(MembrainError):
        graph.node_count()


def test_graph_context_manager_closes_on_exit() -> None:
    with MembrainGraph(config={"embedding_dim": 8}) as graph:
        graph.add_node(memory_id=uid(), embedding=[0.0] * 8, confidence=0.1)
        assert graph.node_count() == 1
    with pytest.raises(MembrainError):
        graph.node_count()


def test_graph_prune_returns_pruning_result() -> None:
    graph = MembrainGraph(config={"embedding_dim": 8})
    try:
        graph.add_node(memory_id=uid(), embedding=[0.0] * 8, confidence=0.1)
        result = graph.prune()
        assert result is not None
    finally:
        graph.close()


def test_graph_add_node_with_wrong_dim_raises() -> None:
    graph = MembrainGraph(config={"embedding_dim": 8})
    try:
        with pytest.raises(MembrainError):
            graph.add_node(memory_id=uid(), embedding=[0.0] * 4, confidence=0.5)
    finally:
        graph.close()
