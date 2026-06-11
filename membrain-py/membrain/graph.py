"""Membrain graph memory layer client — backed by native PyO3 extension."""

from __future__ import annotations

import json
from typing import Any

from membrain._native import MembrainGraph as _NativeGraph

from .types import GraphPruningResult, GraphQueryResult, GraphScoredNode, GraphTraversalStep


class MembrainGraph:
    """Python client for the Membrain graph memory layer (true async via PyO3)."""

    def __init__(self, config: dict[str, Any] | None = None, *, lib_path: str | None = None) -> None:
        embedding_dim = config.get("embedding_dim") if config else None
        self._native = _NativeGraph(embedding_dim=embedding_dim)

    def _get_native(self) -> _NativeGraph:
        if not getattr(self, "_native", None):
            raise RuntimeError("graph is closed")
        return self._native

    def add_node(self, memory_id: str, embedding: list[float], confidence: float = 0.5) -> None:
        """Add a node to the graph with an embedding vector."""
        self._get_native().add_node(memory_id, embedding, confidence=confidence)

    def remove_node(self, memory_id: str) -> None:
        """Remove a node and all its incident edges."""
        self._get_native().remove_node(memory_id)

    async def query(self, embedding: list[float], max_hops: int = -1, top_k: int = 10) -> GraphQueryResult:
        """Run a multi-hop graph query using an embedding vector."""
        actual_hops = 2 if max_hops <= 0 else max_hops
        json_str = await self._get_native().query(embedding, max_hops=actual_hops, top_k=top_k)
        data = json.loads(json_str)
        return GraphQueryResult(
            nodes=[GraphScoredNode(**n) for n in data.get("nodes", [])],
            traversed_edges=[
                GraphTraversalStep(
                    from_id=s["from"],
                    to_id=s["to"],
                    edge_weight=s["edge_weight"],
                    attention_score=s["attention_score"],
                    hop=s["hop"],
                )
                for s in data.get("traversed_edges", [])
            ],
            hops_performed=data.get("hops_performed", 0),
            nodes_visited=data.get("nodes_visited", 0),
        )

    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self._get_native().node_count()

    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._get_native().edge_count()

    def prune(self) -> GraphPruningResult:
        """Manually trigger graph pruning."""
        json_str = self._get_native().prune()
        data = json.loads(json_str)
        return GraphPruningResult(
            edges_removed=data.get("edges_removed", 0),
            nodes_removed=data.get("nodes_removed", 0),
            edges_remaining=data.get("edges_remaining", 0),
            nodes_remaining=data.get("nodes_remaining", 0),
        )

    def save(self) -> str:
        """Save graph state to a base64-encoded string."""
        return self._get_native().save()

    @classmethod
    def load(cls, data: str, *, lib_path: str | None = None) -> MembrainGraph:
        """Load graph state from a base64-encoded string."""
        instance = object.__new__(cls)
        instance._native = _NativeGraph.load(data, lib_path=lib_path)
        return instance

    def close(self) -> None:
        """Explicitly release the underlying native graph."""
        if getattr(self, "_native", None):
            self._native.close()
            self._native = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MembrainGraph:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return repr(self._native) if hasattr(self, "_native") and self._native is not None else "MembrainGraph(closed)"
