"""Membrain distributed HNSW vector index client."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import ClusterInfo, IndexMetrics, IndexSearchResult, NodeInfo


def _setup_distributed_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for distributed index API."""
    # Lifecycle
    lib.memscale_distributed_index_new.restype = ctypes.c_void_p
    lib.memscale_distributed_index_new.argtypes = [ctypes.c_char_p]

    lib.memscale_distributed_index_free.restype = None
    lib.memscale_distributed_index_free.argtypes = [ctypes.c_void_p]

    # Operations
    lib.memscale_distributed_index_add.restype = ctypes.c_int32
    lib.memscale_distributed_index_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.memscale_distributed_index_remove.restype = ctypes.c_int32
    lib.memscale_distributed_index_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
    ]

    lib.memscale_distributed_index_search.restype = ctypes.c_int32
    lib.memscale_distributed_index_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_distributed_index_search_with_filter.restype = ctypes.c_int32
    lib.memscale_distributed_index_search_with_filter.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_distributed_index_batch_search.restype = ctypes.c_int32
    lib.memscale_distributed_index_batch_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    # Info
    lib.memscale_distributed_index_cluster_info.restype = ctypes.c_int32
    lib.memscale_distributed_index_cluster_info.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_distributed_index_len.restype = ctypes.c_int32
    lib.memscale_distributed_index_len.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_distributed_index_dimension.restype = ctypes.c_int32
    lib.memscale_distributed_index_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_distributed_index_metrics.restype = ctypes.c_int32
    lib.memscale_distributed_index_metrics.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    # Lifecycle
    lib.memscale_distributed_index_shutdown.restype = ctypes.c_int32
    lib.memscale_distributed_index_shutdown.argtypes = [ctypes.c_void_p]


class MembrainDistributedIndex:
    """Python client for the Membrain distributed HNSW vector index.

    Spreads vectors across multiple nodes using consistent hashing with
    SWIM-based gossip for membership and failure detection.

    Usage::

        from membrain import MembrainDistributedIndex

        index = MembrainDistributedIndex.connect(
            dimension=128,
            config={"listen_address": "127.0.0.1:9400"},
        )
        index.add("some-uuid", [0.1, 0.2, ...])
        results = index.search([0.5] * 128, k=5)
        for r in results:
            print(r.id, r.score)
        index.close()
    """

    def __init__(self, *, _handle: Any, _lib: Any) -> None:
        """Private constructor. Use connect() class method."""
        self._handle = _handle
        self._lib = _lib

    @classmethod
    def connect(
        cls,
        dimension: int,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> MembrainDistributedIndex:
        """Create a distributed index node and join the cluster.

        Args:
            dimension: Vector dimension.
            config: Optional distributed index configuration with keys:
                - listen_address (str): Address to listen on (default: "127.0.0.1:9400").
                - seed_nodes (list[str]): Seed node addresses to join.
                - replication_factor (int): Number of replicas (default: 3).
                - virtual_nodes_per_node (int): Virtual nodes per physical node (default: 128).
                - rpc_timeout (int): RPC timeout in milliseconds (default: 5000).
                - gossip_interval (int): Gossip interval in milliseconds (default: 1000).
                - suspicion_timeout (int): Suspicion timeout in milliseconds (default: 10000).
                - max_connections_per_peer (int): Max connections per peer (default: 8).
                - local_hnsw_config (dict): Per-node HNSW configuration.
            lib_path: Optional path to the shared library.
        """
        lib = load_library(lib_path)
        _setup_distributed_signatures(lib)

        full_config = {"dimension": dimension, **(config or {})}
        config_bytes = encode(json.dumps(full_config))

        handle = lib.memscale_distributed_index_new(config_bytes)

        if not handle:
            err = get_last_error(lib)
            raise MembrainError(f"failed to create distributed index: {err}")

        return cls(_handle=handle, _lib=lib)

    def _parse_search_results(self, data: list[dict[str, Any]]) -> list[IndexSearchResult]:
        return [
            IndexSearchResult(
                id=r["id"],
                score=r["score"],
                distance=r["distance"],
            )
            for r in data
        ]

    # -------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------

    def add(self, id: str, embedding: list[float]) -> None:
        """Add a vector to the distributed index.

        Args:
            id: UUID string identifying the vector.
            embedding: List of float values (must match index dimension).
        """
        emb_json = json.dumps(embedding)
        code = self._lib.memscale_distributed_index_add(
            self._handle, encode(id), encode(emb_json),
        )
        check(self._lib, code)

    def remove(self, id: str) -> None:
        """Remove a vector from the distributed index by ID."""
        code = self._lib.memscale_distributed_index_remove(
            self._handle, encode(id),
        )
        check(self._lib, code)

    def search(self, query: list[float], k: int = 10) -> list[IndexSearchResult]:
        """Search for the k nearest neighbors across the cluster.

        Args:
            query: Query vector (must match index dimension).
            k: Number of results to return.

        Returns:
            List of IndexSearchResult sorted by score (highest first).
        """
        out = ctypes.c_char_p()
        query_json = json.dumps(query)
        code = self._lib.memscale_distributed_index_search(
            self._handle,
            encode(query_json),
            ctypes.c_uint32(k),
            ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return self._parse_search_results(data)

    def search_with_filter(
        self,
        query: list[float],
        k: int = 10,
        allowed_ids: list[str] | None = None,
    ) -> list[IndexSearchResult]:
        """Search with an ID filter (only return results in allowed_ids).

        Args:
            query: Query vector.
            k: Number of results to return.
            allowed_ids: List of UUID strings to allow in results.

        Returns:
            List of IndexSearchResult.
        """
        out = ctypes.c_char_p()
        query_json = json.dumps(query)
        ids_json = json.dumps(allowed_ids or [])
        code = self._lib.memscale_distributed_index_search_with_filter(
            self._handle,
            encode(query_json),
            ctypes.c_uint32(k),
            encode(ids_json),
            ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return self._parse_search_results(data)

    def batch_search(
        self, queries: list[list[float]], k: int = 10,
    ) -> list[list[IndexSearchResult]]:
        """Run multiple queries across the cluster.

        Args:
            queries: List of query vectors.
            k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        out = ctypes.c_char_p()
        queries_json = json.dumps(queries)
        code = self._lib.memscale_distributed_index_batch_search(
            self._handle,
            encode(queries_json),
            ctypes.c_uint32(k),
            ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return [self._parse_search_results(batch) for batch in data]

    # -------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------

    def __len__(self) -> int:
        """Get the number of vectors on the local node."""
        out = ctypes.c_int64()
        code = self._lib.memscale_distributed_index_len(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_distributed_index_dimension(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    def cluster_info(self) -> ClusterInfo:
        """Get information about the distributed cluster."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_distributed_index_cluster_info(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)

        local = data.get("local_node", {})
        return ClusterInfo(
            node_count=data["node_count"],
            local_node=NodeInfo(
                id=local.get("id", ""),
                address=local.get("address", ""),
                dimension=local.get("dimension", 0),
            ),
            replication_factor=data["replication_factor"],
            members=data.get("members", []),
        )

    def metrics(self) -> IndexMetrics:
        """Get index performance metrics."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_distributed_index_metrics(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return IndexMetrics(
            searches=data.get("searches", 0),
            inserts=data.get("inserts", 0),
            deletes=data.get("deletes", 0),
            compactions=data.get("compactions", 0),
            cache_hits=data.get("cache_hits", 0),
            cache_misses=data.get("cache_misses", 0),
            distance_computations=data.get("distance_computations", 0),
        )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the node and release the underlying native index."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_distributed_index_shutdown(self._handle)
            self._lib.memscale_distributed_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MembrainDistributedIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
