"""Membrain sharded HNSW vector index client."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexMetrics, IndexSearchResult, ShardedIndexInfo, ShardStats


def _setup_sharded_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for sharded index API."""
    # Lifecycle
    lib.memscale_sharded_index_build.restype = ctypes.c_void_p
    lib.memscale_sharded_index_build.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.memscale_sharded_index_free.restype = None
    lib.memscale_sharded_index_free.argtypes = [ctypes.c_void_p]

    lib.memscale_sharded_index_load.restype = ctypes.c_void_p
    lib.memscale_sharded_index_load.argtypes = [ctypes.c_char_p]

    # Operations
    lib.memscale_sharded_index_add.restype = ctypes.c_int32
    lib.memscale_sharded_index_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.memscale_sharded_index_remove.restype = ctypes.c_int32
    lib.memscale_sharded_index_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
    ]

    lib.memscale_sharded_index_search.restype = ctypes.c_int32
    lib.memscale_sharded_index_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_sharded_index_search_with_filter.restype = ctypes.c_int32
    lib.memscale_sharded_index_search_with_filter.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_sharded_index_batch_search.restype = ctypes.c_int32
    lib.memscale_sharded_index_batch_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_sharded_index_rebalance.restype = ctypes.c_int32
    lib.memscale_sharded_index_rebalance.argtypes = [ctypes.c_void_p]

    # Info
    lib.memscale_sharded_index_len.restype = ctypes.c_int32
    lib.memscale_sharded_index_len.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_sharded_index_dimension.restype = ctypes.c_int32
    lib.memscale_sharded_index_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_sharded_index_info.restype = ctypes.c_int32
    lib.memscale_sharded_index_info.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_sharded_index_metrics.restype = ctypes.c_int32
    lib.memscale_sharded_index_metrics.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    # Persistence
    lib.memscale_sharded_index_save.restype = ctypes.c_int32
    lib.memscale_sharded_index_save.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]


class MembrainShardedIndex:
    """Python client for the Membrain sharded HNSW vector index.

    Uses centroid-based routing to partition vectors across independent HNSW
    shards. This scales HNSW to much larger datasets without requiring an
    external vector database.

    Usage::

        from membrain import MembrainShardedIndex
        import uuid

        ids = [str(uuid.uuid4()) for _ in range(1000)]
        vectors = [[float(i) * 0.01] * 128 for i in range(1000)]

        index = MembrainShardedIndex.build(
            dimension=128,
            ids=ids,
            vectors=vectors,
            config={"num_shards": 4, "nprobe": 2},
        )
        results = index.search([0.5] * 128, k=5)
        for r in results:
            print(r.id, r.score)
        index.close()
    """

    def __init__(self, *, _handle: Any, _lib: Any) -> None:
        """Private constructor. Use build() or load() class methods."""
        self._handle = _handle
        self._lib = _lib

    @classmethod
    def build(
        cls,
        dimension: int,
        ids: list[str],
        vectors: list[list[float]],
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> MembrainShardedIndex:
        """Build a sharded index from a set of vectors.

        Args:
            dimension: Vector dimension.
            ids: List of UUID strings identifying each vector.
            vectors: List of float vectors (must match dimension).
            config: Optional sharded index configuration with keys:
                - num_shards (int): Number of partitions.
                - nprobe (int): Shards to probe at query time.
                - overlap_factor (int): Boundary replication factor.
                - training_sample_size (int): Max vectors for k-means.
                - kmeans_iterations (int): K-means iterations.
                - hnsw_config (dict): Per-shard HNSW configuration.
            lib_path: Optional path to the shared library.
        """
        lib = load_library(lib_path)
        _setup_sharded_signatures(lib)

        full_config = {"dimension": dimension, **(config or {})}
        config_bytes = encode(json.dumps(full_config))
        ids_bytes = encode(json.dumps(ids))
        vectors_bytes = encode(json.dumps(vectors))

        handle = lib.memscale_sharded_index_build(
            config_bytes, ids_bytes, vectors_bytes,
        )

        if not handle:
            err = get_last_error(lib)
            raise MembrainError(f"failed to build sharded index: {err}")

        return cls(_handle=handle, _lib=lib)

    @classmethod
    def load(cls, data: str, *, lib_path: str | None = None) -> MembrainShardedIndex:
        """Load a sharded index from a base64-encoded string.

        Args:
            data: Base64-encoded MessagePack data from save().
            lib_path: Optional path to the shared library.
        """
        lib = load_library(lib_path)
        _setup_sharded_signatures(lib)

        handle = lib.memscale_sharded_index_load(encode(data))
        if not handle:
            err = get_last_error(lib)
            raise MembrainError(f"failed to load sharded index: {err}")

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
        """Add a vector to the sharded index.

        Args:
            id: UUID string identifying the vector.
            embedding: List of float values (must match index dimension).
        """
        emb_json = json.dumps(embedding)
        code = self._lib.memscale_sharded_index_add(
            self._handle, encode(id), encode(emb_json),
        )
        check(self._lib, code)

    def remove(self, id: str) -> None:
        """Remove a vector from the sharded index by ID."""
        code = self._lib.memscale_sharded_index_remove(
            self._handle, encode(id),
        )
        check(self._lib, code)

    def search(self, query: list[float], k: int = 10) -> list[IndexSearchResult]:
        """Search for the k nearest neighbors.

        Args:
            query: Query vector (must match index dimension).
            k: Number of results to return.

        Returns:
            List of IndexSearchResult sorted by score (highest first).
        """
        out = ctypes.c_char_p()
        query_json = json.dumps(query)
        code = self._lib.memscale_sharded_index_search(
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
        code = self._lib.memscale_sharded_index_search_with_filter(
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
        """Run multiple queries in parallel.

        Args:
            queries: List of query vectors.
            k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        out = ctypes.c_char_p()
        queries_json = json.dumps(queries)
        code = self._lib.memscale_sharded_index_batch_search(
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

    def rebalance(self) -> None:
        """Retrain centroids and redistribute vectors across shards."""
        code = self._lib.memscale_sharded_index_rebalance(self._handle)
        check(self._lib, code)

    # -------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------

    def __len__(self) -> int:
        """Get the number of unique vectors in the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_sharded_index_len(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_sharded_index_dimension(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    def info(self) -> ShardedIndexInfo:
        """Get per-shard statistics and balance metrics."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_sharded_index_info(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return ShardedIndexInfo(
            num_shards=data["num_shards"],
            total_vectors=data["total_vectors"],
            dimension=data["dimension"],
            shards=[
                ShardStats(
                    shard_index=s["shard_index"],
                    count=s["count"],
                )
                for s in data.get("shards", [])
            ],
            size_stddev=data.get("size_stddev", 0.0),
        )

    def metrics(self) -> IndexMetrics:
        """Get index performance metrics."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_sharded_index_metrics(
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
    # Persistence
    # -------------------------------------------------------------------

    def save(self) -> str:
        """Save index state to a base64-encoded string (MessagePack format)."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_sharded_index_save(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            return out.value.decode("utf-8")  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly release the underlying native index."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_sharded_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MembrainShardedIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
