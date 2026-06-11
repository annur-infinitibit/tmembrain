"""Membrain HNSW vector index clients."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexMetrics, IndexSearchResult


class MembrainIndex:
    """Python client for the Membrain HNSW vector index.

    Uses ctypes to call the shared C library (libmembrain_ffi).
    This is a standalone vector index -- not tied to MembrainClient.

    Usage::

        from membrain import MembrainIndex

        index = MembrainIndex(dimension=128)
        index.add("550e8400-...", [0.1, 0.2, ...])
        results = index.search([0.1, 0.2, ...], k=5)
        for r in results:
            print(r.id, r.score)
        index.close()
    """

    def __init__(
        self,
        dimension: int = 1536,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> None:
        """Initialize an HNSW vector index.

        Args:
            dimension: Vector dimension (ignored if config contains "dimension").
            config: Optional HNSW configuration dictionary. If provided, must include
                "dimension". Other fields: "m", "ef_construction", "ef_search",
                "max_ef_search", "distance_metric", "cache_config".
            lib_path: Optional path to the shared library.
        """
        self._lib = load_library(lib_path)
        self._has_gpu_support = False
        self._setup_index_signatures()
        self._setup_gpu_signatures()

        if config:
            if "dimension" not in config:
                config = {**config, "dimension": dimension}
            config_bytes = encode(json.dumps(config))
            self._handle = self._lib.memscale_index_new_with_config(config_bytes)
        else:
            self._handle = self._lib.memscale_index_new(ctypes.c_uint32(dimension))

        if not self._handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to create index: {err}")

    def _setup_index_signatures(self) -> None:
        """Declare C function signatures for index API."""
        lib = self._lib

        # Lifecycle
        lib.memscale_index_new.restype = ctypes.c_void_p
        lib.memscale_index_new.argtypes = [ctypes.c_uint32]

        lib.memscale_index_new_with_config.restype = ctypes.c_void_p
        lib.memscale_index_new_with_config.argtypes = [ctypes.c_char_p]

        lib.memscale_index_free.restype = None
        lib.memscale_index_free.argtypes = [ctypes.c_void_p]

        # Operations
        lib.memscale_index_add.restype = ctypes.c_int32
        lib.memscale_index_add.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ]

        lib.memscale_index_remove.restype = ctypes.c_int32
        lib.memscale_index_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.memscale_index_search.restype = ctypes.c_int32
        lib.memscale_index_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        lib.memscale_index_search_with_filter.restype = ctypes.c_int32
        lib.memscale_index_search_with_filter.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
        ]

        lib.memscale_index_batch_search.restype = ctypes.c_int32
        lib.memscale_index_batch_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        # Info
        lib.memscale_index_len.restype = ctypes.c_int32
        lib.memscale_index_len.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
        ]

        lib.memscale_index_dimension.restype = ctypes.c_int32
        lib.memscale_index_dimension.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
        ]

        # Metrics
        lib.memscale_index_metrics.restype = ctypes.c_int32
        lib.memscale_index_metrics.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
        ]

        # Configuration
        lib.memscale_index_enable_pq.restype = ctypes.c_int32
        lib.memscale_index_enable_pq.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.memscale_index_enable_wal.restype = ctypes.c_int32
        lib.memscale_index_enable_wal.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.memscale_index_compact.restype = ctypes.c_int32
        lib.memscale_index_compact.argtypes = [ctypes.c_void_p]

        # Persistence
        lib.memscale_index_save.restype = ctypes.c_int32
        lib.memscale_index_save.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
        ]

        lib.memscale_index_load.restype = ctypes.c_void_p
        lib.memscale_index_load.argtypes = [ctypes.c_char_p]

        lib.memscale_index_save_binary.restype = ctypes.c_int32
        lib.memscale_index_save_binary.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.memscale_index_load_binary.restype = ctypes.c_void_p
        lib.memscale_index_load_binary.argtypes = [ctypes.c_char_p]

        lib.memscale_index_mmap_free.restype = None
        lib.memscale_index_mmap_free.argtypes = [ctypes.c_void_p]

        lib.memscale_index_mmap_search.restype = ctypes.c_int32
        lib.memscale_index_mmap_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        lib.memscale_index_mmap_len.restype = ctypes.c_int32
        lib.memscale_index_mmap_len.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
        ]

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
        """Add a vector to the index.

        Args:
            id: UUID string identifying the vector.
            embedding: List of float values (must match index dimension).
        """
        emb_json = json.dumps(embedding)
        code = self._lib.memscale_index_add(
            self._handle,
            encode(id),
            encode(emb_json),
        )
        check(self._lib, code)

    def remove(self, id: str) -> None:
        """Remove a vector from the index by ID."""
        code = self._lib.memscale_index_remove(self._handle, encode(id))
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
        code = self._lib.memscale_index_search(
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
        code = self._lib.memscale_index_search_with_filter(
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
        self, queries: list[list[float]], k: int = 10
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
        code = self._lib.memscale_index_batch_search(
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
        """Get the number of active vectors in the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_index_len(self._handle, ctypes.byref(out))
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_index_dimension(self._handle, ctypes.byref(out))
        check(self._lib, code)
        return out.value

    def metrics(self) -> IndexMetrics:
        """Get index performance metrics."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_index_metrics(self._handle, ctypes.byref(out))
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
    # GPU support
    # -------------------------------------------------------------------

    def _setup_gpu_signatures(self) -> None:
        """Attempt to bind GPU FFI function signatures.

        If the library was built without the `gpu` feature, the symbols will
        not exist and ``_has_gpu_support`` will remain False.
        """
        lib = self._lib
        try:
            lib.memscale_gpu_available.restype = ctypes.c_int32
            lib.memscale_gpu_available.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
            ]

            lib.memscale_index_enable_gpu.restype = ctypes.c_int32
            lib.memscale_index_enable_gpu.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
            ]

            lib.memscale_index_gpu_batch_search.restype = ctypes.c_int32
            lib.memscale_index_gpu_batch_search.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_char_p),
            ]

            self._has_gpu_support = True
        except AttributeError:
            self._has_gpu_support = False

    @staticmethod
    def gpu_available(*, lib_path: str | None = None) -> bool:
        """Check if GPU support is available.

        Returns True if a GPU adapter is found and the library was built with
        the ``gpu`` feature.

        Args:
            lib_path: Optional path to the shared library.
        """
        lib = load_library(lib_path)
        try:
            lib.memscale_gpu_available.restype = ctypes.c_int32
            lib.memscale_gpu_available.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
            ]
        except AttributeError:
            return False

        out = ctypes.c_char_p()
        code = lib.memscale_gpu_available(ctypes.byref(out))
        if code != 0:
            return False
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
            return data.get("available", False)
        except Exception:
            return False
        finally:
            lib.membrain_string_free(out)

    def enable_gpu(self, config: dict[str, Any] | None = None) -> None:
        """Enable GPU-accelerated distance computation.

        Args:
            config: Optional GPU configuration dictionary with keys:
                - max_batch_size (int): Maximum candidates per dispatch (default: 16384).
                - gpu_batch_threshold (int): Minimum candidates for GPU path (default: 256).

        Raises:
            MembrainError: If no GPU is available or the library lacks GPU support.
        """
        if not getattr(self, '_has_gpu_support', False):
            raise MembrainError("library was built without GPU support")

        config_bytes: bytes | None = None
        if config:
            config_bytes = encode(json.dumps(config))

        code = self._lib.memscale_index_enable_gpu(self._handle, config_bytes)
        check(self._lib, code)

    def gpu_batch_search(
        self, queries: list[list[float]], k: int = 10
    ) -> list[list[IndexSearchResult]]:
        """Run multiple queries using GPU brute-force distance computation.

        Falls back to CPU batch search if GPU is not enabled or the library
        lacks GPU support.

        Args:
            queries: List of query vectors.
            k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        if not getattr(self, '_has_gpu_support', False):
            return self.batch_search(queries, k)

        out = ctypes.c_char_p()
        queries_json = json.dumps(queries)
        code = self._lib.memscale_index_gpu_batch_search(
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
    # Configuration
    # -------------------------------------------------------------------

    def enable_pq(self, config: dict[str, Any]) -> None:
        """Enable product quantization.

        Args:
            config: PQ configuration dictionary with keys:
                - num_subspaces (int): Number of subspaces (dimension must be divisible).
                - num_centroids (int): Centroids per subspace (max 256).
                - training_iterations (int): K-means iterations.
                - training_sample_size (int): Max vectors sampled for training.
        """
        config_bytes = encode(json.dumps(config))
        code = self._lib.memscale_index_enable_pq(self._handle, config_bytes)
        check(self._lib, code)

    def enable_wal(self, config: dict[str, Any]) -> None:
        """Enable write-ahead logging for crash recovery.

        Args:
            config: WAL configuration dictionary with keys:
                - log_path (str): Path to the WAL file.
                - checkpoint_dir (str): Directory for checkpoint files.
                - checkpoint_interval (int): Operations between checkpoints.
        """
        config_bytes = encode(json.dumps(config))
        code = self._lib.memscale_index_enable_wal(self._handle, config_bytes)
        check(self._lib, code)

    def compact(self) -> None:
        """Trigger manual graph compaction to remove tombstoned nodes."""
        code = self._lib.memscale_index_compact(self._handle)
        check(self._lib, code)

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self) -> str:
        """Save index state to a base64-encoded string (MessagePack format)."""
        out = ctypes.c_char_p()
        code = self._lib.memscale_index_save(self._handle, ctypes.byref(out))
        check(self._lib, code)
        try:
            return out.value.decode("utf-8")  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)

    @classmethod
    def load(cls, data: str, *, lib_path: str | None = None) -> MembrainIndex:
        """Load index state from a base64-encoded string.

        Args:
            data: Base64-encoded MessagePack data from save().
            lib_path: Optional path to the shared library.
        """
        instance = object.__new__(cls)
        instance._lib = load_library(lib_path)
        instance._has_gpu_support = False
        instance._setup_index_signatures()
        instance._setup_gpu_signatures()
        instance._handle = instance._lib.memscale_index_load(encode(data))
        if not instance._handle:
            err = get_last_error(instance._lib)
            raise MembrainError(f"failed to load index: {err}")
        return instance

    def save_binary(self, path: str) -> None:
        """Save the index to a binary file for memory-mapped loading.

        Args:
            path: File path to write the binary index.
        """
        code = self._lib.memscale_index_save_binary(self._handle, encode(path))
        check(self._lib, code)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly release the underlying native index."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MembrainIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class MembrainMmapIndex:
    """Read-only HNSW index loaded from a binary file.

    This index supports search but not add/remove operations.
    It is designed for production read-heavy workloads.

    Usage::

        from membrain import MembrainMmapIndex

        index = MembrainMmapIndex("index.bin")
        results = index.search([0.1, 0.2, ...], k=5)
        index.close()
    """

    def __init__(self, path: str, *, lib_path: str | None = None) -> None:
        """Load a read-only index from a binary file.

        Args:
            path: Path to the binary index file (created by MembrainIndex.save_binary).
            lib_path: Optional path to the shared library.
        """
        self._lib = load_library(lib_path)
        self._setup_signatures()
        self._handle = self._lib.memscale_index_load_binary(encode(path))
        if not self._handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to load binary index: {err}")

    def _setup_signatures(self) -> None:
        lib = self._lib
        lib.memscale_index_load_binary.restype = ctypes.c_void_p
        lib.memscale_index_load_binary.argtypes = [ctypes.c_char_p]
        lib.memscale_index_mmap_free.restype = None
        lib.memscale_index_mmap_free.argtypes = [ctypes.c_void_p]
        lib.memscale_index_mmap_search.restype = ctypes.c_int32
        lib.memscale_index_mmap_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.memscale_index_mmap_len.restype = ctypes.c_int32
        lib.memscale_index_mmap_len.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
        ]

    def search(self, query: list[float], k: int = 10) -> list[IndexSearchResult]:
        """Search for the k nearest neighbors.

        Args:
            query: Query vector (must match index dimension).
            k: Number of results to return.
        """
        out = ctypes.c_char_p()
        query_json = json.dumps(query)
        code = self._lib.memscale_index_mmap_search(
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
        return [
            IndexSearchResult(id=r["id"], score=r["score"], distance=r["distance"])
            for r in data
        ]

    def __len__(self) -> int:
        """Get the number of vectors in the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_index_mmap_len(self._handle, ctypes.byref(out))
        check(self._lib, code)
        return out.value

    def close(self) -> None:
        """Explicitly release the underlying native index."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_index_mmap_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MembrainMmapIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
