"""Thread-safe concurrent LSH vector index client."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexSearchResult


def _setup_concurrent_lsh_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for concurrent LSH index API."""
    # Lifecycle
    lib.memscale_concurrent_lsh_index_new.restype = ctypes.c_void_p
    lib.memscale_concurrent_lsh_index_new.argtypes = [ctypes.c_uint32]

    lib.memscale_concurrent_lsh_index_new_with_config.restype = ctypes.c_void_p
    lib.memscale_concurrent_lsh_index_new_with_config.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p,
    ]

    lib.memscale_concurrent_lsh_index_clone.restype = ctypes.c_void_p
    lib.memscale_concurrent_lsh_index_clone.argtypes = [ctypes.c_void_p]

    lib.memscale_concurrent_lsh_index_free.restype = None
    lib.memscale_concurrent_lsh_index_free.argtypes = [ctypes.c_void_p]

    # Operations (thread-safe)
    lib.memscale_concurrent_lsh_index_add.restype = ctypes.c_int32
    lib.memscale_concurrent_lsh_index_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
    ]

    lib.memscale_concurrent_lsh_index_remove.restype = ctypes.c_int32
    lib.memscale_concurrent_lsh_index_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_concurrent_lsh_index_search.restype = ctypes.c_int32
    lib.memscale_concurrent_lsh_index_search.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
    ]

    # Info
    lib.memscale_concurrent_lsh_index_len.restype = ctypes.c_int32
    lib.memscale_concurrent_lsh_index_len.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_concurrent_lsh_index_dimension.restype = ctypes.c_int32
    lib.memscale_concurrent_lsh_index_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]


class ConcurrentLshIndex:
    """Thread-safe concurrent LSH (Locality-Sensitive Hashing) vector index.

    LSH uses random hyperplane hashing to partition vectors into buckets,
    then searches only the matching buckets at query time. This provides
    fast approximate nearest neighbor search with sub-linear query time.

    This index can be safely shared across multiple threads. All operations
    (add, remove, search) can run concurrently without external locking.

    Concurrency Model:
        - Concurrent reads (multiple searches in parallel)
        - Per-table locking (writes to different tables don't block each other)
        - Expected 3-5x search throughput on multi-core systems

    Usage:
        from membrain import ConcurrentLshIndex
        import uuid

        index = ConcurrentLshIndex(dimension=128)

        # Add vectors from multiple threads
        for i in range(1000):
            index.add(str(uuid.uuid4()), [0.1] * 128)

        # Search concurrently
        results = index.search([0.1] * 128, k=10)

        index.close()

    Note:
        Python's GIL is released during Rust operations, allowing true
        parallelism for CPU-intensive search and hashing operations.
    """

    def __init__(
        self,
        dimension: int = 1536,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> None:
        """Initialize a thread-safe concurrent LSH vector index.

        Args:
            dimension: Vector dimension.
            config: Optional configuration dictionary. Fields:
                "num_tables", "hash_length", "distance_metric".
            lib_path: Optional path to the shared library.
        """
        self._lib = load_library(lib_path)
        _setup_concurrent_lsh_signatures(self._lib)

        if config:
            config_bytes = encode(json.dumps(config))
            self._handle = self._lib.memscale_concurrent_lsh_index_new_with_config(
                ctypes.c_uint32(dimension), config_bytes,
            )
        else:
            self._handle = self._lib.memscale_concurrent_lsh_index_new(
                ctypes.c_uint32(dimension),
            )

        if not self._handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to create concurrent LSH index: {err}")

    def clone(self) -> ConcurrentLshIndex:
        """Clone this index handle for use in another thread.

        Both the original and cloned handles point to the same underlying
        index and can be used concurrently. Both must be closed separately.

        Returns:
            A new ConcurrentLshIndex instance sharing the same underlying index.
        """
        cloned_handle = self._lib.memscale_concurrent_lsh_index_clone(self._handle)
        if not cloned_handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to clone concurrent LSH index: {err}")

        cloned = object.__new__(ConcurrentLshIndex)
        cloned._lib = self._lib
        cloned._handle = cloned_handle
        return cloned

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
    # Operations (all thread-safe)
    # -------------------------------------------------------------------

    def add(self, id: str, embedding: list[float]) -> None:
        """Add a vector to the index (thread-safe).

        Args:
            id: UUID string identifying the vector.
            embedding: List of float values (must match index dimension).

        Note:
            Multiple threads can call this concurrently. The GIL is released
            during the Rust operation.
        """
        vector_array = (ctypes.c_float * len(embedding))(*embedding)
        code = self._lib.memscale_concurrent_lsh_index_add(
            self._handle,
            encode(id),
            vector_array,
            ctypes.c_uint32(len(embedding)),
        )
        check(self._lib, code)

    def remove(self, id: str) -> bool:
        """Remove a vector from the index by ID (thread-safe).

        Args:
            id: UUID string of the vector to remove.

        Returns:
            True if the vector was found and removed, False otherwise.
        """
        found = ctypes.c_int32()
        code = self._lib.memscale_concurrent_lsh_index_remove(
            self._handle,
            encode(id),
            ctypes.byref(found),
        )
        check(self._lib, code)
        return bool(found.value)

    def search(self, query: list[float], k: int = 10) -> list[IndexSearchResult]:
        """Search for the k nearest neighbors (thread-safe).

        Args:
            query: Query vector (must match index dimension).
            k: Number of results to return.

        Returns:
            List of IndexSearchResult sorted by score (highest first).

        Note:
            Multiple threads can search concurrently. The GIL is released
            during the search operation, allowing true parallelism.
        """
        out = ctypes.c_char_p()
        query_array = (ctypes.c_float * len(query))(*query)
        code = self._lib.memscale_concurrent_lsh_index_search(
            self._handle,
            query_array,
            ctypes.c_uint32(len(query)),
            ctypes.c_uint32(k),
            ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return self._parse_search_results(data)

    # -------------------------------------------------------------------
    # Info (all thread-safe)
    # -------------------------------------------------------------------

    def __len__(self) -> int:
        """Get the number of vectors in the index (thread-safe)."""
        out = ctypes.c_int64()
        code = self._lib.memscale_concurrent_lsh_index_len(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_concurrent_lsh_index_dimension(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Release this handle to the underlying index.

        Other cloned handles remain valid. The index is freed when
        the last handle is closed.
        """
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_concurrent_lsh_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> ConcurrentLshIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
