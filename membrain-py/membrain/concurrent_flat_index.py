"""Thread-safe concurrent flat vector index client."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexSearchResult


def _setup_concurrent_flat_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for concurrent flat index API."""
    # Lifecycle
    lib.memscale_concurrent_flat_index_new.restype = ctypes.c_void_p
    lib.memscale_concurrent_flat_index_new.argtypes = [ctypes.c_uint32]

    lib.memscale_concurrent_flat_index_new_with_config.restype = ctypes.c_void_p
    lib.memscale_concurrent_flat_index_new_with_config.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p,
    ]

    lib.memscale_concurrent_flat_index_clone.restype = ctypes.c_void_p
    lib.memscale_concurrent_flat_index_clone.argtypes = [ctypes.c_void_p]

    lib.memscale_concurrent_flat_index_free.restype = None
    lib.memscale_concurrent_flat_index_free.argtypes = [ctypes.c_void_p]

    # Operations (thread-safe)
    lib.memscale_concurrent_flat_index_add.restype = ctypes.c_int32
    lib.memscale_concurrent_flat_index_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
    ]

    lib.memscale_concurrent_flat_index_remove.restype = ctypes.c_int32
    lib.memscale_concurrent_flat_index_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_concurrent_flat_index_search.restype = ctypes.c_int32
    lib.memscale_concurrent_flat_index_search.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
    ]

    # Info
    lib.memscale_concurrent_flat_index_len.restype = ctypes.c_int32
    lib.memscale_concurrent_flat_index_len.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_concurrent_flat_index_dimension.restype = ctypes.c_int32
    lib.memscale_concurrent_flat_index_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]


class ConcurrentFlatIndex:
    """Thread-safe concurrent flat (brute-force) vector index.

    This index can be safely shared across multiple threads. All operations
    (add, remove, search) can run concurrently without external locking.

    Concurrency Model:
        - Multiple concurrent searches (read parallelism)
        - Writes are serialized but don't block reads
        - Expected 5-6x search throughput on 8 cores

    Usage::

        from membrain import ConcurrentFlatIndex
        from threading import Thread

        index = ConcurrentFlatIndex(dimension=128)

        # Add vectors from multiple threads
        def add_vectors(thread_id):
            for i in range(100):
                index.add(f"vec-{thread_id}-{i}", [0.1] * 128)

        threads = [Thread(target=add_vectors, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Search concurrently
        results = index.search([0.1] * 128, k=10)

        index.close()

    Note:
        Python's GIL is released during Rust operations, allowing true
        parallelism for CPU-intensive search and distance computations.
    """

    def __init__(
        self,
        dimension: int = 1536,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> None:
        """Initialize a thread-safe concurrent flat vector index.

        Args:
            dimension: Vector dimension.
            config: Optional configuration dictionary. Fields:
                "distance_metric", "cache_config".
            lib_path: Optional path to the shared library.
        """
        self._lib = load_library(lib_path)
        _setup_concurrent_flat_signatures(self._lib)

        if config:
            config_bytes = encode(json.dumps(config))
            self._handle = self._lib.memscale_concurrent_flat_index_new_with_config(
                ctypes.c_uint32(dimension), config_bytes,
            )
        else:
            self._handle = self._lib.memscale_concurrent_flat_index_new(
                ctypes.c_uint32(dimension),
            )

        if not self._handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to create concurrent flat index: {err}")

    def clone(self) -> ConcurrentFlatIndex:
        """Clone this index handle for use in another thread.

        Both the original and cloned handles point to the same underlying
        index and can be used concurrently. Both must be closed separately.

        Returns:
            A new ConcurrentFlatIndex instance sharing the same underlying index.

        Example::

            index = ConcurrentFlatIndex(dimension=128)
            clone = index.clone()

            # Use in different threads
            thread1 = Thread(target=lambda: index.search([0.1]*128, 10))
            thread2 = Thread(target=lambda: clone.search([0.2]*128, 10))

            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

            clone.close()
            index.close()
        """
        cloned_handle = self._lib.memscale_concurrent_flat_index_clone(self._handle)
        if not cloned_handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to clone concurrent flat index: {err}")

        cloned = object.__new__(ConcurrentFlatIndex)
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
        code = self._lib.memscale_concurrent_flat_index_add(
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
        code = self._lib.memscale_concurrent_flat_index_remove(
            self._handle,
            encode(id),
            ctypes.byref(found),
        )
        check(self._lib, code)
        return bool(found.value)

    def search(self, query: list[float], k: int = 10) -> list[IndexSearchResult]:
        """Search for the k nearest neighbors (thread-safe, can run concurrently).

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
        code = self._lib.memscale_concurrent_flat_index_search(
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
        code = self._lib.memscale_concurrent_flat_index_len(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_concurrent_flat_index_dimension(
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
            self._lib.memscale_concurrent_flat_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> ConcurrentFlatIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
