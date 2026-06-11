"""Thread-safe concurrent IVF vector index client."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexSearchResult


def _setup_concurrent_ivf_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for concurrent IVF index API."""
    # Lifecycle
    lib.memscale_concurrent_ivf_index_build.restype = ctypes.c_void_p
    lib.memscale_concurrent_ivf_index_build.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p,
    ]

    lib.memscale_concurrent_ivf_index_clone.restype = ctypes.c_void_p
    lib.memscale_concurrent_ivf_index_clone.argtypes = [ctypes.c_void_p]

    lib.memscale_concurrent_ivf_index_free.restype = None
    lib.memscale_concurrent_ivf_index_free.argtypes = [ctypes.c_void_p]

    # Operations (thread-safe)
    lib.memscale_concurrent_ivf_index_add.restype = ctypes.c_int32
    lib.memscale_concurrent_ivf_index_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
    ]

    lib.memscale_concurrent_ivf_index_remove.restype = ctypes.c_int32
    lib.memscale_concurrent_ivf_index_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_concurrent_ivf_index_search.restype = ctypes.c_int32
    lib.memscale_concurrent_ivf_index_search.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
    ]

    # Info
    lib.memscale_concurrent_ivf_index_len.restype = ctypes.c_int32
    lib.memscale_concurrent_ivf_index_len.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_concurrent_ivf_index_dimension.restype = ctypes.c_int32
    lib.memscale_concurrent_ivf_index_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]


class ConcurrentIvfIndex:
    """Thread-safe concurrent IVF (Inverted File) vector index.

    IVF uses k-means clustering to partition vectors into cells, then searches
    only the nearest cells at query time. This provides approximate nearest
    neighbor search with good recall/speed tradeoffs.

    This index can be safely shared across multiple threads. All operations
    (add, remove, search) can run concurrently without external locking.

    Concurrency Model:
        - Concurrent reads (multiple searches in parallel)
        - Per-cell locking (writes to different cells don't block each other)
        - Expected 3-5x search throughput on multi-core systems

    Usage:
        from membrain import ConcurrentIvfIndex

        # Build index from training data
        ids = [str(uuid.uuid4()) for _ in range(1000)]
        vectors = [[random.random() for _ in range(128)] for _ in range(1000)]

        index = ConcurrentIvfIndex.build(ids, vectors, dimension=128)

        # Search concurrently from multiple threads
        results = index.search([0.1] * 128, k=10)

        index.close()

    Note:
        Python's GIL is released during Rust operations, allowing true
        parallelism for CPU-intensive search and distance computations.
    """

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL) -> None:
        """Initialize from an existing handle (internal use)."""
        self._lib = lib
        self._handle = handle

    @classmethod
    def build(
        cls,
        ids: list[str],
        vectors: list[list[float]],
        dimension: int,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> ConcurrentIvfIndex:
        """Build a thread-safe concurrent IVF index from training data.

        Args:
            ids: List of UUID strings for the vectors.
            vectors: List of vectors (each vector is a list of floats).
            dimension: Vector dimension.
            config: Optional configuration dictionary. Fields:
                "num_clusters", "nprobe", "distance_metric".
            lib_path: Optional path to the shared library.

        Returns:
            A new ConcurrentIvfIndex instance.
        """
        lib = load_library(lib_path)
        _setup_concurrent_ivf_signatures(lib)

        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        # Prepare IDs
        id_pointers = (ctypes.c_char_p * len(ids))()
        for i, id_str in enumerate(ids):
            id_pointers[i] = encode(id_str)

        # Prepare vectors
        flat_vectors = []
        for vec in vectors:
            if len(vec) != dimension:
                raise ValueError(f"all vectors must have dimension {dimension}")
            flat_vectors.extend(vec)

        vectors_array = (ctypes.c_float * len(flat_vectors))(*flat_vectors)

        config_bytes = encode(json.dumps(config)) if config else None

        handle = lib.memscale_concurrent_ivf_index_build(
            id_pointers,
            vectors_array,
            ctypes.c_uint32(len(ids)),
            ctypes.c_uint32(dimension),
            config_bytes,
        )

        if not handle:
            err = get_last_error(lib)
            raise MembrainError(f"failed to build concurrent IVF index: {err}")

        return cls(handle, lib)

    def clone(self) -> ConcurrentIvfIndex:
        """Clone this index handle for use in another thread.

        Both the original and cloned handles point to the same underlying
        index and can be used concurrently. Both must be closed separately.

        Returns:
            A new ConcurrentIvfIndex instance sharing the same underlying index.
        """
        cloned_handle = self._lib.memscale_concurrent_ivf_index_clone(self._handle)
        if not cloned_handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to clone concurrent IVF index: {err}")

        return ConcurrentIvfIndex(cloned_handle, self._lib)

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
        code = self._lib.memscale_concurrent_ivf_index_add(
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
        code = self._lib.memscale_concurrent_ivf_index_remove(
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
        code = self._lib.memscale_concurrent_ivf_index_search(
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
        code = self._lib.memscale_concurrent_ivf_index_len(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_concurrent_ivf_index_dimension(
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
            self._lib.memscale_concurrent_ivf_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> ConcurrentIvfIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
