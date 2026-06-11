"""Multi-tenant vector index with namespace-based isolation."""

from __future__ import annotations

import ctypes
import json
from typing import Any

from ._ffi import check, encode, get_last_error, load_library
from .errors import MembrainError
from .types import IndexSearchResult


def _setup_multi_tenant_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for multi-tenant index API."""
    # Lifecycle
    lib.memscale_multi_tenant_index_new.restype = ctypes.c_void_p
    lib.memscale_multi_tenant_index_new.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.memscale_multi_tenant_index_free.restype = None
    lib.memscale_multi_tenant_index_free.argtypes = [ctypes.c_void_p]

    # Tenant management
    lib.memscale_multi_tenant_create_tenant.restype = ctypes.c_int32
    lib.memscale_multi_tenant_create_tenant.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
    ]

    lib.memscale_multi_tenant_delete_tenant.restype = ctypes.c_int32
    lib.memscale_multi_tenant_delete_tenant.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_multi_tenant_has_tenant.restype = ctypes.c_int32
    lib.memscale_multi_tenant_has_tenant.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_multi_tenant_list_tenants.restype = ctypes.c_int32
    lib.memscale_multi_tenant_list_tenants.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_multi_tenant_tenant_count.restype = ctypes.c_int32
    lib.memscale_multi_tenant_tenant_count.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]

    # Per-tenant operations
    lib.memscale_multi_tenant_add.restype = ctypes.c_int32
    lib.memscale_multi_tenant_add.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
    ]

    lib.memscale_multi_tenant_remove.restype = ctypes.c_int32
    lib.memscale_multi_tenant_remove.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int32),
    ]

    lib.memscale_multi_tenant_search.restype = ctypes.c_int32
    lib.memscale_multi_tenant_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.memscale_multi_tenant_tenant_len.restype = ctypes.c_int32
    lib.memscale_multi_tenant_tenant_len.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64),
    ]

    lib.memscale_multi_tenant_dimension.restype = ctypes.c_int32
    lib.memscale_multi_tenant_dimension.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
    ]


class MultiTenantIndex:
    """Multi-tenant vector index with namespace-based isolation.

    Each tenant gets an independent concurrent vector index. The registry lock
    is held only briefly for tenant lookups; all vector operations delegate
    to the tenant's own concurrent index with zero cross-tenant contention.

    Supported index types: ``"flat"``, ``"hnsw"``, ``"lsh"``.

    Usage::

        from membrain import MultiTenantIndex

        index = MultiTenantIndex(dimension=1536, index_type="hnsw")

        index.create_tenant("user-123")
        index.create_tenant("user-456")

        index.add("user-123", "vec-1", [0.1] * 1536)
        results = index.search("user-123", [0.1] * 1536, k=10)

        # user-456 sees nothing from user-123
        results = index.search("user-456", [0.1] * 1536, k=10)
        assert len(results) == 0

        index.close()
    """

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "flat",
        max_tenants: int = 0,
        index_config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
    ) -> None:
        """Initialize a multi-tenant vector index.

        Args:
            dimension: Vector dimension for all tenant indices.
            index_type: Type of index for each tenant: "flat", "hnsw", or "lsh".
            max_tenants: Maximum number of tenants (0 = unlimited).
            index_config: Optional per-index configuration dictionary.
            lib_path: Optional path to the shared library.
        """
        self._lib = load_library(lib_path)
        _setup_multi_tenant_signatures(self._lib)

        mt_config = json.dumps({"dimension": dimension, "max_tenants": max_tenants})
        idx_config = json.dumps(index_config) if index_config else None

        self._handle = self._lib.memscale_multi_tenant_index_new(
            encode(mt_config),
            encode(index_type),
            encode(idx_config) if idx_config else None,
        )

        if not self._handle:
            err = get_last_error(self._lib)
            raise MembrainError(f"failed to create multi-tenant index: {err}")

    # -------------------------------------------------------------------
    # Tenant management
    # -------------------------------------------------------------------

    def create_tenant(self, tenant_id: str) -> None:
        """Create a new tenant namespace.

        Args:
            tenant_id: Unique string identifier for the tenant.

        Raises:
            MembrainError: If tenant already exists or max_tenants reached.
        """
        code = self._lib.memscale_multi_tenant_create_tenant(
            self._handle, encode(tenant_id),
        )
        check(self._lib, code)

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its vectors.

        Args:
            tenant_id: The tenant to delete.

        Returns:
            True if the tenant existed and was removed, False otherwise.
        """
        found = ctypes.c_int32()
        code = self._lib.memscale_multi_tenant_delete_tenant(
            self._handle, encode(tenant_id), ctypes.byref(found),
        )
        check(self._lib, code)
        return bool(found.value)

    def has_tenant(self, tenant_id: str) -> bool:
        """Check if a tenant exists.

        Args:
            tenant_id: The tenant ID to check.

        Returns:
            True if the tenant exists.
        """
        exists = ctypes.c_int32()
        code = self._lib.memscale_multi_tenant_has_tenant(
            self._handle, encode(tenant_id), ctypes.byref(exists),
        )
        check(self._lib, code)
        return bool(exists.value)

    def list_tenants(self) -> list[str]:
        """List all tenant IDs (sorted alphabetically).

        Returns:
            List of tenant ID strings.
        """
        out = ctypes.c_char_p()
        code = self._lib.memscale_multi_tenant_list_tenants(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        try:
            data = json.loads(out.value.decode("utf-8"))  # type: ignore[union-attr]
        finally:
            self._lib.membrain_string_free(out)
        return data

    def tenant_count(self) -> int:
        """Get the number of tenants.

        Returns:
            Number of active tenants.
        """
        out = ctypes.c_int64()
        code = self._lib.memscale_multi_tenant_tenant_count(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    # -------------------------------------------------------------------
    # Per-tenant vector operations
    # -------------------------------------------------------------------

    def add(self, tenant_id: str, id: str, embedding: list[float]) -> None:
        """Add a vector to a tenant's index.

        Args:
            tenant_id: The tenant namespace.
            id: UUID string identifying the vector.
            embedding: List of float values (must match index dimension).
        """
        vector_array = (ctypes.c_float * len(embedding))(*embedding)
        code = self._lib.memscale_multi_tenant_add(
            self._handle,
            encode(tenant_id),
            encode(id),
            vector_array,
            ctypes.c_uint32(len(embedding)),
        )
        check(self._lib, code)

    def remove(self, tenant_id: str, id: str) -> bool:
        """Remove a vector from a tenant's index.

        Args:
            tenant_id: The tenant namespace.
            id: UUID string of the vector to remove.

        Returns:
            True if the vector was found and removed, False otherwise.
        """
        found = ctypes.c_int32()
        code = self._lib.memscale_multi_tenant_remove(
            self._handle,
            encode(tenant_id),
            encode(id),
            ctypes.byref(found),
        )
        check(self._lib, code)
        return bool(found.value)

    def search(
        self, tenant_id: str, query: list[float], k: int = 10,
    ) -> list[IndexSearchResult]:
        """Search a tenant's index for nearest neighbors.

        Args:
            tenant_id: The tenant namespace to search.
            query: Query vector (must match index dimension).
            k: Number of results to return.

        Returns:
            List of IndexSearchResult sorted by score (highest first).
        """
        out = ctypes.c_char_p()
        query_array = (ctypes.c_float * len(query))(*query)
        code = self._lib.memscale_multi_tenant_search(
            self._handle,
            encode(tenant_id),
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
        return [
            IndexSearchResult(id=r["id"], score=r["score"], distance=r["distance"])
            for r in data
        ]

    def tenant_len(self, tenant_id: str) -> int:
        """Get the number of vectors in a tenant's index.

        Args:
            tenant_id: The tenant namespace.

        Returns:
            Number of vectors in the tenant's index.
        """
        out = ctypes.c_int64()
        code = self._lib.memscale_multi_tenant_tenant_len(
            self._handle, encode(tenant_id), ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    @property
    def dimension(self) -> int:
        """Get the vector dimension of the index."""
        out = ctypes.c_int64()
        code = self._lib.memscale_multi_tenant_dimension(
            self._handle, ctypes.byref(out),
        )
        check(self._lib, code)
        return out.value

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Release the multi-tenant index and all tenant data."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.memscale_multi_tenant_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        if hasattr(self, 'close'):
            self.close()

    def __enter__(self) -> MultiTenantIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
