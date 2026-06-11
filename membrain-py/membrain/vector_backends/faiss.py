"""FAISS vector database backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class FAISSBackend(VectorBackend):
    """FAISS vector database backend.

    Requires: pip install faiss-cpu  (or faiss-gpu for GPU support)

    Example::

        backend = FAISSBackend(
            dimension=1536,
            index_type="Flat"  # or "IVF100,Flat" for larger datasets
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "Flat",
        use_gpu: bool = False,
        **config
    ):
        """Initialize FAISS backend.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("Flat", "IVF100,Flat", etc.)
            use_gpu: Whether to use GPU acceleration
            **config: Additional configuration options
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FAISSBackend. "
                "Install it with: pip install faiss-cpu (or faiss-gpu)"
            )

        self.faiss = faiss
        self.dimension = dimension
        self.use_gpu = use_gpu

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = faiss.index_factory(dimension, index_type)

        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )

        # Store metadata separately (FAISS only stores vectors)
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_idx = 0

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in FAISS."""
        import numpy as np

        # Convert to numpy array
        vector = np.array([embedding], dtype=np.float32)

        # Get or assign index
        if memory_id in self._id_to_idx:
            # Update existing (FAISS doesn't support updates, so we just update metadata)
            self._metadata[memory_id] = metadata
        else:
            # Add new
            idx = self._next_idx
            self._next_idx += 1
            self._id_to_idx[memory_id] = idx
            self._idx_to_id[idx] = memory_id
            self._metadata[memory_id] = metadata
            self.index.add(vector)

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in FAISS."""
        import numpy as np

        # Convert query to numpy
        query = np.array([query_embedding], dtype=np.float32)

        # Search (returns distances and indices)
        distances, indices = self.index.search(query, limit)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            memory_id = self._idx_to_id.get(int(idx))
            if not memory_id:
                continue

            metadata = self._metadata.get(memory_id, {})

            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue

            # Convert L2 distance to similarity score
            score = 1.0 / (1.0 + float(distance))
            results.append((memory_id, score, metadata))

        return results[:limit]

    def delete(self, memory_id: str) -> bool:
        """Delete from FAISS (metadata only, vector remains)."""
        if memory_id in self._id_to_idx:
            idx = self._id_to_idx.pop(memory_id)
            self._idx_to_id.pop(idx, None)
            self._metadata.pop(memory_id, None)
            return True
        return False

    def count(self) -> int:
        """Get count from FAISS."""
        return self.index.ntotal

    def health_check(self) -> bool:
        """Check FAISS health."""
        try:
            _ = self.index.ntotal
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get FAISS capabilities."""
        return {
            "supports_metadata_filtering": True,  # Via wrapper
            "supports_hybrid_search": False,
            "max_dimension": 65536,
            "backend_name": "faiss"
        }

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
