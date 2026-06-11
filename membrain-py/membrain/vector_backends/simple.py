"""Simple in-memory vector backend for development/testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class SimpleInMemoryBackend(VectorBackend):
    """Simple in-memory vector backend for development/testing.

    This is a basic implementation using cosine similarity. For production,
    use a proper vector database backend.

    Example::

        backend = SimpleInMemoryBackend()
        backend.store("id1", [0.1, 0.2, 0.3], {"type": "fact"})
        results = backend.search([0.1, 0.2, 0.4], limit=5)
    """

    def __init__(self, **config):
        """Initialize the in-memory backend.

        Args:
            **config: Configuration options (unused for this simple backend)
        """
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store vector in memory."""
        self._vectors[memory_id] = (embedding, metadata)

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using cosine similarity."""
        results = []

        for memory_id, (embedding, metadata) in self._vectors.items():
            # Apply filters if provided
            if filters and not self._matches_filters(metadata, filters):
                continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, embedding)
            results.append((memory_id, score, metadata))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def delete(self, memory_id: str) -> bool:
        """Delete from memory."""
        if memory_id in self._vectors:
            del self._vectors[memory_id]
            return True
        return False

    def count(self) -> int:
        """Get count."""
        return len(self._vectors)

    def health_check(self) -> bool:
        """Always healthy for in-memory."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": False,
            "max_dimension": 10000,
            "backend_name": "simple_in_memory"
        }

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
