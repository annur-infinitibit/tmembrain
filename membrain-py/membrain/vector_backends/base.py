"""Abstract base class for vector database backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class VectorBackend(ABC):
    """Abstract base class for vector database backends.

    Implement this class to create custom vector database backends that can be
    used with Membrain. The backend is responsible for storing and searching
    vector embeddings.

    Example::

        class MyVectorBackend(VectorBackend):
            def __init__(self, **config):
                self.index = {}  # Your storage implementation

            def store(self, memory_id: str, embedding: List[float],
                     metadata: Dict[str, Any]) -> None:
                self.index[memory_id] = (embedding, metadata)

            def search(self, query_embedding: List[float], limit: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
                # Your search implementation
                results = []
                for mem_id, (emb, meta) in self.index.items():
                    score = self._cosine_similarity(query_embedding, emb)
                    results.append((mem_id, score, meta))
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]

            def delete(self, memory_id: str) -> bool:
                return self.index.pop(memory_id, None) is not None

            def count(self) -> int:
                return len(self.index)

            def health_check(self) -> bool:
                return True

            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "supports_metadata_filtering": False,
                    "supports_hybrid_search": False,
                    "max_dimension": 4096
                }
    """

    @abstractmethod
    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store a vector embedding with metadata.

        Args:
            memory_id: Unique identifier for this memory
            embedding: Vector embedding (list of floats)
            metadata: Associated metadata dictionary

        Raises:
            Exception: If storage fails
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            filters: Optional metadata filters

        Returns:
            List of tuples: (memory_id, similarity_score, metadata)
            Scores should be between 0 and 1, higher is more similar.

        Raises:
            Exception: If search fails
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a vector by ID.

        Args:
            memory_id: ID of the vector to delete

        Returns:
            True if the vector was deleted, False if it didn't exist

        Raises:
            Exception: If deletion fails
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the total number of vectors stored.

        Returns:
            Number of vectors in the database
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is healthy and operational.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities.

        Returns:
            Dictionary with keys:
                - supports_metadata_filtering (bool)
                - supports_hybrid_search (bool)
                - max_dimension (int)
                - backend_name (str)
        """
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
