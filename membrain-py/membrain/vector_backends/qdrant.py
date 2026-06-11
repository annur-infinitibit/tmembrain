"""Qdrant vector database backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class QdrantBackend(VectorBackend):
    """Qdrant vector database backend.

    Requires: pip install qdrant-client

    Example::

        backend = QdrantBackend(
            url="http://localhost:6333",
            collection_name="membrain",
            api_key="your-key"  # optional
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "membrain",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        **config
    ):
        """Initialize Qdrant backend.

        Args:
            url: Qdrant server URL
            collection_name: Name of the collection to use
            api_key: Optional API key for authentication
            dimension: Embedding dimension
            **config: Additional configuration options
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantBackend. "
                "Install it with: pip install qdrant-client"
            )

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in Qdrant."""
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload=metadata
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            (str(result.id), result.score, result.payload or {})
            for result in results
        ]

    def delete(self, memory_id: str) -> bool:
        """Delete from Qdrant."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get count from Qdrant."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Qdrant capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": True,
            "max_dimension": 65536,
            "backend_name": "qdrant"
        }

    def close(self) -> None:
        """Close Qdrant client."""
        if hasattr(self.client, 'close'):
            self.client.close()
