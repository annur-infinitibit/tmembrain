"""Pinecone vector database backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class PineconeBackend(VectorBackend):
    """Pinecone vector database backend.

    Requires: pip install pinecone-client

    Example::

        backend = PineconeBackend(
            api_key="your-api-key",
            index_name="membrain",
            namespace="default"
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "membrain",
        namespace: str = "default",
        dimension: int = 1536,
        metric: str = "cosine",
        **config
    ):
        """Initialize Pinecone backend.

        Args:
            api_key: Pinecone API key
            index_name: Name of the index to use
            namespace: Namespace within the index
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            **config: Additional configuration options
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError(
                "pinecone-client is required for PineconeBackend. "
                "Install it with: pip install pinecone-client"
            )

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(index_name)

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in Pinecone."""
        # Pinecone expects tuples: (id, values, metadata)
        self.index.upsert(
            vectors=[(memory_id, embedding, metadata)],
            namespace=self.namespace
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in Pinecone."""
        # Build filter if provided
        pinecone_filter = None
        if filters:
            # Convert filters to Pinecone format: {"key": {"$eq": "value"}}
            pinecone_filter = {
                k: {"$eq": v} for k, v in filters.items()
            }

        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=pinecone_filter,
            namespace=self.namespace
        )

        return [
            (match.id, match.score, match.metadata or {})
            for match in results.matches
        ]

    def delete(self, memory_id: str) -> bool:
        """Delete from Pinecone."""
        try:
            self.index.delete(ids=[memory_id], namespace=self.namespace)
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get count from Pinecone."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count

    def health_check(self) -> bool:
        """Check Pinecone health."""
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Pinecone capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": False,
            "max_dimension": 20000,
            "backend_name": "pinecone"
        }
