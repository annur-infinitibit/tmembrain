"""ChromaDB vector database backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class ChromaBackend(VectorBackend):
    """ChromaDB vector database backend.

    Requires: pip install chromadb

    Example::

        backend = ChromaBackend(
            path="./chroma_db",  # for persistent storage
            collection_name="membrain"
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: str = "membrain",
        **config
    ):
        """Initialize ChromaDB backend.

        Args:
            path: Path for persistent storage (None for in-memory)
            collection_name: Name of the collection to use
            **config: Additional configuration options
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaBackend. "
                "Install it with: pip install chromadb"
            )

        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Membrain memory storage"}
        )

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in ChromaDB."""
        self.collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in ChromaDB."""
        where = filters if filters else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["metadatas", "distances"]
        )

        # ChromaDB returns distances (lower is better), convert to similarity
        output = []
        if results['ids'] and results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                output.append((memory_id, score, metadata))

        return output

    def delete(self, memory_id: str) -> bool:
        """Delete from ChromaDB."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get count from ChromaDB."""
        return self.collection.count()

    def health_check(self) -> bool:
        """Check ChromaDB health."""
        try:
            self.collection.count()
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get ChromaDB capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": False,
            "max_dimension": 4096,
            "backend_name": "chromadb"
        }
