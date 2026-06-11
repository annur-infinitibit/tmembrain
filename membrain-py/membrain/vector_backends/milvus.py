"""Milvus vector database backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class MilvusBackend(VectorBackend):
    """Milvus vector database backend.

    Requires: pip install pymilvus

    Example::

        backend = MilvusBackend(
            uri="http://localhost:19530",
            collection_name="membrain",
            dimension=1536
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "membrain",
        dimension: int = 1536,
        metric_type: str = "COSINE",
        token: Optional[str] = None,
        **config
    ):
        """Initialize Milvus backend.

        Args:
            uri: Milvus server URI (or file path for local Milvus Lite)
            collection_name: Name of the collection to use
            dimension: Embedding dimension
            metric_type: Distance metric (COSINE, L2, IP)
            token: Optional auth token for Milvus Cloud
            **config: Additional configuration options
        """
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusBackend. "
                "Install it with: pip install pymilvus"
            )

        # Connect to Milvus
        if token:
            self.client = MilvusClient(uri=uri, token=token)
        else:
            self.client = MilvusClient(uri=uri)

        self.collection_name = collection_name
        self.dimension = dimension
        self.DataType = DataType

        # Create collection if it doesn't exist
        if not self.client.has_collection(collection_name):
            # Define schema
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )

            # Add fields
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

            # Create index params
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type=metric_type,
                params={"nlist": 128}
            )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in Milvus."""
        # Prepare data row
        data = {
            "id": memory_id,
            "vector": embedding,
            **metadata  # Dynamic fields for metadata
        }

        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in Milvus."""
        # Build filter expression
        filter_expr = None
        if filters:
            # Convert filters to Milvus expression format
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')
            filter_expr = " && ".join(conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            output_fields=["*"]  # Return all fields including metadata
        )

        # Process results
        output = []
        if results and len(results) > 0:
            for hit in results[0]:
                memory_id = hit["id"]
                score = hit["distance"]  # Milvus returns distance, convert to similarity
                metadata = {k: v for k, v in hit["entity"].items() if k not in ["id", "vector"]}

                # Convert distance to similarity (for COSINE, distance is already similarity)
                # For L2, we need to convert: similarity = 1 / (1 + distance)
                similarity = score if score <= 1.0 else 1.0 / (1.0 + score)

                output.append((memory_id, similarity, metadata))

        return output

    def delete(self, memory_id: str) -> bool:
        """Delete from Milvus."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f'id == "{memory_id}"'
            )
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get count from Milvus."""
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)

    def health_check(self) -> bool:
        """Check Milvus health."""
        try:
            self.client.list_collections()
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Milvus capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": True,
            "max_dimension": 32768,
            "backend_name": "milvus"
        }

    def close(self) -> None:
        """Close Milvus client."""
        if hasattr(self.client, 'close'):
            self.client.close()
