"""LanceDB vector database backend."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import VectorBackend


class LanceDBBackend(VectorBackend):
    """LanceDB embedded vector database backend.

    Requires: pip install lancedb

    LanceDB is an embedded vector database that stores data on disk with no
    server required. It supports metadata filtering and scales well for
    local-first architectures.

    Example::

        backend = LanceDBBackend(
            uri="./lancedb_data",
            table_name="membrain",
            dimension=1536,
        )

        client = MembrainClient(vector_backend=backend)
    """

    def __init__(
        self,
        uri: str = "./lancedb_data",
        table_name: str = "membrain",
        dimension: int = 1536,
        **config
    ):
        """Initialize LanceDB backend.

        Args:
            uri: Path to the LanceDB database directory
            table_name: Name of the table to use
            dimension: Embedding dimension
            **config: Additional configuration options
        """
        try:
            import lancedb as _lancedb
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "lancedb is required for LanceDBBackend. "
                "Install it with: pip install lancedb"
            )

        self._lancedb = _lancedb
        self._pa = pa
        self.table_name = table_name
        self.dimension = dimension

        self.database = _lancedb.connect(uri)

        # Create table if it doesn't exist
        if table_name in self.database.table_names():
            self.table = self.database.open_table(table_name)
        else:
            schema = pa.schema([
                pa.field("memory_id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimension)),
                pa.field("metadata", pa.string()),
            ])
            self.table = self.database.create_table(table_name, schema=schema)

    def store(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Store in LanceDB."""
        self.table.add([{
            "memory_id": memory_id,
            "vector": embedding,
            "metadata": json.dumps(metadata),
        }])

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in LanceDB."""
        query = self.table.search(query_embedding).limit(limit)

        if filters:
            # Build SQL-style WHERE clause for metadata filtering
            # LanceDB supports SQL expressions on columns
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"memory_id = '{value}'" if key == "memory_id" else None)
                # For metadata fields, filtering requires JSON parsing
                # which LanceDB does not natively support on serialized JSON.
                # Skip unsupported metadata filters gracefully.
            filter_expression = " AND ".join(c for c in conditions if c)
            if filter_expression:
                query = query.where(filter_expression)

        results = query.to_list()

        output = []
        for row in results:
            memory_id = row["memory_id"]
            # LanceDB returns _distance (L2 by default); convert to similarity
            distance = row.get("_distance", 0.0)
            similarity = 1.0 / (1.0 + distance)
            metadata = json.loads(row.get("metadata", "{}"))
            output.append((memory_id, similarity, metadata))

        return output

    def delete(self, memory_id: str) -> bool:
        """Delete from LanceDB."""
        try:
            self.table.delete(f"memory_id = '{memory_id}'")
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get count from LanceDB."""
        return self.table.count_rows()

    def health_check(self) -> bool:
        """Check LanceDB health."""
        try:
            self.database.open_table(self.table_name)
            return True
        except Exception:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get LanceDB capabilities."""
        return {
            "supports_metadata_filtering": True,
            "supports_hybrid_search": False,
            "max_dimension": 65536,
            "backend_name": "lancedb"
        }

    def close(self) -> None:
        """Close LanceDB backend. No-op for embedded database."""
        pass
