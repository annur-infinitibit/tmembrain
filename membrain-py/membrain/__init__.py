"""Membrain - Python client for the Membrain memory system."""

from .client import MembrainClient
from .concurrent_flat_index import ConcurrentFlatIndex
from .conversation import AsyncConversation, Conversation
from .concurrent_hnsw_index import ConcurrentHnswIndex
from .concurrent_ivf_index import ConcurrentIvfIndex
from .concurrent_lsh_index import ConcurrentLshIndex
from .concurrent_vamana_index import ConcurrentVamanaIndex
from .distributed_index import MembrainDistributedIndex
from .errors import MembrainError
from .flat_index import MembrainFlatIndex
from .multi_tenant_index import MultiTenantIndex
from .graph import MembrainGraph
from .index import MembrainIndex, MembrainMmapIndex
from .ivf_index import MembrainIvfIndex
from .lsh_index import MembrainLshIndex
from .rerankers import (
    AnthropicReranker,
    BaseReranker,
    CohereReranker,
    JinaReranker,
    OpenAIReranker,
    RerankerError,
)
from .sharded_index import MembrainShardedIndex
from .vamana_index import MembrainVamanaIndex
from .types import (
    CaseEntry,
    CaseSearchResults,
    ClusterInfo,
    GraphPruningResult,
    GraphQueryResult,
    GraphScoredNode,
    GraphTraversalStep,
    IndexMetrics,
    IndexSearchResult,
    MemoryEntry,
    MemoryInfo,
    NodeInfo,
    RerankResult,
    RerankResults,
    SearchResults,
    ShardedIndexInfo,
    ShardStats,
    StoreResult,
)
from .vector_backends import (
    ChromaBackend,
    FAISSBackend,
    LanceDBBackend,
    MilvusBackend,
    PineconeBackend,
    QdrantBackend,
    SimpleInMemoryBackend,
    VectorBackend,
)

# Backward compatibility aliases — will be removed in v0.2.0
# The new MembrainClient is natively async, so AsyncMembrainClient is no
# longer needed. Keep the name available so existing imports don't break.
AsyncMembrainClient = MembrainClient

from .conversation import AsyncConversation


__all__ = [
    # Client
    "AsyncMembrainClient",
    "MembrainClient",
    "AsyncConversation",
    "Conversation",
    "MembrainGraph",
    "MembrainIndex",
    "MembrainMmapIndex",
    "MembrainFlatIndex",
    "ConcurrentFlatIndex",
    "ConcurrentHnswIndex",
    "ConcurrentIvfIndex",
    "ConcurrentLshIndex",
    "ConcurrentVamanaIndex",
    "MembrainIvfIndex",
    "MembrainLshIndex",
    "MembrainShardedIndex",
    "MembrainVamanaIndex",
    "MembrainDistributedIndex",
    "MultiTenantIndex",
    "MembrainError",
    # Types
    "CaseEntry",
    "CaseSearchResults",
    "ClusterInfo",
    "GraphPruningResult",
    "GraphQueryResult",
    "GraphScoredNode",
    "GraphTraversalStep",
    "IndexMetrics",
    "IndexSearchResult",
    "MemoryEntry",
    "MemoryInfo",
    "NodeInfo",
    "RerankResult",
    "RerankResults",
    "SearchResults",
    "ShardedIndexInfo",
    "ShardStats",
    "StoreResult",
    # Rerankers
    "BaseReranker",
    "CohereReranker",
    "JinaReranker",
    "OpenAIReranker",
    "AnthropicReranker",
    "RerankerError",
    # Vector Backends
    "VectorBackend",
    "SimpleInMemoryBackend",
    "QdrantBackend",
    "ChromaBackend",
    "FAISSBackend",
    "PineconeBackend",
    "MilvusBackend",
    "LanceDBBackend",
]
