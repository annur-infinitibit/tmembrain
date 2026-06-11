"""Membrain data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StoreResult:
    """Result of a store operation."""

    success: bool
    id: str | None = None
    merged_with: str | None = None
    rejection_reason: str | None = None
    duration_ms: int = 0


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory entry from search results."""

    id: str
    content: str
    score: float
    memory_type: str
    created_at: str = ""


@dataclass(frozen=True)
class SearchResults:
    """Results from a search query."""

    memories: list[MemoryEntry] = field(default_factory=list)
    was_gated: bool = False
    duration_ms: int = 0


@dataclass(frozen=True)
class MemoryInfo:
    """Information about a stored memory."""

    id: str
    content: str
    memory_type: str
    confidence: float


@dataclass(frozen=True)
class GraphScoredNode:
    """A scored node from a graph query."""

    memory_id: str
    score: float
    hop_distance: int


@dataclass(frozen=True)
class GraphTraversalStep:
    """A single traversal step in a graph query."""

    from_id: str
    to_id: str
    edge_weight: float
    attention_score: float
    hop: int


@dataclass(frozen=True)
class GraphQueryResult:
    """Result of a multi-hop graph query."""

    nodes: list[GraphScoredNode] = field(default_factory=list)
    traversed_edges: list[GraphTraversalStep] = field(default_factory=list)
    hops_performed: int = 0
    nodes_visited: int = 0


@dataclass(frozen=True)
class GraphPruningResult:
    """Result of graph pruning."""

    edges_removed: int = 0
    nodes_removed: int = 0
    edges_remaining: int = 0
    nodes_remaining: int = 0


@dataclass(frozen=True)
class IndexSearchResult:
    """A single result from an HNSW index search."""

    id: str
    score: float
    distance: float


@dataclass(frozen=True)
class IndexMetrics:
    """Performance metrics for an HNSW index."""

    searches: int = 0
    inserts: int = 0
    deletes: int = 0
    compactions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    distance_computations: int = 0


@dataclass(frozen=True)
class ShardStats:
    """Per-shard statistics."""

    shard_index: int
    count: int


@dataclass(frozen=True)
class ShardedIndexInfo:
    """Information about a sharded HNSW index."""

    num_shards: int
    total_vectors: int
    dimension: int
    shards: list[ShardStats] = field(default_factory=list)
    size_stddev: float = 0.0


@dataclass(frozen=True)
class NodeInfo:
    """Information about a node in a distributed cluster."""

    id: str
    address: str
    dimension: int


@dataclass(frozen=True)
class ClusterInfo:
    """Information about a distributed MemscaleDB cluster."""

    node_count: int
    local_node: NodeInfo
    replication_factor: int
    members: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class CaseEntry:
    """A single case entry from case-based reasoning search results."""

    id: str
    problem: str
    plan: str
    outcome: str
    reward: float
    score: float


@dataclass(frozen=True)
class CaseSearchResults:
    """Results from a case search, split by reward signal."""

    positive_cases: list[CaseEntry] = field(default_factory=list)
    negative_cases: list[CaseEntry] = field(default_factory=list)
    duration_ms: int = 0


@dataclass(frozen=True)
class RerankResult:
    """A single reranked memory entry."""

    id: str
    content: str
    score: float
    relevance_score: float
    memory_type: str


@dataclass(frozen=True)
class RerankResults:
    """Results from reranking."""

    memories: list[RerankResult] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    duration_ms: int = 0
