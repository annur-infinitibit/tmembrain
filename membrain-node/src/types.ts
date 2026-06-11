/** Result of a store operation. */
export interface StoreResult {
  success: boolean;
  id: string | null;
  merged_with: string | null;
  rejection_reason: string | null;
  duration_ms: number;
}

/** A single memory entry from search results. */
export interface MemoryEntry {
  id: string;
  content: string;
  score: number;
  memory_type: string;
  created_at: string;
}

/** Results from a search query. */
export interface SearchResults {
  memories: MemoryEntry[];
  was_gated: boolean;
  duration_ms: number;
}

/** Information about a stored memory. */
export interface MemoryInfo {
  id: string;
  content: string;
  memory_type: string;
  confidence: number;
}

/** Storage statistics. */
export interface StorageStats {
  total_memories: number;
  by_type: Record<string, number>;
  storage_bytes: number;
  embeddings_count: number;
  avg_confidence: number;
  agent_count: number;
}

/** A scored node from a graph query. */
export interface GraphScoredNode {
  memory_id: string;
  score: number;
  hop_distance: number;
}

/** A single traversal step in a graph query. */
export interface GraphTraversalStep {
  from: string;
  to: string;
  edge_weight: number;
  attention_score: number;
  hop: number;
}

/** Result of a multi-hop graph query. */
export interface GraphQueryResult {
  nodes: GraphScoredNode[];
  traversed_edges: GraphTraversalStep[];
  hops_performed: number;
  nodes_visited: number;
}

/** Filters for search queries. */
export interface SearchFilters {
  /** Filter by memory types (e.g. ["semantic_fact", "episodic_event"]) */
  memory_types?: string[];
  /** Minimum confidence score (0.0-1.0) */
  min_confidence?: number;
  /** Filter by tags (any match) */
  tags?: string[];
  /** Filter by agent ID (UUID string) */
  agent_id?: string;
  /** Filter by metadata key-value pairs (all must match) */
  metadata?: Record<string, unknown>;
  /** Optional query embedding vector for semantic search */
  embedding?: number[];
}

/** Result of graph pruning. */
export interface GraphPruningResult {
  edges_removed: number;
  nodes_removed: number;
  edges_remaining: number;
  nodes_remaining: number;
}

/** A single result from an HNSW index search. */
export interface IndexSearchResult {
  id: string;
  score: number;
  distance: number;
}

/** Performance metrics for an HNSW index. */
export interface IndexMetrics {
  searches: number;
  inserts: number;
  deletes: number;
  compactions: number;
  cache_hits: number;
  cache_misses: number;
  distance_computations: number;
}

/** Configuration for creating an HNSW index. */
export interface IndexConfig {
  dimension: number;
  m?: number;
  ef_construction?: number;
  ef_search?: number;
  max_ef_search?: number;
  distance_metric?: "Cosine" | "Euclidean";
  cache_config?: {
    capacity?: number;
    enabled?: boolean;
  };
}

/** Configuration for product quantization. */
export interface PqConfig {
  num_subspaces: number;
  num_centroids?: number;
  training_iterations?: number;
  training_sample_size?: number;
}

/** Configuration for GPU-accelerated distance computation. */
export interface GpuConfig {
  /** Maximum candidates per single GPU dispatch (default: 16384). */
  max_batch_size?: number;
  /** Minimum candidates before GPU path is used (default: 256). */
  gpu_batch_threshold?: number;
}

/** Configuration for write-ahead logging. */
export interface WalConfig {
  log_path: string;
  checkpoint_dir: string;
  checkpoint_interval: number;
}

/** Configuration for creating a flat (brute-force) index. */
export interface FlatIndexConfig {
  dimension: number;
  distance_metric?: "Cosine" | "Euclidean" | "DotProduct" | "Manhattan" | "Chebyshev";
  cache_config?: {
    capacity?: number;
    enabled?: boolean;
  };
}

/** Configuration for creating an IVF (Inverted File) index. */
export interface IvfIndexConfig {
  dimension: number;
  num_cells?: number;
  nprobe?: number;
  distance_metric?: "Cosine" | "Euclidean" | "DotProduct" | "Manhattan" | "Chebyshev";
  kmeans_iterations?: number;
  training_sample_size?: number;
  seed?: number;
  cache_config?: {
    capacity?: number;
    enabled?: boolean;
  };
}

/** Configuration for creating an LSH (Locality-Sensitive Hashing) index. */
export interface LshIndexConfig {
  dimension: number;
  num_hyperplanes?: number;
  num_tables?: number;
  distance_metric?: "Cosine" | "Euclidean" | "DotProduct" | "Manhattan" | "Chebyshev";
  seed?: number;
  cache_config?: {
    capacity?: number;
    enabled?: boolean;
  };
}

/** Configuration for creating a Vamana (DiskANN-style) index. */
export interface VamanaIndexConfig {
  dimension: number;
  max_degree?: number;
  alpha?: number;
  search_list_size?: number;
  distance_metric?: "Cosine" | "Euclidean" | "DotProduct" | "Manhattan" | "Chebyshev";
  seed?: number;
  cache_config?: {
    capacity?: number;
    enabled?: boolean;
  };
}

/** Configuration for creating a sharded HNSW index. */
export interface ShardedIndexConfig {
  dimension: number;
  num_shards?: number;
  nprobe?: number;
  overlap_factor?: number;
  training_sample_size?: number;
  kmeans_iterations?: number;
  hnsw_config?: Omit<IndexConfig, "dimension">;
}

/** Per-shard statistics. */
export interface ShardStats {
  shard_index: number;
  count: number;
}

/** Information about a sharded HNSW index. */
export interface ShardedIndexInfo {
  num_shards: number;
  total_vectors: number;
  dimension: number;
  shards: ShardStats[];
  size_stddev: number;
}

/** Information about a node in a distributed cluster. */
export interface NodeInfo {
  id: string;
  address: string;
  dimension: number;
}

/** Information about a distributed MemscaleDB cluster. */
export interface ClusterInfo {
  node_count: number;
  local_node: NodeInfo;
  replication_factor: number;
  members: Record<string, unknown>[];
}

/** Configuration for creating a distributed HNSW index. */
export interface DistributedIndexConfig {
  dimension: number;
  listen_address?: string;
  seed_nodes?: string[];
  replication_factor?: number;
  virtual_nodes_per_node?: number;
  read_quorum?: number;
  rpc_timeout?: number;
  gossip_interval?: number;
  suspicion_timeout?: number;
  max_connections_per_peer?: number;
  local_hnsw_config?: Omit<IndexConfig, "dimension">;
}

/** A single case entry from case-based reasoning search results. */
export interface CaseEntry {
  id: string;
  problem: string;
  plan: string;
  outcome: string;
  reward: number;
  score: number;
}

/** Results from a case search, split by reward signal. */
export interface CaseSearchResults {
  positive_cases: CaseEntry[];
  negative_cases: CaseEntry[];
  duration_ms: number;
}

/** A single reranked memory entry. */
export interface RerankResult {
  id: string;
  content: string;
  score: number;
  relevance_score: number;
  memory_type: string;
}

/** Results from reranking. */
export interface RerankResults {
  memories: RerankResult[];
  model: string;
  provider: string;
  duration_ms: number;
}

/** Configuration for reranker instances. */
export interface RerankerConfig {
  apiKey: string;
  model?: string;
  topK?: number;
  endpoint?: string;
  timeout?: number;
}

/** A chat message passed to an LLM callable. */
export interface ChatMessage {
  role: string;
  content: string;
}

/** A single step in a task plan. */
export interface TaskStep {
  id: number;
  description: string;
}

/** A decomposed task plan produced by the planner. */
export interface TaskPlan {
  steps: TaskStep[];
  raw_response: string;
}

/** Result of executing a single task step. */
export interface TaskResult {
  task: TaskStep;
  output: string;
  success: boolean;
  error: string | null;
}

/** A single training example for the relevance classifier. */
export interface TrainingPair {
  query: string;
  case_text: string;
  case_label: "positive" | "negative";
  plan: string;
  truth_label: boolean;
}

/**
 * Judge function used by {@link MembrainAgent.runBatch}.
 *
 * Receives the query, combined output, and per-step results; returns true when
 * the answer is considered correct so the agent can log a positive reward.
 */
export type JudgeCallable = (
  query: string,
  output: string,
  results: TaskResult[]
) => boolean;
