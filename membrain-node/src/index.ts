/**
 * Membrain - Memory system for LLMs.
 *
 * Re-exports all public types and classes.
 */

export { MembrainError, findLibrary } from "./ffi";

export { MembrainClient } from "./client";
export { AsyncConversation, Conversation } from "./conversation";
export type {
  AsyncConversationOptions,
  AsyncLLMCallable,
  ConversationOptions,
  LLMCallable,
} from "./conversation";
export { MembrainGraph } from "./graph";
export { MembrainFlatIndex } from "./flat_index";
export { ConcurrentFlatIndex } from "./concurrent_flat_index";
export { ConcurrentHnswIndex } from "./concurrent_hnsw_index";
export { ConcurrentIvfIndex } from "./concurrent_ivf_index";
export { ConcurrentLshIndex } from "./concurrent_lsh_index";
export { ConcurrentVamanaIndex } from "./concurrent_vamana_index";
export { MembrainIndex, MembrainMmapIndex } from "./hnsw_index";
export { MembrainIvfIndex } from "./ivf_index";
export { MembrainLshIndex } from "./lsh_index";
export { MembrainShardedIndex } from "./sharded_index";
export { MembrainVamanaIndex } from "./vamana_index";
export { MembrainDistributedIndex } from "./distributed_index";
export { MultiTenantIndex } from "./multi_tenant_index";
export type { MultiTenantIndexOptions } from "./multi_tenant_index";

export {
  BaseReranker,
  CohereReranker,
  JinaReranker,
  OpenAIReranker,
  AnthropicReranker,
  RerankerError,
} from "./rerankers";

export * from "./vector_backends";

export {
  MembrainAgent,
  MembrainExecutor,
  MembrainPlanner,
  SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT,
  parsePlan,
} from "./agent";
export type {
  MembrainAgentOptions,
  PlannerOptions,
  ToolFunction,
} from "./agent";

export {
  CasePromptBuilder,
  CaseRetriever,
  ExperienceReplay,
  NonParametricRetriever,
  TrainingDataCollector,
  formatPlan,
} from "./cbr";
export type {
  CasePromptBuilderOptions,
  ExperienceReplayOptions,
  RecordExecutionInput,
} from "./cbr";

export type {
  CaseEntry,
  CaseSearchResults,
  ChatMessage,
  ClusterInfo,
  DistributedIndexConfig,
  FlatIndexConfig,
  GpuConfig,
  GraphPruningResult,
  GraphQueryResult,
  GraphScoredNode,
  GraphTraversalStep,
  IndexConfig,
  IndexMetrics,
  IndexSearchResult,
  IvfIndexConfig,
  JudgeCallable,
  LshIndexConfig,
  MemoryInfo,
  MemoryEntry,
  NodeInfo,
  PqConfig,
  RerankResult,
  RerankResults,
  RerankerConfig,
  SearchFilters,
  SearchResults,
  ShardedIndexConfig,
  ShardedIndexInfo,
  ShardStats,
  StorageStats,
  StoreResult,
  TaskPlan,
  TaskResult,
  TaskStep,
  TrainingPair,
  VamanaIndexConfig,
  WalConfig,
} from "./types";
