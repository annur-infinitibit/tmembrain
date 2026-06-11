/**
 * Membrain C API
 *
 * Shared C ABI for the Membrain memory system.
 * Consumed by Python (ctypes), Node.js (koffi), and any C/C++ consumer.
 *
 * Conventions:
 *   - All functions returning i32 use 0 for success, negative for errors.
 *   - String out-params (*char**) must be freed with membrain_string_free().
 *   - Use membrain_last_error() to retrieve the last error message.
 *   - MembrainClient is an opaque handle; never dereference it directly.
 */

#ifndef MEMBRAIN_H
#define MEMBRAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Error codes
 * ----------------------------------------------------------------------- */

#define MEMBRAIN_OK              0
#define MEMBRAIN_ERR_NULL_PTR   -1
#define MEMBRAIN_ERR_UTF8       -2
#define MEMBRAIN_ERR_CREATE     -3
#define MEMBRAIN_ERR_STORE      -4
#define MEMBRAIN_ERR_QUERY      -5
#define MEMBRAIN_ERR_SERIALIZE  -6
#define MEMBRAIN_ERR_CONFIG     -7
#define MEMBRAIN_ERR_GRAPH      -8
#define MEMBRAIN_ERR_INDEX      -9

/* -----------------------------------------------------------------------
 * Opaque client handle
 * ----------------------------------------------------------------------- */

typedef struct MembrainClient MembrainClient;

/* -----------------------------------------------------------------------
 * Error handling
 * ----------------------------------------------------------------------- */

/**
 * Retrieve the last error message (thread-local).
 * Returns NULL if no error has occurred.
 * The returned pointer is valid until the next FFI call on the same thread.
 * Do NOT free this pointer.
 */
const char* membrain_last_error(void);

/**
 * Free a string previously returned via a membrain_* out-param.
 * Passing NULL is a no-op.
 */
void membrain_string_free(char* s);

/* -----------------------------------------------------------------------
 * Client lifecycle
 * ----------------------------------------------------------------------- */

/**
 * Create a new client with default configuration.
 * Returns NULL on failure (check membrain_last_error()).
 */
MembrainClient* membrain_client_new(void);

/**
 * Create a new client with JSON configuration.
 * Pass NULL for config_json to use defaults.
 * Returns NULL on failure.
 */
MembrainClient* membrain_client_new_with_config(const char* config_json);

/**
 * Destroy a client and free all associated resources.
 * Passing NULL is a no-op.
 */
void membrain_client_free(MembrainClient* client);

/* -----------------------------------------------------------------------
 * Store operations
 *
 * All store functions return 0 on success and write a JSON result string
 * to *out_json. The caller must free the result with membrain_string_free().
 * out_json may be NULL if the caller does not need the result.
 *
 * Result JSON format:
 *   {"success":true,"id":"uuid","merged_with":null,
 *    "rejection_reason":null,"duration_ms":5}
 * ----------------------------------------------------------------------- */

int32_t membrain_store_fact(
    MembrainClient* client,
    const char* statement,
    double confidence,
    char** out_json
);

int32_t membrain_store_preference(
    MembrainClient* client,
    const char* holder,
    const char* subject,
    const char* preference,
    const char* strength,
    char** out_json
);

int32_t membrain_store_event(
    MembrainClient* client,
    const char* event_type,
    const char* description,
    char** out_json
);

int32_t membrain_store_observation(
    MembrainClient* client,
    const char* content,
    char** out_json
);

int32_t membrain_store_concept(
    MembrainClient* client,
    const char* name,
    const char* definition,
    char** out_json
);

int32_t membrain_store_entity(
    MembrainClient* client,
    const char* name,
    const char* entity_type,
    char** out_json
);

int32_t membrain_store_workflow(
    MembrainClient* client,
    const char* name,
    const char* description,
    char** out_json
);

int32_t membrain_store_skill(
    MembrainClient* client,
    const char* name,
    const char* description,
    char** out_json
);

int32_t membrain_store_pattern(
    MembrainClient* client,
    const char* name,
    const char* description,
    const char* pattern_type,
    char** out_json
);

int32_t membrain_store_goal(
    MembrainClient* client,
    const char* description,
    char** out_json
);

int32_t membrain_store_task(
    MembrainClient* client,
    const char* title,
    char** out_json
);

/* -----------------------------------------------------------------------
 * Query operations
 * ----------------------------------------------------------------------- */

/**
 * Search for memories matching a query.
 * limit <= 0 defaults to 10.
 * Writes result JSON to *out_json.
 *
 * Result JSON format:
 *   {"memories":[{"id":"uuid","content":"...","score":0.95,
 *     "memory_type":"semantic_fact"}],"was_gated":false,"duration_ms":12}
 */
int32_t membrain_search(
    MembrainClient* client,
    const char* query,
    int32_t limit,
    char** out_json
);

/**
 * Search for memories with optional filters.
 * filters_json is a nullable JSON string. Pass NULL for no filters.
 * Writes result JSON to *out_json (same format as membrain_search).
 *
 * Filters JSON format:
 *   {"memory_types":["semantic_fact"],"min_confidence":0.7,
 *    "tags":["important"],"agent_id":"uuid",
 *    "metadata":{"source":"arxiv","year":2024}}
 */
int32_t membrain_search_with_filters(
    MembrainClient* client,
    const char* query,
    int32_t limit,
    const char* filters_json,
    char** out_json
);

/**
 * Get a memory by ID. Writes result JSON to *out_json.
 * Writes "null" if not found.
 */
int32_t membrain_get(
    MembrainClient* client,
    const char* id,
    char** out_json
);

/**
 * Delete a memory by ID.
 * Returns 0 on success (whether or not the memory existed).
 */
int32_t membrain_delete(
    MembrainClient* client,
    const char* id
);

/**
 * Get the total memory count. Writes count to *out_count.
 */
int32_t membrain_count(
    MembrainClient* client,
    int64_t* out_count
);

/**
 * Get storage statistics. Writes result JSON to *out_json.
 */
int32_t membrain_stats(
    MembrainClient* client,
    char** out_json
);

/* -----------------------------------------------------------------------
 * Graph memory layer
 *
 * MembrainGraph is a separate opaque handle for the neural graph layer.
 * It manages its own lifecycle independently of MembrainClient.
 * Embeddings are passed as JSON arrays of floats, e.g. "[0.1, 0.2, ...]".
 * ----------------------------------------------------------------------- */

typedef struct MembrainGraph MembrainGraph;

/* --- Graph lifecycle --- */

/**
 * Create a new graph with default configuration.
 * Returns NULL on failure.
 */
MembrainGraph* membrain_graph_new(void);

/**
 * Create a new graph with JSON configuration.
 * Pass NULL for defaults.
 * Returns NULL on failure.
 */
MembrainGraph* membrain_graph_new_with_config(const char* config_json);

/**
 * Destroy a graph and free all associated resources.
 * Passing NULL is a no-op.
 */
void membrain_graph_free(MembrainGraph* graph);

/* --- Graph node operations --- */

/**
 * Add a node to the graph.
 * memory_id: UUID string.
 * embedding_json: JSON array of floats, e.g. "[0.1, 0.2, ...]".
 * confidence: float in [0.0, 1.0].
 */
int32_t membrain_graph_add_node(
    MembrainGraph* graph,
    const char* memory_id,
    const char* embedding_json,
    double confidence
);

/**
 * Remove a node and all its incident edges from the graph.
 * memory_id: UUID string.
 */
int32_t membrain_graph_remove_node(
    MembrainGraph* graph,
    const char* memory_id
);

/* --- Graph query --- */

/**
 * Multi-hop graph query.
 * query_embedding_json: JSON array of floats.
 * max_hops: <= 0 uses configured default.
 * top_k: <= 0 defaults to 10.
 * Writes result JSON to *out_json.
 *
 * Result JSON format:
 *   {"nodes":[{"memory_id":"uuid","score":0.95,"hop_distance":1}],
 *    "traversed_edges":[{"from":"uuid","to":"uuid","edge_weight":0.8,
 *      "attention_score":0.6,"hop":1}],
 *    "hops_performed":3,"nodes_visited":12}
 */
int32_t membrain_graph_query(
    MembrainGraph* graph,
    const char* query_embedding_json,
    int32_t max_hops,
    int32_t top_k,
    char** out_json
);

/* --- Graph info --- */

/** Get graph node count. */
int32_t membrain_graph_node_count(
    MembrainGraph* graph,
    int64_t* out_count
);

/** Get graph edge count. */
int32_t membrain_graph_edge_count(
    MembrainGraph* graph,
    int64_t* out_count
);

/* --- Graph pruning --- */

/**
 * Manually trigger graph pruning.
 * Writes result JSON to *out_json.
 *
 * Result JSON format:
 *   {"edges_removed":5,"nodes_removed":0,
 *    "edges_remaining":42,"nodes_remaining":10}
 */
int32_t membrain_graph_prune(
    MembrainGraph* graph,
    char** out_json
);

/* --- Graph persistence --- */

/**
 * Save graph state to a base64-encoded string.
 * Writes to *out_data. The caller must free with membrain_string_free().
 */
int32_t membrain_graph_save(
    MembrainGraph* graph,
    char** out_data
);

/**
 * Load graph state from a base64-encoded string.
 * Returns an opaque handle, or NULL on failure.
 */
MembrainGraph* membrain_graph_load(const char* data);

/* -----------------------------------------------------------------------
 * MemscaleDB HNSW Vector Index
 *
 * MemscaleIndex is a separate opaque handle for the HNSW vector index.
 * It manages its own lifecycle independently of MembrainClient.
 * Vectors are passed as JSON arrays of floats, e.g. "[0.1, 0.2, ...]".
 * MemscaleMmapIndex is a read-only index loaded from a binary file.
 * ----------------------------------------------------------------------- */

typedef struct HnswIndex MemscaleIndex;
typedef struct MmapIndex MemscaleMmapIndex;

/* --- Index lifecycle --- */

/**
 * Create a new HNSW index with the given vector dimension and default config.
 * Returns an opaque handle.
 */
MemscaleIndex* memscale_index_new(uint32_t dimension);

/**
 * Create a new HNSW index with JSON configuration.
 * config_json must include a "dimension" field.
 *
 * Config JSON format:
 *   {"dimension":1536,"m":32,"ef_construction":400,"ef_search":200,
 *    "max_ef_search":800,"distance_metric":"Cosine",
 *    "cache_config":{"capacity":2048,"enabled":true}}
 *
 * Returns an opaque handle, or NULL on failure.
 */
MemscaleIndex* memscale_index_new_with_config(const char* config_json);

/**
 * Destroy an HNSW index and free all associated resources.
 * Passing NULL is a no-op.
 */
void memscale_index_free(MemscaleIndex* index);

/* --- Index operations --- */

/**
 * Add a vector to the index.
 * id: UUID string.
 * embedding_json: JSON array of floats, e.g. "[0.1, 0.2, ...]".
 */
int32_t memscale_index_add(
    MemscaleIndex* index,
    const char* id,
    const char* embedding_json
);

/**
 * Remove a vector from the index by ID.
 * id: UUID string.
 */
int32_t memscale_index_remove(
    MemscaleIndex* index,
    const char* id
);

/**
 * Search the index for nearest neighbors.
 * query_json: JSON array of floats.
 * k: number of results to return.
 * Writes result JSON to *out_json.
 *
 * Result JSON format:
 *   [{"id":"uuid","score":0.95,"distance":0.05}, ...]
 */
int32_t memscale_index_search(
    const MemscaleIndex* index,
    const char* query_json,
    uint32_t k,
    char** out_json
);

/**
 * Search the index with an ID filter.
 * allowed_ids_json: JSON array of UUID strings that are allowed in results.
 * Writes result JSON to *out_json (same format as memscale_index_search).
 */
int32_t memscale_index_search_with_filter(
    const MemscaleIndex* index,
    const char* query_json,
    uint32_t k,
    const char* allowed_ids_json,
    char** out_json
);

/**
 * Batch search: run multiple queries in parallel.
 * queries_json: JSON array of float arrays, e.g. [[0.1,0.2,...],[0.3,0.4,...]].
 * k: number of results per query.
 * Writes result JSON to *out_json as an array of arrays.
 */
int32_t memscale_index_batch_search(
    const MemscaleIndex* index,
    const char* queries_json,
    uint32_t k,
    char** out_json
);

/* --- Index info --- */

/** Get the number of active vectors in the index. */
int32_t memscale_index_len(
    const MemscaleIndex* index,
    int64_t* out_count
);

/** Get the vector dimension of the index. */
int32_t memscale_index_dimension(
    const MemscaleIndex* index,
    int64_t* out_dimension
);

/* --- Index metrics --- */

/**
 * Get index performance metrics as JSON.
 *
 * Result JSON format:
 *   {"searches":100,"inserts":50,"deletes":2,"compactions":1,
 *    "cache_hits":30,"cache_misses":70,"distance_computations":5000}
 */
int32_t memscale_index_metrics(
    const MemscaleIndex* index,
    char** out_json
);

/* --- Index configuration --- */

/**
 * Enable product quantization on the index.
 * config_json: JSON string with PQ configuration.
 *
 * Config JSON format:
 *   {"num_subspaces":16,"num_centroids":256,
 *    "training_iterations":20,"training_sample_size":10000}
 */
int32_t memscale_index_enable_pq(
    MemscaleIndex* index,
    const char* config_json
);

/**
 * Enable write-ahead logging on the index.
 * config_json: JSON string with WAL configuration.
 *
 * Config JSON format:
 *   {"log_path":"data/index.wal","checkpoint_dir":"data/checkpoints",
 *    "checkpoint_interval":1000}
 */
int32_t memscale_index_enable_wal(
    MemscaleIndex* index,
    const char* config_json
);

/** Trigger manual graph compaction. */
int32_t memscale_index_compact(MemscaleIndex* index);

/* --- Index persistence --- */

/**
 * Save the index to MessagePack format (base64-encoded string).
 * Writes to *out_data. The caller must free with membrain_string_free().
 */
int32_t memscale_index_save(
    const MemscaleIndex* index,
    char** out_data
);

/**
 * Load an index from a base64-encoded MessagePack string.
 * Returns an opaque handle, or NULL on failure.
 */
MemscaleIndex* memscale_index_load(const char* data);

/**
 * Save the index to a binary file at the given path.
 */
int32_t memscale_index_save_binary(
    const MemscaleIndex* index,
    const char* path
);

/**
 * Load a read-only index from a binary file at the given path.
 * Returns an opaque MmapIndex handle, or NULL on failure.
 * The returned index is read-only: add and remove are not supported.
 * Free with memscale_index_mmap_free().
 */
MemscaleMmapIndex* memscale_index_load_binary(const char* path);

/**
 * Destroy a read-only MmapIndex.
 * Passing NULL is a no-op.
 */
void memscale_index_mmap_free(MemscaleMmapIndex* index);

/**
 * Search a read-only MmapIndex for nearest neighbors.
 * Same result format as memscale_index_search.
 */
int32_t memscale_index_mmap_search(
    const MemscaleMmapIndex* index,
    const char* query_json,
    uint32_t k,
    char** out_json
);

/** Get the number of vectors in a read-only MmapIndex. */
int32_t memscale_index_mmap_len(
    const MemscaleMmapIndex* index,
    int64_t* out_count
);

/* -----------------------------------------------------------------------
 * MemscaleDB Sharded HNSW Vector Index
 *
 * MemscaleShardedIndex is a sharded index with centroid-based query routing.
 * It partitions vectors across independent HNSW shards for larger datasets.
 * ----------------------------------------------------------------------- */

typedef struct ShardedIndex MemscaleShardedIndex;

/* --- Sharded index lifecycle --- */

/**
 * Build a sharded index from a set of vectors.
 * config_json must include a "dimension" field and sharded index config.
 * ids_json: JSON array of UUID strings.
 * vectors_json: JSON array of float arrays.
 * Returns an opaque handle, or NULL on failure.
 */
MemscaleShardedIndex* memscale_sharded_index_build(
    const char* config_json,
    const char* ids_json,
    const char* vectors_json
);

/**
 * Destroy a sharded index and free all associated resources.
 * Passing NULL is a no-op.
 */
void memscale_sharded_index_free(MemscaleShardedIndex* index);

/* --- Sharded index operations --- */

/**
 * Add a vector to the sharded index.
 * id: UUID string.
 * embedding_json: JSON array of floats.
 */
int32_t memscale_sharded_index_add(
    MemscaleShardedIndex* index,
    const char* id,
    const char* embedding_json
);

/**
 * Remove a vector from the sharded index by ID.
 * id: UUID string.
 */
int32_t memscale_sharded_index_remove(
    MemscaleShardedIndex* index,
    const char* id
);

/**
 * Search the sharded index for nearest neighbors.
 * query_json: JSON array of floats.
 * k: number of results to return.
 * Writes result JSON to *out_json.
 */
int32_t memscale_sharded_index_search(
    const MemscaleShardedIndex* index,
    const char* query_json,
    uint32_t k,
    char** out_json
);

/**
 * Search the sharded index with an ID filter.
 * allowed_ids_json: JSON array of UUID strings that are allowed in results.
 */
int32_t memscale_sharded_index_search_with_filter(
    const MemscaleShardedIndex* index,
    const char* query_json,
    uint32_t k,
    const char* allowed_ids_json,
    char** out_json
);

/**
 * Batch search the sharded index: run multiple queries in parallel.
 * queries_json: JSON array of float arrays.
 * k: number of results per query.
 */
int32_t memscale_sharded_index_batch_search(
    const MemscaleShardedIndex* index,
    const char* queries_json,
    uint32_t k,
    char** out_json
);

/** Trigger rebalancing of the sharded index. */
int32_t memscale_sharded_index_rebalance(MemscaleShardedIndex* index);

/* --- Sharded index info --- */

/** Get sharded index info as JSON. */
int32_t memscale_sharded_index_info(
    const MemscaleShardedIndex* index,
    char** out_json
);

/** Get the number of active vectors in the sharded index. */
int32_t memscale_sharded_index_len(
    const MemscaleShardedIndex* index,
    int64_t* out_count
);

/** Get the vector dimension of the sharded index. */
int32_t memscale_sharded_index_dimension(
    const MemscaleShardedIndex* index,
    int64_t* out_dimension
);

/** Get sharded index performance metrics as JSON. */
int32_t memscale_sharded_index_metrics(
    const MemscaleShardedIndex* index,
    char** out_json
);

/* --- Sharded index persistence --- */

/**
 * Save the sharded index to MessagePack format (base64-encoded string).
 * Writes to *out_data. The caller must free with membrain_string_free().
 */
int32_t memscale_sharded_index_save(
    const MemscaleShardedIndex* index,
    char** out_data
);

/**
 * Load a sharded index from a base64-encoded MessagePack string.
 * Returns an opaque handle, or NULL on failure.
 */
MemscaleShardedIndex* memscale_sharded_index_load(const char* data);

#ifdef __cplusplus
}
#endif

#endif /* MEMBRAIN_H */
