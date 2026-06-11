use std::collections::HashSet;
use std::os::raw::c_char;
use std::ptr;

use memscaledb::{DistributedConfig, DistributedIndex, VectorId, VectorIndex};

use super::{
    as_mut_or, as_ref_or, cstr_to_str, drop_boxed, parse_vector_json, set_last_error,
    write_json_out, MEMBRAIN_ERR_DISTRIBUTED, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE,
    MEMBRAIN_OK,
};

const NULL_MSG: &str = "null distributed index pointer";

#[inline]
unsafe fn get_distributed_index<'a>(
    index: *const DistributedIndex,
) -> Result<&'a DistributedIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, NULL_MSG) }
}

#[inline]
unsafe fn get_distributed_index_mut<'a>(
    index: *mut DistributedIndex,
) -> Result<&'a mut DistributedIndex, i32> {
    // SAFETY: delegates to `as_mut_or`; caller contract carries forward.
    unsafe { as_mut_or(index, NULL_MSG) }
}

// ---------------------------------------------------------------------------
// Distributed index lifecycle
// ---------------------------------------------------------------------------

/// Create a new distributed index node and join the cluster.
///
/// `config_json` is a JSON string with distributed index configuration:
///   {"dimension": 128, "listen_address": "127.0.0.1:9400", ...}
///
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `config_json` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_new(
    config_json: *const c_char,
) -> *mut DistributedIndex {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let config_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        #[derive(serde::Deserialize)]
        struct DistributedCreateConfig {
            dimension: usize,
            #[serde(flatten)]
            distributed: DistributedConfig,
        }

        let parsed: DistributedCreateConfig = match serde_json::from_str(config_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid distributed index config JSON: {}", e));
                return ptr::null_mut();
            }
        };

        match DistributedIndex::new(parsed.dimension, parsed.distributed) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to create distributed index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Destroy a distributed index and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `index` must have been returned by `memscale_distributed_index_new`, and
///   must not have been freed already, or be null.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_free(index: *mut DistributedIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

// ---------------------------------------------------------------------------
// Distributed index operations
// ---------------------------------------------------------------------------

/// Add a vector to the distributed index.
/// `id` is a UUID string. `embedding_json` is a JSON array of floats.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `id` and `embedding_json` must be non-null and point to valid
///   null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_add(
    index: *mut DistributedIndex,
    id: *const c_char,
    embedding_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(id) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let emb_str = match cstr_to_str(embedding_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let vector_id: VectorId = match id_str.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return MEMBRAIN_ERR_DISTRIBUTED;
            }
        };

        let vector = match parse_vector_json(emb_str) {
            Ok(v) => v,
            Err(code) => return code,
        };

        match index.add(vector_id, &vector) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_DISTRIBUTED
            }
        }
    }
}

/// Remove a vector from the distributed index.
/// `id` is a UUID string. Returns 0 on success.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_remove(
    index: *mut DistributedIndex,
    id: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let vector_id: VectorId = match id_str.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return MEMBRAIN_ERR_DISTRIBUTED;
            }
        };

        match index.remove(&vector_id) {
            Ok(_) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_DISTRIBUTED
            }
        }
    }
}

/// Search the distributed index for nearest neighbors.
/// `query_json` is a JSON array of floats. `k` is the number of results.
/// Writes result JSON to `out_json`.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `query_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_search(
    index: *const DistributedIndex,
    query_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let query_str = match cstr_to_str(query_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let query = match parse_vector_json(query_str) {
            Ok(v) => v,
            Err(code) => return code,
        };

        match index.search(&query, k as usize) {
            Ok(results) => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.id.to_string(),
                            "score": r.score,
                            "distance": r.distance,
                        })
                    })
                    .collect();

                match serde_json::to_string(&json_results) {
                    Ok(json) => write_json_out(out_json, &json),
                    Err(e) => {
                        set_last_error(&format!("failed to serialize results: {}", e));
                        MEMBRAIN_ERR_SERIALIZE
                    }
                }
            }
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_DISTRIBUTED
            }
        }
    }
}

/// Search the distributed index with an ID filter.
/// `allowed_ids_json` is a JSON array of UUID strings.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `query_json` and `allowed_ids_json` must be non-null and point to valid
///   null-terminated C strings.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_search_with_filter(
    index: *const DistributedIndex,
    query_json: *const c_char,
    k: u32,
    allowed_ids_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let query_str = match cstr_to_str(query_json) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let ids_str = match cstr_to_str(allowed_ids_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let query = match parse_vector_json(query_str) {
            Ok(v) => v,
            Err(code) => return code,
        };

        let id_strings: Vec<String> = match serde_json::from_str(ids_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid allowed_ids JSON: {}", e));
                return MEMBRAIN_ERR_DISTRIBUTED;
            }
        };

        let allowed_ids: HashSet<VectorId> =
            id_strings.iter().filter_map(|s| s.parse().ok()).collect();

        match index.search_with_filter(&query, k as usize, &|id| allowed_ids.contains(id)) {
            Ok(results) => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.id.to_string(),
                            "score": r.score,
                            "distance": r.distance,
                        })
                    })
                    .collect();

                match serde_json::to_string(&json_results) {
                    Ok(json) => write_json_out(out_json, &json),
                    Err(e) => {
                        set_last_error(&format!("failed to serialize results: {}", e));
                        MEMBRAIN_ERR_SERIALIZE
                    }
                }
            }
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_DISTRIBUTED
            }
        }
    }
}

/// Batch search the distributed index: run multiple queries in parallel.
/// `queries_json` is a JSON array of float arrays.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `queries_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_batch_search(
    index: *const DistributedIndex,
    queries_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let queries_str = match cstr_to_str(queries_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let raw_queries: Vec<Vec<f32>> = match serde_json::from_str(queries_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid queries JSON: {}", e));
                return MEMBRAIN_ERR_DISTRIBUTED;
            }
        };

        let json_results: Vec<serde_json::Value> = raw_queries
            .iter()
            .map(|query| match index.search(query, k as usize) {
                Ok(results) => {
                    let items: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "id": r.id.to_string(),
                                "score": r.score,
                                "distance": r.distance,
                            })
                        })
                        .collect();
                    serde_json::json!(items)
                }
                Err(e) => serde_json::json!({"error": e.to_string()}),
            })
            .collect();

        match serde_json::to_string(&json_results) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize batch results: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed index info
// ---------------------------------------------------------------------------

/// Get cluster information as JSON.
///
/// Result JSON format:
///   {"node_count":3,"local_node":{...},"replication_factor":3,"members":[...]}
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_cluster_info(
    index: *const DistributedIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let info = index.cluster_info();

        match serde_json::to_string(&info) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize cluster info: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Get the number of active vectors in the local node's index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_len(
    index: *const DistributedIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }
        *out_count = index.len() as i64;
        MEMBRAIN_OK
    }
}

/// Get the vector dimension of the distributed index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `out_dimension` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_dimension(
    index: *const DistributedIndex,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        if out_dimension.is_null() {
            set_last_error("null out_dimension pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }
        *out_dimension = index.dimension() as i64;
        MEMBRAIN_OK
    }
}

/// Get distributed index performance metrics as JSON.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_metrics(
    index: *const DistributedIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let snapshot = index.metrics();

        let json = serde_json::json!({
            "searches": snapshot.searches,
            "inserts": snapshot.inserts,
            "deletes": snapshot.deletes,
            "compactions": snapshot.compactions,
            "cache_hits": snapshot.cache_hits,
            "cache_misses": snapshot.cache_misses,
            "distance_computations": snapshot.distance_computations,
        });

        match serde_json::to_string(&json) {
            Ok(s) => write_json_out(out_json, &s),
            Err(e) => {
                set_last_error(&format!("failed to serialize metrics: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Gracefully shut down the distributed node.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_distributed_index_new`.
#[no_mangle]
pub unsafe extern "C" fn memscale_distributed_index_shutdown(
    index: *const DistributedIndex,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_distributed_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        index.shutdown();
        MEMBRAIN_OK
    }
}
