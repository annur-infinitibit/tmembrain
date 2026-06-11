use std::collections::HashSet;
use std::os::raw::c_char;
use std::ptr;

use memscaledb::{BatchSearch, ShardedConfig, ShardedIndex, VectorId, VectorIndex};

use super::{
    as_mut_or, as_ref_or, cstr_to_str, drop_boxed, parse_vector_json, set_last_error,
    write_json_out, MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE,
    MEMBRAIN_OK,
};

const NULL_MSG: &str = "null sharded index pointer";

#[inline]
unsafe fn get_sharded_index<'a>(index: *const ShardedIndex) -> Result<&'a ShardedIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, NULL_MSG) }
}

#[inline]
unsafe fn get_sharded_index_mut<'a>(index: *mut ShardedIndex) -> Result<&'a mut ShardedIndex, i32> {
    // SAFETY: delegates to `as_mut_or`; caller contract carries forward.
    unsafe { as_mut_or(index, NULL_MSG) }
}

// ---------------------------------------------------------------------------
// Sharded index lifecycle
// ---------------------------------------------------------------------------

/// Create a sharded index by building from data. Use `memscale_sharded_index_build`
/// instead of this function.
///
/// This function exists for API completeness but always returns NULL because a
/// sharded index requires training data for centroid computation. Use
/// `memscale_sharded_index_build` to create a sharded index from vectors, or
/// `memscale_sharded_index_load` to restore a previously saved index.
///
/// # Safety
///
/// - `_config_json` must be null or point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_new(
    _config_json: *const c_char,
) -> *mut ShardedIndex {
    set_last_error("use memscale_sharded_index_build to create a sharded index with training data");
    ptr::null_mut()
}

/// Destroy a sharded index and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `index` must have been returned by `memscale_sharded_index_build` or
///   `memscale_sharded_index_load`, and must not have been freed already,
///   or be null.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_free(index: *mut ShardedIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

/// Build a sharded index from a set of vectors.
///
/// `config_json` is a JSON string with sharded index configuration (see
/// `memscale_sharded_index_new` for format, must include "dimension").
/// `ids_json` is a JSON array of UUID strings.
/// `vectors_json` is a JSON array of float arrays.
///
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `config_json`, `ids_json`, and `vectors_json` must be non-null and point
///   to valid null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_build(
    config_json: *const c_char,
    ids_json: *const c_char,
    vectors_json: *const c_char,
) -> *mut ShardedIndex {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let config_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        let ids_str = match cstr_to_str(ids_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        let vectors_str = match cstr_to_str(vectors_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        #[derive(serde::Deserialize)]
        struct ShardedCreateConfig {
            dimension: usize,
            #[serde(flatten)]
            sharded: ShardedConfig,
        }

        let parsed: ShardedCreateConfig = match serde_json::from_str(config_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid sharded index config JSON: {}", e));
                return ptr::null_mut();
            }
        };

        let id_strings: Vec<String> = match serde_json::from_str(ids_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid ids JSON: {}", e));
                return ptr::null_mut();
            }
        };

        let raw_vectors: Vec<Vec<f32>> = match serde_json::from_str(vectors_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid vectors JSON: {}", e));
                return ptr::null_mut();
            }
        };

        if id_strings.len() != raw_vectors.len() {
            set_last_error("ids and vectors arrays must have the same length");
            return ptr::null_mut();
        }

        let ids: Vec<VectorId> = match id_strings
            .iter()
            .map(|s| s.parse::<VectorId>())
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(ids) => ids,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return ptr::null_mut();
            }
        };

        let entries: Vec<(VectorId, &[f32])> = ids
            .iter()
            .zip(raw_vectors.iter())
            .map(|(&id, vector)| (id, vector.as_slice()))
            .collect();

        match ShardedIndex::build(&entries, parsed.dimension, parsed.sharded) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to build sharded index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sharded index operations
// ---------------------------------------------------------------------------

/// Add a vector to the sharded index.
/// `id` is a UUID string. `embedding_json` is a JSON array of floats.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `id` and `embedding_json` must be non-null and point to valid
///   null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_add(
    index: *mut ShardedIndex,
    id: *const c_char,
    embedding_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index_mut(index) {
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
                return MEMBRAIN_ERR_INDEX;
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
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Remove a vector from the sharded index.
/// `id` is a UUID string. Returns 0 on success.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_remove(
    index: *mut ShardedIndex,
    id: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index_mut(index) {
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
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match index.remove(&vector_id) {
            Ok(_) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Search the sharded index for nearest neighbors.
/// `query_json` is a JSON array of floats. `k` is the number of results.
/// Writes result JSON to `out_json`.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `query_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_search(
    index: *const ShardedIndex,
    query_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Search the sharded index with an ID filter.
/// `allowed_ids_json` is a JSON array of UUID strings.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `query_json` and `allowed_ids_json` must be non-null and point to valid
///   null-terminated C strings.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_search_with_filter(
    index: *const ShardedIndex,
    query_json: *const c_char,
    k: u32,
    allowed_ids_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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
                return MEMBRAIN_ERR_INDEX;
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
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Batch search the sharded index: run multiple queries in parallel.
/// `queries_json` is a JSON array of float arrays.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `queries_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_batch_search(
    index: *const ShardedIndex,
    queries_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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
                return MEMBRAIN_ERR_INDEX;
            }
        };

        let queries: Vec<&[f32]> = raw_queries.iter().map(|v| v.as_slice()).collect();

        let batch_results = index.batch_search(&queries, k as usize);

        let json_results: Vec<serde_json::Value> = batch_results
            .into_iter()
            .map(|query_result| match query_result {
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

/// Trigger rebalancing of the sharded index (retrain centroids and redistribute).
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_rebalance(index: *mut ShardedIndex) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };

        match index.rebalance() {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("failed to rebalance: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sharded index info
// ---------------------------------------------------------------------------

/// Get sharded index info as JSON.
///
/// Result JSON format:
///   {"num_shards":16,"total_vectors":10000,"dimension":1536,
///    "shards":[{"shard_index":0,"count":625},...],
///    "size_stddev":12.5}
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_info(
    index: *const ShardedIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let info = index.shard_info();

        match serde_json::to_string(&info) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize shard info: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Get the number of active vectors in the sharded index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_len(
    index: *const ShardedIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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

/// Get the vector dimension of the sharded index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `out_dimension` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_dimension(
    index: *const ShardedIndex,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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

/// Get sharded index performance metrics as JSON.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_metrics(
    index: *const ShardedIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_sharded_index(index) {
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

// ---------------------------------------------------------------------------
// Sharded index persistence
// ---------------------------------------------------------------------------

/// Save the sharded index to MessagePack format (base64-encoded string).
/// Writes to `out_data`. The caller must free with `membrain_string_free()`.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_sharded_index_build`.
/// - `out_data` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_save(
    index: *const ShardedIndex,
    out_data: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        use base64::Engine;

        let index = match get_sharded_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };

        match index.to_bytes() {
            Ok(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                write_json_out(out_data, &encoded)
            }
            Err(e) => {
                set_last_error(&format!("failed to serialize sharded index: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Load a sharded index from a base64-encoded MessagePack string.
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `data` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_sharded_index_load(data: *const c_char) -> *mut ShardedIndex {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        use base64::Engine;

        let data_str = match cstr_to_str(data) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let bytes = match base64::engine::general_purpose::STANDARD.decode(data_str) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(&format!("invalid base64 data: {}", e));
                return ptr::null_mut();
            }
        };

        match ShardedIndex::from_bytes(&bytes) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to load sharded index: {}", e));
                ptr::null_mut()
            }
        }
    }
}
