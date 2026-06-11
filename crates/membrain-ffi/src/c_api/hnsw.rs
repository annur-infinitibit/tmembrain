use std::collections::HashSet;
use std::os::raw::c_char;
use std::ptr;

use memscaledb::{
    BatchSearch, HnswConfig, HnswIndex, MmapIndex, PqConfig, VectorId, VectorIndex, WalConfig,
};

use super::{
    as_mut_or, as_ref_or, cstr_to_str, drop_boxed, parse_vector_json, set_last_error,
    write_json_out, MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE,
    MEMBRAIN_OK,
};

const NULL_MSG: &str = "null HNSW index pointer";

#[inline]
unsafe fn get_index<'a>(index: *const HnswIndex) -> Result<&'a HnswIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, NULL_MSG) }
}

#[inline]
unsafe fn get_index_mut<'a>(index: *mut HnswIndex) -> Result<&'a mut HnswIndex, i32> {
    // SAFETY: delegates to `as_mut_or`; caller contract carries forward.
    unsafe { as_mut_or(index, NULL_MSG) }
}

// ---------------------------------------------------------------------------
// Index lifecycle
// ---------------------------------------------------------------------------

/// Create a new HNSW index with the given vector dimension and default config.
/// Returns an opaque handle, or NULL on failure.
#[no_mangle]
pub extern "C" fn memscale_index_new(dimension: u32) -> *mut HnswIndex {
    let index = HnswIndex::new(dimension as usize);
    Box::into_raw(Box::new(index))
}

/// Create a new HNSW index with JSON configuration.
/// `config_json` must include a `"dimension"` field.
///
/// Config JSON format:
///   {"dimension":1536,"m":32,"ef_construction":400,"ef_search":200,
///    "max_ef_search":800,"distance_metric":"Cosine",
///    "cache_config":{"capacity":2048,"enabled":true}}
///
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `config_json` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_new_with_config(
    config_json: *const c_char,
) -> *mut HnswIndex {
    // SAFETY: `config_json` validity is part of the fn-level `# Safety` contract.
    unsafe {
        let json_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        #[derive(serde::Deserialize)]
        struct IndexCreateConfig {
            dimension: usize,
            #[serde(flatten)]
            hnsw: HnswConfig,
        }

        let parsed: IndexCreateConfig = match serde_json::from_str(json_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid index config JSON: {}", e));
                return ptr::null_mut();
            }
        };

        let index = HnswIndex::with_config(parsed.dimension, parsed.hnsw);
        Box::into_raw(Box::new(index))
    }
}

/// Destroy an HNSW index and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `index` must have been returned by `memscale_index_new` or
///   `memscale_index_new_with_config`, and must not have been freed already,
///   or be null.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_free(index: *mut HnswIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

// ---------------------------------------------------------------------------
// Index operations
// ---------------------------------------------------------------------------

/// Add a vector to the index.
/// `id` is a UUID string. `embedding_json` is a JSON array of floats.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `id` and `embedding_json` must be non-null and point to valid
///   null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_add(
    index: *mut HnswIndex,
    id: *const c_char,
    embedding_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
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

/// Remove a vector from the index.
/// `id` is a UUID string. Returns 0 on success.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_remove(index: *mut HnswIndex, id: *const c_char) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
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

/// Search the index for nearest neighbors.
/// `query_json` is a JSON array of floats. `k` is the number of results.
/// Writes result JSON to `out_json`.
///
/// Result JSON format:
///   [{"id":"uuid","score":0.95,"distance":0.05}, ...]
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `query_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_search(
    index: *const HnswIndex,
    query_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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

/// Search the index with an ID filter.
/// `allowed_ids_json` is a JSON array of UUID strings that are allowed in results.
/// Writes result JSON to `out_json` (same format as `memscale_index_search`).
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `query_json` and `allowed_ids_json` must be non-null and point to valid
///   null-terminated C strings.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_search_with_filter(
    index: *const HnswIndex,
    query_json: *const c_char,
    k: u32,
    allowed_ids_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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

/// Batch search: run multiple queries in parallel.
/// `queries_json` is a JSON array of float arrays, e.g. `[[0.1,0.2,...],[0.3,0.4,...]]`.
/// `k` is the number of results per query.
/// Writes result JSON to `out_json` as an array of arrays.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `queries_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_batch_search(
    index: *const HnswIndex,
    queries_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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

// ---------------------------------------------------------------------------
// GPU support
// ---------------------------------------------------------------------------

/// Check if GPU support is available.
///
/// Returns a JSON object `{"available": true/false}` indicating whether a GPU
/// adapter can be found on this machine.
///
/// # Safety
///
/// - `out_json` must be non-null and point to valid writable memory.
#[cfg(feature = "gpu")]
#[no_mangle]
pub unsafe extern "C" fn memscale_gpu_available(out_json: *mut *mut c_char) -> i32 {
    // SAFETY: only unsafe op is the `out_json` write inside `write_json_out`;
    // the fn-level `# Safety` contract requires writable out-param memory.
    unsafe {
        let available =
            memscaledb::GpuDistanceContext::try_new(memscaledb::GpuConfig::default()).is_ok();
        let json = serde_json::json!({ "available": available });
        match serde_json::to_string(&json) {
            Ok(s) => write_json_out(out_json, &s),
            Err(e) => {
                set_last_error(&format!("failed to serialize GPU availability: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Enable GPU-accelerated distance computation on an HNSW index.
///
/// `config_json` is an optional JSON configuration string. If null or empty,
/// default configuration is used. Configuration fields:
/// - `max_batch_size` (int): Maximum candidates per GPU dispatch (default: 16384).
/// - `gpu_batch_threshold` (int): Minimum candidates for GPU path (default: 256).
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `config_json` may be null (uses defaults).
#[cfg(feature = "gpu")]
#[no_mangle]
pub unsafe extern "C" fn memscale_index_enable_gpu(
    index: *mut HnswIndex,
    config_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };

        let config = if config_json.is_null() {
            memscaledb::GpuConfig::default()
        } else {
            match cstr_to_str(config_json) {
                Ok("") => memscaledb::GpuConfig::default(),
                Ok(s) => {
                    #[derive(serde::Deserialize)]
                    struct GpuConfigJson {
                        max_batch_size: Option<usize>,
                        gpu_batch_threshold: Option<usize>,
                    }
                    match serde_json::from_str::<GpuConfigJson>(s) {
                        Ok(parsed) => {
                            let mut config = memscaledb::GpuConfig::default();
                            if let Some(max_batch_size) = parsed.max_batch_size {
                                config.max_batch_size = max_batch_size;
                            }
                            if let Some(gpu_batch_threshold) = parsed.gpu_batch_threshold {
                                config.gpu_batch_threshold = gpu_batch_threshold;
                            }
                            config
                        }
                        Err(e) => {
                            set_last_error(&format!("invalid GPU config JSON: {}", e));
                            return MEMBRAIN_ERR_INDEX;
                        }
                    }
                }
                Err(code) => return code,
            }
        };

        match index.enable_gpu(config) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("failed to enable GPU: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// GPU-accelerated batch search across multiple queries.
///
/// Uses GPU brute-force distance computation for all active vectors and
/// returns the top-k results for each query.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `queries_json` must be a valid C string containing a JSON array of arrays.
/// - `out_json` must be non-null and point to valid writable memory.
#[cfg(feature = "gpu")]
#[no_mangle]
pub unsafe extern "C" fn memscale_index_gpu_batch_search(
    index: *const HnswIndex,
    queries_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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
        let batch_results = index.gpu_batch_search(&queries, k as usize);

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
                set_last_error(&format!("failed to serialize GPU batch results: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Index info
// ---------------------------------------------------------------------------

/// Get the number of active vectors in the index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_len(index: *const HnswIndex, out_count: *mut i64) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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

/// Get the vector dimension of the index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `out_dimension` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_dimension(
    index: *const HnswIndex,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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

// ---------------------------------------------------------------------------
// Index metrics
// ---------------------------------------------------------------------------

/// Get index performance metrics as JSON.
///
/// Result JSON format:
///   {"searches":100,"inserts":50,"deletes":2,"compactions":1,
///    "cache_hits":30,"cache_misses":70,"distance_computations":5000}
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_metrics(
    index: *const HnswIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
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
// Index configuration
// ---------------------------------------------------------------------------

/// Enable product quantization on the index.
/// `config_json` is a JSON string with PQ configuration.
///
/// Config JSON format:
///   {"num_subspaces":16,"num_centroids":256,
///    "training_iterations":20,"training_sample_size":10000}
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `config_json` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_enable_pq(
    index: *mut HnswIndex,
    config_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let json_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let config: PqConfig = match serde_json::from_str(json_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid PQ config JSON: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        index.enable_pq(config);
        MEMBRAIN_OK
    }
}

/// Enable write-ahead logging on the index.
/// `config_json` is a JSON string with WAL configuration.
///
/// Config JSON format:
///   {"log_path":"data/index.wal","checkpoint_dir":"data/checkpoints",
///    "checkpoint_interval":1000}
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `config_json` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_enable_wal(
    index: *mut HnswIndex,
    config_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let json_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let config: WalConfig = match serde_json::from_str(json_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid WAL config JSON: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match index.enable_wal(config) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("failed to enable WAL: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Trigger manual graph compaction.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_compact(index: *mut HnswIndex) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index_mut(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        index.compact();
        MEMBRAIN_OK
    }
}

// ---------------------------------------------------------------------------
// Index persistence
// ---------------------------------------------------------------------------

/// Save the index to MessagePack format (base64-encoded string).
/// Writes to `out_data`. The caller must free with `membrain_string_free()`.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `out_data` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_save(
    index: *const HnswIndex,
    out_data: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        use base64::Engine;

        let index = match get_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };

        match index.to_bytes() {
            Ok(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                write_json_out(out_data, &encoded)
            }
            Err(e) => {
                set_last_error(&format!("failed to serialize index: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Load an index from a base64-encoded MessagePack string.
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `data` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_load(data: *const c_char) -> *mut HnswIndex {
    // SAFETY: `data` validity is part of the fn-level `# Safety` contract.
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

        match HnswIndex::from_bytes(&bytes) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to load index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Save the index to a binary file at the given path.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_new`.
/// - `path` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_save_binary(
    index: *const HnswIndex,
    path: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_index(index) {
            Ok(i) => i,
            Err(code) => return code,
        };
        let path_str = match cstr_to_str(path) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let file = match std::fs::File::create(path_str) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&format!("failed to create file: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        let mut writer = std::io::BufWriter::new(file);
        match index.save_binary(&mut writer) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("failed to save binary: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Load a read-only index from a binary file at the given path.
/// Returns an opaque MmapIndex handle, or NULL on failure.
///
/// The returned index is read-only: add() and remove() will return errors.
/// Free with `memscale_index_mmap_free()`.
///
/// # Safety
///
/// - `path` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_load_binary(path: *const c_char) -> *mut MmapIndex {
    // SAFETY: `path` validity is part of the fn-level `# Safety` contract.
    unsafe {
        let path_str = match cstr_to_str(path) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        match MmapIndex::open(path_str) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to load binary index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Destroy a read-only MmapIndex and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `index` must have been returned by `memscale_index_load_binary`, and must
///   not have been freed already, or be null.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_mmap_free(index: *mut MmapIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

/// Search a read-only MmapIndex for nearest neighbors.
/// `query_json` is a JSON array of floats. `k` is the number of results.
/// Writes result JSON to `out_json` (same format as `memscale_index_search`).
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_load_binary`.
/// - `query_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_mmap_search(
    index: *const MmapIndex,
    query_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match as_ref_or(index, "null MmapIndex pointer") {
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

/// Get the number of vectors in a read-only MmapIndex.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_index_load_binary`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_index_mmap_len(
    index: *const MmapIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match as_ref_or(index, "null MmapIndex pointer") {
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
