use std::collections::HashSet;
use std::os::raw::c_char;
use std::ptr;

use memscaledb::{BatchSearch, IvfConfig, IvfIndex, VectorId, VectorIndex};

use super::{
    as_mut_or, as_ref_or, cstr_to_str, drop_boxed, parse_vector_json, set_last_error,
    write_json_out, MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE,
    MEMBRAIN_OK,
};

const NULL_MSG: &str = "null IVF index pointer";

#[inline]
unsafe fn get_ivf_index<'a>(index: *const IvfIndex) -> Result<&'a IvfIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, NULL_MSG) }
}

#[inline]
unsafe fn get_ivf_index_mut<'a>(index: *mut IvfIndex) -> Result<&'a mut IvfIndex, i32> {
    // SAFETY: delegates to `as_mut_or`; caller contract carries forward.
    unsafe { as_mut_or(index, NULL_MSG) }
}

/// Build an IVF index from a set of vectors.
///
/// `config_json` is a JSON string with IVF index configuration. Must include "dimension".
/// `ids_json` is a JSON array of UUID strings.
/// `vectors_json` is a JSON array of float arrays.
///
/// Config JSON format:
///   {"dimension":128,"num_cells":256,"nprobe":10,"distance_metric":"Cosine",
///    "kmeans_iterations":20,"training_sample_size":10000}
///
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `config_json`, `ids_json`, and `vectors_json` must be non-null and point
///   to valid null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_build(
    config_json: *const c_char,
    ids_json: *const c_char,
    vectors_json: *const c_char,
) -> *mut IvfIndex {
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
        struct IvfCreateConfig {
            dimension: usize,
            #[serde(flatten)]
            ivf: IvfConfig,
        }

        let parsed: IvfCreateConfig = match serde_json::from_str(config_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid IVF index config JSON: {}", e));
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

        let vector_refs: Vec<&[f32]> = raw_vectors.iter().map(|v| v.as_slice()).collect();

        match IvfIndex::build(&ids, &vector_refs, parsed.dimension, parsed.ivf) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(e) => {
                set_last_error(&format!("failed to build IVF index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Destroy an IVF index and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `index` must have been returned by `memscale_ivf_index_build`, and must
///   not have been freed already, or be null.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_free(index: *mut IvfIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

/// Add a vector to the IVF index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `id` and `embedding_json` must be non-null and point to valid
///   null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_add(
    index: *mut IvfIndex,
    id: *const c_char,
    embedding_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index_mut(index) {
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

/// Remove a vector from the IVF index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_remove(index: *mut IvfIndex, id: *const c_char) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index_mut(index) {
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

/// Search the IVF index for nearest neighbors.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `query_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_search(
    index: *const IvfIndex,
    query_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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

/// Search the IVF index with an ID filter.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `query_json` and `allowed_ids_json` must be non-null and point to valid
///   null-terminated C strings.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_search_with_filter(
    index: *const IvfIndex,
    query_json: *const c_char,
    k: u32,
    allowed_ids_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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

/// Batch search the IVF index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `queries_json` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_batch_search(
    index: *const IvfIndex,
    queries_json: *const c_char,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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

/// Get the number of vectors in the IVF index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_len(
    index: *const IvfIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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

/// Get the vector dimension of the IVF index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `out_dimension` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_dimension(
    index: *const IvfIndex,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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

/// Get IVF index performance metrics as JSON.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_ivf_index_build`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_ivf_index_metrics(
    index: *const IvfIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let index = match get_ivf_index(index) {
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
