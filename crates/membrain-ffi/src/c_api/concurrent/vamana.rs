use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use memscaledb::{ConcurrentVamanaIndex, ConcurrentVectorIndex, VamanaConfig, VectorId};

use super::super::{
    as_ref_or, cstr_to_str, drop_boxed, safe_slice, set_last_error, write_json_out,
    MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE, MEMBRAIN_OK,
};

// ---------------------------------------------------------------------------
// ConcurrentVamanaIndex (thread-safe Vamana/DiskANN-style index)
// ---------------------------------------------------------------------------

#[inline]
unsafe fn get_concurrent_vamana_index<'a>(
    index: *const Arc<ConcurrentVamanaIndex>,
) -> Result<&'a Arc<ConcurrentVamanaIndex>, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, "null concurrent Vamana index pointer") }
}

/// Build a new thread-safe concurrent Vamana index from training data.
///
/// # Safety
///
/// - `ids_data` must point to `num_vectors` valid null-terminated C strings.
/// - `vectors_data` must point to `num_vectors * dimension` floats.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_build(
    ids_data: *const *const c_char,
    vectors_data: *const f32,
    num_vectors: u32,
    dimension: u32,
    config_json: *const c_char,
) -> *mut Arc<ConcurrentVamanaIndex> {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        if ids_data.is_null() || vectors_data.is_null() {
            set_last_error("null ids_data or vectors_data pointer");
            return ptr::null_mut();
        }

        let mut ids = Vec::new();
        for i in 0..num_vectors as isize {
            let id_str = match cstr_to_str(*ids_data.offset(i)) {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            };
            let id: VectorId = match id_str.parse() {
                Ok(id) => id,
                Err(e) => {
                    set_last_error(&format!("invalid vector ID: {}", e));
                    return ptr::null_mut();
                }
            };
            ids.push(id);
        }

        let mut vectors = Vec::new();
        for i in 0..num_vectors {
            let offset = match (i as usize).checked_mul(dimension as usize) {
                Some(o) => o,
                None => {
                    set_last_error("vectors_data offset overflow");
                    return ptr::null_mut();
                }
            };
            let slice = match safe_slice(vectors_data.add(offset), dimension as usize) {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            };
            vectors.push(slice.to_vec());
        }

        let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = if config_json.is_null() {
            VamanaConfig::default()
        } else {
            let json_str = match cstr_to_str(config_json) {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            };

            match serde_json::from_str(json_str) {
                Ok(c) => c,
                Err(e) => {
                    set_last_error(&format!("invalid Vamana config JSON: {}", e));
                    return ptr::null_mut();
                }
            }
        };

        match ConcurrentVamanaIndex::build(&ids, &vector_refs, dimension as usize, config) {
            Ok(index) => Box::into_raw(Box::new(Arc::new(index))),
            Err(e) => {
                set_last_error(&format!("failed to build concurrent Vamana index: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Clone a concurrent Vamana index handle for use in another thread.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_concurrent_vamana_index_build`.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_clone(
    index: *const Arc<ConcurrentVamanaIndex>,
) -> *mut Arc<ConcurrentVamanaIndex> {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        Box::into_raw(Box::new(Arc::clone(arc)))
    }
}

/// Free a concurrent Vamana index handle.
///
/// # Safety
///
/// - `index` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_free(
    index: *mut Arc<ConcurrentVamanaIndex>,
) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

/// Add a vector to the concurrent Vamana index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `vector_data` must point to `dimension` floats.
/// - `id_str` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_add(
    index: *const Arc<ConcurrentVamanaIndex>,
    id_str: *const c_char,
    vector_data: *const f32,
    dimension: u32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(code) => return code,
        };

        let id_string = match cstr_to_str(id_str) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let vector = match safe_slice(vector_data, dimension as usize) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let id: VectorId = match id_string.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid vector ID: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match arc.add_concurrent(id, vector) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Remove a vector from the concurrent Vamana index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `id_str` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_remove(
    index: *const Arc<ConcurrentVamanaIndex>,
    id_str: *const c_char,
    out_found: *mut i32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(code) => return code,
        };

        let id_string = match cstr_to_str(id_str) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let id: VectorId = match id_string.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid vector ID: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match arc.remove_concurrent(&id) {
            Ok(found) => {
                if !out_found.is_null() {
                    *out_found = if found { 1 } else { 0 };
                }
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Search the concurrent Vamana index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `query_data` must point to `dimension` floats.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_search(
    index: *const Arc<ConcurrentVamanaIndex>,
    query_data: *const f32,
    dimension: u32,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(code) => return code,
        };

        let query = match safe_slice(query_data, dimension as usize) {
            Ok(s) => s,
            Err(code) => return code,
        };

        match arc.search(query, k as usize) {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.id.to_string(),
                            "score": r.score,
                            "distance": r.distance,
                        })
                    })
                    .collect();

                match serde_json::to_string(&results_json) {
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

/// Get the number of vectors in the concurrent Vamana index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_count` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_len(
    index: *const Arc<ConcurrentVamanaIndex>,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(code) => return code,
        };

        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        *out_count = arc.len() as i64;
        MEMBRAIN_OK
    }
}

/// Get the dimension of vectors in the concurrent Vamana index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_dimension` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_vamana_index_dimension(
    index: *const Arc<ConcurrentVamanaIndex>,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_vamana_index(index) {
            Ok(a) => a,
            Err(code) => return code,
        };

        if out_dimension.is_null() {
            set_last_error("null out_dimension pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        *out_dimension = arc.dimension() as i64;
        MEMBRAIN_OK
    }
}
