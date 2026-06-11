use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use memscaledb::{ConcurrentFlatIndex, ConcurrentVectorIndex, FlatConfig, VectorId};

use super::super::{
    as_ref_or, cstr_to_str, drop_boxed, safe_slice, set_last_error, write_json_out,
    MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE, MEMBRAIN_OK,
};

// ---------------------------------------------------------------------------
// ConcurrentFlatIndex (thread-safe brute-force exact search)
// ---------------------------------------------------------------------------

#[inline]
unsafe fn get_concurrent_flat_index<'a>(
    index: *const Arc<ConcurrentFlatIndex>,
) -> Result<&'a Arc<ConcurrentFlatIndex>, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, "null concurrent flat index pointer") }
}

/// Create a new thread-safe concurrent flat index.
/// Returns an Arc-wrapped handle for safe sharing across threads.
///
/// # Example (C)
/// ```c
/// void* index = memscale_concurrent_flat_index_new(128);
/// void* clone = memscale_concurrent_flat_index_clone(index);
/// // Use both handles from different threads
/// memscale_concurrent_flat_index_free(clone);
/// memscale_concurrent_flat_index_free(index);
/// ```
#[no_mangle]
pub extern "C" fn memscale_concurrent_flat_index_new(
    dimension: u32,
) -> *mut Arc<ConcurrentFlatIndex> {
    let index = ConcurrentFlatIndex::new(dimension as usize);
    Box::into_raw(Box::new(Arc::new(index)))
}

/// Create a new thread-safe concurrent flat index with custom configuration.
///
/// # Safety
///
/// - `config_json` must be null or point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_new_with_config(
    dimension: u32,
    config_json: *const c_char,
) -> *mut Arc<ConcurrentFlatIndex> {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let config = if config_json.is_null() {
            FlatConfig::default()
        } else {
            let json_str = match cstr_to_str(config_json) {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            };

            match serde_json::from_str(json_str) {
                Ok(c) => c,
                Err(e) => {
                    set_last_error(&format!("invalid flat config JSON: {}", e));
                    return ptr::null_mut();
                }
            }
        };

        let index = ConcurrentFlatIndex::with_config(dimension as usize, config);
        Box::into_raw(Box::new(Arc::new(index)))
    }
}

/// Clone a concurrent flat index handle for use in another thread.
/// Both handles point to the same index and must be freed separately.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_concurrent_flat_index_new`.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_clone(
    index: *const Arc<ConcurrentFlatIndex>,
) -> *mut Arc<ConcurrentFlatIndex> {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        Box::into_raw(Box::new(Arc::clone(arc)))
    }
}

/// Free a concurrent flat index handle. Other cloned handles remain valid.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_concurrent_flat_index_new` or `_clone`.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_free(index: *mut Arc<ConcurrentFlatIndex>) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

/// Add a vector to the concurrent flat index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `vector_data` must point to `dimension` floats.
/// - `id_str` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_add(
    index: *const Arc<ConcurrentFlatIndex>,
    id_str: *const c_char,
    vector_data: *const f32,
    dimension: u32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
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

/// Remove a vector from the concurrent flat index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `id_str` must be a valid null-terminated C string.
/// - `out_found` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_remove(
    index: *const Arc<ConcurrentFlatIndex>,
    id_str: *const c_char,
    out_found: *mut i32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
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

/// Search the concurrent flat index (thread-safe, can run concurrently).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `query_data` must point to `dimension` floats.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_search(
    index: *const Arc<ConcurrentFlatIndex>,
    query_data: *const f32,
    dimension: u32,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
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

/// Get the number of vectors in the concurrent flat index (thread-safe).
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_count` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_len(
    index: *const Arc<ConcurrentFlatIndex>,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
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

/// Get the dimension of vectors in the concurrent flat index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_dimension` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_concurrent_flat_index_dimension(
    index: *const Arc<ConcurrentFlatIndex>,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let arc = match get_concurrent_flat_index(index) {
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
