//! HNSW index with per-vector metadata for scoped vector search (FFI).
//!
//! Parallel to the raw `memscale_index_*` family in `hnsw.rs`. Bundles an
//! `HnswIndex` with an in-memory metadata sidecar so callers can attach
//! arbitrary `HashMap<String, Value>` to each vector and later pre-filter a
//! search by metadata equality constraints.

use std::collections::HashMap;
use std::os::raw::c_char;
use std::sync::RwLock;

use memscaledb::storage::metadata::canonicalize_metadata_value;
use memscaledb::{HnswIndex, VectorId, VectorIndex};

use super::{
    as_mut_or, as_ref_or, cstr_to_str, drop_boxed, parse_vector_json, set_last_error,
    write_json_out, MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE,
    MEMBRAIN_OK,
};

const NULL_MSG: &str = "null scoped HNSW index pointer";

/// Opaque handle: HNSW index plus per-vector metadata.
pub struct HnswScopedIndex {
    inner: HnswIndex,
    metadata: RwLock<HashMap<VectorId, HashMap<String, serde_json::Value>>>,
}

impl HnswScopedIndex {
    fn new(dimension: usize) -> Self {
        Self {
            inner: HnswIndex::new(dimension),
            metadata: RwLock::new(HashMap::new()),
        }
    }
}

#[inline]
unsafe fn get_handle<'a>(handle: *const HnswScopedIndex) -> Result<&'a HnswScopedIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(handle, NULL_MSG) }
}

#[inline]
unsafe fn get_handle_mut<'a>(handle: *mut HnswScopedIndex) -> Result<&'a mut HnswScopedIndex, i32> {
    // SAFETY: delegates to `as_mut_or`; caller contract carries forward.
    unsafe { as_mut_or(handle, NULL_MSG) }
}

/// Create a new scoped HNSW index with the given vector dimension.
///
/// # Safety
///
/// The returned pointer must be freed via `memscale_hnsw_scoped_free`.
#[no_mangle]
pub extern "C" fn memscale_hnsw_scoped_new(dimension: u32) -> *mut HnswScopedIndex {
    Box::into_raw(Box::new(HnswScopedIndex::new(dimension as usize)))
}

/// Destroy a scoped HNSW index. Passing NULL is a no-op.
///
/// # Safety
///
/// - `handle` must have been returned by `memscale_hnsw_scoped_new` (or be null)
///   and must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_free(handle: *mut HnswScopedIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer.
    unsafe { drop_boxed(handle) }
}

/// Add a vector with optional per-vector metadata.
///
/// `metadata_json` is null or a valid JSON object (e.g. `{"user_id":"alice"}`).
///
/// # Safety
///
/// - `handle` must be valid.
/// - `id` and `embedding_json` must be non-null, null-terminated C strings.
/// - `metadata_json` must be null or a non-null null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_add(
    handle: *mut HnswScopedIndex,
    id: *const c_char,
    embedding_json: *const c_char,
    metadata_json: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`.
    unsafe {
        let handle = match get_handle_mut(handle) {
            Ok(h) => h,
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
            Ok(parsed) => parsed,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };
        let vector = match parse_vector_json(emb_str) {
            Ok(v) => v,
            Err(code) => return code,
        };

        let meta_map: HashMap<String, serde_json::Value> = if metadata_json.is_null() {
            HashMap::new()
        } else {
            match cstr_to_str(metadata_json) {
                Ok("") => HashMap::new(),
                Ok(s) => match serde_json::from_str(s) {
                    Ok(m) => m,
                    Err(e) => {
                        set_last_error(&format!("invalid metadata JSON: {}", e));
                        return MEMBRAIN_ERR_INDEX;
                    }
                },
                Err(code) => return code,
            }
        };

        if let Err(e) = handle.inner.add(vector_id, &vector) {
            set_last_error(&format!("{}", e));
            return MEMBRAIN_ERR_INDEX;
        }
        match handle.metadata.write() {
            Ok(mut guard) => {
                guard.insert(vector_id, meta_map);
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("metadata lock poisoned: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Remove a vector and its metadata.
///
/// # Safety
///
/// - `handle` must be valid.
/// - `id` must be non-null, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_remove(
    handle: *mut HnswScopedIndex,
    id: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`.
    unsafe {
        let handle = match get_handle_mut(handle) {
            Ok(h) => h,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(id) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let vector_id: VectorId = match id_str.parse() {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };
        if let Err(e) = handle.inner.remove(&vector_id) {
            set_last_error(&format!("{}", e));
            return MEMBRAIN_ERR_INDEX;
        }
        match handle.metadata.write() {
            Ok(mut guard) => {
                guard.remove(&vector_id);
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("metadata lock poisoned: {}", e));
                MEMBRAIN_ERR_INDEX
            }
        }
    }
}

/// Search with an optional metadata equality filter.
///
/// `filter_json` is null or a JSON object. Keys in the object are matched by
/// equality against each vector's stored metadata; all keys must match.
/// Pass null for an unfiltered search.
///
/// # Safety
///
/// - `handle` must be valid.
/// - `query_json` must be non-null, null-terminated.
/// - `filter_json` must be null or non-null null-terminated.
/// - `out_json` must be null or point to writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_search(
    handle: *const HnswScopedIndex,
    query_json: *const c_char,
    k: u32,
    filter_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`.
    unsafe {
        let handle = match get_handle(handle) {
            Ok(h) => h,
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

        let filter_map: Option<HashMap<String, serde_json::Value>> = if filter_json.is_null() {
            None
        } else {
            match cstr_to_str(filter_json) {
                Ok("") => None,
                Ok(s) => match serde_json::from_str(s) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        set_last_error(&format!("invalid filter JSON: {}", e));
                        return MEMBRAIN_ERR_INDEX;
                    }
                },
                Err(code) => return code,
            }
        };

        let search_outcome = match filter_map {
            None => handle.inner.search(&query, k as usize),
            Some(filter) => {
                let canonical_filter: HashMap<String, Vec<u8>> = filter
                    .iter()
                    .map(|(k, v)| (k.clone(), canonicalize_metadata_value(v)))
                    .collect();
                let metadata_guard = match handle.metadata.read() {
                    Ok(g) => g,
                    Err(e) => {
                        set_last_error(&format!("metadata lock poisoned: {}", e));
                        return MEMBRAIN_ERR_INDEX;
                    }
                };
                let passes = |vid: &VectorId| -> bool {
                    let Some(vector_meta) =
                        metadata_guard.get(vid) as Option<&HashMap<String, serde_json::Value>>
                    else {
                        return false;
                    };
                    canonical_filter.iter().all(|(key, expected)| {
                        vector_meta
                            .get(key)
                            .is_some_and(|actual| canonicalize_metadata_value(actual) == *expected)
                    })
                };
                handle.inner.search_with_filter(&query, k as usize, &passes)
            }
        };

        match search_outcome {
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

/// Get the metadata for a single vector as a JSON object.
///
/// Returns MEMBRAIN_OK and writes `"{}"` for missing vectors.
///
/// # Safety
///
/// - `handle` must be valid.
/// - `id` must be non-null, null-terminated.
/// - `out_json` must be null or writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_get_metadata(
    handle: *const HnswScopedIndex,
    id: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`.
    unsafe {
        let handle = match get_handle(handle) {
            Ok(h) => h,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(id) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let vector_id: VectorId = match id_str.parse() {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid vector_id: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };
        let guard = match handle.metadata.read() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(&format!("metadata lock poisoned: {}", e));
                return MEMBRAIN_ERR_INDEX;
            }
        };
        let meta = guard.get(&vector_id);
        match serde_json::to_string(&meta) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize metadata: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Get the number of active vectors in a scoped HNSW index.
///
/// # Safety
///
/// - `handle` must be valid.
/// - `out_count` must be non-null and point to writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_hnsw_scoped_len(
    handle: *const HnswScopedIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`.
    unsafe {
        let handle = match get_handle(handle) {
            Ok(h) => h,
            Err(code) => return code,
        };
        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }
        *out_count = handle.inner.len() as i64;
        MEMBRAIN_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    fn c(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    unsafe fn read_out(ptr: *mut c_char) -> String {
        // SAFETY: pointer comes from `write_json_out` inside the FFI fn.
        let s = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        // Caller frees in Python; tests leak here intentionally.
        s
    }

    #[test]
    fn scoped_add_search_filter_roundtrip() {
        let handle = memscale_hnsw_scoped_new(4);
        assert!(!handle.is_null());
        let alice_id = VectorId::new().to_string();
        let bob_id = VectorId::new().to_string();
        let emb = c("[1.0, 0.0, 0.0, 0.0]");
        let alice_meta = c(r#"{"user_id":"alice"}"#);
        let bob_meta = c(r#"{"user_id":"bob"}"#);
        assert_eq!(
            unsafe {
                memscale_hnsw_scoped_add(
                    handle,
                    c(&alice_id).as_ptr(),
                    emb.as_ptr(),
                    alice_meta.as_ptr(),
                )
            },
            MEMBRAIN_OK
        );
        assert_eq!(
            unsafe {
                memscale_hnsw_scoped_add(
                    handle,
                    c(&bob_id).as_ptr(),
                    emb.as_ptr(),
                    bob_meta.as_ptr(),
                )
            },
            MEMBRAIN_OK
        );

        // Unfiltered: both hit
        let mut out: *mut c_char = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                memscale_hnsw_scoped_search(
                    handle,
                    emb.as_ptr(),
                    10,
                    std::ptr::null(),
                    &mut out,
                )
            },
            MEMBRAIN_OK
        );
        let raw = unsafe { read_out(out) };
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.len(), 2);

        // Filtered: only alice
        let filter = c(r#"{"user_id":"alice"}"#);
        let mut out2: *mut c_char = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                memscale_hnsw_scoped_search(
                    handle,
                    emb.as_ptr(),
                    10,
                    filter.as_ptr(),
                    &mut out2,
                )
            },
            MEMBRAIN_OK
        );
        let raw2 = unsafe { read_out(out2) };
        let parsed2: Vec<serde_json::Value> = serde_json::from_str(&raw2).unwrap();
        assert_eq!(parsed2.len(), 1);
        assert_eq!(parsed2[0]["id"], serde_json::json!(alice_id));

        // Used for the test; drops metadata/index.
        unsafe { memscale_hnsw_scoped_free(handle) };
    }

}
