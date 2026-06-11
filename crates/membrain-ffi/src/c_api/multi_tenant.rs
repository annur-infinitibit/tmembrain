//! FFI bindings for MultiTenantIndex.
//!
//! The index is stored as a `Box<MultiTenantIndex>` on the heap. The `index_type`
//! parameter (`"flat"`, `"hnsw"`, `"lsh"`) maps to a factory closure that creates
//! the appropriate `ConcurrentXxxIndex` for each new tenant.

use std::os::raw::c_char;
use std::sync::Arc;

use memscaledb::{
    ConcurrentFlatIndex, ConcurrentHnswIndex, ConcurrentLshIndex, ConcurrentVectorIndex,
    FlatConfig, HnswConfig, LshConfig, MultiTenantConfig, MultiTenantIndex, VectorId,
};

use super::{
    as_ref_or, cstr_to_str, drop_boxed, safe_slice, set_last_error, write_json_out,
    MEMBRAIN_ERR_INDEX, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE, MEMBRAIN_OK,
};

/// Tenant-specific error code.
pub(crate) const MEMBRAIN_ERR_TENANT: i32 = -11;

/// Factory closure that creates a concurrent index for a given dimension.
type IndexFactory = Box<dyn Fn(usize) -> Arc<dyn ConcurrentVectorIndex> + Send + Sync>;

const NULL_MSG: &str = "null multi-tenant index pointer";

#[inline]
unsafe fn get_multi_tenant_index<'a>(
    index: *const MultiTenantIndex,
) -> Result<&'a MultiTenantIndex, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(index, NULL_MSG) }
}

/// Build a factory closure from an index_type string and optional config JSON.
fn build_factory(index_type: &str, index_config_json: Option<&str>) -> Result<IndexFactory, i32> {
    match index_type {
        "flat" => {
            let config: FlatConfig = if let Some(json) = index_config_json {
                serde_json::from_str(json).map_err(|e| {
                    set_last_error(&format!("invalid flat config JSON: {e}"));
                    MEMBRAIN_ERR_SERIALIZE
                })?
            } else {
                FlatConfig::default()
            };
            Ok(Box::new(move |dimension| {
                Arc::new(ConcurrentFlatIndex::with_config(dimension, config.clone()))
            }))
        }
        "hnsw" => {
            let config: HnswConfig = if let Some(json) = index_config_json {
                serde_json::from_str(json).map_err(|e| {
                    set_last_error(&format!("invalid HNSW config JSON: {e}"));
                    MEMBRAIN_ERR_SERIALIZE
                })?
            } else {
                HnswConfig::default()
            };
            Ok(Box::new(move |dimension| {
                Arc::new(ConcurrentHnswIndex::with_config(dimension, config.clone()))
            }))
        }
        "lsh" => {
            let config: LshConfig = if let Some(json) = index_config_json {
                serde_json::from_str(json).map_err(|e| {
                    set_last_error(&format!("invalid LSH config JSON: {e}"));
                    MEMBRAIN_ERR_SERIALIZE
                })?
            } else {
                LshConfig::default()
            };
            Ok(Box::new(move |dimension| {
                Arc::new(ConcurrentLshIndex::with_config(dimension, config.clone()))
            }))
        }
        other => {
            set_last_error(&format!(
                "unsupported index type '{other}': must be 'flat', 'hnsw', or 'lsh'"
            ));
            Err(MEMBRAIN_ERR_INDEX)
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a new multi-tenant index.
///
/// # Safety
///
/// - `config_json` must be null or point to a valid null-terminated C string (MultiTenantConfig JSON).
/// - `index_type` must point to a valid null-terminated C string (`"flat"`, `"hnsw"`, or `"lsh"`).
/// - `index_config_json` must be null or point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_index_new(
    config_json: *const c_char,
    index_type: *const c_char,
    index_config_json: *const c_char,
) -> *mut MultiTenantIndex {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt_config: MultiTenantConfig = if config_json.is_null() {
            MultiTenantConfig::default()
        } else {
            let json_str = match cstr_to_str(config_json) {
                Ok(s) => s,
                Err(_) => return std::ptr::null_mut(),
            };
            match serde_json::from_str(json_str) {
                Ok(c) => c,
                Err(e) => {
                    set_last_error(&format!("invalid multi-tenant config JSON: {e}"));
                    return std::ptr::null_mut();
                }
            }
        };

        let type_str = match cstr_to_str(index_type) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let idx_config_str = if index_config_json.is_null() {
            None
        } else {
            match cstr_to_str(index_config_json) {
                Ok(s) => Some(s),
                Err(_) => return std::ptr::null_mut(),
            }
        };

        let factory = match build_factory(type_str, idx_config_str) {
            Ok(f) => f,
            Err(_) => return std::ptr::null_mut(),
        };

        let index = MultiTenantIndex::new(mt_config, factory);
        Box::into_raw(Box::new(index))
    }
}

/// Free a multi-tenant index.
///
/// # Safety
///
/// - `index` must be a valid pointer returned by `memscale_multi_tenant_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_index_free(index: *mut MultiTenantIndex) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(index) }
}

// ---------------------------------------------------------------------------
// Tenant management
// ---------------------------------------------------------------------------

/// Create a new tenant.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_create_tenant(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        match mt.create_tenant(tid) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{e}"));
                MEMBRAIN_ERR_TENANT
            }
        }
    }
}

/// Delete a tenant.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `out_found` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_delete_tenant(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    out_found: *mut i32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let found = mt.delete_tenant(tid);
        if !out_found.is_null() {
            *out_found = i32::from(found);
        }
        MEMBRAIN_OK
    }
}

/// Check if a tenant exists.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `out_exists` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_has_tenant(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    out_exists: *mut i32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        if out_exists.is_null() {
            set_last_error("null out_exists pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        *out_exists = i32::from(mt.has_tenant(tid));
        MEMBRAIN_OK
    }
}

/// List all tenant IDs as a JSON array string.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_list_tenants(
    index: *const MultiTenantIndex,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tenants = mt.list_tenants();
        match serde_json::to_string(&tenants) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize tenant list: {e}"));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Get the number of tenants.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_count` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_tenant_count(
    index: *const MultiTenantIndex,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        *out_count = mt.tenant_count() as i64;
        MEMBRAIN_OK
    }
}

// ---------------------------------------------------------------------------
// Per-tenant vector operations
// ---------------------------------------------------------------------------

/// Add a vector to a tenant's index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `id_str` must point to a valid null-terminated C string.
/// - `vector_data` must point to `dimension` floats.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_add(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    id_str: *const c_char,
    vector_data: *const f32,
    dimension: u32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
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
                set_last_error(&format!("invalid vector ID: {e}"));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match mt.add(tid, id, vector) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{e}"));
                MEMBRAIN_ERR_TENANT
            }
        }
    }
}

/// Remove a vector from a tenant's index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `id_str` must point to a valid null-terminated C string.
/// - `out_found` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_remove(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    id_str: *const c_char,
    out_found: *mut i32,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let id_string = match cstr_to_str(id_str) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let id: VectorId = match id_string.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid vector ID: {e}"));
                return MEMBRAIN_ERR_INDEX;
            }
        };

        match mt.remove(tid, &id) {
            Ok(found) => {
                if !out_found.is_null() {
                    *out_found = i32::from(found);
                }
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("{e}"));
                MEMBRAIN_ERR_TENANT
            }
        }
    }
}

/// Search a tenant's index for nearest neighbors.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `query_data` must point to `dimension` floats.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_search(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    query_data: *const f32,
    dimension: u32,
    k: u32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let query = match safe_slice(query_data, dimension as usize) {
            Ok(s) => s,
            Err(code) => return code,
        };

        match mt.search(tid, query, k as usize) {
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
                        set_last_error(&format!("failed to serialize results: {e}"));
                        MEMBRAIN_ERR_SERIALIZE
                    }
                }
            }
            Err(e) => {
                set_last_error(&format!("{e}"));
                MEMBRAIN_ERR_TENANT
            }
        }
    }
}

/// Get the number of vectors in a tenant's index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `tenant_id` must point to a valid null-terminated C string.
/// - `out_count` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_tenant_len(
    index: *const MultiTenantIndex,
    tenant_id: *const c_char,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        let tid = match cstr_to_str(tenant_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        match mt.tenant_len(tid) {
            Ok(count) => {
                *out_count = count as i64;
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("{e}"));
                MEMBRAIN_ERR_TENANT
            }
        }
    }
}

/// Get the vector dimension of the multi-tenant index.
///
/// # Safety
///
/// - `index` must be a valid pointer.
/// - `out_dimension` must point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn memscale_multi_tenant_dimension(
    index: *const MultiTenantIndex,
    out_dimension: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let mt = match get_multi_tenant_index(index) {
            Ok(m) => m,
            Err(code) => return code,
        };

        if out_dimension.is_null() {
            set_last_error("null out_dimension pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        *out_dimension = mt.dimension() as i64;
        MEMBRAIN_OK
    }
}
