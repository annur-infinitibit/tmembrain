//! C ABI layer for Membrain
//!
//! All functions are `extern "C"`, `#[no_mangle]`, and use raw C types.
//! Strings are C null-terminated `*const c_char` / `*mut c_char`.
//! The client is an opaque `*mut MembrainClient` handle.
//!
//! Return values:
//! - `i32` status code: 0 = success, negative = error
//! - String results via `*mut *mut c_char` out-param, freed with `membrain_string_free()`
//! - Error details via `membrain_last_error()`

mod client;
mod concurrent;
mod distributed;
mod flat;
mod graph;
mod handle;
mod hnsw;
mod hnsw_scoped;
mod ivf;
mod lsh;
mod multi_tenant;
mod safety;
mod sharded;
mod vamana;

pub(crate) use handle::{as_mut_or, as_ref_or, drop_boxed};
pub(crate) use safety::safe_slice;

// Re-export all #[no_mangle] symbols so they remain visible to the linker.
pub use client::*;
pub use concurrent::*;
pub use distributed::*;
pub use flat::*;
pub use graph::*;
pub use hnsw::*;
pub use hnsw_scoped::*;
pub use ivf::*;
pub use lsh::*;
pub use multi_tenant::*;
pub use sharded::*;
pub use vamana::*;

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use membrain_core::types::Embedding;

use crate::MembrainClient;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

pub(crate) const MEMBRAIN_OK: i32 = 0;
pub(crate) const MEMBRAIN_ERR_NULL_POINTER: i32 = -1;
pub(crate) const MEMBRAIN_ERR_INVALID_UTF8: i32 = -2;
pub(crate) const MEMBRAIN_ERR_STORE: i32 = -4;
pub(crate) const MEMBRAIN_ERR_QUERY: i32 = -5;
pub(crate) const MEMBRAIN_ERR_SERIALIZE: i32 = -6;
pub(crate) const MEMBRAIN_ERR_GRAPH: i32 = -8;
pub(crate) const MEMBRAIN_ERR_INDEX: i32 = -9;
pub(crate) const MEMBRAIN_ERR_DISTRIBUTED: i32 = -10;

// ---------------------------------------------------------------------------
// Thread-local last-error storage
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// Thread-local Tokio runtime for C ABI callers (Node.js / JavaScript)
// ---------------------------------------------------------------------------
//
// The MembrainClient methods are now `async fn`. C ABI functions are
// inherently synchronous (`extern "C"`), so we need a Tokio runtime to
// drive those futures. A thread-local runtime avoids the old pattern of
// embedding the runtime inside the client struct, keeping the client
// runtime-agnostic.
//
// Each thread that calls into the C ABI gets its own independent runtime.
// This is safe because multiple Tokio runtimes can coexist in a single
// process. Node.js uses libuv's thread pool, so each worker thread will
// lazily create its own runtime on first use.
thread_local! {
    static C_API_RUNTIME: tokio::runtime::Runtime = tokio::runtime::Runtime::new()
        .expect("failed to create C API Tokio runtime");
}

/// Execute an async Rust future synchronously for C ABI callers.
///
/// This is the **only** bridge between the synchronous C ABI and the
/// async `MembrainClient` methods. It is never called from Python —
/// Python uses `pyo3-async-runtimes` instead.
pub(crate) fn block_in_c<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    C_API_RUNTIME.with(|rt| rt.block_on(fut))
}
pub(crate) fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Retrieve the last error message as a C string.
/// Returns NULL if no error. The returned pointer is valid until the next
/// FFI call on the same thread. Do NOT free this pointer.
#[no_mangle]
pub extern "C" fn membrain_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|cs| cs.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Free a string previously returned by a `membrain_*` out-param.
///
/// # Safety
///
/// - `s` must have been returned by a `membrain_*` out-param, or be null.
/// - `s` must not have been freed already and must not be used after this
///   call returns.
#[no_mangle]
pub unsafe extern "C" fn membrain_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: non-null checked; caller contract guarantees `s` was
        // produced by `CString::into_raw` in a prior FFI call and has not
        // been freed.
        drop(unsafe { CString::from_raw(s) });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `*const c_char` to `&str`.
///
/// # Safety
///
/// - `ptr` must be null or point to a valid null-terminated C string that
///   stays alive for the duration of the returned borrow.
pub(crate) unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, i32> {
    if ptr.is_null() {
        set_last_error("null pointer passed for string argument");
        return Err(MEMBRAIN_ERR_NULL_POINTER);
    }
    // SAFETY: non-null checked above; caller contract guarantees the
    // pointer references a null-terminated C string for the borrow's lifetime.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().map_err(|e| {
        set_last_error(&format!("invalid UTF-8: {}", e));
        MEMBRAIN_ERR_INVALID_UTF8
    })
}

/// Parse optional embedding JSON to Embedding struct.
/// Returns None if ptr is null, Err if invalid JSON.
///
/// # Safety
///
/// - `embedding_json` must be null or point to a valid null-terminated C string.
pub(crate) unsafe fn parse_embedding_json(
    embedding_json: *const c_char,
) -> Result<Option<Embedding>, i32> {
    if embedding_json.is_null() {
        return Ok(None);
    }
    // SAFETY: non-null checked above; caller contract covers C string validity.
    let json_str = unsafe { cstr_to_str(embedding_json) }?;

    let values: Vec<f32> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("invalid embedding JSON: {}", e));
            return Err(MEMBRAIN_ERR_SERIALIZE);
        }
    };

    Ok(Some(Embedding::new(values)))
}

/// Parse optional metadata JSON to HashMap.
/// Returns None if ptr is null, Err if invalid JSON.
///
/// # Safety
///
/// - `metadata_json` must be null or point to a valid null-terminated C string.
pub(crate) unsafe fn parse_metadata_json(
    metadata_json: *const c_char,
) -> Result<Option<std::collections::HashMap<String, serde_json::Value>>, i32> {
    if metadata_json.is_null() {
        return Ok(None);
    }
    // SAFETY: non-null checked above; caller contract covers C string validity.
    let json_str = unsafe { cstr_to_str(metadata_json) }?;

    match serde_json::from_str(json_str) {
        Ok(v) => Ok(Some(v)),
        Err(e) => {
            set_last_error(&format!("invalid metadata JSON: {}", e));
            Err(MEMBRAIN_ERR_SERIALIZE)
        }
    }
}

/// Write a JSON string to an out-param `*mut *mut c_char`.
///
/// The allocation is transferred to the caller; they must release it with
/// `membrain_string_free`.
pub(crate) fn write_json_out(out: *mut *mut c_char, json: &str) -> i32 {
    if out.is_null() {
        return MEMBRAIN_OK; // caller doesn't want the result
    }
    match CString::new(json) {
        Ok(cs) => {
            // SAFETY: `out` was checked non-null above; caller guarantees it
            // points to a writable `*mut c_char` slot.
            unsafe { *out = cs.into_raw() };
            MEMBRAIN_OK
        }
        Err(e) => {
            set_last_error(&format!("failed to create C string: {}", e));
            MEMBRAIN_ERR_SERIALIZE
        }
    }
}

/// Serialize a store result to JSON and write to out-param.
pub(crate) fn write_store_result(
    result: Result<crate::StoreResult, membrain_core::error::Error>,
    out_json: *mut *mut c_char,
) -> i32 {
    match result {
        Ok(store_result) => match serde_json::to_string(&store_result) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize result: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        },
        Err(e) => {
            set_last_error(&format!("{}", e));
            MEMBRAIN_ERR_STORE
        }
    }
}

/// Validate that a client pointer is non-null and return a shared reference.
///
/// # Safety
///
/// - `client` must be null or have been returned by `membrain_client_new*` and
///   not yet freed.
pub(crate) unsafe fn get_client<'a>(
    client: *mut MembrainClient,
) -> Result<&'a MembrainClient, i32> {
    // SAFETY: delegated to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(client, "null client pointer") }
}

/// Parse a JSON array of floats into a `Vec<f32>` vector.
pub(crate) fn parse_vector_json(json_str: &str) -> Result<Vec<f32>, i32> {
    match serde_json::from_str(json_str) {
        Ok(v) => Ok(v),
        Err(e) => {
            set_last_error(&format!("invalid embedding JSON: {}", e));
            Err(MEMBRAIN_ERR_INDEX)
        }
    }
}
