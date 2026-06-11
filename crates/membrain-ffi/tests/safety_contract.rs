//! Safety-boundary tests: null pointers, bad JSON, invalid UTF-8.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

mod common;

use std::ffi::CString;
use std::ptr;

use membrain_ffi::c_api::*;

const MEMBRAIN_OK: i32 = 0;
const MEMBRAIN_ERR_NULL_POINTER: i32 = -1;

#[test]
fn null_handle_rejected_by_flat_add() {
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_flat_index_add(ptr::null_mut(), id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn null_handle_rejected_by_hnsw_add() {
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_index_add(ptr::null_mut(), id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn null_handle_rejected_by_ivf_add() {
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_ivf_index_add(ptr::null_mut(), id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn null_handle_rejected_by_lsh_add() {
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_lsh_index_add(ptr::null_mut(), id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn null_handle_rejected_by_vamana_add() {
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let embedding = common::embedding_json(4);
    let code =
        unsafe { memscale_vamana_index_add(ptr::null_mut(), id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn flat_search_null_handle_rejected() {
    let query = common::embedding_json(4);
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let code = unsafe {
        memscale_flat_index_search(ptr::null_mut(), query.as_ptr(), 5, &mut out as *mut _)
    };
    assert_ne!(code, MEMBRAIN_OK);
    assert!(out.is_null());
}

#[test]
fn hnsw_search_null_handle_rejected() {
    let query = common::embedding_json(4);
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let code =
        unsafe { memscale_index_search(ptr::null(), query.as_ptr(), 5, &mut out as *mut _) };
    assert_ne!(code, MEMBRAIN_OK);
}

#[test]
fn flat_add_rejects_null_id() {
    let handle = memscale_flat_index_new(4);
    assert!(!handle.is_null());
    let embedding = common::embedding_json(4);
    let code =
        unsafe { memscale_flat_index_add(handle, common::null_id(), embedding.as_ptr()) };
    assert_eq!(code, MEMBRAIN_ERR_NULL_POINTER);
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_add_rejects_invalid_vector_json() {
    let handle = memscale_flat_index_new(4);
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let bad_json = common::cstring("{not-json");
    let code = unsafe { memscale_flat_index_add(handle, id.as_ptr(), bad_json.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
    let message = unsafe { common::read_last_error(membrain_last_error) };
    assert!(message.is_some());
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_add_rejects_invalid_vector_id() {
    let handle = memscale_flat_index_new(4);
    let id = common::cstring("not-a-uuid");
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_add_rejects_invalid_utf8_id() {
    let handle = memscale_flat_index_new(4);
    let bad_bytes: [u8; 4] = [0x80, 0x80, 0x80, 0x00];
    let bad_ptr = bad_bytes.as_ptr() as *const std::os::raw::c_char;
    let embedding = common::embedding_json(4);
    let code = unsafe { memscale_flat_index_add(handle, bad_ptr, embedding.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_index_new_with_bad_config_returns_null() {
    let bad_config = common::cstring("not-json");
    let handle = unsafe { memscale_flat_index_new_with_config(bad_config.as_ptr()) };
    assert!(handle.is_null());
    let message = unsafe { common::read_last_error(membrain_last_error) };
    assert!(message.is_some());
}

#[test]
fn flat_index_free_null_is_safe() {
    unsafe { memscale_flat_index_free(ptr::null_mut()) };
}

#[test]
fn hnsw_index_free_null_is_safe() {
    unsafe { memscale_index_free(ptr::null_mut()) };
}

#[test]
fn ivf_index_free_null_is_safe() {
    unsafe { memscale_ivf_index_free(ptr::null_mut()) };
}

#[test]
fn lsh_index_free_null_is_safe() {
    unsafe { memscale_lsh_index_free(ptr::null_mut()) };
}

#[test]
fn vamana_index_free_null_is_safe() {
    unsafe { memscale_vamana_index_free(ptr::null_mut()) };
}

#[test]
fn membrain_string_free_handles_null() {
    unsafe { membrain_string_free(ptr::null_mut()) };
}

#[test]
fn last_error_populated_after_failure() {
    let handle = memscale_flat_index_new(4);
    let bad_json = common::cstring("not json");
    let id = common::cstring("00000000-0000-0000-0000-000000000000");
    let code = unsafe { memscale_flat_index_add(handle, id.as_ptr(), bad_json.as_ptr()) };
    assert_ne!(code, MEMBRAIN_OK);
    let message = unsafe { common::read_last_error(membrain_last_error) }.expect("message");
    assert!(!message.is_empty());
    unsafe { memscale_flat_index_free(handle) };
    // Avoid dead_code warning for CString-based null id variant.
    let _ = CString::new("").unwrap();
}
