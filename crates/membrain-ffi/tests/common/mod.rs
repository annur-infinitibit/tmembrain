//! Shared test helpers for FFI integration tests.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

pub fn cstring(value: &str) -> CString {
    CString::new(value).expect("cstring")
}

pub fn embedding_json(dimension: usize) -> CString {
    let values: Vec<f32> = (0..dimension).map(|i| (i as f32) * 0.01).collect();
    let json = serde_json::to_string(&values).expect("serialize");
    cstring(&json)
}

pub fn embedding_json_for(index: usize, dimension: usize) -> CString {
    let values: Vec<f32> = (0..dimension)
        .map(|slot| ((index as f32 * 0.1) + (slot as f32 * 0.001)).sin())
        .collect();
    let json = serde_json::to_string(&values).expect("serialize");
    cstring(&json)
}

/// Read a `*mut c_char` out-param into a Rust String then free it via the
/// given free function.
pub unsafe fn consume_string_out<F: FnOnce(*mut c_char)>(
    slot: *mut c_char,
    free_fn: F,
) -> String {
    if slot.is_null() {
        return String::new();
    }
    let cstr = unsafe { CStr::from_ptr(slot) };
    let text = cstr.to_string_lossy().into_owned();
    free_fn(slot);
    text
}

/// Call any `*const c_char` returning function, capture the string, free it.
pub unsafe fn read_last_error(fetch: unsafe extern "C" fn() -> *const c_char) -> Option<String> {
    let ptr = unsafe { fetch() };
    if ptr.is_null() {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    Some(cstr.to_string_lossy().into_owned())
}

pub fn null_id() -> *const c_char {
    ptr::null()
}
