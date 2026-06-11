//! FFI roundtrip: the scoped HNSW index accepts per-vector metadata and
//! filters equality queries correctly.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

mod common;

use std::os::raw::c_char;
use std::ptr;

use common::{consume_string_out, cstring};
use membrain_ffi::c_api::{
    memscale_hnsw_scoped_add, memscale_hnsw_scoped_free, memscale_hnsw_scoped_new,
    memscale_hnsw_scoped_search, membrain_string_free,
};
use memscaledb::VectorId;

fn vec_id() -> String {
    VectorId::new().to_string()
}

#[test]
fn scoped_hnsw_filters_by_metadata_equality() {
    let handle = memscale_hnsw_scoped_new(4);
    assert!(!handle.is_null());

    let emb = cstring("[1.0, 0.0, 0.0, 0.0]");
    let alice_id = vec_id();
    let bob_id = vec_id();
    let carol_id = vec_id();
    unsafe {
        assert_eq!(
            memscale_hnsw_scoped_add(
                handle,
                cstring(&alice_id).as_ptr(),
                emb.as_ptr(),
                cstring(r#"{"user_id":"alice","role":"admin"}"#).as_ptr(),
            ),
            0
        );
        assert_eq!(
            memscale_hnsw_scoped_add(
                handle,
                cstring(&bob_id).as_ptr(),
                emb.as_ptr(),
                cstring(r#"{"user_id":"bob","role":"admin"}"#).as_ptr(),
            ),
            0
        );
        assert_eq!(
            memscale_hnsw_scoped_add(
                handle,
                cstring(&carol_id).as_ptr(),
                emb.as_ptr(),
                cstring(r#"{"user_id":"carol","role":"viewer"}"#).as_ptr(),
            ),
            0
        );
    }

    // Unfiltered: all three
    let mut out: *mut c_char = ptr::null_mut();
    unsafe {
        assert_eq!(
            memscale_hnsw_scoped_search(handle, emb.as_ptr(), 10, ptr::null(), &mut out),
            0
        );
    }
    let results = unsafe { consume_string_out(out, |p| membrain_string_free(p)) };
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&results).unwrap();
    assert_eq!(parsed.len(), 3);

    // Filter by user_id=alice: only alice
    let mut out2: *mut c_char = ptr::null_mut();
    unsafe {
        assert_eq!(
            memscale_hnsw_scoped_search(
                handle,
                emb.as_ptr(),
                10,
                cstring(r#"{"user_id":"alice"}"#).as_ptr(),
                &mut out2,
            ),
            0
        );
    }
    let results = unsafe { consume_string_out(out2, |p| membrain_string_free(p)) };
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&results).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0]["id"], serde_json::json!(alice_id));

    // Filter by role=admin (both alice and bob)
    let mut out3: *mut c_char = ptr::null_mut();
    unsafe {
        assert_eq!(
            memscale_hnsw_scoped_search(
                handle,
                emb.as_ptr(),
                10,
                cstring(r#"{"role":"admin"}"#).as_ptr(),
                &mut out3,
            ),
            0
        );
    }
    let results = unsafe { consume_string_out(out3, |p| membrain_string_free(p)) };
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&results).unwrap();
    assert_eq!(parsed.len(), 2);

    // Filter combo user_id+role: only alice
    let mut out4: *mut c_char = ptr::null_mut();
    unsafe {
        assert_eq!(
            memscale_hnsw_scoped_search(
                handle,
                emb.as_ptr(),
                10,
                cstring(r#"{"user_id":"alice","role":"admin"}"#).as_ptr(),
                &mut out4,
            ),
            0
        );
    }
    let results = unsafe { consume_string_out(out4, |p| membrain_string_free(p)) };
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&results).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0]["id"], serde_json::json!(alice_id));

    unsafe { memscale_hnsw_scoped_free(handle) };
}
