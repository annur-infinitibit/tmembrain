//! End-to-end tests for the HNSW index C ABI.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

mod common;

use std::os::raw::c_char;
use std::ptr;

use membrain_ffi::c_api::*;
use serde_json::Value;

const MEMBRAIN_OK: i32 = 0;

fn dimension() -> usize {
    32
}

fn sample_id(index: usize) -> String {
    format!("00000000-0000-0000-0000-{:012x}", index + 100)
}

#[test]
fn hnsw_add_search_roundtrip() {
    let dim = dimension();
    let handle = memscale_index_new(dim as u32);
    for index in 0..6 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        let code = unsafe { memscale_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
        assert_eq!(code, MEMBRAIN_OK);
    }
    let query = common::embedding_json_for(0, dim);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { memscale_index_search(handle, query.as_ptr(), 3, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let hits: Vec<Value> = serde_json::from_str(&json).expect("parse");
    assert_eq!(hits.len(), 3);
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_remove_decreases_len() {
    let dim = dimension();
    let handle = memscale_index_new(dim as u32);
    let id = common::cstring(&sample_id(1));
    let embedding = common::embedding_json_for(1, dim);
    unsafe { memscale_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    let mut length: i64 = 0;
    unsafe { memscale_index_len(handle, &mut length) };
    assert_eq!(length, 1);
    unsafe { memscale_index_remove(handle, id.as_ptr()) };
    unsafe { memscale_index_len(handle, &mut length) };
    assert_eq!(length, 0);
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_len_and_dimension() {
    let dim = dimension();
    let handle = memscale_index_new(dim as u32);
    let mut dim_out: i64 = 0;
    let code = unsafe { memscale_index_dimension(handle, &mut dim_out) };
    assert_eq!(code, MEMBRAIN_OK);
    assert_eq!(dim_out as usize, dim);
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_metrics_returns_json() {
    let dim = dimension();
    let handle = memscale_index_new(dim as u32);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { memscale_index_metrics(handle, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let value: Value = serde_json::from_str(&json).expect("parse");
    assert!(value.is_object());
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_batch_search_returns_array_per_query() {
    let dim = dimension();
    let handle = memscale_index_new(dim as u32);
    for index in 0..5 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        unsafe { memscale_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    }
    let queries: Vec<Vec<f32>> = (0..3)
        .map(|index| {
            (0..dim)
                .map(|slot| ((index as f32 * 0.1) + (slot as f32 * 0.001)).sin())
                .collect()
        })
        .collect();
    let batch = common::cstring(&serde_json::to_string(&queries).expect("serialize"));
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { memscale_index_batch_search(handle, batch.as_ptr(), 2, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let value: Value = serde_json::from_str(&json).expect("parse");
    assert_eq!(value.as_array().expect("array").len(), 3);
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_new_with_valid_config_succeeds() {
    let config = serde_json::json!({
        "dimension": 16,
        "m": 16,
        "ef_construction": 100,
        "ef": 50,
        "distance_metric": "Cosine"
    });
    let config_c = common::cstring(&config.to_string());
    let handle = unsafe { memscale_index_new_with_config(config_c.as_ptr()) };
    assert!(!handle.is_null());
    unsafe { memscale_index_free(handle) };
}

#[test]
fn hnsw_new_with_bad_config_returns_null() {
    let config_c = common::cstring("not valid json");
    let handle = unsafe { memscale_index_new_with_config(config_c.as_ptr()) };
    assert!(handle.is_null());
}
