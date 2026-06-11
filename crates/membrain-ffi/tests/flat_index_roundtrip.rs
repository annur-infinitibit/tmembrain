//! End-to-end tests for the flat index C ABI.
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
    16
}

fn sample_id(index: usize) -> String {
    format!("00000000-0000-0000-0000-{:012x}", index)
}

#[test]
fn flat_add_then_search_returns_inserted_id() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    for index in 0..4 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        let code = unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
        assert_eq!(code, MEMBRAIN_OK, "add #{index} should succeed");
    }

    let query = common::embedding_json_for(0, dim);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { memscale_flat_index_search(handle, query.as_ptr(), 3, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);

    let json = unsafe {
        common::consume_string_out(out, |ptr| {
            membrain_string_free(ptr);
        })
    };
    let parsed: Vec<Value> = serde_json::from_str(&json).expect("parse");
    assert_eq!(parsed.len(), 3);
    let first_id = parsed[0]["id"].as_str().expect("id");
    assert_eq!(first_id, sample_id(0));

    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_remove_removes_from_index() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    let id = common::cstring(&sample_id(1));
    let embedding = common::embedding_json_for(1, dim);
    unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    let code = unsafe { memscale_flat_index_remove(handle, id.as_ptr()) };
    assert_eq!(code, MEMBRAIN_OK);

    let mut out_len: i64 = -1;
    let code = unsafe { memscale_flat_index_len(handle, &mut out_len) };
    assert_eq!(code, MEMBRAIN_OK);
    assert_eq!(out_len, 0);

    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_len_and_dimension_report_state() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    for index in 0..5 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    }
    let mut length: i64 = 0;
    assert_eq!(
        unsafe { memscale_flat_index_len(handle, &mut length) },
        MEMBRAIN_OK
    );
    assert_eq!(length, 5);
    let mut reported_dim: i64 = 0;
    assert_eq!(
        unsafe { memscale_flat_index_dimension(handle, &mut reported_dim) },
        MEMBRAIN_OK
    );
    assert_eq!(reported_dim as usize, dim);
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_metrics_returns_valid_json() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { memscale_flat_index_metrics(handle, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let value: Value = serde_json::from_str(&json).expect("parse metrics");
    assert!(value.is_object());
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_batch_search_returns_result_per_query() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    for index in 0..4 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    }
    let queries: Vec<Vec<f32>> = (0..2)
        .map(|index| {
            (0..dim)
                .map(|slot| ((index as f32 * 0.1) + (slot as f32 * 0.001)).sin())
                .collect()
        })
        .collect();
    let batch_json = common::cstring(&serde_json::to_string(&queries).expect("serialize"));
    let mut out: *mut c_char = ptr::null_mut();
    let code =
        unsafe { memscale_flat_index_batch_search(handle, batch_json.as_ptr(), 2, &mut out) };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let value: Value = serde_json::from_str(&json).expect("parse");
    let array = value.as_array().expect("array");
    assert_eq!(array.len(), 2);
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_new_with_config_accepts_valid_json() {
    let config = serde_json::json!({
        "dimension": 32,
        "distance_metric": "Cosine"
    });
    let config_c = common::cstring(&config.to_string());
    let handle = unsafe { memscale_flat_index_new_with_config(config_c.as_ptr()) };
    assert!(!handle.is_null());
    unsafe { memscale_flat_index_free(handle) };
}

#[test]
fn flat_search_with_filter_honours_allowed_ids() {
    let dim = dimension();
    let handle = memscale_flat_index_new(dim as u32);
    for index in 0..3 {
        let id = common::cstring(&sample_id(index));
        let embedding = common::embedding_json_for(index, dim);
        unsafe { memscale_flat_index_add(handle, id.as_ptr(), embedding.as_ptr()) };
    }
    let allowed_ids = serde_json::json!([sample_id(1)]);
    let allowed_c = common::cstring(&allowed_ids.to_string());
    let query = common::embedding_json_for(0, dim);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        memscale_flat_index_search_with_filter(
            handle,
            query.as_ptr(),
            5,
            allowed_c.as_ptr(),
            &mut out,
        )
    };
    assert_eq!(code, MEMBRAIN_OK);
    let json = unsafe { common::consume_string_out(out, |ptr| membrain_string_free(ptr)) };
    let hits: Vec<Value> = serde_json::from_str(&json).expect("parse");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["id"].as_str().expect("id"), sample_id(1));
    unsafe { memscale_flat_index_free(handle) };
}
