//! Concurrency stress tests for the concurrent C API wrappers.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

mod common;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use memscaledb::{ConcurrentFlatIndex, ConcurrentVectorIndex};
use membrain_ffi::c_api::*;

const MEMBRAIN_OK: i32 = 0;

fn sample_id(index: usize) -> String {
    format!("00000000-0000-0000-0000-{:012x}", index)
}

fn test_vector(offset: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|slot| ((offset as f32 * 0.01) + (slot as f32 * 0.001)).sin())
        .collect()
}

#[test]
fn concurrent_flat_parallel_adds_preserve_count() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 25;
    const DIM: usize = 16;

    let handle = memscale_concurrent_flat_index_new(DIM as u32);
    assert!(!handle.is_null());
    // SAFETY: `handle` was returned by `memscale_concurrent_flat_index_new`.
    let arc: Arc<ConcurrentFlatIndex> = unsafe { (*handle).clone() };

    let mut handles = Vec::new();
    for thread_index in 0..THREADS {
        let arc = Arc::clone(&arc);
        handles.push(thread::spawn(move || {
            for slot in 0..PER_THREAD {
                let offset = thread_index * PER_THREAD + slot;
                let id: memscaledb::VectorId = sample_id(offset).parse().expect("uuid parse");
                let vector = test_vector(offset, DIM);
                arc.add_concurrent(id, &vector).expect("add");
            }
        }));
    }
    for handle in handles {
        handle.join().expect("join");
    }

    let mut length: i64 = 0;
    let code = unsafe { memscale_concurrent_flat_index_len(handle, &mut length) };
    assert_eq!(code, MEMBRAIN_OK);
    assert_eq!(length as usize, THREADS * PER_THREAD);

    unsafe { memscale_concurrent_flat_index_free(handle) };
}

#[test]
fn concurrent_flat_clone_shares_state() {
    const DIM: usize = 8;
    let original = memscale_concurrent_flat_index_new(DIM as u32);
    let cloned = unsafe { memscale_concurrent_flat_index_clone(original) };
    assert!(!cloned.is_null());

    let id = common::cstring(&sample_id(42));
    let vector = test_vector(42, DIM);
    let code = unsafe {
        memscale_concurrent_flat_index_add(original, id.as_ptr(), vector.as_ptr(), DIM as u32)
    };
    assert_eq!(code, MEMBRAIN_OK);

    let mut length_original: i64 = 0;
    let mut length_clone: i64 = 0;
    unsafe { memscale_concurrent_flat_index_len(original, &mut length_original) };
    unsafe { memscale_concurrent_flat_index_len(cloned, &mut length_clone) };
    assert_eq!(length_original, length_clone);
    assert_eq!(length_original, 1);

    unsafe { memscale_concurrent_flat_index_free(original) };
    unsafe { memscale_concurrent_flat_index_free(cloned) };
}

#[test]
fn concurrent_flat_search_during_insert_survives() {
    const DIM: usize = 16;
    let handle = memscale_concurrent_flat_index_new(DIM as u32);
    let arc: Arc<ConcurrentFlatIndex> = unsafe { (*handle).clone() };
    static SEARCH_HITS: AtomicUsize = AtomicUsize::new(0);

    let inserter = {
        let arc = Arc::clone(&arc);
        thread::spawn(move || {
            for offset in 0..200 {
                let id: memscaledb::VectorId = sample_id(offset).parse().expect("uuid");
                let vector = test_vector(offset, DIM);
                arc.add_concurrent(id, &vector).expect("add");
            }
        })
    };

    let searcher = {
        let arc = Arc::clone(&arc);
        thread::spawn(move || {
            let query = test_vector(0, DIM);
            for _ in 0..50 {
                match arc.search(&query, 3) {
                    Ok(hits) => {
                        SEARCH_HITS.fetch_add(hits.len(), Ordering::SeqCst);
                    }
                    Err(_) => {}
                }
            }
        })
    };

    inserter.join().expect("inserter");
    searcher.join().expect("searcher");

    assert!(arc.len() == 200);
    let _ = SEARCH_HITS.load(Ordering::SeqCst);
    unsafe { memscale_concurrent_flat_index_free(handle) };
}

#[test]
fn concurrent_flat_dimension_via_c_api() {
    const DIM: usize = 24;
    let handle = memscale_concurrent_flat_index_new(DIM as u32);
    let mut dim: i64 = 0;
    let code = unsafe { memscale_concurrent_flat_index_dimension(handle, &mut dim) };
    assert_eq!(code, MEMBRAIN_OK);
    assert_eq!(dim as usize, DIM);
    unsafe { memscale_concurrent_flat_index_free(handle) };
}

#[test]
fn concurrent_flat_len_zero_after_new() {
    let handle = memscale_concurrent_flat_index_new(8);
    let mut length: i64 = -1;
    let code = unsafe { memscale_concurrent_flat_index_len(handle, &mut length) };
    assert_eq!(code, MEMBRAIN_OK);
    assert_eq!(length, 0);
    unsafe { memscale_concurrent_flat_index_free(handle) };
}

#[test]
fn concurrent_flat_remove_reduces_len() {
    const DIM: usize = 8;
    let handle = memscale_concurrent_flat_index_new(DIM as u32);
    let id = common::cstring(&sample_id(7));
    let vector = test_vector(7, DIM);
    unsafe {
        memscale_concurrent_flat_index_add(handle, id.as_ptr(), vector.as_ptr(), DIM as u32)
    };
    let mut found: i32 = 0;
    unsafe { memscale_concurrent_flat_index_remove(handle, id.as_ptr(), &mut found) };
    let mut length: i64 = -1;
    unsafe { memscale_concurrent_flat_index_len(handle, &mut length) };
    assert_eq!(length, 0);
    unsafe { memscale_concurrent_flat_index_free(handle) };
}

#[test]
fn concurrent_flat_search_returns_match() {
    const DIM: usize = 8;
    let handle = memscale_concurrent_flat_index_new(DIM as u32);
    let id = common::cstring(&sample_id(1));
    let vector = test_vector(1, DIM);
    unsafe {
        memscale_concurrent_flat_index_add(handle, id.as_ptr(), vector.as_ptr(), DIM as u32)
    };
    let query = test_vector(1, DIM);
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe {
        memscale_concurrent_flat_index_search(handle, query.as_ptr(), DIM as u32, 1, &mut out)
    };
    assert_eq!(code, MEMBRAIN_OK);
    let text = if out.is_null() {
        String::new()
    } else {
        let cstr = unsafe { CStr::from_ptr(out) };
        let text = cstr.to_string_lossy().into_owned();
        unsafe { membrain_string_free(out) };
        text
    };
    assert!(text.contains(&sample_id(1)));
    unsafe { memscale_concurrent_flat_index_free(handle) };
}
