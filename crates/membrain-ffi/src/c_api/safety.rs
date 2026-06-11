//! Bounds-checked FFI helpers.
//!
//! Every C boundary that receives a raw pointer + length pair MUST route the
//! pointer through `safe_slice`. It enforces:
//! 1. non-null pointer
//! 2. proper alignment for `T`
//! 3. overflow-free byte-length computation (`len * size_of::<T>()` fits in `usize`)
//!
//! It cannot validate that the buffer actually contains `len` elements of
//! valid memory — that is the caller's contract.

use std::mem;

use super::{set_last_error, MEMBRAIN_ERR_NULL_POINTER};

pub(crate) const MEMBRAIN_ERR_OVERFLOW: i32 = -20;
pub(crate) const MEMBRAIN_ERR_MISALIGNED: i32 = -21;

/// Convert a `(ptr, len)` pair into a safely-bounded `&[T]`.
///
/// # Safety
/// Caller must guarantee that `ptr` references at least `len` contiguous,
/// initialized elements of `T` for the duration of the returned borrow. The
/// helper itself only checks null / alignment / length-overflow; it cannot
/// verify the backing allocation size.
pub(crate) unsafe fn safe_slice<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], i32> {
    unsafe {
        if ptr.is_null() {
            set_last_error("null pointer");
            return Err(MEMBRAIN_ERR_NULL_POINTER);
        }
        if len.checked_mul(mem::size_of::<T>()).is_none() {
            set_last_error("length * size_of::<T> overflows usize");
            return Err(MEMBRAIN_ERR_OVERFLOW);
        }
        if !(ptr as usize).is_multiple_of(mem::align_of::<T>()) {
            set_last_error("pointer misaligned for target type");
            return Err(MEMBRAIN_ERR_MISALIGNED);
        }
        // SAFETY: caller contract — ptr is non-null, aligned, and references at
        // least `len` initialized elements. Length overflow was checked above.
        Ok(std::slice::from_raw_parts(ptr, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_null() {
        let err = unsafe { safe_slice::<f32>(std::ptr::null(), 4) }.unwrap_err();
        assert_eq!(err, MEMBRAIN_ERR_NULL_POINTER);
    }

    #[test]
    fn rejects_misaligned() {
        let buf = [0u8; 16];
        let misaligned = unsafe { buf.as_ptr().add(1) } as *const f32;
        let err = unsafe { safe_slice::<f32>(misaligned, 2) }.unwrap_err();
        assert_eq!(err, MEMBRAIN_ERR_MISALIGNED);
    }

    #[test]
    fn rejects_overflow() {
        let buf = [0f32; 4];
        let err = unsafe { safe_slice::<f32>(buf.as_ptr(), usize::MAX) }.unwrap_err();
        assert_eq!(err, MEMBRAIN_ERR_OVERFLOW);
    }

    #[test]
    fn accepts_valid() {
        let buf = [1.0f32, 2.0, 3.0, 4.0];
        let slice = unsafe { safe_slice(buf.as_ptr(), 4) }.unwrap();
        assert_eq!(slice, &buf);
    }

    // --- end-to-end: exercise safe_slice through a real FFI boundary fn ---

    use std::ffi::{CStr, CString};

    use crate::c_api::{
        membrain_last_error, memscale_concurrent_flat_index_add,
        memscale_concurrent_flat_index_free, memscale_concurrent_flat_index_new,
        memscale_concurrent_flat_index_search, MEMBRAIN_OK,
    };

    fn last_error_str() -> String {
        let ptr = membrain_last_error();
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    #[test]
    fn ffi_add_null_vector_rejected() {
        let index = memscale_concurrent_flat_index_new(4);
        assert!(!index.is_null());
        let id = CString::new("00000000-0000-0000-0000-000000000001").unwrap();
        // NULL vector pointer should surface as NULL_POINTER, not a crash.
        let code =
            unsafe { memscale_concurrent_flat_index_add(index, id.as_ptr(), std::ptr::null(), 4) };
        assert_eq!(code, MEMBRAIN_ERR_NULL_POINTER);
        assert!(!last_error_str().is_empty());
        unsafe { memscale_concurrent_flat_index_free(index) };
    }

    #[test]
    fn ffi_search_misaligned_query_rejected() {
        let index = memscale_concurrent_flat_index_new(4);
        assert!(!index.is_null());
        // Build a misaligned f32 pointer from a byte buffer.
        let buf = [0u8; 32];
        let misaligned = unsafe { buf.as_ptr().add(1) } as *const f32;
        let mut out_json: *mut i8 = std::ptr::null_mut();
        let code = unsafe {
            memscale_concurrent_flat_index_search(index, misaligned, 4, 5, &mut out_json as *mut _)
        };
        assert_eq!(code, MEMBRAIN_ERR_MISALIGNED);
        unsafe { memscale_concurrent_flat_index_free(index) };
    }

    #[test]
    fn ffi_search_oversize_len_rejected() {
        let index = memscale_concurrent_flat_index_new(4);
        assert!(!index.is_null());
        let buf = [0.0f32; 4];
        let mut out_json: *mut i8 = std::ptr::null_mut();
        // dimension is u32 so the widest len we can pass is u32::MAX. Pair it
        // with a real (small) buffer; safe_slice should flag the overflow
        // when len * size_of::<f32>() exceeds usize on 32-bit, or succeed on
        // 64-bit. On 64-bit, use the separate overflow unit test (above) —
        // here we just verify the call does not segfault.
        let code = unsafe {
            memscale_concurrent_flat_index_search(
                index,
                buf.as_ptr(),
                u32::MAX,
                5,
                &mut out_json as *mut _,
            )
        };
        // Accept either overflow (32-bit) or a different index error: the
        // point is that we get a structured return, never UB.
        assert!(code != MEMBRAIN_OK);
        unsafe { memscale_concurrent_flat_index_free(index) };
    }
}
