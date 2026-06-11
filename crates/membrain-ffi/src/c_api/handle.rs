//! Generic opaque-handle helpers for the C FFI.
//!
//! Every `pub unsafe extern "C" fn` in the C ABI accepts opaque pointers
//! produced by `Box::into_raw`. Dereferencing those pointers is identical
//! across index variants: null check → set thread-local error → cast. This
//! module centralises that pattern so each variant no longer carries a
//! private copy.
//!
//! These helpers are `unsafe fn` because callers pass raw pointers whose
//! validity we cannot prove locally. The helpers only enforce the checks
//! that are machine-checkable (null, alignment is implicit for `Box` output);
//! validity of the backing allocation is still the caller's contract.

use super::{set_last_error, MEMBRAIN_ERR_NULL_POINTER};

/// Dereference a `*const T` opaque FFI handle.
///
/// # Safety
///
/// - `ptr` must either be null, or point to a live `T` that was allocated
///   by `Box::into_raw::<T>` and not yet freed.
/// - While the returned reference is alive, the backing allocation must not
///   be dropped and must not be mutably aliased.
pub(crate) unsafe fn as_ref_or<'a, T>(ptr: *const T, null_msg: &str) -> Result<&'a T, i32> {
    if ptr.is_null() {
        set_last_error(null_msg);
        return Err(MEMBRAIN_ERR_NULL_POINTER);
    }
    // SAFETY: non-null checked above; caller contract guarantees the
    // allocation is live and not mutably aliased for the borrow's lifetime.
    Ok(unsafe { &*ptr })
}

/// Dereference a `*mut T` opaque FFI handle as a unique mutable reference.
///
/// # Safety
///
/// - `ptr` must either be null, or point to a live `T` that was allocated
///   by `Box::into_raw::<T>` and not yet freed.
/// - While the returned reference is alive, no other reference (shared or
///   mutable) to the same `T` may exist.
pub(crate) unsafe fn as_mut_or<'a, T>(ptr: *mut T, null_msg: &str) -> Result<&'a mut T, i32> {
    if ptr.is_null() {
        set_last_error(null_msg);
        return Err(MEMBRAIN_ERR_NULL_POINTER);
    }
    // SAFETY: non-null checked above; caller contract guarantees exclusive
    // access for the borrow's lifetime.
    Ok(unsafe { &mut *ptr })
}

/// Free an opaque handle previously returned by `Box::into_raw::<T>`.
///
/// Passing null is a no-op.
///
/// # Safety
///
/// - `ptr` must be null, or have been returned by `Box::into_raw::<T>` and
///   not yet freed. Callers that cloned the handle via an `Arc` must still
///   call this exactly once per cloned handle — the `Arc` refcount is
///   tracked inside the boxed value.
pub(crate) unsafe fn drop_boxed<T>(ptr: *mut T) {
    if !ptr.is_null() {
        // SAFETY: non-null checked; `ptr` came from `Box::into_raw` by contract.
        drop(unsafe { Box::from_raw(ptr) });
    }
}
