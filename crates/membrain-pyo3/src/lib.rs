//! PyO3-based Python bindings for Membrain — true async via `pyo3-async-runtimes`.
//!
//! This module is imported as `membrain._native` in Python. Every method on
//! `MembrainClient` that performs I/O returns a Python coroutine backed by a
//! Rust `Future` running on a shared Tokio runtime.
//!
//! # Architecture
//!
//! ```text
//! Python asyncio event loop
//! │
//! └─→ await client.store_fact(...)       ← Python coroutine
//!       │
//!       └─→ pyo3-async-runtimes          ← bridges Future ↔ coroutine
//!             │
//!             └─→ Tokio runtime          ← shared, managed by pyo3-async-runtimes
//!                   │
//!                   └─→ MembrainClient   ← pure async Rust (no block_on)
//! ```

use pyo3::prelude::*;

mod client;
mod graph;
mod types;

/// Native Python extension for Membrain — true async via PyO3.
/// Imported as `membrain._native` in Python.
#[pymodule]
#[pyo3(name = "_native")]
fn membrain_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    let _ = pyo3_async_runtimes::tokio::init(builder);
    // Register the client class
    m.add_class::<client::PyMembrainClient>()?;

    // Register the graph class
    m.add_class::<graph::PyMembrainGraph>()?;

    // Result types — registered so Python can inspect them
    m.add_class::<types::PyStoreResult>()?;
    m.add_class::<types::PySearchResults>()?;
    m.add_class::<types::PyMemoryEntry>()?;
    m.add_class::<types::PyMemoryInfo>()?;
    m.add_class::<types::PyCaseEntry>()?;
    m.add_class::<types::PyCaseSearchResults>()?;

    Ok(())
}
