//! Thread-safe FFI bindings for all concurrent index variants.
//!
//! Each submodule mirrors the same lifecycle (`new`, `new_with_config`,
//! `clone`, `free`) and operation (`add`, `remove`, `search`, `len`,
//! `dimension`) surface for its index variant. Helpers live in the parent
//! `c_api` module.

mod flat;
mod hnsw;
mod ivf;
mod lsh;
mod vamana;

pub use flat::*;
pub use hnsw::*;
pub use ivf::*;
pub use lsh::*;
pub use vamana::*;
