use std::os::raw::c_char;
use std::ptr;

use membrain_core::types::{Confidence, Embedding, MemoryId};
use membrain_graph::{GraphConfig, MemoryGraph};

use super::{
    as_ref_or, cstr_to_str, drop_boxed, set_last_error, write_json_out, MEMBRAIN_ERR_GRAPH,
    MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_SERIALIZE, MEMBRAIN_OK,
};

const NULL_MSG: &str = "null graph pointer";

#[inline]
unsafe fn get_graph<'a>(graph: *mut MemoryGraph) -> Result<&'a MemoryGraph, i32> {
    // SAFETY: delegates to `as_ref_or`; caller contract carries forward.
    unsafe { as_ref_or(graph, NULL_MSG) }
}

// ---------------------------------------------------------------------------
// Graph lifecycle
// ---------------------------------------------------------------------------

/// Create a new MemoryGraph with default configuration.
/// Returns an opaque handle, or NULL on failure.
#[no_mangle]
pub extern "C" fn membrain_graph_new() -> *mut MemoryGraph {
    let graph = MemoryGraph::new(GraphConfig::default());
    Box::into_raw(Box::new(graph))
}

/// Create a new MemoryGraph with JSON configuration.
/// Pass NULL for defaults.
///
/// # Safety
///
/// - `config_json` must be null or point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_new_with_config(
    config_json: *const c_char,
) -> *mut MemoryGraph {
    // SAFETY: `config_json` validity is part of the fn-level `# Safety` contract.
    unsafe {
        if config_json.is_null() {
            return membrain_graph_new();
        }

        let json_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let config: GraphConfig = match serde_json::from_str(json_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid graph config JSON: {}", e));
                return ptr::null_mut();
            }
        };

        let graph = MemoryGraph::new(config);
        Box::into_raw(Box::new(graph))
    }
}

/// Destroy a MemoryGraph and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `graph` must have been returned by `membrain_graph_new` or
///   `membrain_graph_new_with_config`, and must not have been freed already,
///   or be null.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_free(graph: *mut MemoryGraph) {
    // SAFETY: `drop_boxed` accepts null or a `Box::into_raw` pointer — our
    // caller contract guarantees exactly that.
    unsafe { drop_boxed(graph) }
}

// ---------------------------------------------------------------------------
// Graph node operations
// ---------------------------------------------------------------------------

/// Add a node to the graph.
/// `memory_id` is a UUID string.
/// `embedding_json` is a JSON array of floats, e.g. "[0.1, 0.2, ...]".
/// `confidence` is a float in [0.0, 1.0].
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `memory_id` and `embedding_json` must be non-null and point to valid
///   null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_add_node(
    graph: *mut MemoryGraph,
    memory_id: *const c_char,
    embedding_json: *const c_char,
    confidence: f64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(memory_id) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let emb_str = match cstr_to_str(embedding_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let mid: MemoryId = match id_str.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid memory_id: {}", e));
                return MEMBRAIN_ERR_GRAPH;
            }
        };

        let values: Vec<f32> = match serde_json::from_str(emb_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid embedding JSON: {}", e));
                return MEMBRAIN_ERR_GRAPH;
            }
        };

        let embedding = Embedding::new(values);
        let conf = Confidence::new(confidence);

        match graph.add_node(mid, &embedding, conf) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_GRAPH
            }
        }
    }
}

/// Remove a node (and its incident edges) from the graph.
/// `memory_id` is a UUID string.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `memory_id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_remove_node(
    graph: *mut MemoryGraph,
    memory_id: *const c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        let id_str = match cstr_to_str(memory_id) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let mid: MemoryId = match id_str.parse() {
            Ok(id) => id,
            Err(e) => {
                set_last_error(&format!("invalid memory_id: {}", e));
                return MEMBRAIN_ERR_GRAPH;
            }
        };

        match graph.remove_node(&mid) {
            Ok(()) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_GRAPH
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Graph query
// ---------------------------------------------------------------------------

/// Multi-hop graph query.
/// `query_embedding_json` is a JSON array of floats.
/// `max_hops` <= 0 uses the configured default.
/// `top_k` <= 0 defaults to 10.
/// Writes result JSON to `out_json`.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `query_embedding_json` must be non-null and point to a valid
///   null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_query(
    graph: *mut MemoryGraph,
    query_embedding_json: *const c_char,
    max_hops: i32,
    top_k: i32,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        let emb_str = match cstr_to_str(query_embedding_json) {
            Ok(s) => s,
            Err(code) => return code,
        };

        let values: Vec<f32> = match serde_json::from_str(emb_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("invalid embedding JSON: {}", e));
                return MEMBRAIN_ERR_GRAPH;
            }
        };

        let hops = if max_hops <= 0 {
            None
        } else {
            Some(max_hops as usize)
        };
        let k = if top_k <= 0 { 10 } else { top_k as usize };

        let result = membrain_graph::graph_query(graph, &values, hops, k);
        let json_result: crate::GraphQueryResultJson = result.into();

        match serde_json::to_string(&json_result) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize graph query result: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Graph info
// ---------------------------------------------------------------------------

/// Get graph node count.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_node_count(
    graph: *mut MemoryGraph,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }
        *out_count = graph.node_count() as i64;
        MEMBRAIN_OK
    }
}

/// Get graph edge count.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_edge_count(
    graph: *mut MemoryGraph,
    out_count: *mut i64,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }
        *out_count = graph.edge_count() as i64;
        MEMBRAIN_OK
    }
}

// ---------------------------------------------------------------------------
// Graph pruning
// ---------------------------------------------------------------------------

/// Manually trigger graph pruning. Writes result JSON to `out_json`.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_prune(
    graph: *mut MemoryGraph,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };
        let result = membrain_graph::prune(graph);
        let json_result: crate::GraphPruningResultJson = result.into();

        match serde_json::to_string(&json_result) {
            Ok(json) => write_json_out(out_json, &json),
            Err(e) => {
                set_last_error(&format!("failed to serialize pruning result: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Graph persistence
// ---------------------------------------------------------------------------

/// Save graph state to a base64-encoded string. Writes to `out_data`.
/// The caller must free the returned string with `membrain_string_free()`.
///
/// # Safety
///
/// - `graph` must be a valid pointer returned by `membrain_graph_new`.
/// - `out_data` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_save(
    graph: *mut MemoryGraph,
    out_data: *mut *mut c_char,
) -> i32 {
    // SAFETY: raw-pointer preconditions are stated in the fn-level `# Safety`
    // doc. Each unsafe call within this block relies on the same contract.
    unsafe {
        use base64::Engine;

        let graph = match get_graph(graph) {
            Ok(g) => g,
            Err(code) => return code,
        };

        match membrain_graph::save_to_bytes(graph) {
            Ok(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                write_json_out(out_data, &encoded)
            }
            Err(e) => {
                set_last_error(&format!("failed to save graph: {}", e));
                MEMBRAIN_ERR_SERIALIZE
            }
        }
    }
}

/// Load graph state from a base64-encoded string.
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `data` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn membrain_graph_load(data: *const c_char) -> *mut MemoryGraph {
    // SAFETY: `data` validity is part of the fn-level `# Safety` contract.
    unsafe {
        use base64::Engine;

        let data_str = match cstr_to_str(data) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let bytes = match base64::engine::general_purpose::STANDARD.decode(data_str) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(&format!("invalid base64 data: {}", e));
                return ptr::null_mut();
            }
        };

        match membrain_graph::load_from_bytes(&bytes) {
            Ok(graph) => Box::into_raw(Box::new(graph)),
            Err(e) => {
                set_last_error(&format!("failed to load graph: {}", e));
                ptr::null_mut()
            }
        }
    }
}
