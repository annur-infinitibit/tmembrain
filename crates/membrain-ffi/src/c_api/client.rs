use std::os::raw::c_char;
use std::ptr;

use membrain_core::config::Config;

use crate::MembrainClient;

use super::{
    block_in_c, cstr_to_str, get_client, parse_embedding_json, parse_metadata_json, set_last_error,
    write_json_out, write_store_result, MEMBRAIN_ERR_NULL_POINTER, MEMBRAIN_ERR_QUERY,
    MEMBRAIN_ERR_SERIALIZE, MEMBRAIN_OK,
};

/// Create a new Membrain client with default configuration.
/// Returns an opaque handle, or NULL on failure (check `membrain_last_error()`).
#[no_mangle]
pub extern "C" fn membrain_client_new() -> *mut MembrainClient {
    match block_in_c(MembrainClient::new()) {
        Ok(client) => Box::into_raw(Box::new(client)),
        Err(e) => {
            set_last_error(&format!("failed to create client: {}", e));
            ptr::null_mut()
        }
    }
}

/// Create a new Membrain client with JSON configuration.
/// `config_json` is a null-terminated JSON string. Pass NULL for defaults.
/// Returns an opaque handle, or NULL on failure.
///
/// # Safety
///
/// - `config_json` must be null or point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn membrain_client_new_with_config(config_json: *const c_char) -> *mut MembrainClient {
    // SAFETY: This function's caller contract defines `config_json` validity.
    unsafe {
        if config_json.is_null() {
            return membrain_client_new();
        }

        let json_str = match cstr_to_str(config_json) {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let config: Config = match serde_json::from_str(json_str) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&format!("invalid config JSON: {}", e));
                return ptr::null_mut();
            }
        };

        match block_in_c(MembrainClient::with_config(config)) {
            Ok(client) => Box::into_raw(Box::new(client)),
            Err(e) => {
                set_last_error(&format!("failed to create client: {}", e));
                ptr::null_mut()
            }
        }
    }
}

/// Destroy a Membrain client and free all associated resources.
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `client` must have been returned by `membrain_client_new` or
///   `membrain_client_new_with_config`, and must not have been freed already,
///   or be null.
#[no_mangle]
pub unsafe extern "C" fn membrain_client_free(client: *mut MembrainClient) {
    // SAFETY: This function's caller contract defines `client` provenance.
    unsafe {
        if !client.is_null() {
            drop(Box::from_raw(client));
        }
    }
}

// ---------------------------------------------------------------------------
// Store functions
// ---------------------------------------------------------------------------

/// Store a fact. Returns 0 on success, writes result JSON to `out_json`.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `statement` must be non-null and point to a valid null-terminated C string.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `metadata_json` must be null or point to a valid JSON object.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_fact(
    client: *mut MembrainClient,
    statement: *const c_char,
    confidence: f64,
    embedding_json: *const c_char,
    metadata_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    // SAFETY: All raw pointers are governed by the fn-level `# Safety` contract.
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let statement = match cstr_to_str(statement) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        let metadata = match parse_metadata_json(metadata_json) {
            Ok(meta) => meta,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_fact_with_embedding(statement, confidence, embedding, metadata)),
            out_json,
        )
    }
}

/// Store a preference.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `holder`, `subject`, `preference`, and `strength` must be non-null and
///   point to valid null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_preference(
    client: *mut MembrainClient,
    holder: *const c_char,
    subject: *const c_char,
    preference: *const c_char,
    strength: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let holder = match cstr_to_str(holder) { Ok(s) => s, Err(code) => return code };
        let subject = match cstr_to_str(subject) { Ok(s) => s, Err(code) => return code };
        let preference = match cstr_to_str(preference) { Ok(s) => s, Err(code) => return code };
        let strength = match cstr_to_str(strength) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_preference_with_embedding(
                holder,
                subject,
                preference,
                strength,
                embedding,
            )),
            out_json,
        )
    }
}

/// Store an event.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `event_type` and `description` must be non-null and point to valid
///   null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_event(
    client: *mut MembrainClient,
    event_type: *const c_char,
    description: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let event_type = match cstr_to_str(event_type) { Ok(s) => s, Err(code) => return code };
        let description = match cstr_to_str(description) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_event_with_embedding(event_type, description, embedding)),
            out_json,
        )
    }
}

/// Store an observation.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `content` must be non-null and point to a valid null-terminated C string.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_observation(
    client: *mut MembrainClient,
    content: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let content = match cstr_to_str(content) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_observation_with_embedding(content, embedding)),
            out_json,
        )
    }
}

/// Store a concept.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `name` and `definition` must be non-null and point to valid null-terminated
///   C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_concept(
    client: *mut MembrainClient,
    name: *const c_char,
    definition: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let name = match cstr_to_str(name) { Ok(s) => s, Err(code) => return code };
        let definition = match cstr_to_str(definition) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_concept_with_embedding(name, definition, embedding)),
            out_json,
        )
    }
}

/// Store an entity.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `name` and `entity_type` must be non-null and point to valid
///   null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_entity(
    client: *mut MembrainClient,
    name: *const c_char,
    entity_type: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let name = match cstr_to_str(name) { Ok(s) => s, Err(code) => return code };
        let entity_type = match cstr_to_str(entity_type) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_entity_with_embedding(name, entity_type, embedding)),
            out_json,
        )
    }
}

/// Store a workflow.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `name` and `description` must be non-null and point to valid
///   null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_workflow(
    client: *mut MembrainClient,
    name: *const c_char,
    description: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let name = match cstr_to_str(name) { Ok(s) => s, Err(code) => return code };
        let description = match cstr_to_str(description) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_workflow_with_embedding(name, description, embedding)),
            out_json,
        )
    }
}

/// Store a skill.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `name` and `description` must be non-null and point to valid
///   null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_skill(
    client: *mut MembrainClient,
    name: *const c_char,
    description: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let name = match cstr_to_str(name) { Ok(s) => s, Err(code) => return code };
        let description = match cstr_to_str(description) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_skill_with_embedding(name, description, embedding)),
            out_json,
        )
    }
}

/// Store a pattern.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `name`, `description`, and `pattern_type` must be non-null and point to
///   valid null-terminated C strings.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_pattern(
    client: *mut MembrainClient,
    name: *const c_char,
    description: *const c_char,
    pattern_type: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let name = match cstr_to_str(name) { Ok(s) => s, Err(code) => return code };
        let description = match cstr_to_str(description) { Ok(s) => s, Err(code) => return code };
        let pattern_type = match cstr_to_str(pattern_type) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_pattern_with_embedding(
                name,
                description,
                pattern_type,
                embedding,
            )),
            out_json,
        )
    }
}

/// Store a case (experience for case-based reasoning).
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `problem`, `plan`, and `outcome` must be non-null and point to valid
///   null-terminated C strings.
/// - `reward` is a f64 in range [0.0, 1.0] (clamped internally).
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_case(
    client: *mut MembrainClient,
    problem: *const c_char,
    plan: *const c_char,
    outcome: *const c_char,
    reward: f64,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let problem = match cstr_to_str(problem) { Ok(s) => s, Err(code) => return code };
        let plan = match cstr_to_str(plan) { Ok(s) => s, Err(code) => return code };
        let outcome = match cstr_to_str(outcome) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_case_with_embedding(
                problem, plan, outcome, reward, embedding,
            )),
            out_json,
        )
    }
}

/// Store a goal.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `description` must be non-null and point to a valid null-terminated C string.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_goal(
    client: *mut MembrainClient,
    description: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let description = match cstr_to_str(description) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_goal_with_embedding(description, embedding)),
            out_json,
        )
    }
}

/// Store a task.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `title` must be non-null and point to a valid null-terminated C string.
/// - `embedding_json` must be null or point to a valid JSON array of f32 values.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_store_task(
    client: *mut MembrainClient,
    title: *const c_char,
    embedding_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let title = match cstr_to_str(title) { Ok(s) => s, Err(code) => return code };
        let embedding = match parse_embedding_json(embedding_json) {
            Ok(emb) => emb,
            Err(code) => return code,
        };
        write_store_result(
            block_in_c(client.store_task_with_embedding(title, embedding)),
            out_json,
        )
    }
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

/// Search for memories. Writes result JSON to `out_json`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `query` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_search(
    client: *mut MembrainClient,
    query: *const c_char,
    limit: i32,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let query = match cstr_to_str(query) { Ok(s) => s, Err(code) => return code };
        let limit = if limit <= 0 { 10 } else { limit as usize };

        match block_in_c(client.search(query, limit)) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(json) => write_json_out(out_json, &json),
                Err(e) => {
                    set_last_error(&format!("failed to serialize results: {}", e));
                    MEMBRAIN_ERR_SERIALIZE
                }
            },
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Search for memories with filters. Writes result JSON to `out_json`.
/// `filters_json` is a nullable JSON string with filter criteria.
/// Pass NULL for no filters (equivalent to `membrain_search`).
///
/// Filters JSON format:
///   {"memory_types":["semantic_fact"],"min_confidence":0.7,
///    "tags":["important"],"agent_id":"uuid",
///    "metadata":{"source":"arxiv","year":2024}}
///
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `query` must be non-null and point to a valid null-terminated C string.
/// - `filters_json` must be null or point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_search_with_filters(
    client: *mut MembrainClient,
    query: *const c_char,
    limit: i32,
    filters_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let query = match cstr_to_str(query) { Ok(s) => s, Err(code) => return code };
        let limit = if limit <= 0 { 10 } else { limit as usize };

        let filters = if filters_json.is_null() {
            None
        } else {
            let json_str = match cstr_to_str(filters_json) { Ok(s) => s, Err(code) => return code };
            match serde_json::from_str::<crate::SearchFiltersJson>(json_str) {
                Ok(f) => Some(f),
                Err(e) => {
                    set_last_error(&format!("invalid filters JSON: {}", e));
                    return MEMBRAIN_ERR_QUERY;
                }
            }
        };

        match block_in_c(client.search_with_filters(query, limit, filters)) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(json) => write_json_out(out_json, &json),
                Err(e) => {
                    set_last_error(&format!("failed to serialize results: {}", e));
                    MEMBRAIN_ERR_SERIALIZE
                }
            },
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Get a memory by ID. Writes result JSON to `out_json`.
/// If the memory is not found, writes `"null"` to `out_json`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `id` must be non-null and point to a valid null-terminated C string.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_get(
    client: *mut MembrainClient,
    id: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let id = match cstr_to_str(id) { Ok(s) => s, Err(code) => return code };

        match block_in_c(client.get(id)) {
            Ok(info) => {
                let json = match &info {
                    Some(m) => match serde_json::to_string(m) {
                        Ok(j) => j,
                        Err(e) => {
                            set_last_error(&format!("failed to serialize memory: {}", e));
                            return MEMBRAIN_ERR_SERIALIZE;
                        }
                    },
                    None => "null".to_string(),
                };
                write_json_out(out_json, &json)
            }
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Delete a memory by ID. Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `id` must be non-null and point to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn membrain_delete(
    client: *mut MembrainClient,
    id: *const c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        let id = match cstr_to_str(id) { Ok(s) => s, Err(code) => return code };

        match block_in_c(client.delete(id)) {
            Ok(_) => MEMBRAIN_OK,
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Get the total memory count. Writes the count to `out_count`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `out_count` must be non-null and point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_count(
    client: *mut MembrainClient,
    out_count: *mut i64,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };
        if out_count.is_null() {
            set_last_error("null out_count pointer");
            return MEMBRAIN_ERR_NULL_POINTER;
        }

        match block_in_c(client.count()) {
            Ok(count) => {
                // SAFETY: `out_count` validated non-null by this function.
                *out_count = count as i64;
                MEMBRAIN_OK
            }
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Get storage statistics. Writes result JSON to `out_json`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `out_json` must be null or point to valid writable memory.
#[no_mangle]
pub unsafe extern "C" fn membrain_stats(
    client: *mut MembrainClient,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };

        match block_in_c(client.stats()) {
            Ok(stats) => match serde_json::to_string(&stats) {
                Ok(json) => write_json_out(out_json, &json),
                Err(e) => {
                    set_last_error(&format!("failed to serialize stats: {}", e));
                    MEMBRAIN_ERR_SERIALIZE
                }
            },
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vector backend health & stats
// ---------------------------------------------------------------------------

/// Check vector backend health status. Writes result JSON to `out_json`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `out_json` must be a valid pointer. Free the result with `membrain_string_free`.
#[no_mangle]
pub unsafe extern "C" fn membrain_vector_backend_health(
    client: *mut MembrainClient,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };

        match block_in_c(client.vector_backend_health()) {
            Ok(health) => match serde_json::to_string(&health) {
                Ok(json) => write_json_out(out_json, &json),
                Err(e) => {
                    set_last_error(&format!("failed to serialize health result: {}", e));
                    MEMBRAIN_ERR_SERIALIZE
                }
            },
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}

/// Get vector backend capabilities and statistics. Writes result JSON to `out_json`.
/// Returns 0 on success.
///
/// # Safety
///
/// - `client` must be a valid pointer returned by `membrain_client_new`.
/// - `out_json` must be a valid pointer. Free the result with `membrain_string_free`.
#[no_mangle]
pub unsafe extern "C" fn membrain_vector_backend_stats(
    client: *mut MembrainClient,
    out_json: *mut *mut c_char,
) -> i32 {
    unsafe {
        let client = match get_client(client) { Ok(c) => c, Err(code) => return code };

        match block_in_c(client.vector_backend_stats()) {
            Ok(stats) => match serde_json::to_string(&stats) {
                Ok(json) => write_json_out(out_json, &json),
                Err(e) => {
                    set_last_error(&format!("failed to serialize backend stats: {}", e));
                    MEMBRAIN_ERR_SERIALIZE
                }
            },
            Err(e) => {
                set_last_error(&format!("{}", e));
                MEMBRAIN_ERR_QUERY
            }
        }
    }
}
