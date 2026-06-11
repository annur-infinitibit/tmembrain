/**
 * Membrain client for the memory system.
 */

import koffi from "koffi";

import type {
  CaseEntry,
  CaseSearchResults,
  MemoryInfo,
  SearchFilters,
  SearchResults,
  StorageStats,
  StoreResult,
} from "./types";

import {
  findLibrary,
  int64Ptr,
  MEMBRAIN_OK,
  MembrainError,
  newInt64Out,
  newStringOut,
  readInt64,
  readStringOut,
  voidPtr,
  voidPtrPtr,
} from "./ffi";

// Koffi's `func()` returns a callable `KoffiFunction`. It is still variadic
// at the type level (koffi can't derive per-binding signatures from our
// declaration strings) but at least the identifier is known to be a
// foreign function, not `any`. Per-binding signature migration is tracked
// as a follow-up.
type FFIFn = ReturnType<ReturnType<typeof koffi.load>["func"]>;

/**
 * Options for constructing a MembrainClient.
 */
export interface MembrainClientOptions {
  /** Optional path to the membrain FFI shared library. */
  libPath?: string;
  /**
   * Default metadata scope auto-applied on every store and every search.
   * Per-call metadata / filter entries override on the same key.
   */
  scope?: Record<string, unknown>;
  /**
   * Metadata keys to index at the storage layer. Queries constraining only
   * these keys pre-filter in O(matching) instead of O(total).
   */
  indexedMetadataKeys?: string[];
}

export class MembrainClient {
  private handle: unknown;
  private lib: ReturnType<typeof koffi.load>;
  private _scope: Record<string, unknown>;

  // Bound function references (all are koffi foreign functions).
  private fn_last_error!: FFIFn;
  private fn_string_free!: FFIFn;
  private fn_client_free!: FFIFn;
  private fn_store_fact!: FFIFn;
  private fn_store_preference!: FFIFn;
  private fn_store_event!: FFIFn;
  private fn_store_observation!: FFIFn;
  private fn_store_concept!: FFIFn;
  private fn_store_entity!: FFIFn;
  private fn_store_workflow!: FFIFn;
  private fn_store_skill!: FFIFn;
  private fn_store_pattern!: FFIFn;
  private fn_store_case!: FFIFn;
  private fn_store_goal!: FFIFn;
  private fn_store_task!: FFIFn;
  private fn_search!: FFIFn;
  private fn_search_with_filters!: FFIFn;
  private fn_get!: FFIFn;
  private fn_delete!: FFIFn;
  private fn_count!: FFIFn;
  private fn_stats!: FFIFn;
  private fn_vector_backend_health!: FFIFn;
  private fn_vector_backend_stats!: FFIFn;

  constructor(
    config?: Record<string, any>,
    libPathOrOptions?: string | MembrainClientOptions,
  ) {
    const options: MembrainClientOptions =
      typeof libPathOrOptions === "string"
        ? { libPath: libPathOrOptions }
        : libPathOrOptions ?? {};

    const libFile = options.libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    const effectiveConfig: Record<string, any> = { ...(config ?? {}) };
    if (options.scope) {
      const scopeSection = { ...(effectiveConfig.scope ?? {}) };
      const defaultScope = { ...(scopeSection.default_scope ?? {}) };
      for (const [key, value] of Object.entries(options.scope)) {
        if (!(key in defaultScope)) {
          defaultScope[key] = value;
        }
      }
      scopeSection.default_scope = defaultScope;
      effectiveConfig.scope = scopeSection;
    }
    if (options.indexedMetadataKeys && options.indexedMetadataKeys.length > 0) {
      const storageSection = { ...(effectiveConfig.storage ?? {}) };
      const existing: string[] = Array.isArray(storageSection.indexed_metadata_keys)
        ? [...storageSection.indexed_metadata_keys]
        : [];
      for (const key of options.indexedMetadataKeys) {
        if (!existing.includes(key)) {
          existing.push(key);
        }
      }
      storageSection.indexed_metadata_keys = existing;
      effectiveConfig.storage = storageSection;
    }

    this._scope = { ...(effectiveConfig.scope?.default_scope ?? {}) };

    if (Object.keys(effectiveConfig).length > 0) {
      const createWithConfig = this.lib.func(
        "membrain_client_new_with_config",
        voidPtr,
        ["str"],
      );
      this.handle = createWithConfig(JSON.stringify(effectiveConfig));
    } else {
      const create = this.lib.func("membrain_client_new", voidPtr, []);
      this.handle = create();
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create client: ${err}`);
    }
  }

  /**
   * Read-only copy of the default metadata scope applied to this client.
   */
  get scope(): Record<string, unknown> {
    return { ...this._scope };
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_client_free = lib.func("membrain_client_free", "void", [voidPtr]);

    this.fn_store_fact = lib.func("membrain_store_fact", "int32_t", [
      voidPtr, "str", "double", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_preference = lib.func("membrain_store_preference", "int32_t", [
      voidPtr, "str", "str", "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_event = lib.func("membrain_store_event", "int32_t", [
      voidPtr, "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_observation = lib.func("membrain_store_observation", "int32_t", [
      voidPtr, "str", "str", voidPtrPtr,
    ]);
    this.fn_store_concept = lib.func("membrain_store_concept", "int32_t", [
      voidPtr, "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_entity = lib.func("membrain_store_entity", "int32_t", [
      voidPtr, "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_workflow = lib.func("membrain_store_workflow", "int32_t", [
      voidPtr, "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_skill = lib.func("membrain_store_skill", "int32_t", [
      voidPtr, "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_pattern = lib.func("membrain_store_pattern", "int32_t", [
      voidPtr, "str", "str", "str", "str", voidPtrPtr,
    ]);
    this.fn_store_case = lib.func("membrain_store_case", "int32_t", [
      voidPtr, "str", "str", "str", "double", "str", voidPtrPtr,
    ]);
    this.fn_store_goal = lib.func("membrain_store_goal", "int32_t", [
      voidPtr, "str", "str", voidPtrPtr,
    ]);
    this.fn_store_task = lib.func("membrain_store_task", "int32_t", [
      voidPtr, "str", "str", voidPtrPtr,
    ]);

    this.fn_search = lib.func("membrain_search", "int32_t", [
      voidPtr, "str", "int32_t", voidPtrPtr,
    ]);
    this.fn_search_with_filters = lib.func("membrain_search_with_filters", "int32_t", [
      voidPtr, "str", "int32_t", "str", voidPtrPtr,
    ]);
    this.fn_get = lib.func("membrain_get", "int32_t", [
      voidPtr, "str", voidPtrPtr,
    ]);
    this.fn_delete = lib.func("membrain_delete", "int32_t", [voidPtr, "str"]);
    this.fn_count = lib.func("membrain_count", "int32_t", [voidPtr, int64Ptr]);
    this.fn_stats = lib.func("membrain_stats", "int32_t", [
      voidPtr, voidPtrPtr,
    ]);
    this.fn_vector_backend_health = lib.func("membrain_vector_backend_health", "int32_t", [
      voidPtr, voidPtrPtr,
    ]);
    this.fn_vector_backend_stats = lib.func("membrain_vector_backend_stats", "int32_t", [
      voidPtr, voidPtrPtr,
    ]);
  }

  private getLastError(): string {
    const err = this.fn_last_error();
    return err ?? "unknown error";
  }

  private check(code: number): void {
    if (code !== MEMBRAIN_OK) {
      throw new MembrainError(this.getLastError(), code);
    }
  }

  private callStore(fn: any, ...args: any[]): StoreResult {
    const buf = newStringOut();
    const code = fn(this.handle, ...args, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return JSON.parse(value) as StoreResult;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  private embeddingToJson(embedding?: number[]): string | null {
    return embedding ? JSON.stringify(embedding) : null;
  }

  // -----------------------------------------------------------------------
  // Store methods
  // -----------------------------------------------------------------------

  storeFact(
    statement: string,
    confidence: number = 0.8,
    embedding?: number[],
    metadata?: Record<string, unknown>
  ): StoreResult {
    return this.callStore(
      this.fn_store_fact,
      statement,
      confidence,
      this.embeddingToJson(embedding),
      metadata ? JSON.stringify(metadata) : null
    );
  }

  storePreference(
    holder: string,
    subject: string,
    preference: string,
    strength: string = "moderate",
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_preference,
      holder,
      subject,
      preference,
      strength,
      this.embeddingToJson(embedding)
    );
  }

  storeEvent(
    eventType: string,
    description: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_event,
      eventType,
      description,
      this.embeddingToJson(embedding)
    );
  }

  storeObservation(content: string, embedding?: number[]): StoreResult {
    return this.callStore(
      this.fn_store_observation,
      content,
      this.embeddingToJson(embedding)
    );
  }

  storeConcept(
    name: string,
    definition: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_concept,
      name,
      definition,
      this.embeddingToJson(embedding)
    );
  }

  storeEntity(
    name: string,
    entityType: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_entity,
      name,
      entityType,
      this.embeddingToJson(embedding)
    );
  }

  storeWorkflow(
    name: string,
    description: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_workflow,
      name,
      description,
      this.embeddingToJson(embedding)
    );
  }

  storeSkill(
    name: string,
    description: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_skill,
      name,
      description,
      this.embeddingToJson(embedding)
    );
  }

  storePattern(
    name: string,
    description: string,
    patternType: string,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_pattern,
      name,
      description,
      patternType,
      this.embeddingToJson(embedding)
    );
  }

  storeCase(
    problem: string,
    plan: string,
    outcome: string,
    reward: number = 1.0,
    embedding?: number[]
  ): StoreResult {
    return this.callStore(
      this.fn_store_case,
      problem,
      plan,
      outcome,
      reward,
      this.embeddingToJson(embedding)
    );
  }

  searchCases(
    query: string,
    limit: number = 5,
    positiveRewardThreshold: number = 0.5
  ): CaseSearchResults {
    const results = this.search(query, limit, {
      memory_types: ["procedural_case"],
    });

    const positiveCases: CaseEntry[] = [];
    const negativeCases: CaseEntry[] = [];

    for (const memory of results.memories) {
      const entry = parseCaseEntry(memory.id, memory.content, memory.score);
      if (!entry) continue;
      if (entry.reward >= positiveRewardThreshold) {
        positiveCases.push(entry);
      } else {
        negativeCases.push(entry);
      }
    }

    return {
      positive_cases: positiveCases,
      negative_cases: negativeCases,
      duration_ms: results.duration_ms,
    };
  }

  storeGoal(description: string, embedding?: number[]): StoreResult {
    return this.callStore(
      this.fn_store_goal,
      description,
      this.embeddingToJson(embedding)
    );
  }

  storeTask(title: string, embedding?: number[]): StoreResult {
    return this.callStore(
      this.fn_store_task,
      title,
      this.embeddingToJson(embedding)
    );
  }

  // -----------------------------------------------------------------------
  // Query methods
  // -----------------------------------------------------------------------

  search(
    query: string,
    limit: number = 10,
    filters?: SearchFilters,
    embedding?: number[]
  ): SearchResults {
    let finalFilters = filters;
    if (embedding !== undefined) {
      finalFilters = { ...filters, embedding };
    }

    const hasFilters =
      finalFilters !== undefined && Object.keys(finalFilters).length > 0;

    const buf = newStringOut();
    const code = hasFilters
      ? this.fn_search_with_filters(
          this.handle,
          query,
          limit,
          JSON.stringify(finalFilters),
          buf
        )
      : this.fn_search(this.handle, query, limit, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return JSON.parse(value) as SearchResults;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  get(id: string): MemoryInfo | null {
    const buf = newStringOut();
    const code = this.fn_get(this.handle, id, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      if (value === "null" || value === "") return null;
      return JSON.parse(value) as MemoryInfo;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  delete(id: string): boolean {
    const code = this.fn_delete(this.handle, id);
    return code === MEMBRAIN_OK;
  }

  count(): number {
    const out = newInt64Out();
    const code = this.fn_count(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  stats(): StorageStats {
    const buf = newStringOut();
    const code = this.fn_stats(this.handle, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return JSON.parse(value) as StorageStats;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  // -----------------------------------------------------------------------
  // Vector Database Methods
  // -----------------------------------------------------------------------

  vectorBackendHealth(): Record<string, any> {
    const buf = newStringOut();
    const code = this.fn_vector_backend_health(this.handle, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return JSON.parse(value);
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  vectorBackendStats(): Record<string, any> {
    const buf = newStringOut();
    const code = this.fn_vector_backend_stats(this.handle, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return JSON.parse(value);
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  close(): void {
    if (this.handle) {
      this.fn_client_free(this.handle);
      this.handle = null;
    }
  }
}

function parseCaseEntry(
  id: string,
  content: string,
  score: number
): CaseEntry | null {
  let problem = "";
  let plan = "";
  let outcome = "";
  let reward = 0.0;

  for (const line of content.split("\n")) {
    if (line.startsWith("Problem: ")) {
      problem = line.slice("Problem: ".length);
    } else if (line.startsWith("Plan: ")) {
      plan = line.slice("Plan: ".length);
    } else if (line.startsWith("Outcome: ")) {
      outcome = line.slice("Outcome: ".length);
    } else if (line.startsWith("Result: ")) {
      const resultText = line.slice("Result: ".length);
      reward = resultText === "success" ? 1.0 : 0.0;
    }
  }

  if (!problem) return null;

  return { id, problem, plan, outcome, reward, score };
}
