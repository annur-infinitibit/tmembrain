/**
 * Thread-safe concurrent HNSW (Hierarchical Navigable Small World) vector index client.
 */

import type {
  IndexConfig,
  IndexSearchResult,
} from "./types";

import {
  findLibrary,
  MEMBRAIN_OK,
  MembrainError,
  newInt64Out,
  newStringOut,
  readInt64,
  readStringOut,
  voidPtr,
  voidPtrPtr,
} from "./ffi";

import koffi from "koffi";

/**
 * Thread-safe concurrent HNSW (Hierarchical Navigable Small World) vector index.
 *
 * HNSW builds a multi-layer navigable small world graph providing fast
 * approximate nearest neighbor search with excellent recall. This is the
 * production-grade graph-based index for high-dimensional vectors.
 *
 * This index can be safely shared across multiple worker threads. All operations
 * (add, remove, search) can run concurrently without external locking.
 *
 * Concurrency Model:
 * - Concurrent reads (multiple searches in parallel, read-lock acquired)
 * - Serialized writes (write-lock acquired per operation)
 * - Expected 4-6x search throughput on multi-core systems
 *
 * @example
 * ```typescript
 * import { ConcurrentHnswIndex } from 'membrain';
 *
 * const index = new ConcurrentHnswIndex(1536);
 *
 * // Add vectors from multiple workers
 * for (let i = 0; i < 1000; i++) {
 *   index.add(crypto.randomUUID(), Array(1536).fill(0.1));
 * }
 *
 * // Search concurrently
 * const results = index.search(Array(1536).fill(0.1), 10);
 *
 * index.close();
 * ```
 */
export class ConcurrentHnswIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_concurrent_hnsw_free: any;
  private fn_concurrent_hnsw_clone: any;
  private fn_concurrent_hnsw_add: any;
  private fn_concurrent_hnsw_remove: any;
  private fn_concurrent_hnsw_search: any;
  private fn_concurrent_hnsw_len: any;
  private fn_concurrent_hnsw_dimension: any;

  constructor(dimension?: number, config?: IndexConfig, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    if (config) {
      const createWithConfig = this.lib.func(
        "memscale_concurrent_hnsw_index_new_with_config",
        voidPtr,
        ["uint32_t", "str"],
      );
      this.handle = createWithConfig(dimension ?? 1536, JSON.stringify(config));
    } else {
      const create = this.lib.func(
        "memscale_concurrent_hnsw_index_new",
        voidPtr,
        ["uint32_t"],
      );
      this.handle = create(dimension ?? 1536);
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create concurrent HNSW index: ${err}`);
    }
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_concurrent_hnsw_free = lib.func("memscale_concurrent_hnsw_index_free", "void", [voidPtr]);
    this.fn_concurrent_hnsw_clone = lib.func("memscale_concurrent_hnsw_index_clone", voidPtr, [voidPtr]);

    this.fn_concurrent_hnsw_add = lib.func("memscale_concurrent_hnsw_index_add", "int32_t", [
      voidPtr, "str", "float *", "uint32_t",
    ]);
    this.fn_concurrent_hnsw_remove = lib.func("memscale_concurrent_hnsw_index_remove", "int32_t", [
      voidPtr, "str", "int32_t *",
    ]);
    this.fn_concurrent_hnsw_search = lib.func("memscale_concurrent_hnsw_index_search", "int32_t", [
      voidPtr, "float *", "uint32_t", "uint32_t", voidPtrPtr,
    ]);
    this.fn_concurrent_hnsw_len = lib.func("memscale_concurrent_hnsw_index_len", "int32_t", [
      voidPtr, "int64_t *",
    ]);
    this.fn_concurrent_hnsw_dimension = lib.func(
      "memscale_concurrent_hnsw_index_dimension",
      "int32_t",
      [voidPtr, "int64_t *"],
    );
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

  /**
   * Clone this index handle for use in another worker thread.
   *
   * @returns A new ConcurrentHnswIndex instance sharing the same underlying index
   */
  clone(): ConcurrentHnswIndex {
    const clonedHandle = this.fn_concurrent_hnsw_clone(this.handle);
    if (!clonedHandle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to clone concurrent HNSW index: ${err}`);
    }

    const cloned = Object.create(ConcurrentHnswIndex.prototype);
    cloned.handle = clonedHandle;
    cloned.lib = this.lib;
    cloned.fn_last_error = this.fn_last_error;
    cloned.fn_string_free = this.fn_string_free;
    cloned.fn_concurrent_hnsw_free = this.fn_concurrent_hnsw_free;
    cloned.fn_concurrent_hnsw_clone = this.fn_concurrent_hnsw_clone;
    cloned.fn_concurrent_hnsw_add = this.fn_concurrent_hnsw_add;
    cloned.fn_concurrent_hnsw_remove = this.fn_concurrent_hnsw_remove;
    cloned.fn_concurrent_hnsw_search = this.fn_concurrent_hnsw_search;
    cloned.fn_concurrent_hnsw_len = this.fn_concurrent_hnsw_len;
    cloned.fn_concurrent_hnsw_dimension = this.fn_concurrent_hnsw_dimension;
    return cloned;
  }

  /**
   * Add a vector to the index (thread-safe).
   *
   * @param id - UUID string identifying the vector
   * @param embedding - Array of numbers (must match index dimension)
   */
  add(id: string, embedding: number[]): void {
    const vectorArray = new Float32Array(embedding);
    const code = this.fn_concurrent_hnsw_add(
      this.handle,
      id,
      vectorArray,
      embedding.length,
    );
    this.check(code);
  }

  /**
   * Remove a vector from the index by ID (thread-safe).
   *
   * @param id - UUID string of the vector to remove
   * @returns True if the vector was found and removed, false otherwise
   */
  remove(id: string): boolean {
    const out = new Int32Array(1);
    const code = this.fn_concurrent_hnsw_remove(this.handle, id, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * Search for the k nearest neighbors (thread-safe).
   *
   * Multiple threads can search concurrently. Read-locks are acquired
   * to allow maximum read parallelism.
   *
   * @param query - Query vector (must match index dimension)
   * @param k - Number of results to return
   * @returns Array of search results sorted by score (highest first)
   */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryArray = new Float32Array(query);
    const code = this.fn_concurrent_hnsw_search(
      this.handle,
      queryArray,
      query.length,
      k,
      buf,
    );
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /**
   * Get the number of vectors in the index (thread-safe).
   */
  len(): number {
    const out = newInt64Out();
    const code = this.fn_concurrent_hnsw_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Get the vector dimension of the index.
   */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_concurrent_hnsw_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Release this handle to the underlying index.
   */
  close(): void {
    if (this.handle) {
      this.fn_concurrent_hnsw_free(this.handle);
      this.handle = null;
    }
  }
}
