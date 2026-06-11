/**
 * Thread-safe concurrent flat (brute-force) vector index client.
 */

import type {
  FlatIndexConfig,
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
 * Thread-safe concurrent flat (brute-force) vector index.
 *
 * This index can be safely shared across multiple worker threads. All operations
 * (add, remove, search) can run concurrently without external locking.
 *
 * Concurrency Model:
 * - Multiple concurrent searches (read parallelism)
 * - Writes are serialized but don't block reads
 * - Expected 5-6x search throughput on 8 cores
 *
 * @example
 * ```typescript
 * import { ConcurrentFlatIndex } from 'membrain';
 *
 * const index = new ConcurrentFlatIndex(128);
 *
 * // Add vectors from multiple workers
 * index.add('vec-1', [0.1, 0.2, ...]);
 *
 * // Search concurrently
 * const results = index.search([0.1, 0.2, ...], 10);
 *
 * index.close();
 * ```
 */
export class ConcurrentFlatIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_concurrent_flat_free: any;
  private fn_concurrent_flat_clone: any;
  private fn_concurrent_flat_add: any;
  private fn_concurrent_flat_remove: any;
  private fn_concurrent_flat_search: any;
  private fn_concurrent_flat_len: any;
  private fn_concurrent_flat_dimension: any;

  constructor(dimension?: number, config?: FlatIndexConfig, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    if (config) {
      const createWithConfig = this.lib.func(
        "memscale_concurrent_flat_index_new_with_config",
        voidPtr,
        ["uint32_t", "str"],
      );
      this.handle = createWithConfig(dimension ?? 1536, JSON.stringify(config));
    } else {
      const create = this.lib.func(
        "memscale_concurrent_flat_index_new",
        voidPtr,
        ["uint32_t"],
      );
      this.handle = create(dimension ?? 1536);
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create concurrent flat index: ${err}`);
    }
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_concurrent_flat_free = lib.func("memscale_concurrent_flat_index_free", "void", [voidPtr]);
    this.fn_concurrent_flat_clone = lib.func("memscale_concurrent_flat_index_clone", voidPtr, [voidPtr]);

    this.fn_concurrent_flat_add = lib.func("memscale_concurrent_flat_index_add", "int32_t", [
      voidPtr, "str", "float *", "uint32_t",
    ]);
    this.fn_concurrent_flat_remove = lib.func("memscale_concurrent_flat_index_remove", "int32_t", [
      voidPtr, "str", "int32_t *",
    ]);
    this.fn_concurrent_flat_search = lib.func("memscale_concurrent_flat_index_search", "int32_t", [
      voidPtr, "float *", "uint32_t", "uint32_t", voidPtrPtr,
    ]);
    this.fn_concurrent_flat_len = lib.func("memscale_concurrent_flat_index_len", "int32_t", [
      voidPtr, "int64_t *",
    ]);
    this.fn_concurrent_flat_dimension = lib.func(
      "memscale_concurrent_flat_index_dimension",
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
   * Both the original and cloned handles point to the same underlying
   * index and can be used concurrently. Both must be closed separately.
   *
   * @returns A new ConcurrentFlatIndex instance sharing the same underlying index.
   *
   * @example
   * ```typescript
   * const index = new ConcurrentFlatIndex(128);
   * const clone = index.clone();
   *
   * // Use in different workers
   * // Both handles are valid and can be used concurrently
   *
   * clone.close();
   * index.close();
   * ```
   */
  clone(): ConcurrentFlatIndex {
    const clonedHandle = this.fn_concurrent_flat_clone(this.handle);
    if (!clonedHandle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to clone concurrent flat index: ${err}`);
    }

    const cloned = Object.create(ConcurrentFlatIndex.prototype);
    cloned.handle = clonedHandle;
    cloned.lib = this.lib;
    cloned.fn_last_error = this.fn_last_error;
    cloned.fn_string_free = this.fn_string_free;
    cloned.fn_concurrent_flat_free = this.fn_concurrent_flat_free;
    cloned.fn_concurrent_flat_clone = this.fn_concurrent_flat_clone;
    cloned.fn_concurrent_flat_add = this.fn_concurrent_flat_add;
    cloned.fn_concurrent_flat_remove = this.fn_concurrent_flat_remove;
    cloned.fn_concurrent_flat_search = this.fn_concurrent_flat_search;
    cloned.fn_concurrent_flat_len = this.fn_concurrent_flat_len;
    cloned.fn_concurrent_flat_dimension = this.fn_concurrent_flat_dimension;
    return cloned;
  }

  /**
   * Add a vector to the index (thread-safe).
   *
   * Multiple threads can call this concurrently. Writes are serialized
   * internally but don't block concurrent reads.
   *
   * @param id - UUID string identifying the vector
   * @param embedding - Array of numbers (must match index dimension)
   */
  add(id: string, embedding: number[]): void {
    const vectorArray = new Float32Array(embedding);
    const code = this.fn_concurrent_flat_add(
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
    const code = this.fn_concurrent_flat_remove(this.handle, id, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * Search for the k nearest neighbors (thread-safe).
   *
   * Multiple threads can search concurrently without blocking each other.
   * This enables true parallelism for search-heavy workloads.
   *
   * @param query - Query vector (must match index dimension)
   * @param k - Number of results to return
   * @returns Array of search results sorted by score (highest first)
   */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryArray = new Float32Array(query);
    const code = this.fn_concurrent_flat_search(
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
    const code = this.fn_concurrent_flat_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Get the vector dimension of the index.
   */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_concurrent_flat_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Release this handle to the underlying index.
   *
   * Other cloned handles remain valid. The index is freed when
   * the last handle is closed.
   */
  close(): void {
    if (this.handle) {
      this.fn_concurrent_flat_free(this.handle);
      this.handle = null;
    }
  }
}
