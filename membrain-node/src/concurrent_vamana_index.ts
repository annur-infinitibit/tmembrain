/**
 * Thread-safe concurrent Vamana (DiskANN-style) vector index client.
 */

import type {
  VamanaIndexConfig,
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
 * Thread-safe concurrent Vamana (DiskANN-style) vector index.
 *
 * Vamana builds a graph-based index using greedy search and pruning,
 * providing excellent recall and query performance. This implementation
 * is based on the DiskANN paper by Microsoft Research.
 *
 * This index can be safely shared across multiple worker threads. All operations
 * (add, remove, search) can run concurrently without external locking.
 *
 * Concurrency Model:
 * - Concurrent reads (multiple searches in parallel)
 * - Coarse-grained locking for writes (graph-level lock)
 * - Expected 4-6x search throughput on multi-core systems
 *
 * @example
 * ```typescript
 * import { ConcurrentVamanaIndex } from 'membrain';
 *
 * // Build index from training data
 * const ids = Array.from({ length: 1000 }, () => crypto.randomUUID());
 * const vectors = Array.from({ length: 1000 }, () => Array(128).fill(0.1));
 *
 * const index = ConcurrentVamanaIndex.build(ids, vectors, 128);
 *
 * // Search concurrently
 * const results = index.search(Array(128).fill(0.1), 10);
 *
 * index.close();
 * ```
 */
export class ConcurrentVamanaIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_concurrent_vamana_free: any;
  private fn_concurrent_vamana_clone: any;
  private fn_concurrent_vamana_add: any;
  private fn_concurrent_vamana_remove: any;
  private fn_concurrent_vamana_search: any;
  private fn_concurrent_vamana_len: any;
  private fn_concurrent_vamana_dimension: any;

  private constructor(handle: any, lib: ReturnType<typeof koffi.load>) {
    this.handle = handle;
    this.lib = lib;
    this.bindFunctions();
  }

  /**
   * Build a thread-safe concurrent Vamana index from training data.
   *
   * @param ids - Array of UUID strings for the vectors
   * @param vectors - Array of vectors (each vector is an array of numbers)
   * @param dimension - Vector dimension
   * @param config - Optional configuration object
   * @param libPath - Optional path to the shared library
   * @returns A new ConcurrentVamanaIndex instance
   */
  static build(
    ids: string[],
    vectors: number[][],
    dimension: number,
    config?: VamanaIndexConfig,
    libPath?: string,
  ): ConcurrentVamanaIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    if (ids.length !== vectors.length) {
      throw new Error("ids and vectors must have the same length");
    }

    // Prepare IDs array - koffi will automatically convert string[] to char**
    const idsArray = ids;

    // Prepare flat vectors array
    const flatVectors: number[] = [];
    for (const vec of vectors) {
      if (vec.length !== dimension) {
        throw new Error(`all vectors must have dimension ${dimension}`);
      }
      flatVectors.push(...vec);
    }
    const vectorsArray = new Float32Array(flatVectors);

    const configJson = config ? JSON.stringify(config) : null;

    const buildFunc = lib.func(
      "memscale_concurrent_vamana_index_build",
      voidPtr,
      [koffi.pointer("char *"), "float *", "uint32_t", "uint32_t", "str"],
    );

    const handle = buildFunc(
      idsArray,
      vectorsArray,
      ids.length,
      dimension,
      configJson,
    );

    if (!handle) {
      const lastErrorFunc = lib.func("membrain_last_error", "str", []);
      const err = lastErrorFunc() ?? "unknown error";
      throw new MembrainError(`failed to build concurrent Vamana index: ${err}`);
    }

    return new ConcurrentVamanaIndex(handle, lib);
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_concurrent_vamana_free = lib.func("memscale_concurrent_vamana_index_free", "void", [voidPtr]);
    this.fn_concurrent_vamana_clone = lib.func("memscale_concurrent_vamana_index_clone", voidPtr, [voidPtr]);

    this.fn_concurrent_vamana_add = lib.func("memscale_concurrent_vamana_index_add", "int32_t", [
      voidPtr, "str", "float *", "uint32_t",
    ]);
    this.fn_concurrent_vamana_remove = lib.func("memscale_concurrent_vamana_index_remove", "int32_t", [
      voidPtr, "str", "int32_t *",
    ]);
    this.fn_concurrent_vamana_search = lib.func("memscale_concurrent_vamana_index_search", "int32_t", [
      voidPtr, "float *", "uint32_t", "uint32_t", voidPtrPtr,
    ]);
    this.fn_concurrent_vamana_len = lib.func("memscale_concurrent_vamana_index_len", "int32_t", [
      voidPtr, "int64_t *",
    ]);
    this.fn_concurrent_vamana_dimension = lib.func(
      "memscale_concurrent_vamana_index_dimension",
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
   * @returns A new ConcurrentVamanaIndex instance sharing the same underlying index
   */
  clone(): ConcurrentVamanaIndex {
    const clonedHandle = this.fn_concurrent_vamana_clone(this.handle);
    if (!clonedHandle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to clone concurrent Vamana index: ${err}`);
    }
    return new ConcurrentVamanaIndex(clonedHandle, this.lib);
  }

  /**
   * Add a vector to the index (thread-safe).
   *
   * @param id - UUID string identifying the vector
   * @param embedding - Array of numbers (must match index dimension)
   */
  add(id: string, embedding: number[]): void {
    const vectorArray = new Float32Array(embedding);
    const code = this.fn_concurrent_vamana_add(
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
    const code = this.fn_concurrent_vamana_remove(this.handle, id, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * Search for the k nearest neighbors (thread-safe).
   *
   * @param query - Query vector (must match index dimension)
   * @param k - Number of results to return
   * @returns Array of search results sorted by score (highest first)
   */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryArray = new Float32Array(query);
    const code = this.fn_concurrent_vamana_search(
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
    const code = this.fn_concurrent_vamana_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Get the vector dimension of the index.
   */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_concurrent_vamana_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Release this handle to the underlying index.
   */
  close(): void {
    if (this.handle) {
      this.fn_concurrent_vamana_free(this.handle);
      this.handle = null;
    }
  }
}
