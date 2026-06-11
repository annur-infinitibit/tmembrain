/**
 * Membrain Vamana (DiskANN-style) vector index client.
 */

import type {
  VamanaIndexConfig,
  IndexMetrics,
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

/** Vamana (DiskANN-style graph) vector index client. */
export class MembrainVamanaIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_vamana_free: any;
  private fn_vamana_add: any;
  private fn_vamana_remove: any;
  private fn_vamana_search: any;
  private fn_vamana_search_with_filter: any;
  private fn_vamana_batch_search: any;
  private fn_vamana_len: any;
  private fn_vamana_dimension: any;
  private fn_vamana_metrics: any;

  private constructor(handle: any, lib: ReturnType<typeof koffi.load>) {
    this.handle = handle;
    this.lib = lib;
    this.bindFunctions();
  }

  /** Build a Vamana index from a set of vectors. */
  static build(
    ids: string[],
    vectors: number[][],
    config: VamanaIndexConfig,
    libPath?: string,
  ): MembrainVamanaIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_build = lib.func(
      "memscale_vamana_index_build",
      voidPtr,
      ["str", "str", "str"],
    );

    const handle = fn_build(
      JSON.stringify(config),
      JSON.stringify(ids),
      JSON.stringify(vectors),
    );

    if (!handle) {
      const fn_err = lib.func("membrain_last_error", "str", []);
      throw new MembrainError(
        `failed to build Vamana index: ${fn_err() ?? "unknown error"}`,
      );
    }

    return new MembrainVamanaIndex(handle, lib);
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_vamana_free = lib.func("memscale_vamana_index_free", "void", [
      voidPtr,
    ]);

    this.fn_vamana_add = lib.func("memscale_vamana_index_add", "int32_t", [
      voidPtr, "str", "str",
    ]);
    this.fn_vamana_remove = lib.func(
      "memscale_vamana_index_remove",
      "int32_t",
      [voidPtr, "str"],
    );
    this.fn_vamana_search = lib.func(
      "memscale_vamana_index_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", voidPtrPtr],
    );
    this.fn_vamana_search_with_filter = lib.func(
      "memscale_vamana_index_search_with_filter",
      "int32_t",
      [voidPtr, "str", "uint32_t", "str", voidPtrPtr],
    );
    this.fn_vamana_batch_search = lib.func(
      "memscale_vamana_index_batch_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", voidPtrPtr],
    );
    this.fn_vamana_len = lib.func("memscale_vamana_index_len", "int32_t", [
      voidPtr, "int64_t *",
    ]);
    this.fn_vamana_dimension = lib.func(
      "memscale_vamana_index_dimension",
      "int32_t",
      [voidPtr, "int64_t *"],
    );
    this.fn_vamana_metrics = lib.func(
      "memscale_vamana_index_metrics",
      "int32_t",
      [voidPtr, voidPtrPtr],
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

  /** Add a vector to the index. */
  add(id: string, embedding: number[]): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_vamana_add(this.handle, id, embJson);
    this.check(code);
  }

  /** Remove a vector from the index by ID. */
  remove(id: string): void {
    const code = this.fn_vamana_remove(this.handle, id);
    this.check(code);
  }

  /** Search for the k nearest neighbors. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryJson = JSON.stringify(query);
    const code = this.fn_vamana_search(this.handle, queryJson, k, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Search with an ID filter (only return results in allowedIds). */
  searchWithFilter(
    query: number[],
    k: number = 10,
    allowedIds: string[] = [],
  ): IndexSearchResult[] {
    const buf = newStringOut();
    const queryJson = JSON.stringify(query);
    const idsJson = JSON.stringify(allowedIds);
    const code = this.fn_vamana_search_with_filter(
      this.handle, queryJson, k, idsJson, buf,
    );
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Run multiple queries. */
  batchSearch(queries: number[][], k: number = 10): IndexSearchResult[][] {
    const buf = newStringOut();
    const queriesJson = JSON.stringify(queries);
    const code = this.fn_vamana_batch_search(
      this.handle, queriesJson, k, buf,
    );
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[][];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Get the number of vectors in the index. */
  len(): number {
    const out = newInt64Out();
    const code = this.fn_vamana_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get the vector dimension of the index. */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_vamana_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get index performance metrics. */
  metrics(): IndexMetrics {
    const buf = newStringOut();
    const code = this.fn_vamana_metrics(this.handle, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexMetrics;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Destroy the index and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_vamana_free(this.handle);
      this.handle = null;
    }
  }
}
