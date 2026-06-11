/**
 * Membrain flat (brute-force) vector index client.
 */

import type {
  FlatIndexConfig,
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

/** Flat (brute-force exact search) vector index client. */
export class MembrainFlatIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_flat_free: any;
  private fn_flat_add: any;
  private fn_flat_remove: any;
  private fn_flat_search: any;
  private fn_flat_search_with_filter: any;
  private fn_flat_batch_search: any;
  private fn_flat_len: any;
  private fn_flat_dimension: any;
  private fn_flat_metrics: any;

  constructor(dimension?: number, config?: FlatIndexConfig, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    if (config) {
      const createWithConfig = this.lib.func(
        "memscale_flat_index_new_with_config",
        voidPtr,
        ["str"],
      );
      this.handle = createWithConfig(JSON.stringify(config));
    } else {
      const create = this.lib.func(
        "memscale_flat_index_new",
        voidPtr,
        ["uint32_t"],
      );
      this.handle = create(dimension ?? 1536);
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create flat index: ${err}`);
    }
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_flat_free = lib.func("memscale_flat_index_free", "void", [voidPtr]);

    this.fn_flat_add = lib.func("memscale_flat_index_add", "int32_t", [
      voidPtr, "str", "str",
    ]);
    this.fn_flat_remove = lib.func("memscale_flat_index_remove", "int32_t", [
      voidPtr, "str",
    ]);
    this.fn_flat_search = lib.func("memscale_flat_index_search", "int32_t", [
      voidPtr, "str", "uint32_t", voidPtrPtr,
    ]);
    this.fn_flat_search_with_filter = lib.func(
      "memscale_flat_index_search_with_filter",
      "int32_t",
      [voidPtr, "str", "uint32_t", "str", voidPtrPtr],
    );
    this.fn_flat_batch_search = lib.func(
      "memscale_flat_index_batch_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", voidPtrPtr],
    );
    this.fn_flat_len = lib.func("memscale_flat_index_len", "int32_t", [
      voidPtr, "int64_t *",
    ]);
    this.fn_flat_dimension = lib.func(
      "memscale_flat_index_dimension",
      "int32_t",
      [voidPtr, "int64_t *"],
    );
    this.fn_flat_metrics = lib.func("memscale_flat_index_metrics", "int32_t", [
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

  /** Add a vector to the index. */
  add(id: string, embedding: number[]): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_flat_add(this.handle, id, embJson);
    this.check(code);
  }

  /** Remove a vector from the index by ID. */
  remove(id: string): void {
    const code = this.fn_flat_remove(this.handle, id);
    this.check(code);
  }

  /** Search for the k nearest neighbors. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryJson = JSON.stringify(query);
    const code = this.fn_flat_search(this.handle, queryJson, k, buf);
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
    const code = this.fn_flat_search_with_filter(
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
    const code = this.fn_flat_batch_search(this.handle, queriesJson, k, buf);
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
    const code = this.fn_flat_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get the vector dimension of the index. */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_flat_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get index performance metrics. */
  metrics(): IndexMetrics {
    const buf = newStringOut();
    const code = this.fn_flat_metrics(this.handle, buf);
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
      this.fn_flat_free(this.handle);
      this.handle = null;
    }
  }
}
