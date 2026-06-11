/**
 * Membrain sharded HNSW vector index client.
 */

import koffi from "koffi";

import type {
  IndexMetrics,
  IndexSearchResult,
  ShardedIndexConfig,
  ShardedIndexInfo,
} from "./types";

import {
  charPtr,
  charPtrPtr,
  findLibrary,
  int64Ptr,
  MEMBRAIN_OK,
  MembrainError,
  voidPtr,
} from "./ffi";

/** Sharded HNSW index with centroid-based routing. */
export class MembrainShardedIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_sharded_free: any;
  private fn_sharded_add: any;
  private fn_sharded_remove: any;
  private fn_sharded_search: any;
  private fn_sharded_search_with_filter: any;
  private fn_sharded_batch_search: any;
  private fn_sharded_rebalance: any;
  private fn_sharded_info: any;
  private fn_sharded_len: any;
  private fn_sharded_dimension: any;
  private fn_sharded_metrics: any;
  private fn_sharded_save: any;

  private constructor(handle: any, lib: ReturnType<typeof koffi.load>) {
    this.handle = handle;
    this.lib = lib;
    this.bindFunctions();
  }

  static build(
    ids: string[],
    vectors: number[][],
    config: ShardedIndexConfig,
    libPath?: string,
  ): MembrainShardedIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_build = lib.func(
      "memscale_sharded_index_build",
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
        `failed to build sharded index: ${fn_err() ?? "unknown error"}`,
      );
    }

    return new MembrainShardedIndex(handle, lib);
  }

  static load(data: string, libPath?: string): MembrainShardedIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_load = lib.func("memscale_sharded_index_load", voidPtr, ["str"]);
    const handle = fn_load(data);

    if (!handle) {
      const fn_err = lib.func("membrain_last_error", "str", []);
      throw new MembrainError(
        `failed to load sharded index: ${fn_err() ?? "unknown error"}`,
      );
    }

    return new MembrainShardedIndex(handle, lib);
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [charPtr]);
    this.fn_sharded_free = lib.func("memscale_sharded_index_free", "void", [
      voidPtr,
    ]);

    this.fn_sharded_add = lib.func("memscale_sharded_index_add", "int32_t", [
      voidPtr, "str", "str",
    ]);
    this.fn_sharded_remove = lib.func(
      "memscale_sharded_index_remove",
      "int32_t",
      [voidPtr, "str"],
    );
    this.fn_sharded_search = lib.func(
      "memscale_sharded_index_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", charPtrPtr],
    );
    this.fn_sharded_search_with_filter = lib.func(
      "memscale_sharded_index_search_with_filter",
      "int32_t",
      [voidPtr, "str", "uint32_t", "str", charPtrPtr],
    );
    this.fn_sharded_batch_search = lib.func(
      "memscale_sharded_index_batch_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", charPtrPtr],
    );
    this.fn_sharded_rebalance = lib.func(
      "memscale_sharded_index_rebalance",
      "int32_t",
      [voidPtr],
    );
    this.fn_sharded_info = lib.func(
      "memscale_sharded_index_info",
      "int32_t",
      [voidPtr, charPtrPtr],
    );
    this.fn_sharded_len = lib.func(
      "memscale_sharded_index_len",
      "int32_t",
      [voidPtr, int64Ptr],
    );
    this.fn_sharded_dimension = lib.func(
      "memscale_sharded_index_dimension",
      "int32_t",
      [voidPtr, int64Ptr],
    );
    this.fn_sharded_metrics = lib.func(
      "memscale_sharded_index_metrics",
      "int32_t",
      [voidPtr, charPtrPtr],
    );
    this.fn_sharded_save = lib.func(
      "memscale_sharded_index_save",
      "int32_t",
      [voidPtr, charPtrPtr],
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

  /** Add a vector to the sharded index. */
  add(id: string, embedding: number[]): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_sharded_add(this.handle, id, embJson);
    this.check(code);
  }

  /** Remove a vector from the sharded index by ID. */
  remove(id: string): void {
    const code = this.fn_sharded_remove(this.handle, id);
    this.check(code);
  }

  /** Search for the k nearest neighbors. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const out = [null] as any;
    const queryJson = JSON.stringify(query);
    const code = this.fn_sharded_search(this.handle, queryJson, k, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Search with an ID filter (only return results in allowedIds). */
  searchWithFilter(
    query: number[],
    k: number = 10,
    allowedIds: string[] = [],
  ): IndexSearchResult[] {
    const out = [null] as any;
    const queryJson = JSON.stringify(query);
    const idsJson = JSON.stringify(allowedIds);
    const code = this.fn_sharded_search_with_filter(
      this.handle,
      queryJson,
      k,
      idsJson,
      out,
    );
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Run multiple queries in parallel. */
  batchSearch(queries: number[][], k: number = 10): IndexSearchResult[][] {
    const out = [null] as any;
    const queriesJson = JSON.stringify(queries);
    const code = this.fn_sharded_batch_search(
      this.handle,
      queriesJson,
      k,
      out,
    );
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as IndexSearchResult[][];
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Retrain centroids and redistribute vectors across shards. */
  rebalance(): void {
    const code = this.fn_sharded_rebalance(this.handle);
    this.check(code);
  }

  /** Get the number of active vectors in the sharded index. */
  len(): number {
    const out = [BigInt(0)] as any;
    const code = this.fn_sharded_len(this.handle, out);
    this.check(code);
    return Number(out[0]);
  }

  /** Get the vector dimension of the sharded index. */
  dimension(): number {
    const out = [BigInt(0)] as any;
    const code = this.fn_sharded_dimension(this.handle, out);
    this.check(code);
    return Number(out[0]);
  }

  /** Get per-shard statistics. */
  info(): ShardedIndexInfo {
    const out = [null] as any;
    const code = this.fn_sharded_info(this.handle, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as ShardedIndexInfo;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Get index performance metrics. */
  metrics(): IndexMetrics {
    const out = [null] as any;
    const code = this.fn_sharded_metrics(this.handle, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as IndexMetrics;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Save index state to a base64-encoded string (MessagePack format). */
  save(): string {
    const out = [null] as any;
    const code = this.fn_sharded_save(this.handle, out);
    this.check(code);
    const data = out[0] as string;
    try {
      return data;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Destroy the index and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_sharded_free(this.handle);
      this.handle = null;
    }
  }
}
