/**
 * Membrain distributed HNSW vector index client.
 */

import koffi from "koffi";

import type {
  ClusterInfo,
  DistributedIndexConfig,
  IndexMetrics,
  IndexSearchResult,
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

export class MembrainDistributedIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_distributed_free: any;
  private fn_distributed_add: any;
  private fn_distributed_remove: any;
  private fn_distributed_search: any;
  private fn_distributed_search_with_filter: any;
  private fn_distributed_batch_search: any;
  private fn_distributed_cluster_info: any;
  private fn_distributed_len: any;
  private fn_distributed_dimension: any;
  private fn_distributed_metrics: any;
  private fn_distributed_shutdown: any;

  private constructor(handle: any, lib: ReturnType<typeof koffi.load>) {
    this.handle = handle;
    this.lib = lib;
    this.bindFunctions();
  }

  static connect(
    config: DistributedIndexConfig,
    libPath?: string,
  ): MembrainDistributedIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_new = lib.func(
      "memscale_distributed_index_new",
      voidPtr,
      ["str"],
    );

    const handle = fn_new(JSON.stringify(config));

    if (!handle) {
      const fn_err = lib.func("membrain_last_error", "str", []);
      throw new MembrainError(
        `failed to create distributed index: ${fn_err() ?? "unknown error"}`,
      );
    }

    return new MembrainDistributedIndex(handle, lib);
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [charPtr]);
    this.fn_distributed_free = lib.func(
      "memscale_distributed_index_free",
      "void",
      [voidPtr],
    );

    this.fn_distributed_add = lib.func(
      "memscale_distributed_index_add",
      "int32_t",
      [voidPtr, "str", "str"],
    );
    this.fn_distributed_remove = lib.func(
      "memscale_distributed_index_remove",
      "int32_t",
      [voidPtr, "str"],
    );
    this.fn_distributed_search = lib.func(
      "memscale_distributed_index_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", charPtrPtr],
    );
    this.fn_distributed_search_with_filter = lib.func(
      "memscale_distributed_index_search_with_filter",
      "int32_t",
      [voidPtr, "str", "uint32_t", "str", charPtrPtr],
    );
    this.fn_distributed_batch_search = lib.func(
      "memscale_distributed_index_batch_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", charPtrPtr],
    );
    this.fn_distributed_cluster_info = lib.func(
      "memscale_distributed_index_cluster_info",
      "int32_t",
      [voidPtr, charPtrPtr],
    );
    this.fn_distributed_len = lib.func(
      "memscale_distributed_index_len",
      "int32_t",
      [voidPtr, int64Ptr],
    );
    this.fn_distributed_dimension = lib.func(
      "memscale_distributed_index_dimension",
      "int32_t",
      [voidPtr, int64Ptr],
    );
    this.fn_distributed_metrics = lib.func(
      "memscale_distributed_index_metrics",
      "int32_t",
      [voidPtr, charPtrPtr],
    );
    this.fn_distributed_shutdown = lib.func(
      "memscale_distributed_index_shutdown",
      "int32_t",
      [voidPtr],
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

  /** Add a vector to the distributed index. */
  add(id: string, embedding: number[]): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_distributed_add(this.handle, id, embJson);
    this.check(code);
  }

  /** Remove a vector from the distributed index by ID. */
  remove(id: string): void {
    const code = this.fn_distributed_remove(this.handle, id);
    this.check(code);
  }

  /** Search for the k nearest neighbors across the cluster. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const out = [null] as any;
    const queryJson = JSON.stringify(query);
    const code = this.fn_distributed_search(this.handle, queryJson, k, out);
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
    const code = this.fn_distributed_search_with_filter(
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

  /** Run multiple queries across the cluster. */
  batchSearch(queries: number[][], k: number = 10): IndexSearchResult[][] {
    const out = [null] as any;
    const queriesJson = JSON.stringify(queries);
    const code = this.fn_distributed_batch_search(
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

  /** Get the number of vectors on the local node. */
  len(): number {
    const out = [BigInt(0)] as any;
    const code = this.fn_distributed_len(this.handle, out);
    this.check(code);
    return Number(out[0]);
  }

  /** Get the vector dimension of the distributed index. */
  dimension(): number {
    const out = [BigInt(0)] as any;
    const code = this.fn_distributed_dimension(this.handle, out);
    this.check(code);
    return Number(out[0]);
  }

  /** Get cluster information. */
  clusterInfo(): ClusterInfo {
    const out = [null] as any;
    const code = this.fn_distributed_cluster_info(this.handle, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as ClusterInfo;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Get index performance metrics. */
  metrics(): IndexMetrics {
    const out = [null] as any;
    const code = this.fn_distributed_metrics(this.handle, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as IndexMetrics;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Shut down the node and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_distributed_shutdown(this.handle);
      this.fn_distributed_free(this.handle);
      this.handle = null;
    }
  }
}
