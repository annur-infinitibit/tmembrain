/** Membrain HNSW vector index clients. */

import koffi from "koffi";

import type {
  GpuConfig,
  IndexConfig,
  IndexMetrics,
  IndexSearchResult,
  PqConfig,
  WalConfig,
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

/** HNSW vector index client. */
export class MembrainIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  private fn_last_error: any;
  private fn_string_free: any;
  private fn_index_free: any;
  private fn_index_add: any;
  private fn_index_remove: any;
  private fn_index_search: any;
  private fn_index_search_with_filter: any;
  private fn_index_batch_search: any;
  private fn_index_len: any;
  private fn_index_dimension: any;
  private fn_index_metrics: any;
  private fn_index_enable_pq: any;
  private fn_index_enable_wal: any;
  private fn_index_compact: any;
  private fn_index_save: any;
  private fn_index_load: any;
  private fn_index_save_binary: any;

  private fn_gpu_available: any | null = null;
  private fn_index_enable_gpu: any | null = null;
  private fn_index_gpu_batch_search: any | null = null;
  private hasGpuSupport: boolean = false;

  constructor(dimension?: number, config?: IndexConfig, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindIndexFunctions();
    this.bindGpuFunctions();

    if (config) {
      const createWithConfig = this.lib.func(
        "memscale_index_new_with_config",
        voidPtr,
        ["str"],
      );
      const configJson = JSON.stringify({
        ...config,
        dimension: config.dimension ?? dimension ?? 1536,
      });
      this.handle = createWithConfig(configJson);
    } else {
      const create = this.lib.func("memscale_index_new", voidPtr, ["uint32_t"]);
      this.handle = create(dimension ?? 1536);
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create index: ${err}`);
    }
  }

  /** Load an index from a base64-encoded MessagePack string. */
  static load(data: string, libPath?: string): MembrainIndex {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_load = lib.func("memscale_index_load", voidPtr, ["str"]);
    const handle = fn_load(data);
    if (!handle) {
      const fn_err = lib.func("membrain_last_error", "str", []);
      throw new MembrainError(
        `failed to load index: ${fn_err() ?? "unknown error"}`,
      );
    }

    const instance = Object.create(MembrainIndex.prototype) as MembrainIndex;
    (instance as any).lib = lib;
    (instance as any).handle = handle;
    instance.bindIndexFunctions();
    instance.bindGpuFunctions();
    return instance;
  }

  private bindIndexFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_index_free = lib.func("memscale_index_free", "void", [voidPtr]);

    this.fn_index_add = lib.func("memscale_index_add", "int32_t", [
      voidPtr,
      "str",
      "str",
    ]);
    this.fn_index_remove = lib.func("memscale_index_remove", "int32_t", [
      voidPtr,
      "str",
    ]);
    this.fn_index_search = lib.func("memscale_index_search", "int32_t", [
      voidPtr,
      "str",
      "uint32_t",
      voidPtrPtr,
    ]);
    this.fn_index_search_with_filter = lib.func(
      "memscale_index_search_with_filter",
      "int32_t",
      [voidPtr, "str", "uint32_t", "str", voidPtrPtr],
    );
    this.fn_index_batch_search = lib.func(
      "memscale_index_batch_search",
      "int32_t",
      [voidPtr, "str", "uint32_t", voidPtrPtr],
    );
    this.fn_index_len = lib.func("memscale_index_len", "int32_t", [
      voidPtr,
      "int64_t *",
    ]);
    this.fn_index_dimension = lib.func(
      "memscale_index_dimension",
      "int32_t",
      [voidPtr, "int64_t *"],
    );
    this.fn_index_metrics = lib.func("memscale_index_metrics", "int32_t", [
      voidPtr,
      voidPtrPtr,
    ]);
    this.fn_index_enable_pq = lib.func("memscale_index_enable_pq", "int32_t", [
      voidPtr,
      "str",
    ]);
    this.fn_index_enable_wal = lib.func(
      "memscale_index_enable_wal",
      "int32_t",
      [voidPtr, "str"],
    );
    this.fn_index_compact = lib.func("memscale_index_compact", "int32_t", [
      voidPtr,
    ]);
    this.fn_index_save = lib.func("memscale_index_save", "int32_t", [
      voidPtr,
      voidPtrPtr,
    ]);
    this.fn_index_load = lib.func("memscale_index_load", voidPtr, ["str"]);
    this.fn_index_save_binary = lib.func(
      "memscale_index_save_binary",
      "int32_t",
      [voidPtr, "str"],
    );
  }

  private bindGpuFunctions(): void {
    try {
      this.fn_gpu_available = this.lib.func(
        "memscale_gpu_available",
        "int32_t",
        [voidPtrPtr],
      );
      this.fn_index_enable_gpu = this.lib.func(
        "memscale_index_enable_gpu",
        "int32_t",
        [voidPtr, "str"],
      );
      this.fn_index_gpu_batch_search = this.lib.func(
        "memscale_index_gpu_batch_search",
        "int32_t",
        [voidPtr, "str", "uint32_t", voidPtrPtr],
      );
      this.hasGpuSupport = true;
    } catch {
      this.hasGpuSupport = false;
    }
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

  /** Check if GPU support is available in the loaded library and hardware. */
  static gpuAvailable(libPath?: string): boolean {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);
    try {
      const fn_gpu_available = lib.func(
        "memscale_gpu_available",
        "int32_t",
        [voidPtrPtr],
      );
      const buf = newStringOut();
      const code = fn_gpu_available(buf);
      if (code !== MEMBRAIN_OK) return false;
      const { value: json, ptr } = readStringOut(buf);
      try {
        const parsed = JSON.parse(json);
        return parsed.available === true;
      } catch {
        return false;
      } finally {
        if (ptr) {
          const fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
          fn_string_free(ptr);
        }
      }
    } catch {
      return false;
    }
  }

  /** Enable GPU-accelerated distance computation. */
  enableGpu(config?: GpuConfig): void {
    if (!this.hasGpuSupport) {
      throw new MembrainError("library was built without GPU support");
    }
    const configJson = config ? JSON.stringify(config) : "";
    const code = this.fn_index_enable_gpu(this.handle, configJson);
    this.check(code);
  }

  /** Run multiple queries using GPU brute-force distance computation. */
  gpuBatchSearch(queries: number[][], k: number = 10): IndexSearchResult[][] {
    if (!this.hasGpuSupport) {
      return this.batchSearch(queries, k);
    }
    const buf = newStringOut();
    const queriesJson = JSON.stringify(queries);
    const code = this.fn_index_gpu_batch_search(
      this.handle,
      queriesJson,
      k,
      buf,
    );
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[][];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Add a vector to the index. */
  add(id: string, embedding: number[]): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_index_add(this.handle, id, embJson);
    this.check(code);
  }

  /** Remove a vector from the index by ID. */
  remove(id: string): void {
    const code = this.fn_index_remove(this.handle, id);
    this.check(code);
  }

  /** Search for the k nearest neighbors. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryJson = JSON.stringify(query);
    const code = this.fn_index_search(this.handle, queryJson, k, buf);
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
    const code = this.fn_index_search_with_filter(
      this.handle,
      queryJson,
      k,
      idsJson,
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

  /** Run multiple queries in parallel. */
  batchSearch(queries: number[][], k: number = 10): IndexSearchResult[][] {
    const buf = newStringOut();
    const queriesJson = JSON.stringify(queries);
    const code = this.fn_index_batch_search(this.handle, queriesJson, k, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[][];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Get the number of active vectors in the index. */
  len(): number {
    const out = newInt64Out();
    const code = this.fn_index_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get the vector dimension of the index. */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_index_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get index performance metrics. */
  metrics(): IndexMetrics {
    const buf = newStringOut();
    const code = this.fn_index_metrics(this.handle, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexMetrics;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Enable product quantization. */
  enablePq(config: PqConfig): void {
    const code = this.fn_index_enable_pq(this.handle, JSON.stringify(config));
    this.check(code);
  }

  /** Enable write-ahead logging for crash recovery. */
  enableWal(config: WalConfig): void {
    const code = this.fn_index_enable_wal(this.handle, JSON.stringify(config));
    this.check(code);
  }

  /** Trigger manual graph compaction. */
  compact(): void {
    const code = this.fn_index_compact(this.handle);
    this.check(code);
  }

  /** Save index state to a base64-encoded string (MessagePack format). */
  save(): string {
    const buf = newStringOut();
    const code = this.fn_index_save(this.handle, buf);
    this.check(code);
    const { value, ptr } = readStringOut(buf);
    try {
      return value;
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Save the index to a binary file for memory-mapped loading. */
  saveBinary(path: string): void {
    const code = this.fn_index_save_binary(this.handle, path);
    this.check(code);
  }

  /** Destroy the index and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_index_free(this.handle);
      this.handle = null;
    }
  }
}

/** Read-only HNSW index loaded from a binary file. */
export class MembrainMmapIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  private fn_last_error: any;
  private fn_string_free: any;
  private fn_mmap_free: any;
  private fn_mmap_search: any;
  private fn_mmap_len: any;

  constructor(path: string, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    const fn_load = this.lib.func("memscale_index_load_binary", voidPtr, ["str"]);
    this.handle = fn_load(path);
    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to load binary index: ${err}`);
    }
  }

  private bindFunctions(): void {
    const lib = this.lib;
    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_mmap_free = lib.func("memscale_index_mmap_free", "void", [voidPtr]);
    this.fn_mmap_search = lib.func("memscale_index_mmap_search", "int32_t", [
      voidPtr,
      "str",
      "uint32_t",
      voidPtrPtr,
    ]);
    this.fn_mmap_len = lib.func("memscale_index_mmap_len", "int32_t", [
      voidPtr,
      "int64_t *",
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

  /** Search for the k nearest neighbors. */
  search(query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryJson = JSON.stringify(query);
    const code = this.fn_mmap_search(this.handle, queryJson, k, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as IndexSearchResult[];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /** Get the number of vectors in the index. */
  len(): number {
    const out = newInt64Out();
    const code = this.fn_mmap_len(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Destroy the index and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_mmap_free(this.handle);
      this.handle = null;
    }
  }
}
