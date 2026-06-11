/**
 * Multi-tenant vector index with namespace-based isolation.
 *
 * Each tenant gets an independent concurrent vector index. The registry lock
 * is held only briefly for tenant lookups; all vector operations delegate
 * to the tenant's own concurrent index with zero cross-tenant contention.
 *
 * Supported index types: "flat", "hnsw", "lsh".
 *
 * @example
 * ```typescript
 * import { MultiTenantIndex } from "membrain";
 *
 * const index = new MultiTenantIndex({ dimension: 1536, indexType: "hnsw" });
 *
 * index.createTenant("user-123");
 * index.createTenant("user-456");
 *
 * index.add("user-123", "vec-1", [0.1, 0.2, ...]);
 * const results = index.search("user-123", [0.1, 0.2, ...], 10);
 *
 * // user-456 sees nothing from user-123
 * const empty = index.search("user-456", [0.1, 0.2, ...], 10);
 *
 * index.close();
 * ```
 */

import type { IndexSearchResult } from "./types";

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

export interface MultiTenantIndexOptions {
  dimension?: number;
  indexType?: string;
  maxTenants?: number;
  indexConfig?: Record<string, unknown>;
  libPath?: string;
}

export class MultiTenantIndex {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_free: any;
  private fn_create_tenant: any;
  private fn_delete_tenant: any;
  private fn_has_tenant: any;
  private fn_list_tenants: any;
  private fn_tenant_count: any;
  private fn_add: any;
  private fn_remove: any;
  private fn_search: any;
  private fn_tenant_len: any;
  private fn_dimension: any;

  constructor(options: MultiTenantIndexOptions = {}) {
    const {
      dimension = 1536,
      indexType = "flat",
      maxTenants = 0,
      indexConfig,
      libPath,
    } = options;

    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindFunctions();

    const mtConfig = JSON.stringify({ dimension, max_tenants: maxTenants });
    const idxConfig = indexConfig ? JSON.stringify(indexConfig) : null;

    const create = this.lib.func(
      "memscale_multi_tenant_index_new",
      voidPtr,
      ["str", "str", "str"],
    );

    this.handle = create(mtConfig, indexType, idxConfig);
    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create multi-tenant index: ${err}`);
    }
  }

  private bindFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [voidPtr]);
    this.fn_free = lib.func("memscale_multi_tenant_index_free", "void", [voidPtr]);

    this.fn_create_tenant = lib.func(
      "memscale_multi_tenant_create_tenant", "int32_t", [voidPtr, "str"],
    );
    this.fn_delete_tenant = lib.func(
      "memscale_multi_tenant_delete_tenant", "int32_t", [voidPtr, "str", "int32_t *"],
    );
    this.fn_has_tenant = lib.func(
      "memscale_multi_tenant_has_tenant", "int32_t", [voidPtr, "str", "int32_t *"],
    );
    this.fn_list_tenants = lib.func(
      "memscale_multi_tenant_list_tenants", "int32_t", [voidPtr, voidPtrPtr],
    );
    this.fn_tenant_count = lib.func(
      "memscale_multi_tenant_tenant_count", "int32_t", [voidPtr, "int64_t *"],
    );

    this.fn_add = lib.func(
      "memscale_multi_tenant_add", "int32_t",
      [voidPtr, "str", "str", "float *", "uint32_t"],
    );
    this.fn_remove = lib.func(
      "memscale_multi_tenant_remove", "int32_t",
      [voidPtr, "str", "str", "int32_t *"],
    );
    this.fn_search = lib.func(
      "memscale_multi_tenant_search", "int32_t",
      [voidPtr, "str", "float *", "uint32_t", "uint32_t", voidPtrPtr],
    );
    this.fn_tenant_len = lib.func(
      "memscale_multi_tenant_tenant_len", "int32_t",
      [voidPtr, "str", "int64_t *"],
    );
    this.fn_dimension = lib.func(
      "memscale_multi_tenant_dimension", "int32_t",
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

  // -------------------------------------------------------------------
  // Tenant management
  // -------------------------------------------------------------------

  /**
   * Create a new tenant namespace.
   *
   * @param tenantId - Unique string identifier for the tenant.
   * @throws MembrainError if tenant already exists or max_tenants reached.
   */
  createTenant(tenantId: string): void {
    const code = this.fn_create_tenant(this.handle, tenantId);
    this.check(code);
  }

  /**
   * Delete a tenant and all its vectors.
   *
   * @param tenantId - The tenant to delete.
   * @returns True if the tenant existed and was removed.
   */
  deleteTenant(tenantId: string): boolean {
    const out = new Int32Array(1);
    const code = this.fn_delete_tenant(this.handle, tenantId, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * Check if a tenant exists.
   *
   * @param tenantId - The tenant ID to check.
   * @returns True if the tenant exists.
   */
  hasTenant(tenantId: string): boolean {
    const out = new Int32Array(1);
    const code = this.fn_has_tenant(this.handle, tenantId, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * List all tenant IDs (sorted alphabetically).
   *
   * @returns Array of tenant ID strings.
   */
  listTenants(): string[] {
    const buf = newStringOut();
    const code = this.fn_list_tenants(this.handle, buf);
    this.check(code);
    const { value: json, ptr } = readStringOut(buf);
    try {
      return JSON.parse(json) as string[];
    } finally {
      if (ptr) this.fn_string_free(ptr);
    }
  }

  /**
   * Get the number of tenants.
   */
  tenantCount(): number {
    const out = newInt64Out();
    const code = this.fn_tenant_count(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  // -------------------------------------------------------------------
  // Per-tenant vector operations
  // -------------------------------------------------------------------

  /**
   * Add a vector to a tenant's index.
   *
   * @param tenantId - The tenant namespace.
   * @param id - UUID string identifying the vector.
   * @param embedding - Array of numbers (must match index dimension).
   */
  add(tenantId: string, id: string, embedding: number[]): void {
    const vectorArray = new Float32Array(embedding);
    const code = this.fn_add(
      this.handle, tenantId, id, vectorArray, embedding.length,
    );
    this.check(code);
  }

  /**
   * Remove a vector from a tenant's index.
   *
   * @param tenantId - The tenant namespace.
   * @param id - UUID string of the vector to remove.
   * @returns True if the vector was found and removed.
   */
  remove(tenantId: string, id: string): boolean {
    const out = new Int32Array(1);
    const code = this.fn_remove(this.handle, tenantId, id, out);
    this.check(code);
    return out[0] !== 0;
  }

  /**
   * Search a tenant's index for nearest neighbors.
   *
   * @param tenantId - The tenant namespace to search.
   * @param query - Query vector (must match index dimension).
   * @param k - Number of results to return.
   * @returns Array of search results sorted by score (highest first).
   */
  search(tenantId: string, query: number[], k: number = 10): IndexSearchResult[] {
    const buf = newStringOut();
    const queryArray = new Float32Array(query);
    const code = this.fn_search(
      this.handle, tenantId, queryArray, query.length, k, buf,
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
   * Get the number of vectors in a tenant's index.
   *
   * @param tenantId - The tenant namespace.
   */
  tenantLen(tenantId: string): number {
    const out = newInt64Out();
    const code = this.fn_tenant_len(this.handle, tenantId, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Get the vector dimension of the index.
   */
  dimension(): number {
    const out = newInt64Out();
    const code = this.fn_dimension(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /**
   * Release the multi-tenant index and all tenant data.
   */
  close(): void {
    if (this.handle) {
      this.fn_free(this.handle);
      this.handle = null;
    }
  }
}
