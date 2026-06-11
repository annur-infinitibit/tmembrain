/**
 * Membrain graph memory layer client.
 */

import koffi from "koffi";

import type { GraphPruningResult, GraphQueryResult } from "./types";

import {
  charPtr,
  charPtrPtr,
  findLibrary,
  int64Ptr,
  MEMBRAIN_OK,
  MembrainError,
  newInt64Out,
  readInt64,
  voidPtr,
} from "./ffi";

/** Graph memory layer client. */
export class MembrainGraph {
  private handle: any;
  private lib: ReturnType<typeof koffi.load>;

  // Bound function references
  private fn_last_error: any;
  private fn_string_free: any;
  private fn_graph_free: any;
  private fn_graph_add_node: any;
  private fn_graph_remove_node: any;
  private fn_graph_query: any;
  private fn_graph_node_count: any;
  private fn_graph_edge_count: any;
  private fn_graph_prune: any;
  private fn_graph_save: any;
  private fn_graph_load: any;

  constructor(config?: Record<string, any>, libPath?: string) {
    const libFile = libPath ?? findLibrary();
    this.lib = koffi.load(libFile);
    this.bindGraphFunctions();

    if (config) {
      const createWithConfig = this.lib.func(
        "membrain_graph_new_with_config",
        voidPtr,
        ["str"]
      );
      this.handle = createWithConfig(JSON.stringify(config));
    } else {
      const create = this.lib.func("membrain_graph_new", voidPtr, []);
      this.handle = create();
    }

    if (!this.handle) {
      const err = this.getLastError();
      throw new MembrainError(`failed to create graph: ${err}`);
    }
  }

  /** Load a graph from a base64-encoded string. */
  static load(data: string, libPath?: string): MembrainGraph {
    const libFile = libPath ?? findLibrary();
    const lib = koffi.load(libFile);

    const fn_load = lib.func("membrain_graph_load", voidPtr, ["str"]);
    const handle = fn_load(data);
    if (!handle) {
      const fn_err = lib.func("membrain_last_error", "str", []);
      throw new MembrainError(`failed to load graph: ${fn_err() ?? "unknown error"}`);
    }

    const instance = Object.create(MembrainGraph.prototype) as MembrainGraph;
    (instance as any).lib = lib;
    (instance as any).handle = handle;
    instance.bindGraphFunctions();
    return instance;
  }

  private bindGraphFunctions(): void {
    const lib = this.lib;

    this.fn_last_error = lib.func("membrain_last_error", "str", []);
    this.fn_string_free = lib.func("membrain_string_free", "void", [charPtr]);
    this.fn_graph_free = lib.func("membrain_graph_free", "void", [voidPtr]);

    this.fn_graph_add_node = lib.func("membrain_graph_add_node", "int32_t", [
      voidPtr, "str", "str", "double",
    ]);
    this.fn_graph_remove_node = lib.func("membrain_graph_remove_node", "int32_t", [
      voidPtr, "str",
    ]);
    this.fn_graph_query = lib.func("membrain_graph_query", "int32_t", [
      voidPtr, "str", "int32_t", "int32_t", charPtrPtr,
    ]);
    this.fn_graph_node_count = lib.func("membrain_graph_node_count", "int32_t", [
      voidPtr, int64Ptr,
    ]);
    this.fn_graph_edge_count = lib.func("membrain_graph_edge_count", "int32_t", [
      voidPtr, int64Ptr,
    ]);
    this.fn_graph_prune = lib.func("membrain_graph_prune", "int32_t", [
      voidPtr, charPtrPtr,
    ]);
    this.fn_graph_save = lib.func("membrain_graph_save", "int32_t", [
      voidPtr, charPtrPtr,
    ]);
    this.fn_graph_load = lib.func("membrain_graph_load", voidPtr, ["str"]);
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

  /** Add a node to the graph with an embedding vector. */
  addNode(memoryId: string, embedding: number[], confidence: number = 0.5): void {
    const embJson = JSON.stringify(embedding);
    const code = this.fn_graph_add_node(this.handle, memoryId, embJson, confidence);
    this.check(code);
  }

  /** Remove a node and all its incident edges. */
  removeNode(memoryId: string): void {
    const code = this.fn_graph_remove_node(this.handle, memoryId);
    this.check(code);
  }

  /** Run a multi-hop graph query using an embedding vector. */
  query(embedding: number[], maxHops: number = -1, topK: number = 10): GraphQueryResult {
    const embJson = JSON.stringify(embedding);
    const out = [null] as any;
    const code = this.fn_graph_query(this.handle, embJson, maxHops, topK, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as GraphQueryResult;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Get the number of nodes in the graph. */
  nodeCount(): number {
    const out = newInt64Out();
    const code = this.fn_graph_node_count(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Get the number of edges in the graph. */
  edgeCount(): number {
    const out = newInt64Out();
    const code = this.fn_graph_edge_count(this.handle, out);
    this.check(code);
    return readInt64(out);
  }

  /** Manually trigger graph pruning. */
  prune(): GraphPruningResult {
    const out = [null] as any;
    const code = this.fn_graph_prune(this.handle, out);
    this.check(code);
    const json = out[0] as string;
    try {
      return JSON.parse(json) as GraphPruningResult;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Save graph state to a base64-encoded string. */
  save(): string {
    const out = [null] as any;
    const code = this.fn_graph_save(this.handle, out);
    this.check(code);
    const data = out[0] as string;
    try {
      return data;
    } finally {
      if (out[0]) this.fn_string_free(out[0]);
    }
  }

  /** Destroy the graph and free all associated resources. */
  close(): void {
    if (this.handle) {
      this.fn_graph_free(this.handle);
      this.handle = null;
    }
  }
}
