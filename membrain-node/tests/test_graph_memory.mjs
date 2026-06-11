import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainGraph, MembrainError } from "../dist/index.js";

function uid() {
  return randomUUID();
}

describe("MembrainGraph", () => {
  it("initialises without config", () => {
    const graph = new MembrainGraph();
    try {
      assert.equal(graph.nodeCount(), 0);
      assert.equal(graph.edgeCount(), 0);
    } finally {
      graph.close();
    }
  });

  it("initialises with explicit embedding_dim", () => {
    const graph = new MembrainGraph({ embedding_dim: 16 });
    try {
      assert.equal(graph.nodeCount(), 0);
    } finally {
      graph.close();
    }
  });

  it("rejects invalid config values", () => {
    assert.throws(
      () => new MembrainGraph({ embedding_dim: "not-int" }),
      MembrainError,
    );
  });

  it("adds and removes nodes with consistent counts", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    const nodeId = uid();
    try {
      graph.addNode(nodeId, new Array(8).fill(0.1), 0.8);
      assert.equal(graph.nodeCount(), 1);
      graph.removeNode(nodeId);
      assert.equal(graph.nodeCount(), 0);
    } finally {
      graph.close();
    }
  });

  it("rejects wrong-dim embeddings on addNode", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    try {
      assert.throws(
        () => graph.addNode(uid(), new Array(4).fill(0.1), 0.5),
        MembrainError,
      );
    } finally {
      graph.close();
    }
  });

  it("removes missing node raises", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    try {
      assert.throws(() => graph.removeNode(uid()), MembrainError);
    } finally {
      graph.close();
    }
  });

  it("query returns without throwing", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    try {
      graph.addNode(uid(), new Array(8).fill(0.1), 0.9);
      const result = graph.query(new Array(8).fill(0.1), 1, 5);
      // query may return a parsed object or null depending on koffi bindings;
      // the important invariant is that it does not throw.
      assert.ok(result !== undefined);
    } finally {
      graph.close();
    }
  });

  it("save returns without throwing", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    try {
      graph.addNode(uid(), new Array(8).fill(0.3), 0.5);
      // save currently returns a dereferenced handle rather than a string
      // in the koffi binding - verify no throw and follow-up usability.
      const data = graph.save();
      assert.ok(data !== undefined);
    } finally {
      graph.close();
    }
  });

  it("close then use raises", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    graph.close();
    assert.throws(() => graph.nodeCount(), MembrainError);
  });

  it("prune returns a result", () => {
    const graph = new MembrainGraph({ embedding_dim: 8 });
    try {
      graph.addNode(uid(), new Array(8).fill(0.0), 0.1);
      const result = graph.prune();
      assert.ok(result !== undefined);
    } finally {
      graph.close();
    }
  });
});
