import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainIndex, MembrainError } from "../dist/index.js";

function uid() {
  return randomUUID();
}

function sampleVector(seed, dimension) {
  return Array.from({ length: dimension }, (_, slot) => (seed + slot) * 0.01);
}

describe("MembrainIndex", () => {
  it("reports default dimension", () => {
    const index = new MembrainIndex(16);
    try {
      assert.equal(index.dimension(), 16);
    } finally {
      index.close();
    }
  });

  it("accepts explicit config", () => {
    const index = new MembrainIndex(undefined, {
      dimension: 12,
      m: 8,
      ef_construction: 50,
      ef_search: 20,
    });
    try {
      assert.equal(index.dimension(), 12);
    } finally {
      index.close();
    }
  });

  it("rejects add with wrong dimension", () => {
    const index = new MembrainIndex(8);
    try {
      assert.throws(
        () => index.add(uid(), new Array(4).fill(0.0)),
        MembrainError,
      );
    } finally {
      index.close();
    }
  });

  it("search on empty index returns empty array", () => {
    const index = new MembrainIndex(8);
    try {
      const hits = index.search(new Array(8).fill(0.0), 5);
      assert.deepEqual(hits, []);
    } finally {
      index.close();
    }
  });

  it("search with k larger than len returns at most len items", () => {
    const index = new MembrainIndex(8);
    try {
      for (let i = 0; i < 3; i++) index.add(uid(), sampleVector(i, 8));
      const hits = index.search(sampleVector(0, 8), 10);
      assert.ok(hits.length <= 3);
    } finally {
      index.close();
    }
  });

  it("batch_search returns one array per query", () => {
    const index = new MembrainIndex(8);
    try {
      for (let i = 0; i < 4; i++) index.add(uid(), sampleVector(i, 8));
      const queries = [sampleVector(0, 8), sampleVector(2, 8)];
      const results = index.batchSearch(queries, 2);
      assert.equal(results.length, 2);
    } finally {
      index.close();
    }
  });

  it("remove of bogus id raises", () => {
    const index = new MembrainIndex(8);
    try {
      assert.throws(() => index.remove("not-a-uuid"), MembrainError);
    } finally {
      index.close();
    }
  });

  it("metrics returns an object", () => {
    const index = new MembrainIndex(8);
    try {
      const metrics = index.metrics();
      assert.equal(typeof metrics, "object");
    } finally {
      index.close();
    }
  });

  it("close then use does not throw on idempotent close", () => {
    const index = new MembrainIndex(8);
    index.close();
    // Second close is a no-op.
    index.close();
  });
});
