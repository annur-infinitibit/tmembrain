import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainLshIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("MembrainLshIndex", () => {
  it("should create and close", () => {
    const index = new MembrainLshIndex(DIMENSION);
    assert.equal(index.len(), 0);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const index = new MembrainLshIndex(DIMENSION);
    const ids = [];
    const vectors = [];
    for (let i = 0; i < 100; i++) {
      const id = randomUUID();
      const vec = randomVector();
      index.add(id, vec);
      ids.push(id);
      vectors.push(vec);
    }
    assert.equal(index.len(), 100);

    const results = index.search(vectors[0], 5);
    assert.ok(results.length <= 5);
    // Exact vector should be found (high probability with LSH)
    const resultIds = results.map((r) => r.id);
    assert.ok(resultIds.includes(ids[0]));
    index.close();
  });

  it("should remove", () => {
    const index = new MembrainLshIndex(DIMENSION);
    const id1 = randomUUID();
    const id2 = randomUUID();
    index.add(id1, randomVector());
    index.add(id2, randomVector());
    assert.equal(index.len(), 2);

    index.remove(id1);
    assert.equal(index.len(), 1);
    index.close();
  });

  it("should search with filter", () => {
    const index = new MembrainLshIndex(DIMENSION);
    const ids = [];
    for (let i = 0; i < 50; i++) {
      const id = randomUUID();
      index.add(id, randomVector());
      ids.push(id);
    }

    const allowed = ids.slice(0, 10);
    const results = index.searchWithFilter(randomVector(), 5, allowed);
    for (const r of results) {
      assert.ok(allowed.includes(r.id));
    }
    index.close();
  });

  it("should batch search", () => {
    const index = new MembrainLshIndex(DIMENSION);
    for (let i = 0; i < 50; i++) {
      index.add(randomUUID(), randomVector());
    }

    const queries = [randomVector(), randomVector(), randomVector()];
    const batchResults = index.batchSearch(queries, 5);
    assert.equal(batchResults.length, 3);
    index.close();
  });

  it("should report metrics", () => {
    const index = new MembrainLshIndex(DIMENSION);
    index.add(randomUUID(), randomVector());
    index.search(randomVector(), 1);

    const m = index.metrics();
    assert.ok(m.inserts >= 1);
    assert.ok(m.searches >= 1);
    index.close();
  });

  it("should work with config", () => {
    const index = new MembrainLshIndex(undefined, {
      dimension: DIMENSION,
      num_hyperplanes: 8,
      num_tables: 4,
      seed: 42,
    });
    const id = randomUUID();
    const vec = randomVector();
    index.add(id, vec);
    const results = index.search(vec, 1);
    assert.equal(results.length, 1);
    index.close();
  });
});
