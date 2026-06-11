import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainFlatIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("MembrainFlatIndex", () => {
  it("should create and close", () => {
    const index = new MembrainFlatIndex(DIMENSION);
    assert.equal(index.len(), 0);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const index = new MembrainFlatIndex(DIMENSION);
    const id = randomUUID();
    const vector = randomVector();
    index.add(id, vector);
    assert.equal(index.len(), 1);

    const results = index.search(vector, 1);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, id);
    assert.ok(results[0].distance < 1e-5);
    index.close();
  });

  it("should remove", () => {
    const index = new MembrainFlatIndex(DIMENSION);
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
    const index = new MembrainFlatIndex(DIMENSION);
    const ids = [];
    for (let i = 0; i < 10; i++) {
      const id = randomUUID();
      index.add(id, randomVector());
      ids.push(id);
    }

    const allowed = ids.slice(0, 5);
    const results = index.searchWithFilter(randomVector(), 3, allowed);
    for (const r of results) {
      assert.ok(allowed.includes(r.id));
    }
    index.close();
  });

  it("should batch search", () => {
    const index = new MembrainFlatIndex(DIMENSION);
    for (let i = 0; i < 20; i++) {
      index.add(randomUUID(), randomVector());
    }

    const queries = [randomVector(), randomVector(), randomVector()];
    const batchResults = index.batchSearch(queries, 5);
    assert.equal(batchResults.length, 3);
    index.close();
  });

  it("should report metrics", () => {
    const index = new MembrainFlatIndex(DIMENSION);
    index.add(randomUUID(), randomVector());
    index.search(randomVector(), 1);

    const m = index.metrics();
    assert.ok(m.inserts >= 1);
    assert.ok(m.searches >= 1);
    index.close();
  });

  it("should achieve exact recall", () => {
    const index = new MembrainFlatIndex(DIMENSION);
    const pairs = [];
    for (let i = 0; i < 50; i++) {
      const id = randomUUID();
      const vec = randomVector();
      index.add(id, vec);
      pairs.push({ id, vec });
    }

    for (const { id, vec } of pairs) {
      const results = index.search(vec, 1);
      assert.equal(results[0].id, id);
    }
    index.close();
  });
});
