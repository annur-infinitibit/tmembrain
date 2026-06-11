import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("MembrainIndex (HNSW)", () => {
  it("should create and close", () => {
    const index = new MembrainIndex(DIMENSION);
    assert.equal(index.len(), 0);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const index = new MembrainIndex(DIMENSION);
    const id = randomUUID();
    const vector = randomVector();
    index.add(id, vector);
    assert.equal(index.len(), 1);
    const results = index.search(vector, 1);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, id);
    index.close();
  });

  it("should remove", () => {
    const index = new MembrainIndex(DIMENSION);
    const id = randomUUID();
    index.add(id, randomVector());
    index.remove(id);
    assert.equal(index.len(), 0);
    index.close();
  });

  it("should search with filter", () => {
    const index = new MembrainIndex(DIMENSION);
    const ids = [];
    for (let i = 0; i < 20; i++) {
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
    const index = new MembrainIndex(DIMENSION);
    for (let i = 0; i < 30; i++) {
      index.add(randomUUID(), randomVector());
    }
    const queries = [randomVector(), randomVector(), randomVector()];
    const batchResults = index.batchSearch(queries, 5);
    assert.equal(batchResults.length, 3);
    index.close();
  });

  it("should report metrics", () => {
    const index = new MembrainIndex(DIMENSION);
    index.add(randomUUID(), randomVector());
    index.search(randomVector(), 1);
    const m = index.metrics();
    assert.ok(m.inserts >= 1);
    assert.ok(m.searches >= 1);
    index.close();
  });

  it("should save and load via base64", () => {
    const index = new MembrainIndex(DIMENSION);
    const id = randomUUID();
    const vector = randomVector();
    index.add(id, vector);
    const data = index.save();
    index.close();

    const restored = MembrainIndex.load(data);
    const results = restored.search(vector, 1);
    assert.equal(results[0].id, id);
    restored.close();
  });

  it("should accept custom config", () => {
    const index = new MembrainIndex(undefined, {
      dimension: DIMENSION,
      distance_metric: "Cosine",
      m: 8,
      ef_construction: 64,
    });
    index.add(randomUUID(), randomVector());
    assert.equal(index.search(randomVector(), 1).length, 1);
    index.close();
  });
});
