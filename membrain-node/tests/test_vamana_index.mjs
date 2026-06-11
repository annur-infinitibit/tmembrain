import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MembrainVamanaIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

function buildIndex(count = 200) {
  const ids = Array.from({ length: count }, () => randomUUID());
  const vectors = ids.map(() => randomVector());
  const index = MembrainVamanaIndex.build(ids, vectors, {
    dimension: DIMENSION,
    max_degree: 32,
    alpha: 1.2,
    search_list_size: 64,
  });
  return { index, ids, vectors };
}

describe("MembrainVamanaIndex", () => {
  it("should build and search", () => {
    const { index, ids, vectors } = buildIndex();
    assert.equal(index.len(), 200);
    assert.equal(index.dimension(), DIMENSION);

    const results = index.search(vectors[0], 5);
    assert.equal(results.length, 5);
    assert.equal(results[0].id, ids[0]);
    index.close();
  });

  it("should add after build", () => {
    const { index } = buildIndex(100);
    const newId = randomUUID();
    index.add(newId, randomVector());
    assert.equal(index.len(), 101);
    index.close();
  });

  it("should remove", () => {
    const { index, ids } = buildIndex(100);
    index.remove(ids[0]);
    assert.equal(index.len(), 99);
    index.close();
  });

  it("should search with filter", () => {
    const { index, ids } = buildIndex();
    const allowed = ids.slice(0, 20);
    const results = index.searchWithFilter(randomVector(), 5, allowed);
    for (const r of results) {
      assert.ok(allowed.includes(r.id));
    }
    index.close();
  });

  it("should batch search", () => {
    const { index } = buildIndex();
    const queries = [randomVector(), randomVector(), randomVector()];
    const batchResults = index.batchSearch(queries, 5);
    assert.equal(batchResults.length, 3);
    index.close();
  });

  it("should report metrics", () => {
    const { index, vectors } = buildIndex();
    index.search(vectors[0], 1);
    const m = index.metrics();
    assert.ok(m.searches >= 1);
    assert.ok(m.distance_computations > 0);
    index.close();
  });
});
