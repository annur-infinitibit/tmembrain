import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { ConcurrentIvfIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

function buildIndex(count = 100, config = { num_cells: 4, nprobe: 2 }) {
  const ids = Array.from({ length: count }, () => randomUUID());
  const vectors = ids.map(() => randomVector());
  return {
    index: ConcurrentIvfIndex.build(ids, vectors, DIMENSION, config),
    ids,
    vectors,
  };
}

describe("ConcurrentIvfIndex", () => {
  it("should build and inspect", () => {
    const { index } = buildIndex(100);
    assert.equal(index.len(), 100);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const { index, vectors, ids } = buildIndex(100);
    const results = index.search(vectors[0], 5);
    assert.ok(results.length > 0);
    assert.ok(results.some((r) => r.id === ids[0]));
    index.close();
  });

  it("should remove", () => {
    const { index, ids } = buildIndex(50);
    assert.equal(index.remove(ids[0]), true);
    assert.equal(index.remove(ids[0]), false);
    assert.equal(index.len(), 49);
    index.close();
  });

  it("should share state across cloned handles", () => {
    const { index } = buildIndex(50);
    const clone = index.clone();
    assert.equal(clone.len(), index.len());
    clone.close();
    index.close();
  });

  it("should sustain overlapping writes and reads", async () => {
    const { index, vectors } = buildIndex(200);
    const writers = Array.from({ length: 4 }, () =>
      Promise.resolve().then(() => {
        for (let i = 0; i < 20; i++) {
          index.add(randomUUID(), randomVector());
        }
      }),
    );
    const readers = Array.from({ length: 4 }, () =>
      Promise.resolve().then(() => index.search(vectors[0], 5)),
    );
    await Promise.all([...writers, ...readers]);
    assert.equal(index.len(), 280);
    index.close();
  });
});
