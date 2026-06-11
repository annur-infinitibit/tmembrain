import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { ConcurrentHnswIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("ConcurrentHnswIndex", () => {
  it("should create and close", () => {
    const index = new ConcurrentHnswIndex(DIMENSION);
    assert.equal(index.len(), 0);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const index = new ConcurrentHnswIndex(DIMENSION);
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
    const index = new ConcurrentHnswIndex(DIMENSION);
    const id = randomUUID();
    index.add(id, randomVector());
    assert.equal(index.remove(id), true);
    assert.equal(index.remove(id), false);
    index.close();
  });

  it("should share state across cloned handles", () => {
    const index = new ConcurrentHnswIndex(DIMENSION);
    const clone = index.clone();

    index.add(randomUUID(), randomVector());
    clone.add(randomUUID(), randomVector());

    assert.equal(index.len(), 2);
    assert.equal(clone.len(), 2);

    clone.close();
    index.close();
  });

  it("should sustain overlapping writes and reads", async () => {
    const index = new ConcurrentHnswIndex(DIMENSION);
    const writers = Array.from({ length: 4 }, () =>
      Promise.resolve().then(() => {
        for (let i = 0; i < 25; i++) {
          index.add(randomUUID(), randomVector());
        }
      }),
    );
    const readers = Array.from({ length: 4 }, () =>
      Promise.resolve().then(() => index.search(randomVector(), 5)),
    );
    await Promise.all([...writers, ...readers]);

    assert.equal(index.len(), 100);
    index.close();
  });

  it("should accept custom config", () => {
    const index = new ConcurrentHnswIndex(DIMENSION, {
      distance_metric: "Cosine",
      m: 8,
      ef_construction: 64,
    });
    index.add(randomUUID(), randomVector());
    assert.equal(index.search(randomVector(), 1).length, 1);
    index.close();
  });
});
