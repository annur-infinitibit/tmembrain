import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { ConcurrentFlatIndex } from "../dist/index.js";

const DIMENSION = 32;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("ConcurrentFlatIndex", () => {
  it("should create and close", () => {
    const index = new ConcurrentFlatIndex(DIMENSION);
    assert.equal(index.len(), 0);
    assert.equal(index.dimension(), DIMENSION);
    index.close();
  });

  it("should add and search", () => {
    const index = new ConcurrentFlatIndex(DIMENSION);
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
    const index = new ConcurrentFlatIndex(DIMENSION);
    const id = randomUUID();
    index.add(id, randomVector());
    assert.equal(index.remove(id), true);
    assert.equal(index.remove(id), false);
    assert.equal(index.len(), 0);
    index.close();
  });

  it("should share state across cloned handles", () => {
    const index = new ConcurrentFlatIndex(DIMENSION);
    const clone = index.clone();

    const id1 = randomUUID();
    const id2 = randomUUID();
    index.add(id1, randomVector());
    clone.add(id2, randomVector());

    assert.equal(index.len(), 2);
    assert.equal(clone.len(), 2);

    clone.close();
    index.close();
  });

  it("should sustain overlapping writes and reads", async () => {
    const index = new ConcurrentFlatIndex(DIMENSION);
    const writers = Array.from({ length: 4 }, (_, workerId) =>
      Promise.resolve().then(() => {
        for (let i = 0; i < 25; i++) {
          index.add(randomUUID(), randomVector());
        }
        return workerId;
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
    const index = new ConcurrentFlatIndex(DIMENSION, {
      distance_metric: "Euclidean",
    });
    index.add(randomUUID(), randomVector());
    assert.equal(index.search(randomVector(), 1).length, 1);
    index.close();
  });
});
