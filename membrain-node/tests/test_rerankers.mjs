import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  RerankerError,
  buildLlmUserPrompt,
  buildRerankResults,
  parseLlmScores,
} from "../dist/rerankers/base.js";

function sampleMemories() {
  return [
    { id: "id-0", content: "first", score: 0.9, memory_type: "semantic_fact" },
    { id: "id-1", content: "second", score: 0.8, memory_type: "semantic_fact" },
    { id: "id-2", content: "third", score: 0.7, memory_type: "semantic_fact" },
  ];
}

describe("buildLlmUserPrompt", () => {
  it("lists documents with numbered prefixes", () => {
    const prompt = buildLlmUserPrompt("q", ["alpha", "beta"]);
    assert.ok(prompt.includes("Query: q"));
    assert.ok(prompt.includes("[0] alpha"));
    assert.ok(prompt.includes("[1] beta"));
  });
});

describe("parseLlmScores", () => {
  it("parses valid JSON arrays into sorted tuples", () => {
    const response = JSON.stringify([
      { index: 0, score: 9 },
      { index: 1, score: 4 },
      { index: 2, score: 8 },
    ]);
    const parsed = parseLlmScores(response, 3, 2);
    assert.equal(parsed.length, 2);
    assert.equal(parsed[0][0], 0);
    assert.equal(parsed[1][0], 2);
  });

  it("throws RerankerError on invalid JSON", () => {
    assert.throws(
      () => parseLlmScores("not-json", 1, 1),
      RerankerError,
    );
  });

  it("throws RerankerError on non-array JSON", () => {
    assert.throws(
      () => parseLlmScores('{"foo": 1}', 1, 1),
      RerankerError,
    );
  });

  it("ignores out-of-range indices", () => {
    const response = JSON.stringify([
      { index: 0, score: 10 },
      { index: 5, score: 10 },
      { index: -1, score: 10 },
    ]);
    const parsed = parseLlmScores(response, 2, 10);
    assert.deepEqual(parsed, [[0, 1.0]]);
  });

  it("clamps scores to [0,1]", () => {
    const response = JSON.stringify([
      { index: 0, score: 100 },
      { index: 1, score: -10 },
    ]);
    const parsed = parseLlmScores(response, 2, 2);
    assert.equal(parsed[0][1], 1.0);
    assert.equal(parsed[parsed.length - 1][1], 0.0);
  });
});

describe("buildRerankResults", () => {
  it("returns ordered results", () => {
    const result = buildRerankResults(
      sampleMemories(),
      [[2, 0.9], [0, 0.5]],
      "test",
      "deterministic",
      42,
    );
    assert.equal(result.memories.length, 2);
    assert.equal(result.memories[0].id, "id-2");
    assert.equal(result.memories[0].relevance_score, 0.9);
    assert.equal(result.duration_ms, 42);
  });

  it("skips invalid indices silently", () => {
    const result = buildRerankResults(
      sampleMemories(),
      [[9, 0.8], [0, 0.1]],
      "test",
      "deterministic",
      0,
    );
    assert.equal(result.memories.length, 1);
    assert.equal(result.memories[0].id, "id-0");
  });

  it("returns empty memories on empty input", () => {
    const result = buildRerankResults(sampleMemories(), [], "t", "d", 0);
    assert.deepEqual(result.memories, []);
  });
});

describe("RerankerError", () => {
  it("is an Error subclass", () => {
    const error = new RerankerError("boom");
    assert.ok(error instanceof Error);
    assert.equal(error.message, "boom");
  });
});
