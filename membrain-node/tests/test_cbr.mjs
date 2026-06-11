/**
 * Tests for the case-based reasoning helpers (Node.js).
 *
 * Includes cross-language JSONL parity: the Python test writes
 * `tests/fixtures/cbr/training_data.jsonl` and this test reads it to assert
 * identical field values. The reverse direction is tested by writing from JS
 * and reading back via the same loader.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { rm } from "node:fs/promises";

import {
  CasePromptBuilder,
  ExperienceReplay,
  MembrainClient,
  NonParametricRetriever,
  TrainingDataCollector,
} from "../dist/index.js";
import { MembrainError } from "../dist/ffi.js";

const FIXTURE_DIR = join(
  import.meta.dirname,
  "..",
  "..",
  "tests",
  "fixtures",
  "cbr",
);
const GOLDEN_PROMPT = join(FIXTURE_DIR, "prompt_golden.txt");
const SHARED_JSONL = join(FIXTURE_DIR, "training_data.jsonl");

function makeTmpPath() {
  return join(tmpdir(), `membrain_cbr_${randomUUID()}`);
}

function sampleCaseSet() {
  return {
    positive_cases: [
      {
        id: "c1",
        problem: "Deploy the payment service",
        plan: JSON.stringify({
          plan: [
            { id: 1, description: "Build image" },
            { id: 2, description: "Push to registry" },
          ],
        }),
        outcome: "Service went live without downtime",
        reward: 1.0,
        score: 0.92,
      },
    ],
    negative_cases: [
      {
        id: "c2",
        problem: "Deploy the payment service",
        plan: "Restart production database first",
        outcome: "Caused a 20-minute outage",
        reward: 0.0,
        score: 0.41,
      },
    ],
    duration_ms: 5,
  };
}

describe("NonParametricRetriever", () => {
  it("round-trips a stored case", () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      client.storeCase(
        "Flaky integration test",
        "Increase timeout and add retries",
        "Tests pass on first run",
        1.0,
      );
      client.storeCase(
        "Database deadlock",
        "Reorder locks alphabetically",
        "Still deadlocks intermittently",
        0.0,
      );
      const retriever = new NonParametricRetriever(client);
      const result = retriever.retrieve("flaky test", 5);
      assert.ok(
        result.positive_cases.length + result.negative_cases.length >= 1,
      );
    } finally {
      client.close();
    }
  });
});

describe("CasePromptBuilder", () => {
  it("matches the Python golden prompt", () => {
    assert.ok(
      existsSync(GOLDEN_PROMPT),
      `Run pytest first to create ${GOLDEN_PROMPT}`,
    );
    const builder = new CasePromptBuilder();
    const context = builder.buildContext(sampleCaseSet());
    const golden = readFileSync(GOLDEN_PROMPT, "utf8");
    assert.equal(context, golden);
  });

  it("returns empty string when no cases are available", () => {
    const builder = new CasePromptBuilder();
    assert.equal(
      builder.buildContext({
        positive_cases: [],
        negative_cases: [],
        duration_ms: 0,
      }),
      "",
    );
  });
});

describe("TrainingDataCollector", () => {
  it("round-trips pairs through JSONL", async () => {
    const collector = new TrainingDataCollector();
    collector.record(
      "fix flaky",
      [
        {
          id: "a",
          problem: "Flaky test",
          plan: "Add retry",
          outcome: "Green",
          reward: 1.0,
          score: 0.9,
        },
        {
          id: "b",
          problem: "Deadlock",
          plan: "Reorder locks",
          outcome: "Still flaky",
          reward: 0.0,
          score: 0.2,
        },
      ],
      true,
    );
    const path = join(tmpdir(), `membrain_training_${randomUUID()}.jsonl`);
    const count = await collector.flush(path);
    assert.equal(count, 2);
    assert.equal(collector.size, 0);

    const pairs = await TrainingDataCollector.load(path);
    assert.equal(pairs.length, 2);
    assert.equal(pairs[0].query, "fix flaky");
    assert.equal(pairs[0].case_label, "positive");
    assert.equal(pairs[1].case_label, "negative");
    assert.equal(pairs[0].truth_label, true);
    await rm(path, { force: true });
  });

  it("reads the Python-written fixture with identical fields", async () => {
    assert.ok(
      existsSync(SHARED_JSONL),
      `Run pytest first to create ${SHARED_JSONL}`,
    );
    const pairs = await TrainingDataCollector.load(SHARED_JSONL);
    assert.equal(pairs.length, 2);
    assert.equal(pairs[0].case_label, "positive");
    assert.equal(pairs[1].case_label, "negative");
    assert.equal(pairs[0].plan, "Use weak references");
    assert.ok(pairs[0].case_text.startsWith("[CASE]"));
  });

  it("writes JSONL records with the shared schema", async () => {
    const collector = new TrainingDataCollector();
    collector.record(
      "js origin",
      [
        {
          id: "x",
          problem: "JS wrote this",
          plan: "Load in Python",
          outcome: "Fields match",
          reward: 1.0,
          score: 0.8,
        },
      ],
      true,
    );
    const path = join(tmpdir(), `membrain_jsonl_${randomUUID()}.jsonl`);
    try {
      await collector.flush(path);
      const content = readFileSync(path, "utf8").trim().split("\n");
      assert.equal(content.length, 1);
      const record = JSON.parse(content[0]);
      assert.equal(record.case_label, "positive");
      assert.equal(record.plan, "Load in Python");
      assert.equal(record.truth_label, true);
    } finally {
      await rm(path, { force: true });
    }
  });
});

describe("ExperienceReplay", () => {
  it("records and stores an execution", async () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    const path = join(tmpdir(), `membrain_replay_${randomUUID()}.jsonl`);
    try {
      const replay = new ExperienceReplay(client, {
        trainingDataPath: path,
      });
      const id = replay.recordExecution({
        problem: "Fix CI cache",
        plan: "Bump cache key",
        outcome: "Faster CI",
        reward: 1.0,
        query: "ci cache",
        retrievedCases: [],
        isCorrect: true,
      });
      assert.ok(id);
      assert.equal(replay.executionsSinceFlush, 1);
      const written = await replay.flushTrainingData();
      assert.equal(written, 0);
      assert.equal(replay.executionsSinceFlush, 0);
    } finally {
      client.close();
      await rm(path, { force: true });
    }
  });

  it("throws a MembrainError when retrain is invoked", async () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    const path = join(tmpdir(), `membrain_replay_${randomUUID()}.jsonl`);
    try {
      const replay = new ExperienceReplay(client, { trainingDataPath: path });
      await assert.rejects(() => replay.retrain("./out"), MembrainError);
    } finally {
      client.close();
      await rm(path, { force: true });
    }
  });
});
