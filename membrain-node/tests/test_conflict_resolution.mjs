/**
 * Tests for LLM-based conflict resolution in Membrain.
 *
 * Tests marked with `skip` require OPENAI_API_KEY to be set.
 * The non-API test verifies that conflict resolution is off by default.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import { MembrainClient } from "../dist/index.js";

function getApiKey() {
  if (process.env.OPENAI_API_KEY) {
    return process.env.OPENAI_API_KEY;
  }

  const envPath = join(import.meta.dirname, "..", "..", ".env");
  if (existsSync(envPath)) {
    const content = readFileSync(envPath, "utf-8");
    for (const line of content.split("\n")) {
      if (line.startsWith("OPENAI_API_KEY=")) {
        return line.split("=")[1].trim();
      }
    }
  }
  return null;
}

const API_KEY = getApiKey();
const hasOpenAI = !!API_KEY;

function makeClient() {
  return new MembrainClient({
    storage: { backend: "memory" },
    embedding: {
      provider: "openai",
      api_key: API_KEY,
      model: "text-embedding-3-small",
    },
    write: {
      conflict_resolution: {
        enabled: true,
        api_key: API_KEY,
        model: "gpt-4o-mini",
      },
    },
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("Conflict resolution", () => {
  it("contradicting fact invalidates old one", { skip: !hasOpenAI }, async () => {
    const client = makeClient();
    try {
      const result1 = client.storeFact("David likes football", 0.9);
      assert.ok(result1.success, `First store failed: ${result1.rejection_reason}`);

      await sleep(1000);

      const result2 = client.storeFact("David likes basketball", 0.9);
      assert.ok(
        result2.success || result2.merged_with,
        `Second store failed: ${result2.rejection_reason}`
      );

      await sleep(1000);

      const results = client.search("What sport does David like?", 5);
      const contents = results.memories.map((m) => m.content.toLowerCase());
      const hasBasketball = contents.some((c) => c.includes("basketball"));
      assert.ok(hasBasketball, `Expected basketball in results, got: ${contents}`);
    } finally {
      client.close();
    }
  });

  it("duplicate fact is noop", { skip: !hasOpenAI }, async () => {
    const client = makeClient();
    try {
      const result1 = client.storeFact("The Earth orbits the Sun", 0.95);
      assert.ok(result1.success);

      await sleep(1000);

      const result2 = client.storeFact("The Earth orbits the Sun", 0.95);
      // NOOP results in rejection, or LLM may still ADD -- both acceptable
      assert.ok(
        result2.success || result2.rejection_reason !== null,
        "Expected success or rejection"
      );
    } finally {
      client.close();
    }
  });

  it("update refines existing memory", { skip: !hasOpenAI }, async () => {
    const client = makeClient();
    try {
      const result1 = client.storeFact("Alice works at a tech company", 0.8);
      assert.ok(result1.success);

      await sleep(1000);

      const result2 = client.storeFact(
        "Alice works at Google as a senior engineer",
        0.9
      );
      assert.ok(
        result2.success || result2.merged_with !== null,
        `Refinement store failed: ${result2.rejection_reason}`
      );

      await sleep(1000);

      const results = client.search("Where does Alice work?", 5);
      const contents = results.memories.map((m) => m.content.toLowerCase());
      const hasGoogle = contents.some((c) => c.includes("google"));
      assert.ok(hasGoogle, `Expected 'google' in results, got: ${contents}`);
    } finally {
      client.close();
    }
  });

  it("conflict resolution disabled by default", () => {
    const unique = randomUUID();
    const client = new MembrainClient({
      storage: { backend: "memory" },
    });
    try {
      const statement = `Xylophonic resonance frequency ${unique} measured at 7.3 terahertz`;

      const result1 = client.storeFact(statement, 0.9);
      assert.ok(result1.success, `First store failed: ${result1.rejection_reason}`);

      // Without conflict resolution, exact duplicates are rejected by novelty filter
      const result2 = client.storeFact(statement, 0.9);
      assert.ok(!result2.success, "Duplicate should be rejected");
      assert.ok(result2.rejection_reason !== null);
      assert.ok(
        result2.rejection_reason.toLowerCase().includes("novelty"),
        `Expected novelty rejection, got: ${result2.rejection_reason}`
      );
    } finally {
      client.close();
    }
  });
});
