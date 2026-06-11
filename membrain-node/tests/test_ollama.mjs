/**
 * Tests for Ollama integration with Membrain.
 *
 * Requires a running Ollama instance at localhost:11434 with:
 * - nomic-embed-text (embedding model)
 * - llama3.1:8b or similar (chat model for conflict resolution)
 *
 * Skips automatically if Ollama is not reachable.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { MembrainClient } from "../dist/index.js";

async function ollamaReachable() {
  try {
    const response = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function getModels() {
  try {
    const response = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(2000),
    });
    const data = await response.json();
    return (data.models || []).map((m) => m.name);
  } catch {
    return [];
  }
}

async function hasModel(name) {
  const models = await getModels();
  return models.some((m) => m.startsWith(name));
}

async function findChatModel() {
  const preferred = ["llama3.1", "llama3.2", "qwen2.5", "qwen3", "qwen3.5"];
  const models = await getModels();
  for (const prefix of preferred) {
    const found = models.find((m) => m.startsWith(prefix));
    if (found) return found;
  }
  return "";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const isReachable = await ollamaReachable();
const hasEmbedding = isReachable && (await hasModel("nomic-embed-text"));
const chatModel = isReachable ? await findChatModel() : "";

function makeEmbeddingClient() {
  return new MembrainClient({
    storage: { backend: "memory" },
    embedding: {
      provider: "ollama",
      model: "nomic-embed-text",
    },
  });
}

function makeFullClient() {
  return new MembrainClient({
    storage: { backend: "memory" },
    embedding: {
      provider: "ollama",
      model: "nomic-embed-text",
    },
    write: {
      conflict_resolution: {
        enabled: true,
        provider: "ollama",
        model: chatModel,
      },
    },
  });
}

describe("Ollama integration", () => {
  it("store and search with Ollama embeddings", { skip: !hasEmbedding }, async () => {
    const client = makeEmbeddingClient();
    try {
      const result = client.storeFact(
        "Python was created by Guido van Rossum",
        0.9
      );
      assert.ok(result.success, `Store failed: ${result.rejection_reason}`);

      await sleep(500);

      const results = client.search("Who created Python?", 5);
      assert.ok(results.memories.length > 0, "Expected search results");

      const contents = results.memories.map((m) => m.content.toLowerCase());
      const hasGuido = contents.some((c) => c.includes("guido"));
      assert.ok(hasGuido, `Expected Guido in results: ${contents}`);
    } finally {
      client.close();
    }
  });

  it("conflict resolution with Ollama", {
    skip: !hasEmbedding || !chatModel,
    timeout: 60000,
  }, async () => {
    const client = makeFullClient();
    try {
      const result1 = client.storeFact(
        "Bob's favorite color is blue",
        0.9
      );
      assert.ok(result1.success, `First store failed: ${result1.rejection_reason}`);

      await sleep(1000);

      const result2 = client.storeFact(
        "Bob's favorite color is green",
        0.9
      );
      assert.ok(
        result2.success || result2.merged_with,
        `Second store failed: ${result2.rejection_reason}`
      );

      await sleep(1000);

      const results = client.search("What is Bob's favorite color?", 5);
      const contents = results.memories.map((m) => m.content.toLowerCase());
      const hasGreen = contents.some((c) => c.includes("green"));
      assert.ok(hasGreen, `Expected 'green' in results, got: ${contents}`);
    } finally {
      client.close();
    }
  });

  it("multiple facts with Ollama embeddings", { skip: !hasEmbedding }, async () => {
    const client = makeEmbeddingClient();
    try {
      const facts = [
        "The Earth is the third planet from the Sun",
        "Water freezes at 0 degrees Celsius",
        "Light travels at approximately 300000 km per second",
      ];

      for (const fact of facts) {
        const result = client.storeFact(fact, 0.9);
        assert.ok(result.success, `Store failed for '${fact}': ${result.rejection_reason}`);
      }

      await sleep(500);

      const results = client.search("What temperature does water freeze?", 5);
      assert.ok(results.memories.length > 0, "Expected search results");

      const topContent = results.memories[0].content.toLowerCase();
      assert.ok(
        topContent.includes("water") || topContent.includes("freeze") || topContent.includes("0"),
        `Expected water/freeze as top result, got: ${topContent}`
      );
    } finally {
      client.close();
    }
  });
});
