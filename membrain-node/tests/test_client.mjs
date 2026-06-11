import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import { MembrainClient } from "../dist/index.js";

function makeTmpPath() {
  return join(tmpdir(), `membrain_test_${randomUUID()}`);
}

describe("MembrainClient search results", () => {
  it("created_at should be present in search results", () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      const stored = client.storeFact("The sky is blue", 0.9);
      assert.ok(stored.success);

      const results = client.search("sky", 10);
      assert.ok(results.memories.length > 0);

      for (const memory of results.memories) {
        assert.ok(
          typeof memory.created_at === "string",
          "created_at should be a string"
        );
        assert.ok(
          memory.created_at.length > 0,
          "created_at should be non-empty"
        );
        assert.ok(
          memory.created_at.includes("T"),
          `created_at should be RFC 3339 format, got: ${memory.created_at}`
        );
      }
    } finally {
      client.close();
    }
  });

  it("search() should gate greetings", () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      client.storeFact("Important fact", 0.9);
      const results = client.search("hello", 10);
      assert.ok(results.was_gated, "hello should be gated");
      assert.equal(results.memories.length, 0);
    } finally {
      client.close();
    }
  });
});
