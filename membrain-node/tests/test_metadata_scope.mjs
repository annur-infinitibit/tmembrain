import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import { MembrainClient } from "../dist/index.js";

function makeTmpPath() {
  return join(tmpdir(), `membrain_scope_test_${randomUUID()}`);
}

function storageConfig(path, indexedKeys) {
  const config = {
    storage: {
      backend: "memscaledb",
      path,
    },
  };
  if (indexedKeys) {
    config.storage.indexed_metadata_keys = indexedKeys;
  }
  return config;
}

describe("MembrainClient metadata scope", () => {
  it("exposes the scope set at construction time", () => {
    const client = new MembrainClient(storageConfig(makeTmpPath()), {
      scope: { user_id: "alice", tenant_id: "acme" },
    });
    try {
      assert.deepEqual(client.scope, {
        user_id: "alice",
        tenant_id: "acme",
      });
    } finally {
      client.close();
    }
  });

  it("injects scope into stored memory metadata and filters searches by it", () => {
    const path = makeTmpPath();
    const alice = new MembrainClient(
      storageConfig(path, ["user_id"]),
      { scope: { user_id: "alice" } },
    );
    try {
      const stored = alice.storeFact("alice favors rust", 0.9);
      assert.ok(stored.success);

      const filtered = alice.search("rust", 10, {
        metadata: { user_id: "alice" },
      });
      assert.ok(
        filtered.memories.some((m) => m.content.includes("alice favors rust")),
        "stored memory should carry user_id=alice and match filter",
      );
    } finally {
      alice.close();
    }
  });

  it("isolates consecutive scoped clients on shared storage", () => {
    const path = makeTmpPath();
    const config = storageConfig(path, ["user_id"]);

    const alice = new MembrainClient(config, { scope: { user_id: "alice" } });
    try {
      alice.storeFact("alice likes rust", 0.9);
      alice.storeFact("alice likes python", 0.9);
    } finally {
      alice.close();
    }

    const bob = new MembrainClient(config, { scope: { user_id: "bob" } });
    try {
      bob.storeFact("bob likes go", 0.9);
      bob.storeFact("bob likes typescript", 0.9);

      const bobView = bob.search("likes", 20);
      for (const memory of bobView.memories) {
        assert.ok(
          memory.content.includes("bob"),
          `bob should not see alice's rows: ${memory.content}`,
        );
      }
      assert.ok(bobView.memories.some((m) => m.content.includes("bob")));
    } finally {
      bob.close();
    }

    const aliceAgain = new MembrainClient(config, {
      scope: { user_id: "alice" },
    });
    try {
      const aliceView = aliceAgain.search("likes", 20);
      for (const memory of aliceView.memories) {
        assert.ok(
          memory.content.includes("alice"),
          `alice should not see bob's rows: ${memory.content}`,
        );
      }
      assert.ok(aliceView.memories.some((m) => m.content.includes("alice")));
    } finally {
      aliceAgain.close();
    }
  });

  it("per-call metadata overrides scope on the same key at write time", () => {
    const client = new MembrainClient(
      storageConfig(makeTmpPath(), ["user_id"]),
      { scope: { user_id: "alice" } },
    );
    try {
      const stored = client.storeFact("shared context item", 0.9, undefined, {
        user_id: "bob",
      });
      assert.ok(stored.success);

      const bobView = client.search("shared", 5, {
        metadata: { user_id: "bob" },
      });
      assert.ok(
        bobView.memories.some((m) => m.content.includes("shared")),
        "override metadata should be visible under bob's filter",
      );

      // Default scope {user_id: alice} auto-applies here, so this client's own
      // search should NOT see the bob-tagged row.
      const aliceView = client.search("shared", 5);
      assert.ok(
        aliceView.memories.every((m) => !m.content.includes("shared")),
        "default scope should hide the overridden row",
      );
    } finally {
      client.close();
    }
  });
});
