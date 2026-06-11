import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { AsyncConversation, MembrainClient } from "../dist/index.js";

function makeFakeAsyncLlm(reply, extractionJson) {
  const calls = [];
  const fn = async (messages) => {
    calls.push(messages);
    const system = messages[0]?.content ?? "";
    if (system.includes("Analyze the following conversation turn")) {
      return extractionJson;
    }
    return reply;
  };
  fn.calls = calls;
  return fn;
}

describe("AsyncConversation", () => {
  it("stores extracted memory round-trip", async () => {
    const client = new MembrainClient();
    try {
      const llm = makeFakeAsyncLlm(
        "I have noted your preference for dark mode.",
        '[{"type": "preference", "holder": "user", "subject": "theme",' +
          ' "preference": "dark mode", "strength": "strong"}]',
      );

      const conv = new AsyncConversation(llm, { client, autoExtract: true });
      const reply = await conv.reply("I prefer dark mode.");

      assert.equal(reply, "I have noted your preference for dark mode.");
      assert.equal(llm.calls.length, 2);

      const results = client.search("dark mode", 5);
      assert.ok(
        results.memories.some((m) => m.content.toLowerCase().includes("dark")),
      );
    } finally {
      client.close();
    }
  });

  it("invokes onExtractionError when extraction LLM throws", async () => {
    const client = new MembrainClient();
    try {
      const captured = [];
      const llm = async (messages) => {
        const system = messages[0]?.content ?? "";
        if (system.includes("Analyze the following conversation turn")) {
          throw new Error("extraction upstream failed");
        }
        return "ok";
      };

      const conv = new AsyncConversation(llm, {
        client,
        autoExtract: true,
        onExtractionError: (err) => captured.push(err),
      });
      const reply = await conv.reply("hello");

      assert.equal(reply, "ok");
      assert.equal(captured.length, 1);
      assert.match(String(captured[0]), /extraction upstream failed/);
    } finally {
      client.close();
    }
  });

  it("stores a case on end()", async () => {
    const client = new MembrainClient();
    try {
      const llm = async () => "assistant reply";
      const conv = new AsyncConversation(llm, { client, autoExtract: false });
      await conv.reply("hi there");
      await conv.end("positive conclusion", 1.0);

      const cases = client.searchCases("hi there", 5, 0.0);
      assert.ok(cases.positive_cases.length > 0);
    } finally {
      client.close();
    }
  });

  it("closes cleanly without an owned client", async () => {
    const client = new MembrainClient();
    try {
      const llm = async () => "ok";
      const conv = new AsyncConversation(llm, { client });
      await conv.close();
    } finally {
      client.close();
    }
  });
});
