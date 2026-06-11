// Scoped memory per user: two users share a DB but not their memories.
// Run: node examples/javascript/metadata_scope_example.mjs
// Requires: OPENAI_API_KEY for the LLM replies.

import { tmpdir } from "node:os";
import { join } from "node:path";

import OpenAI from "openai";
import { MembrainClient } from "membrain";

const DB_DIR = join(tmpdir(), "membrain_scope_example");
const INDEXED = ["user_id"];
const llm = new OpenAI();

async function chatAs(userId, message) {
  const client = new MembrainClient(
    {
      storage: {
        backend: "memscaledb",
        path: DB_DIR,
        indexed_metadata_keys: INDEXED,
      },
    },
    { scope: { user_id: userId } },
  );
  try {
    const prior = client
      .search(message, 5)
      .memories.map((m) => m.content)
      .join("\n");
    const reply = (
      await llm.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "user", content: `Context:\n${prior}\n\nUser: ${message}` },
        ],
      })
    ).choices[0].message.content;
    client.storeObservation(`${userId}: ${message}`);
    client.storeObservation(`assistant to ${userId}: ${reply}`);
    return reply;
  } finally {
    client.close();
  }
}

console.log(await chatAs("alice", "I love rust programming"));
console.log(await chatAs("bob", "I love go programming"));
console.log(await chatAs("alice", "what do I like?"));
