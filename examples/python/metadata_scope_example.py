"""Scoped memory per user: two users share a DB but not their memories.

Run: python examples/python/metadata_scope_example.py
Requires: OPENAI_API_KEY in .env (for the LLM replies; the scope itself is
  database-only and needs no network).
"""

from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

from membrain import MembrainClient

load_dotenv()

DB_DIR = os.path.join(tempfile.gettempdir(), "membrain_scope_example")
INDEXED = ["user_id"]
llm = OpenAI()


def chat_as(user_id: str, message: str) -> str:
    client = MembrainClient(
        config={"storage": {"backend": "memscaledb", "path": DB_DIR,
                            "indexed_metadata_keys": INDEXED}},
        scope={"user_id": user_id},
    )
    try:
        prior = "\n".join(m.content for m in client.search(message, limit=5).memories)
        reply = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": f"Context:\n{prior}\n\nUser: {message}"}],
        ).choices[0].message.content
        client.store_observation(f"{user_id}: {message}")
        client.store_observation(f"assistant to {user_id}: {reply}")
        return reply
    finally:
        client.close()


if __name__ == "__main__":
    print(chat_as("alice", "I love rust programming"))
    print(chat_as("bob", "I love go programming"))
    # alice asks for her own memory; only alice-scoped rows are visible.
    print(chat_as("alice", "what do I like?"))
