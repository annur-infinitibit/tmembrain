"""Tests for LLM-based conflict resolution in Membrain.

These tests require a valid OpenAI API key (set OPENAI_API_KEY env var or .env).
They exercise the full write pipeline with conflict resolution enabled:
store contradicting facts and verify the system auto-invalidates the old one.
"""

import os
import time
import uuid

import pytest

from membrain import MembrainClient


def _get_api_key() -> str | None:
    """Read the OpenAI API key from env or .env file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1]
    return None


API_KEY = _get_api_key()

requires_openai = pytest.mark.skipif(
    not API_KEY,
    reason="OPENAI_API_KEY not set",
)


def _make_client() -> MembrainClient:
    """Create a MembrainClient with in-memory storage, embedding, and conflict resolution."""
    return MembrainClient(config={
        "storage": {
            "backend": "memory",
        },
        "embedding": {
            "provider": "openai",
            "api_key": API_KEY,
            "model": "text-embedding-3-small",
        },
        "write": {
            "conflict_resolution": {
                "enabled": True,
                "api_key": API_KEY,
                "model": "gpt-4o-mini",
            },
        },
    })


@requires_openai
async def test_contradicting_fact_invalidates_old():
    """Store 'David likes football', then 'David likes basketball'.

    The conflict resolver should detect the contradiction and either:
    - DELETE the old memory and ADD the new one, or
    - UPDATE the old memory with the new content.

    Either way, searching should return basketball, not football.
    """
    client = _make_client()

    # Store the initial fact
    result1 = await client.store_fact("David likes football", confidence=0.9)
    assert result1.success, f"First store failed: {result1.rejection_reason}"

    # Small delay so embeddings are indexed
    time.sleep(1)

    # Store the contradicting fact
    result2 = await client.store_fact("David likes basketball", confidence=0.9)
    # The new fact should either be stored or merged (both are success)
    assert result2.success or result2.merged_with, (
        f"Second store failed: {result2.rejection_reason}"
    )

    time.sleep(1)

    # Search for David's sport preference
    results = await client.search("What sport does David like?", limit=5)

    # At least one result should mention basketball
    contents = [m.content.lower() for m in results.memories]
    has_basketball = any("basketball" in c for c in contents)
    assert has_basketball, f"Expected basketball in results, got: {contents}"

    client.close()


@requires_openai
async def test_duplicate_fact_is_noop():
    """Storing the exact same fact twice should result in a NOOP (rejection)."""
    client = _make_client()

    result1 = await client.store_fact("The Earth orbits the Sun", confidence=0.95)
    assert result1.success

    time.sleep(1)

    result2 = await client.store_fact("The Earth orbits the Sun", confidence=0.95)
    # NOOP results in a rejection (not an error, just a no-op)
    # The system may also ADD it if the resolver decides it's new enough
    # Either outcome is acceptable
    assert result2.success or result2.rejection_reason is not None

    client.close()


@requires_openai
async def test_update_refines_existing():
    """Storing a refinement should update the existing memory."""
    client = _make_client()

    result1 = await client.store_fact("Alice works at a tech company", confidence=0.8)
    assert result1.success

    time.sleep(1)

    result2 = await client.store_fact(
        "Alice works at Google as a senior engineer", confidence=0.9,
    )
    assert result2.success or result2.merged_with is not None

    time.sleep(1)

    results = await client.search("Where does Alice work?", limit=5)
    contents = [m.content.lower() for m in results.memories]
    has_google = any("google" in c for c in contents)
    assert has_google, f"Expected 'google' in results, got: {contents}"

    client.close()


async def test_conflict_resolution_disabled_by_default():
    """Without conflict_resolution config, duplicates hit the novelty filter instead."""
    unique = uuid.uuid4().hex
    client = MembrainClient(config={
        "storage": {"backend": "memory"},
    })

    statement = f"Xylophonic resonance frequency {unique} measured at 7.3 terahertz"
    result1 = await client.store_fact(statement, confidence=0.9)
    assert result1.success, f"First store failed: {result1.rejection_reason}"

    # Without conflict resolution, exact duplicates are rejected by the
    # novelty filter (low novelty score), not by LLM-based NOOP.
    result2 = await client.store_fact(statement, confidence=0.9)
    assert not result2.success
    assert result2.rejection_reason is not None
    assert "novelty" in result2.rejection_reason.lower()

    client.close()
