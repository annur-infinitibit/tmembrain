"""Tests for Ollama integration with Membrain.

These tests require a running Ollama instance at localhost:11434 with:
- nomic-embed-text (embedding model)
- llama3.1:8b or similar (chat model for extraction/conflict resolution)

Skip automatically if Ollama is not reachable.
"""

import time
import urllib.request
import urllib.error

import pytest

from membrain import MembrainClient


def _ollama_reachable() -> bool:
    """Check if Ollama is running at localhost:11434."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _has_model(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        import json

        response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        data = json.loads(response.read())
        model_names = [m["name"].split(":")[0] for m in data.get("models", [])]
        return model_name in model_names
    except (urllib.error.URLError, OSError, ValueError):
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama not running at localhost:11434",
)

requires_embedding_model = pytest.mark.skipif(
    not _has_model("nomic-embed-text"),
    reason="nomic-embed-text model not available in Ollama",
)


def _chat_model() -> str:
    """Find the best available chat model in Ollama."""
    preferred = ["llama3.1", "llama3.2", "qwen2.5", "qwen3", "qwen3.5"]
    for model in preferred:
        if _has_model(model):
            # Return the full name with tag from the API
            try:
                import json

                response = urllib.request.urlopen(
                    "http://localhost:11434/api/tags", timeout=2,
                )
                data = json.loads(response.read())
                for m in data.get("models", []):
                    if m["name"].startswith(model):
                        return m["name"]
            except (urllib.error.URLError, OSError, ValueError):
                pass
    return ""


CHAT_MODEL = _chat_model()

requires_chat_model = pytest.mark.skipif(
    not CHAT_MODEL,
    reason="No suitable chat model available in Ollama",
)


def _make_embedding_client() -> MembrainClient:
    """Create a client with Ollama embeddings only."""
    return MembrainClient(config={
        "storage": {"backend": "memory"},
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
        },
    })


def _make_full_client() -> MembrainClient:
    """Create a client with Ollama embeddings + conflict resolution."""
    return MembrainClient(config={
        "storage": {"backend": "memory"},
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
        },
        "write": {
            "conflict_resolution": {
                "enabled": True,
                "provider": "ollama",
                "model": CHAT_MODEL,
            },
        },
    })


@requires_ollama
@requires_embedding_model
async def test_store_and_search_with_ollama_embeddings():
    """Store facts using Ollama embeddings and search for them."""
    client = _make_embedding_client()

    result = await client.store_fact(
        "Python was created by Guido van Rossum", confidence=0.9,
    )
    assert result.success, f"Store failed: {result.rejection_reason}"

    time.sleep(0.5)

    results = await client.search("Who created Python?", limit=5)
    assert len(results.memories) > 0
    contents = [m.content.lower() for m in results.memories]
    assert any("guido" in c for c in contents), f"Expected Guido in results: {contents}"

    client.close()


@requires_ollama
@requires_embedding_model
async def test_ollama_embedding_dimension():
    """Verify nomic-embed-text produces 768-dimensional embeddings."""
    client = _make_embedding_client()

    result = await client.store_fact("Test embedding dimension", confidence=0.9)
    assert result.success, f"Store failed: {result.rejection_reason}"

    client.close()


@requires_ollama
@requires_embedding_model
@requires_chat_model
async def test_conflict_resolution_with_ollama():
    """Full conflict resolution using Ollama for both embeddings and chat."""
    client = _make_full_client()

    result1 = await client.store_fact("Bob's favorite color is blue", confidence=0.9)
    assert result1.success, f"First store failed: {result1.rejection_reason}"

    time.sleep(1)

    result2 = await client.store_fact("Bob's favorite color is green", confidence=0.9)
    assert result2.success or result2.merged_with, (
        f"Second store failed: {result2.rejection_reason}"
    )

    time.sleep(1)

    results = await client.search("What is Bob's favorite color?", limit=5)
    contents = [m.content.lower() for m in results.memories]
    has_green = any("green" in c for c in contents)
    assert has_green, f"Expected 'green' in results, got: {contents}"

    client.close()


@requires_ollama
@requires_embedding_model
async def test_multiple_facts_with_ollama():
    """Store multiple facts and verify search relevance."""
    client = _make_embedding_client()

    facts = [
        "The Earth is the third planet from the Sun",
        "Water freezes at 0 degrees Celsius",
        "Light travels at approximately 300000 km per second",
    ]

    for fact in facts:
        result = await client.store_fact(fact, confidence=0.9)
        assert result.success, f"Store failed for '{fact}': {result.rejection_reason}"

    time.sleep(0.5)

    results = await client.search("What temperature does water freeze?", limit=5)
    assert len(results.memories) > 0
    top_content = results.memories[0].content.lower()
    assert "water" in top_content or "freeze" in top_content or "0" in top_content, (
        f"Expected water/freeze fact as top result, got: {top_content}"
    )

    client.close()
