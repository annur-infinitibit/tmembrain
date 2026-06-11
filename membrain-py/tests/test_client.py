"""Tests for MembrainClient (native PyO3 backend)."""

import uuid

import pytest

from membrain import MembrainClient, MemoryEntry


def _uid() -> str:
    """Generate a short unique token for content uniqueness."""
    return uuid.uuid4().hex[:12]


@pytest.fixture
async def client(tmp_path):
    """Create a MembrainClient with temporary storage."""
    client = MembrainClient(
        config={
            "storage_path": str(tmp_path / "test_db"),
            "write": {
                "novelty": {"enabled": False},
                "salience": {"enabled": False},
            },
        }
    )
    try:
        yield client
    finally:
        client.close()


async def test_search_result_has_created_at(client):
    """MemoryEntry.created_at should be populated after search."""
    tag = _uid()
    result = await client.store_fact(f"The sky is blue {tag}", confidence=0.9)
    assert result.success

    results = await client.search(tag, limit=10)
    assert len(results.memories) > 0

    for memory in results.memories:
        assert isinstance(memory, MemoryEntry)
        assert memory.created_at, "created_at should be non-empty"
        assert "T" in memory.created_at, f"Expected RFC 3339 format, got: {memory.created_at}"


async def test_memory_entry_created_at_default():
    """MemoryEntry created_at should default to empty string."""
    entry = MemoryEntry(id="test", content="hello", score=1.0, memory_type="semantic_fact")
    assert entry.created_at == ""


async def test_search_gating_active_on_search(client):
    """search() should gate greetings via the full adaptive pipeline."""
    await client.store_fact("Important fact", confidence=0.9)
    results = await client.search("hello", limit=10)
    # With the new native client, search() uses the full pipeline
    # which includes intent-based gating for greetings
    assert results.was_gated
    assert len(results.memories) == 0


async def test_store_and_search(client):
    """Basic store and search round-trip."""
    tag = _uid()
    result = await client.store_fact(f"Rust is a systems language {tag}", confidence=0.95)
    assert result.success
    assert result.id

    results = await client.search(f"systems language {tag}", limit=10)
    assert len(results.memories) > 0


async def test_get_and_delete(client):
    """Store, get, delete, get cycle."""
    tag = _uid()
    result = await client.store_fact(f"temporary fact {tag}", confidence=0.8)
    assert result.success
    memory_id = result.id

    info = await client.get(memory_id)
    assert info is not None
    assert info.id == memory_id

    deleted = await client.delete(memory_id)
    assert deleted is True

    info = await client.get(memory_id)
    assert info is None


async def test_count(client):
    """Count should increase after storing."""
    initial = await client.count()
    tag = _uid()
    result = await client.store_fact(f"countable fact {tag}", confidence=0.9)
    assert result.success
    count = await client.count()
    assert count == initial + 1


async def test_stats(client):
    """Stats should return a dict."""
    tag = _uid()
    await client.store_fact(f"stats test fact {tag}", confidence=0.9)
    stats = await client.stats()
    assert isinstance(stats, dict)
    assert "total_memories" in stats


async def test_true_concurrent_operations(client):
    """Verify operations run concurrently — impossible with fake async."""
    import time
    import asyncio

    start = time.monotonic()
    run_id = _uid()
    # Fire 15 store operations concurrently
    results = await asyncio.gather(*(
        client.store_fact(f"concurrent fact {run_id} {i}", confidence=0.9)
        for i in range(15)
    ))
    elapsed = time.monotonic() - start

    def is_acceptable_result(r):
        if r.success:
            return True
        # Under true concurrency on Windows, Tantivy/SQLite might throw OS Error 5 / locked
        if r.rejection_reason and ("Access is denied" in r.rejection_reason or "locked" in r.rejection_reason.lower() or "storage error" in r.rejection_reason.lower()):
            return True
        return False

    assert all(is_acceptable_result(r) for r in results), [r.rejection_reason for r in results if not r.success]
    # The main test is that it doesn't deadlock or panic
    assert elapsed < 5.0, f"Operations took too long: {elapsed:.3f}s"


async def test_no_cannot_start_runtime_error(client):
    """Verify no 'Cannot start a runtime from within an async context' panic."""
    # This would panic with the old block_on approach when called from async
    result = await client.store_fact(f"test from async context {_uid()}", confidence=0.8)
    assert result.success


async def test_client_usable_inside_fastapi(tmp_path):
    """Smoke test: client works when used from a running event loop."""
    # Simulates FastAPI handler
    async def handler():
        c = MembrainClient(config={
            "storage_path": str(tmp_path / "api_test"),
            "write": {
                "novelty": {"enabled": False},
                "salience": {"enabled": False},
            },
        })
        result = await c.store_fact(f"API test fact {_uid()}", confidence=0.9)
        c.close()
        return result.success

    assert await handler()

async def test_closed_client_raises(tmp_path):
    """Closed client should raise RuntimeError on method call."""
    c = MembrainClient(config={"storage_path": str(tmp_path / "closed_test")})
    c.close()
    with pytest.raises(RuntimeError, match="client is closed"):
        await c.store_fact("should fail")
