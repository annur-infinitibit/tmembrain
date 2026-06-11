"""
Metadata scope tests for the Membrain Python client.

Verify end-to-end that:
- ``scope`` injected at construction is applied on writes and reads.
- Two clients pointed at the same storage but with different scopes see only
  their own memories.
- Per-call ``metadata`` on store and ``metadata`` on filters override the
  default scope on a per-key basis.
- ``indexed_metadata_keys`` in config is accepted and does not break normal
  behavior.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from membrain import MembrainClient


def _storage_config(tmp_path, indexed_keys: list[str] | None = None) -> dict[str, Any]:
    storage_path = tmp_path / f"scope_{uuid.uuid4().hex[:8]}"
    config: dict[str, Any] = {
        "storage": {"backend": "memscaledb", "path": str(storage_path)},
    }
    if indexed_keys is not None:
        config["storage"]["indexed_metadata_keys"] = indexed_keys
    return config


def test_scope_attribute_reflects_ctor_arg(tmp_path):
    client = MembrainClient(
        config=_storage_config(tmp_path),
        scope={"user_id": "alice", "tenant_id": "acme"},
    )
    try:
        assert client.scope == {"user_id": "alice", "tenant_id": "acme"}
    finally:
        client.close()


def test_scope_injected_into_stored_metadata(tmp_path):
    """A stored memory should carry the client's scope keys in its metadata."""
    client = MembrainClient(
        config=_storage_config(tmp_path, indexed_keys=["user_id"]),
        scope={"user_id": "alice"},
    )
    try:
        result = client.store_fact("alice favors rust", confidence=0.9)
        assert result.success

        results = client.search("rust", limit=10, filters={"metadata": {"user_id": "alice"}})
        assert any("alice favors rust" in m.content for m in results.memories)
    finally:
        client.close()


def test_scope_isolates_consecutive_clients_on_same_storage(tmp_path):
    """Sequentially open two scoped clients on the same DB; each sees only its own rows.

    MemscaleDB (Redb) holds an exclusive lock, so this scenario mirrors a
    multi-process deployment where each process opens the DB, writes, closes,
    and a new process opens it with a different scope.
    """
    storage_path = tmp_path / f"shared_{uuid.uuid4().hex[:8]}"
    shared_config = {
        "storage": {
            "backend": "memscaledb",
            "path": str(storage_path),
            "indexed_metadata_keys": ["user_id"],
        },
    }

    alice = MembrainClient(config=shared_config, scope={"user_id": "alice"})
    try:
        alice.store_fact("alice likes rust", confidence=0.9)
        alice.store_fact("alice likes python", confidence=0.9)
    finally:
        alice.close()

    bob = MembrainClient(config=shared_config, scope={"user_id": "bob"})
    try:
        bob.store_fact("bob likes go", confidence=0.9)
        bob.store_fact("bob likes typescript", confidence=0.9)

        bob_view = bob.search("likes", limit=20)
        for memory in bob_view.memories:
            assert "bob" in memory.content, (
                f"Bob's search saw alice's row: {memory.content!r}"
            )
        assert any("bob" in memory.content for memory in bob_view.memories)
    finally:
        bob.close()

    alice_again = MembrainClient(config=shared_config, scope={"user_id": "alice"})
    try:
        alice_view = alice_again.search("likes", limit=20)
        for memory in alice_view.memories:
            assert "alice" in memory.content, (
                f"Alice's search saw bob's row: {memory.content!r}"
            )
        assert any("alice" in memory.content for memory in alice_view.memories)
    finally:
        alice_again.close()


def test_per_call_metadata_overrides_scope_key_on_store(tmp_path):
    """Per-call metadata on the same key wins over the client's default scope."""
    client = MembrainClient(
        config=_storage_config(tmp_path, indexed_keys=["user_id"]),
        scope={"user_id": "alice"},
    )
    try:
        # Write a memory with an explicit override — it should be tagged as 'bob'.
        result = client.store_fact(
            "shared context item",
            confidence=0.9,
            metadata={"user_id": "bob"},
        )
        assert result.success

        bob_view = client.search(
            "shared", limit=5, filters={"metadata": {"user_id": "bob"}}
        )
        assert any("shared" in m.content for m in bob_view.memories), (
            "override metadata should be visible under bob's filter"
        )

        # Alice's own scope still filters: the overridden memory has user_id=bob,
        # so alice's default-scope search should not see it (but alice's own
        # memories are not here — just the bob-tagged one).
        alice_view = client.search("shared", limit=5)
        # The default scope {user_id: alice} is applied, and no alice-tagged
        # memories exist, so alice's view is empty.
        assert all("shared" not in m.content for m in alice_view.memories)
    finally:
        client.close()


def test_non_indexed_metadata_still_filters(tmp_path):
    """Filtering on a non-indexed key falls back to post-filter correctness."""
    client = MembrainClient(
        config=_storage_config(tmp_path, indexed_keys=["user_id"]),
        scope={"user_id": "alice"},
    )
    try:
        client.store_fact("admin note", confidence=0.9, metadata={"role": "admin"})
        client.store_fact("viewer note", confidence=0.9, metadata={"role": "viewer"})

        admin_view = client.search(
            "note", limit=10, filters={"metadata": {"role": "admin"}}
        )
        assert any("admin" in m.content for m in admin_view.memories)
        assert all(
            "viewer" not in m.content for m in admin_view.memories
        ), "viewer memories should not appear when filtering role=admin"
    finally:
        client.close()
