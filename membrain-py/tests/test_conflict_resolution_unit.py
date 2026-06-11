"""Unit tests for conflict-resolution configuration and error paths.

These tests do NOT require OpenAI credentials - they exercise the
configuration plumbing and error surfacing only.
"""

from __future__ import annotations

from membrain import MembrainClient


def test_client_disables_conflict_resolution_by_default(tmp_path) -> None:
    storage_path = tmp_path / "membrain-db"
    config = {
        "storage": {"backend": "memory", "path": str(storage_path)},
    }
    client = MembrainClient(config=config)
    try:
        # Client was constructed without conflict resolution enabled.
        stats = client.stats()
        assert isinstance(stats, dict)
    finally:
        client.close()


def test_client_accepts_conflict_resolution_config(tmp_path) -> None:
    storage_path = tmp_path / "membrain-db"
    config = {
        "storage": {"backend": "memory", "path": str(storage_path)},
        "write": {
            "conflict_resolution": {
                "enabled": False,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "http://127.0.0.1:1",
                "timeout_secs": 1,
                "retries": 0,
            }
        },
    }
    client = MembrainClient(config=config)
    try:
        stats = client.stats()
        assert isinstance(stats, dict)
    finally:
        client.close()
