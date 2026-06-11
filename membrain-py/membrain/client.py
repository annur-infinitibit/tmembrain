"""Membrain Python client — backed by native PyO3 extension (true async).

The main ``MembrainClient`` class is implemented in Rust and exposed
via the ``membrain._native`` PyO3 module. This file re-exports it and
adds the ``search_cases()`` convenience method.

Usage::

    from membrain import MembrainClient

    client = MembrainClient()
    result = await client.store_fact("User prefers dark mode", confidence=0.9)
    results = await client.search("dark mode", limit=10)
    for memory in results.memories:
        print(f"{memory.content}: {memory.score}")
    client.close()
"""

from __future__ import annotations

import json
from typing import Any

from membrain._native import MembrainClient as _NativeClient

from .errors import MembrainError
from .types import (
    CaseEntry,
    CaseSearchResults,
    MemoryEntry,
    MemoryInfo,
    SearchResults,
    StoreResult,
)


class MembrainClient:
    """Python client for the Membrain memory system.

    All I/O methods are async (return coroutines). The constructor is
    synchronous for convenience::

        client = MembrainClient()
        result = await client.store_fact("some fact", confidence=0.9)
        results = await client.search("some query")
        client.close()

    Context manager support::

        with MembrainClient() as client:
            ...
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        lib_path: str | None = None,
        scope: dict[str, Any] | None = None,
        indexed_metadata_keys: list[str] | None = None,
    ) -> None:
        """Initialize Membrain client.

        Args:
            config: Optional configuration dictionary.
            lib_path: Ignored (kept for backward compatibility with old ctypes client).
            scope: Optional default scope dict (e.g. ``{"user_id": "alice"}``).
                Merged into every stored memory's metadata and applied as a filter
                on every search. Per-call metadata / filter entries override the
                default on the same key.
            indexed_metadata_keys: Optional list of metadata keys to index at the
                storage layer for faster scoped filtering.
        """
        effective_config: dict[str, Any] = dict(config) if config else {}

        if scope:
            scope_section = dict(effective_config.get("scope", {}))
            default_scope = dict(scope_section.get("default_scope", {}))
            # Explicit scope passed here must override any defaults from `config`
            # to avoid accidental cross-tenant bleed when reusing config dicts.
            for key, value in scope.items():
                default_scope[key] = value
            scope_section["default_scope"] = default_scope
            effective_config["scope"] = scope_section

        if indexed_metadata_keys:
            storage_section = dict(effective_config.get("storage", {}))
            existing_keys = list(storage_section.get("indexed_metadata_keys", []))
            merged_keys = list(dict.fromkeys(existing_keys + list(indexed_metadata_keys)))
            storage_section["indexed_metadata_keys"] = merged_keys
            effective_config["storage"] = storage_section

        self._scope: dict[str, Any] = dict(
            effective_config.get("scope", {}).get("default_scope", {})
        )
        self._native = _NativeClient(
            config=effective_config if effective_config else None,
            lib_path=lib_path,
        )

    @property
    def scope(self) -> dict[str, Any]:
        """Default metadata scope applied to this client."""
        return dict(self._scope)

    # -------------------------------------------------------------------
    # Store methods — delegate to native async methods
    # -------------------------------------------------------------------

    async def store_fact(
        self,
        statement: str,
        confidence: float = 0.8,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoreResult:
        """Store a factual statement."""
        # Convert metadata values to strings for the native layer
        str_meta = None
        if metadata:
            str_meta = {k: str(v) for k, v in metadata.items()}
        result = await self._native.store_fact(statement, confidence, embedding, str_meta)
        return _to_store_result(result)

    async def store_preference(
        self,
        holder: str,
        subject: str,
        preference: str,
        strength: str = "moderate",
        embedding: list[float] | None = None,
    ) -> StoreResult:
        """Store a preference."""
        result = await self._native.store_preference(holder, subject, preference, strength, embedding)
        return _to_store_result(result)

    async def store_event(
        self,
        event_type: str,
        description: str,
        embedding: list[float] | None = None,
    ) -> StoreResult:
        """Store an event."""
        result = await self._native.store_event(event_type, description, embedding)
        return _to_store_result(result)

    async def store_observation(
        self, content: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store an observation."""
        result = await self._native.store_observation(content, embedding)
        return _to_store_result(result)

    async def store_concept(
        self, name: str, definition: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store a concept."""
        result = await self._native.store_concept(name, definition, embedding)
        return _to_store_result(result)

    async def store_entity(
        self, name: str, entity_type: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store an entity."""
        result = await self._native.store_entity(name, entity_type, embedding)
        return _to_store_result(result)

    async def store_workflow(
        self, name: str, description: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store a workflow."""
        result = await self._native.store_workflow(name, description, embedding)
        return _to_store_result(result)

    async def store_skill(
        self, name: str, description: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store a skill."""
        result = await self._native.store_skill(name, description, embedding)
        return _to_store_result(result)

    async def store_pattern(
        self,
        name: str,
        description: str,
        pattern_type: str,
        embedding: list[float] | None = None,
    ) -> StoreResult:
        """Store a pattern."""
        result = await self._native.store_pattern(name, description, pattern_type, embedding)
        return _to_store_result(result)

    async def store_case(
        self,
        problem: str,
        plan: str,
        outcome: str,
        reward: float = 1.0,
        embedding: list[float] | None = None,
    ) -> StoreResult:
        """Store an experience case for case-based reasoning."""
        result = await self._native.store_case(problem, plan, outcome, reward, embedding)
        return _to_store_result(result)

    async def store_goal(
        self, description: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store a goal."""
        result = await self._native.store_goal(description, embedding)
        return _to_store_result(result)

    async def store_task(
        self, title: str, embedding: list[float] | None = None
    ) -> StoreResult:
        """Store a task."""
        result = await self._native.store_task(title, embedding)
        return _to_store_result(result)

    # -------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> SearchResults:
        """Search for memories matching a query.

        Args:
            query: The search query text.
            limit: Maximum number of results.
            filters: Optional filter criteria dict.
            embedding: Optional pre-computed query embedding vector.
        """
        filters_json = json.dumps(filters) if filters else None
        result = await self._native.search(query, limit, filters_json, embedding)
        return SearchResults(
            memories=[
                MemoryEntry(
                    id=m.id,
                    content=m.content,
                    score=m.score,
                    memory_type=m.memory_type,
                    created_at=getattr(m, "created_at", ""),
                )
                for m in result.memories
            ],
            was_gated=result.was_gated,
            duration_ms=result.duration_ms,
        )

    async def search_cases(
        self,
        query: str,
        limit: int = 5,
        min_reward: float | None = None,
        positive_reward_threshold: float = 0.5,
    ) -> CaseSearchResults:
        """Search for similar cases, split into positive and negative.

        Args:
            query: The search query text.
            limit: Maximum number of results.
            min_reward: If set, only return cases with reward >= this value.
            positive_reward_threshold: Reward threshold to split positive/negative.
        """
        filters = {"memory_types": ["procedural_case"]}
        results = await self.search(query, limit=limit, filters=filters)

        positive_cases: list[CaseEntry] = []
        negative_cases: list[CaseEntry] = []

        for memory in results.memories:
            entry = _parse_case_entry(memory, positive_reward_threshold)
            if entry is None:
                continue
            if min_reward is not None and entry.reward < min_reward:
                continue
            if entry.reward >= positive_reward_threshold:
                positive_cases.append(entry)
            else:
                negative_cases.append(entry)

        return CaseSearchResults(
            positive_cases=positive_cases,
            negative_cases=negative_cases,
            duration_ms=results.duration_ms,
        )

    # -------------------------------------------------------------------
    # Get / Delete / Count
    # -------------------------------------------------------------------

    async def get(self, id: str) -> MemoryInfo | None:
        """Get a memory by ID. Returns None if not found."""
        result = await self._native.get(id)
        if result is None:
            return None
        return MemoryInfo(
            id=result.id,
            content=result.content,
            memory_type=result.memory_type,
            confidence=result.confidence,
        )

    async def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        return await self._native.delete(id)

    async def count(self) -> int:
        """Get the total number of stored memories."""
        return await self._native.count()

    # -------------------------------------------------------------------
    # Stats and health
    # -------------------------------------------------------------------

    async def stats(self) -> dict[str, Any]:
        """Get storage statistics as a dictionary."""
        json_str = await self._native.stats()
        return json.loads(json_str)

    async def vector_backend_health(self) -> dict[str, Any]:
        """Check vector backend health status."""
        json_str = await self._native.vector_backend_health()
        return json.loads(json_str)

    async def vector_backend_stats(self) -> dict[str, Any]:
        """Get vector backend capabilities and statistics."""
        json_str = await self._native.vector_backend_stats()
        return json.loads(json_str)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly release the underlying native client."""
        if hasattr(self, "_native") and self._native is not None:
            self._native.close()

    def __del__(self) -> None:
        if hasattr(self, "close"):
            self.close()

    def __enter__(self) -> MembrainClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_store_result(native_result) -> StoreResult:
    """Convert a native PyO3 StoreResult to the Python dataclass."""
    return StoreResult(
        success=native_result.success,
        id=native_result.id,
        merged_with=native_result.merged_with,
        rejection_reason=native_result.rejection_reason,
        duration_ms=native_result.duration_ms,
    )


def _parse_case_entry(
    memory: MemoryEntry, positive_reward_threshold: float
) -> CaseEntry | None:
    """Parse a MemoryEntry into a CaseEntry by extracting fields from text content."""
    content = memory.content
    problem = ""
    plan = ""
    outcome = ""
    reward = 0.0

    for line in content.split("\n"):
        if line.startswith("Problem: "):
            problem = line[len("Problem: "):]
        elif line.startswith("Plan: "):
            plan = line[len("Plan: "):]
        elif line.startswith("Outcome: "):
            outcome = line[len("Outcome: "):]
        elif line.startswith("Result: "):
            result_text = line[len("Result: "):]
            reward = 1.0 if result_text == "success" else 0.0

    if not problem and not plan and not outcome:
        return None

    return CaseEntry(
        id=memory.id,
        problem=problem,
        plan=plan,
        outcome=outcome,
        reward=reward,
        score=memory.score,
    )
