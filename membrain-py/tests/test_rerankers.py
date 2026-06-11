"""Unit tests for reranker helpers and implementations.

External-service tests (OpenAI/Cohere/Jina/Anthropic) use skipif markers and
pull credentials from the repo-root `.env` via conftest.
"""

from __future__ import annotations

import json
import os

import pytest

from membrain import RerankerError
from membrain.rerankers.base import (
    build_llm_user_prompt,
    build_rerank_results,
    parse_llm_scores,
)
from membrain.types import MemoryEntry, RerankResults, SearchResults


def sample_memories() -> list[MemoryEntry]:
    return [
        MemoryEntry(
            id="id-0",
            memory_type="semantic_fact",
            content="first document",
            score=0.9,
        ),
        MemoryEntry(
            id="id-1",
            memory_type="semantic_fact",
            content="second document",
            score=0.8,
        ),
        MemoryEntry(
            id="id-2",
            memory_type="semantic_fact",
            content="third document",
            score=0.7,
        ),
    ]


def test_build_llm_user_prompt_lists_documents_numbered() -> None:
    prompt = build_llm_user_prompt("q", ["alpha", "beta"])
    assert "Query: q" in prompt
    assert "[0] alpha" in prompt
    assert "[1] beta" in prompt


def test_parse_llm_scores_accepts_valid_json_array() -> None:
    response = json.dumps(
        [
            {"index": 0, "score": 9},
            {"index": 1, "score": 4},
            {"index": 2, "score": 8},
        ]
    )
    scored = parse_llm_scores(response, document_count=3, top_k=2)
    assert len(scored) == 2
    # Sorted descending by normalised score.
    assert scored[0][0] == 0
    assert scored[1][0] == 2


def test_parse_llm_scores_rejects_malformed_json() -> None:
    with pytest.raises(RerankerError):
        parse_llm_scores("not json", document_count=1, top_k=1)


def test_parse_llm_scores_rejects_non_array_json() -> None:
    with pytest.raises(RerankerError):
        parse_llm_scores('{"foo": 1}', document_count=1, top_k=1)


def test_parse_llm_scores_ignores_out_of_range_indices() -> None:
    response = json.dumps(
        [
            {"index": 0, "score": 10},
            {"index": 99, "score": 10},
            {"index": -1, "score": 10},
        ]
    )
    scored = parse_llm_scores(response, document_count=2, top_k=10)
    assert scored == [(0, 1.0)]


def test_parse_llm_scores_clamps_to_unit_interval() -> None:
    response = json.dumps([{"index": 0, "score": 20}, {"index": 1, "score": -5}])
    scored = parse_llm_scores(response, document_count=2, top_k=2)
    assert scored[0][1] == 1.0
    assert scored[-1][1] == 0.0


def test_build_rerank_results_sorts_and_truncates() -> None:
    memories = sample_memories()
    scored_indices = [(2, 0.9), (0, 0.5)]
    result = build_rerank_results(
        memories,
        scored_indices,
        model="test",
        provider="deterministic",
        duration_ms=42,
    )
    assert isinstance(result, RerankResults)
    assert len(result.memories) == 2
    assert result.memories[0].id == "id-2"
    assert result.memories[0].relevance_score == pytest.approx(0.9)


def test_build_rerank_results_skips_invalid_indices() -> None:
    memories = sample_memories()
    result = build_rerank_results(
        memories,
        [(5, 0.8), (0, 0.2)],
        model="test",
        provider="deterministic",
        duration_ms=0,
    )
    assert len(result.memories) == 1
    assert result.memories[0].id == "id-0"


def test_build_rerank_results_with_empty_scored_indices() -> None:
    memories = sample_memories()
    result = build_rerank_results(
        memories,
        [],
        model="test",
        provider="deterministic",
        duration_ms=0,
    )
    assert result.memories == []


def test_reranker_error_is_membrain_error() -> None:
    from membrain import MembrainError

    error = RerankerError("boom")
    assert isinstance(error, MembrainError)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_reranker_smoke(requires_openai: str) -> None:
    from membrain import OpenAIReranker

    reranker = OpenAIReranker(api_key=requires_openai, model="gpt-4o-mini")
    memories = SearchResults(memories=sample_memories())
    try:
        result = reranker.rerank("ranking query", memories, top_k=2)
    except RerankerError as error:
        if "401" in str(error) or "403" in str(error) or "invalid_api_key" in str(error):
            pytest.skip(f"OpenAI credentials rejected: {error}")
        raise
    assert len(result.memories) <= 2


@pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY"),
    reason="COHERE_API_KEY not set",
)
def test_cohere_reranker_smoke(requires_cohere: str) -> None:
    from membrain import CohereReranker

    reranker = CohereReranker(api_key=requires_cohere)
    memories = SearchResults(memories=sample_memories())
    try:
        result = reranker.rerank("cohere query", memories, top_k=2)
    except RerankerError as error:
        if "401" in str(error) or "403" in str(error):
            pytest.skip(f"Cohere credentials rejected: {error}")
        raise
    assert len(result.memories) <= 2


@pytest.mark.skipif(
    not os.environ.get("JINA_API_KEY"),
    reason="JINA_API_KEY not set",
)
def test_jina_reranker_smoke(requires_jina: str) -> None:
    from membrain import JinaReranker

    reranker = JinaReranker(api_key=requires_jina)
    memories = SearchResults(memories=sample_memories())
    try:
        result = reranker.rerank("jina query", memories, top_k=2)
    except RerankerError as error:
        if "401" in str(error) or "403" in str(error):
            pytest.skip(f"Jina credentials rejected: {error}")
        raise
    assert len(result.memories) <= 2


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_anthropic_reranker_smoke(requires_anthropic: str) -> None:
    from membrain import AnthropicReranker

    reranker = AnthropicReranker(api_key=requires_anthropic)
    memories = SearchResults(memories=sample_memories())
    try:
        result = reranker.rerank("anthropic query", memories, top_k=2)
    except RerankerError as error:
        if "401" in str(error) or "403" in str(error):
            pytest.skip(f"Anthropic credentials rejected: {error}")
        raise
    assert len(result.memories) <= 2
