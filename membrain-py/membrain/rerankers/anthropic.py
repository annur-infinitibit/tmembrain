"""Anthropic LLM-based reranker."""

from __future__ import annotations

import time

from ..types import RerankResults, SearchResults
from .base import (
    DEFAULT_TIMEOUT,
    LLM_SYSTEM_PROMPT,
    BaseReranker,
    build_llm_user_prompt,
    build_rerank_results,
    http_post,
    parse_llm_scores,
)

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicReranker(BaseReranker):
    """LLM-based reranker using the Anthropic Messages API.

    Supports: claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001, etc.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _ANTHROPIC_DEFAULT_MODEL,
        top_k: int = 5,
        endpoint: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.endpoint = endpoint or _ANTHROPIC_MESSAGES_URL
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        results: SearchResults,
        top_k: int | None = None,
    ) -> RerankResults:
        effective_top_k = top_k if top_k is not None else self.top_k
        memories = results.memories
        if not memories:
            return RerankResults(model=self.model, provider="anthropic")

        documents = [memory.content for memory in memories]
        user_prompt = build_llm_user_prompt(query, documents)

        start = time.monotonic()
        response = http_post(
            url=self.endpoint,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            body={
                "model": self.model,
                "max_tokens": 1024,
                "system": LLM_SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        content_blocks = response.get("content", [])
        response_text = "[]"
        for block in content_blocks:
            if block.get("type") == "text":
                response_text = block.get("text", "[]")
                break

        scored_indices = parse_llm_scores(
            response_text, len(documents), effective_top_k
        )

        return build_rerank_results(
            memories, scored_indices, self.model, "anthropic", duration_ms
        )
