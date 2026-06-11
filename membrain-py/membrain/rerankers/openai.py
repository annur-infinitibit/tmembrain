"""OpenAI LLM-based reranker."""

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

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIReranker(BaseReranker):
    """LLM-based reranker using the OpenAI Chat Completions API.

    Supports any chat model: gpt-4o-mini, gpt-4o, etc.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _OPENAI_DEFAULT_MODEL,
        top_k: int = 5,
        endpoint: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.endpoint = endpoint or _OPENAI_CHAT_URL
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
            return RerankResults(model=self.model, provider="openai")

        documents = [memory.content for memory in memories]
        user_prompt = build_llm_user_prompt(query, documents)

        start = time.monotonic()
        response = http_post(
            url=self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            body={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        response_text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "[]")
        )
        scored_indices = parse_llm_scores(
            response_text, len(documents), effective_top_k
        )

        return build_rerank_results(
            memories, scored_indices, self.model, "openai", duration_ms
        )
