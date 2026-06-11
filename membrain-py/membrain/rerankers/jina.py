"""Jina cross encoder reranker."""

from __future__ import annotations

import time

from ..types import RerankResults, SearchResults
from .base import (
    DEFAULT_TIMEOUT,
    BaseReranker,
    build_rerank_results,
    http_post,
)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"


class JinaReranker(BaseReranker):
    """Cross encoder reranker using the Jina Rerank API.

    Supported models: jina-reranker-v2-base-multilingual, jina-colbert-v2.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _JINA_DEFAULT_MODEL,
        top_k: int = 5,
        endpoint: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.endpoint = endpoint or _JINA_RERANK_URL
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
            return RerankResults(model=self.model, provider="jina")

        documents = [memory.content for memory in memories]

        start = time.monotonic()
        response = http_post(
            url=self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            body={
                "query": query,
                "documents": documents,
                "model": self.model,
                "top_n": effective_top_k,
            },
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        scored_indices: list[tuple[int, float]] = []
        for item in response.get("results", []):
            scored_indices.append(
                (item["index"], item["relevance_score"])
            )
        scored_indices.sort(key=lambda pair: pair[1], reverse=True)

        return build_rerank_results(
            memories, scored_indices, self.model, "jina", duration_ms
        )
