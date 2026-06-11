"""Base reranker class and shared utilities."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from ..errors import MembrainError
from ..types import MemoryEntry, RerankResult, RerankResults, SearchResults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RerankerError(MembrainError):
    """Error from a reranking operation."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: SearchResults,
        top_k: int = 5,
    ) -> RerankResults:
        """Rerank search results by semantic relevance to the query.

        Args:
            query: The original search query.
            results: SearchResults from membrain.search().
            top_k: Number of top results to keep after reranking.

        Returns:
            RerankResults ordered by descending relevance_score.
        """
        ...


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def http_post(
    url: str,
    headers: dict[str, str],
    body: dict,
    timeout: int,
) -> dict:
    """Send an HTTP POST request and return the parsed JSON response.

    Retries up to MAX_RETRIES times for rate limit (429) and server (5xx)
    errors with exponential backoff.
    """
    data = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            status = error.code
            if status == 429 or status >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Reranker API returned %d, retrying in %ds (attempt %d/%d)",
                    status,
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            # Non-retryable HTTP error
            response_body = error.read().decode("utf-8", errors="replace")
            raise RerankerError(
                f"Reranker API returned HTTP {status}: {response_body}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            wait = 2 ** attempt
            logger.warning(
                "Reranker API request failed: %s, retrying in %ds (attempt %d/%d)",
                error,
                wait,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(wait)

    raise RerankerError(
        f"Reranker API failed after {MAX_RETRIES} retries: {last_error}"
    ) from last_error


def build_rerank_results(
    memories: list[MemoryEntry],
    scored_indices: list[tuple[int, float]],
    model: str,
    provider: str,
    duration_ms: int,
) -> RerankResults:
    """Build RerankResults from scored index/score pairs."""
    reranked: list[RerankResult] = []
    for index, relevance_score in scored_indices:
        if 0 <= index < len(memories):
            memory = memories[index]
            reranked.append(
                RerankResult(
                    id=memory.id,
                    content=memory.content,
                    score=memory.score,
                    relevance_score=relevance_score,
                    memory_type=memory.memory_type,
                )
            )
    return RerankResults(
        memories=reranked,
        model=model,
        provider=provider,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# LLM Reranker: prompt construction
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a relevance scoring assistant. Given a query and a list of "
    "documents, score each document's relevance to the query on a scale "
    "of 0 to 10. Respond with ONLY a JSON array of objects: "
    '[{"index": 0, "score": 7}, ...]'
)


def build_llm_user_prompt(query: str, documents: list[str]) -> str:
    """Build the user prompt for LLM-based reranking."""
    parts = [f"Query: {query}\n\nDocuments:"]
    for index, document in enumerate(documents):
        parts.append(f"[{index}] {document}")
    parts.append("\nScore each document's relevance to the query (0-10).")
    return "\n".join(parts)


def parse_llm_scores(
    response_text: str,
    document_count: int,
    top_k: int,
) -> list[tuple[int, float]]:
    """Parse LLM response into scored index pairs, normalized to 0.0-1.0."""
    try:
        scores = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RerankerError(
            f"LLM reranker returned invalid JSON: {response_text[:200]}"
        ) from error

    if not isinstance(scores, list):
        raise RerankerError(
            f"LLM reranker returned non-array JSON: {type(scores).__name__}"
        )

    scored_indices: list[tuple[int, float]] = []
    for item in scores:
        index = item.get("index")
        score = item.get("score")
        if isinstance(index, int) and isinstance(score, (int, float)):
            if 0 <= index < document_count:
                normalized = max(0.0, min(1.0, score / 10.0))
                scored_indices.append((index, normalized))

    scored_indices.sort(key=lambda pair: pair[1], reverse=True)
    return scored_indices[:top_k]
