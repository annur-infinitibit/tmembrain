"""Case retrieval strategies for case-based reasoning.

Provides an abstract CaseRetriever base class and a NonParametricRetriever
that uses embedding similarity via MembrainClient.search_cases().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..client import MembrainClient
from ..types import CaseSearchResults


class CaseRetriever(ABC):
    """Abstract base class for case retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        positive_reward_threshold: float = 0.5,
    ) -> CaseSearchResults:
        """Retrieve similar cases for a given query.

        Args:
            query: The problem description to find similar cases for.
            limit: Maximum number of cases to return.
            positive_reward_threshold: Reward threshold separating
                positive from negative cases.

        Returns:
            CaseSearchResults with positive and negative case lists.
        """


class NonParametricRetriever(CaseRetriever):
    """Embedding-similarity retriever using MembrainClient.search_cases().

    No training required -- works immediately with stored cases by using
    cosine similarity on embeddings.
    """

    def __init__(self, client: MembrainClient) -> None:
        self._client = client

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        positive_reward_threshold: float = 0.5,
    ) -> CaseSearchResults:
        """Retrieve cases ranked by embedding similarity."""
        return await self._client.search_cases(
            query,
            limit=limit,
            positive_reward_threshold=positive_reward_threshold,
        )
