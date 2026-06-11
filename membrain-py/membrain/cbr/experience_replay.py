"""Experience replay loop tying storage, retrieval, and training.

ExperienceReplay is the main orchestrator that:
1. Stores new execution experiences as cases in Membrain
2. Records training data from each execution
3. Triggers retraining of the neural retriever on accumulated data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..client import MembrainClient
from ..types import CaseEntry
from .retriever import CaseRetriever, NonParametricRetriever
from .training_data import TrainingDataCollector

logger = logging.getLogger(__name__)


class ExperienceReplay:
    """Orchestrates the store-retrieve-train cycle.

    Usage::

        from membrain import MembrainClient
        from membrain.cbr import NonParametricRetriever
        from membrain.cbr.experience_replay import ExperienceReplay

        client = MembrainClient()
        retriever = NonParametricRetriever(client)
        replay = ExperienceReplay(client, retriever, training_data_path="data.jsonl")

        # After each agent execution:
        replay.record_execution(
            problem="How to fix the flaky test",
            plan="Add retry logic and increase timeout",
            outcome="Tests pass consistently now",
            reward=1.0,
            query="fix flaky test",
            retrieved_cases=cases_used,
            is_correct=True,
        )

        # Periodically retrain:
        result = replay.retrain(output_dir="./checkpoints")
    """

    def __init__(
        self,
        client: MembrainClient,
        retriever: CaseRetriever | None = None,
        training_data_path: str | Path = "training_data.jsonl",
        positive_reward_threshold: float = 0.5,
    ) -> None:
        self._client = client
        self._retriever = retriever or NonParametricRetriever(client)
        self._training_data_path = Path(training_data_path)
        self._positive_reward_threshold = positive_reward_threshold
        self._collector = TrainingDataCollector()
        self._execution_count = 0

    @property
    def retriever(self) -> CaseRetriever:
        """The current case retriever."""
        return self._retriever

    @retriever.setter
    def retriever(self, value: CaseRetriever) -> None:
        """Swap the retriever (e.g. after training a parametric one)."""
        self._retriever = value

    @property
    def execution_count(self) -> int:
        """Number of executions recorded since last flush."""
        return self._execution_count

    @property
    def pending_training_pairs(self) -> int:
        """Number of training pairs not yet flushed to disk."""
        return self._collector.size

    async def record_execution(
        self,
        problem: str,
        plan: str,
        outcome: str,
        reward: float,
        query: str,
        retrieved_cases: list[CaseEntry],
        is_correct: bool,
    ) -> str | None:
        """Store a new case and record training data.

        Args:
            problem: The problem that was addressed.
            plan: The plan that was executed.
            outcome: The observed outcome.
            reward: Reward signal (1.0 = success, 0.0 = failure).
            query: The original search query used for retrieval.
            retrieved_cases: The cases that were retrieved and shown to the LLM.
            is_correct: Whether the final answer was judged correct.

        Returns:
            The memory ID of the stored case, or None if storage failed.
        """
        result = await self._client.store_case(
            problem=problem,
            plan=plan,
            outcome=outcome,
            reward=reward,
        )

        self._collector.record(
            query=query,
            retrieved_cases=retrieved_cases,
            is_correct=is_correct,
            positive_reward_threshold=self._positive_reward_threshold,
        )

        self._execution_count += 1
        case_id = result.id if result.success else None

        if case_id:
            logger.info(
                "Stored case %s (reward=%.2f, correct=%s)",
                case_id,
                reward,
                is_correct,
            )

        return case_id

    def flush_training_data(self) -> int:
        """Write accumulated training pairs to disk.

        Returns:
            Number of pairs written.
        """
        count = self._collector.flush(self._training_data_path)
        logger.info(
            "Flushed %d training pairs to %s", count, self._training_data_path
        )
        self._execution_count = 0
        return count

    def retrain(
        self,
        output_dir: str,
        **training_kwargs: Any,
    ) -> Any:
        """Flush pending data and retrain the neural retriever.

        Args:
            output_dir: Directory to save checkpoints.
            **training_kwargs: Additional arguments passed to
                RetrieverTrainer.train().

        Returns:
            TrainingResult from the training run.
        """
        self.flush_training_data()

        if not self._training_data_path.exists():
            logger.warning(
                "No training data found at %s", self._training_data_path
            )
            return None

        from .trainer import RetrieverTrainer

        trainer = RetrieverTrainer(
            training_data_path=str(self._training_data_path),
            output_dir=output_dir,
        )
        result = trainer.train(**training_kwargs)

        logger.info(
            "Training complete: best_metric=%.4f, checkpoint=%s",
            result.best_metric,
            result.best_checkpoint_path,
        )
        return result
