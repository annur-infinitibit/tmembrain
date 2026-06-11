"""Training data collection and management for the neural retriever.

Accumulates (query, case_text, case_label, plan, truth_label) tuples during
agent execution and serialises them to JSONL for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..types import CaseEntry


@dataclass(frozen=True)
class TrainingPair:
    """A single training example for the relevance classifier.

    Attributes:
        query: The user's problem / question.
        case_text: The retrieved case formatted as "[CASE]...".
        case_label: "positive" or "negative" (quality of the case itself).
        plan: The plan from the case.
        truth_label: Whether using this case led to a correct answer.
    """

    query: str
    case_text: str
    case_label: str
    plan: str
    truth_label: bool


class TrainingDataCollector:
    """Accumulates training pairs during agent execution and writes JSONL."""

    def __init__(self) -> None:
        self._buffer: list[TrainingPair] = []

    @property
    def size(self) -> int:
        """Number of accumulated training pairs."""
        return len(self._buffer)

    def record(
        self,
        query: str,
        retrieved_cases: list[CaseEntry],
        is_correct: bool,
        positive_reward_threshold: float = 0.5,
    ) -> None:
        """Record training pairs from a single execution step.

        For each retrieved case, creates a TrainingPair recording whether
        the case contributed to a correct answer.

        Args:
            query: The problem that was being solved.
            retrieved_cases: Cases that were retrieved and shown to the LLM.
            is_correct: Whether the final answer was judged correct.
            positive_reward_threshold: Reward threshold to label cases.
        """
        for case in retrieved_cases:
            case_label = (
                "positive"
                if case.reward >= positive_reward_threshold
                else "negative"
            )
            case_text = (
                f"[CASE]\n"
                f"Problem: {case.problem}\n"
                f"Outcome: {case.outcome}\n"
                f"Reward: {case.reward}"
            )
            pair = TrainingPair(
                query=query,
                case_text=case_text,
                case_label=case_label,
                plan=case.plan,
                truth_label=is_correct,
            )
            self._buffer.append(pair)

    def flush(self, path: str | Path) -> int:
        """Write accumulated pairs to a JSONL file and clear the buffer.

        Appends to the file if it already exists.

        Args:
            path: Destination JSONL file path.

        Returns:
            Number of pairs written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self._buffer)
        with path.open("a", encoding="utf-8") as file_handle:
            for pair in self._buffer:
                record = {
                    "query": pair.query,
                    "case": pair.case_text,
                    "case_label": pair.case_label,
                    "plan": pair.plan,
                    "truth_label": pair.truth_label,
                }
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()
        return count

    @staticmethod
    def load(path: str | Path) -> list[TrainingPair]:
        """Load training pairs from a JSONL file.

        Args:
            path: Source JSONL file path.

        Returns:
            List of TrainingPair instances.
        """
        pairs: list[TrainingPair] = []
        with Path(path).open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pairs.append(
                    TrainingPair(
                        query=record["query"],
                        case_text=record["case"],
                        case_label=record["case_label"],
                        plan=record["plan"],
                        truth_label=record["truth_label"],
                    )
                )
        return pairs
