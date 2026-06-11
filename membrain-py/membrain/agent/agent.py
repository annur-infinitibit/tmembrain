"""Top-level Membrain agent combining planning, execution, and learning.

MembrainAgent ties together the planner, executor, and experience replay
loop into a single ``run()`` method.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..cbr.experience_replay import ExperienceReplay
from ..cbr.prompt_builder import CasePromptBuilder
from ..cbr.retriever import CaseRetriever
from ..client import MembrainClient
from .executor import MembrainExecutor, TaskResult
from .planner import LLMCallable, MembrainPlanner

logger = logging.getLogger(__name__)

JudgeCallable = Callable[[str, str, list[TaskResult]], bool]


class MembrainAgent:
    """Full plan-execute-learn agent.

    Usage::

        from membrain import MembrainClient
        from membrain.cbr import NonParametricRetriever, CasePromptBuilder
        from membrain.agent import MembrainAgent

        client = MembrainClient()
        agent = MembrainAgent(
            client=client,
            llm_callable=my_llm_function,
        )
        answer = agent.run("Deploy the new microservice")
    """

    def __init__(
        self,
        client: MembrainClient,
        llm_callable: LLMCallable,
        retriever: CaseRetriever | None = None,
        prompt_builder: CasePromptBuilder | None = None,
        training_data_path: str = "training_data.jsonl",
        case_limit: int = 5,
        max_cycles: int = 3,
    ) -> None:
        """Initialize the agent.

        Args:
            client: MembrainClient for memory storage and retrieval.
            llm_callable: Function that takes messages and returns a response.
            retriever: Optional case retriever (defaults to NonParametricRetriever).
            prompt_builder: Optional prompt builder (defaults to CasePromptBuilder).
            training_data_path: Path for accumulating training data.
            case_limit: Maximum cases to retrieve per planning step.
            max_cycles: Maximum plan-execute cycles before giving up.
        """
        from ..cbr.retriever import NonParametricRetriever

        self._client = client
        self._llm_callable = llm_callable
        self._max_cycles = max_cycles

        actual_retriever = retriever or NonParametricRetriever(client)
        actual_builder = prompt_builder or CasePromptBuilder()

        self._planner = MembrainPlanner(
            retriever=actual_retriever,
            prompt_builder=actual_builder,
            llm_callable=llm_callable,
            case_limit=case_limit,
        )
        self._executor = MembrainExecutor(llm_callable=llm_callable)
        self._replay = ExperienceReplay(
            client=client,
            retriever=actual_retriever,
            training_data_path=training_data_path,
        )

    @property
    def executor(self) -> MembrainExecutor:
        """Access the executor to register tools."""
        return self._executor

    @property
    def replay(self) -> ExperienceReplay:
        """Access the experience replay for manual control."""
        return self._replay

    async def run(self, query: str) -> str:
        """Run the full plan-execute loop for a query.

        Args:
            query: The user's problem or request.

        Returns:
            The final combined output from all executed steps.
        """
        for cycle in range(self._max_cycles):
            logger.info("Cycle %d/%d for query: %s", cycle + 1, self._max_cycles, query)

            plan = await self._planner.plan(query)
            results: list[TaskResult] = []

            all_succeeded = True
            for step in plan.steps:
                result = await self._executor.execute(step)
                results.append(result)
                if not result.success:
                    all_succeeded = False
                    break

            output = "\n".join(
                result.output for result in results if result.output
            )

            if all_succeeded:
                return output

            logger.warning(
                "Cycle %d failed, %d/%d steps succeeded",
                cycle + 1,
                sum(1 for r in results if r.success),
                len(plan.steps),
            )

        return output

    async def run_batch(
        self,
        queries: list[str],
        judge_callable: JudgeCallable,
    ) -> list[str]:
        """Run multiple queries with automatic training data collection.

        Args:
            queries: List of queries to process.
            judge_callable: A function that takes (query, output, results)
                and returns True if the answer is correct.

        Returns:
            List of outputs, one per query.
        """
        retrieved_cases_cache = self._replay.retriever

        outputs: list[str] = []
        for query in queries:
            cases_result = await retrieved_cases_cache.retrieve(query)
            all_cases = (
                cases_result.positive_cases + cases_result.negative_cases
            )

            output = await self.run(query)
            outputs.append(output)

            is_correct = judge_callable(query, output, [])

            reward = 1.0 if is_correct else 0.0
            await self._replay.record_execution(
                problem=query,
                plan=output,
                outcome="correct" if is_correct else "incorrect",
                reward=reward,
                query=query,
                retrieved_cases=all_cases,
                is_correct=is_correct,
            )

        return outputs
