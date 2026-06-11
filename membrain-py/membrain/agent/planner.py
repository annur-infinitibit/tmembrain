"""LLM-agnostic task planner with case-based context injection.

Retrieves similar past cases, builds an in-context learning prompt, then
asks an LLM to decompose a query into a minimal task plan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from ..cbr.prompt_builder import CasePromptBuilder
from ..cbr.retriever import CaseRetriever

logger = logging.getLogger(__name__)

LLMCallable = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = """\
You are a task planner. Given a user query and optional context from past \
experiences, decompose the query into a minimal sequence of steps.

Return ONLY a JSON object in this format:
{"plan": [{"id": 1, "description": "..."}, {"id": 2, "description": "..."}]}

Rules:
- Each step must be a concrete, actionable task.
- Use the fewest steps possible.
- If past successful approaches are provided, learn from them.
- If approaches to avoid are provided, do not repeat those mistakes.
- Return valid JSON only, no markdown or explanation.\
"""


@dataclass(frozen=True)
class TaskStep:
    """A single step in a task plan."""

    id: int
    description: str


@dataclass(frozen=True)
class TaskPlan:
    """A decomposed task plan."""

    steps: list[TaskStep] = field(default_factory=list)
    raw_response: str = ""


class MembrainPlanner:
    """Plans tasks by retrieving cases and calling an LLM.

    Usage::

        from membrain.cbr import NonParametricRetriever, CasePromptBuilder
        from membrain.agent import MembrainPlanner

        planner = MembrainPlanner(
            retriever=retriever,
            prompt_builder=CasePromptBuilder(),
            llm_callable=my_llm_function,
        )
        plan = planner.plan("Deploy the new microservice")
        for step in plan.steps:
            print(f"{step.id}. {step.description}")
    """

    def __init__(
        self,
        retriever: CaseRetriever,
        prompt_builder: CasePromptBuilder,
        llm_callable: LLMCallable,
        case_limit: int = 5,
    ) -> None:
        """Initialize the planner.

        Args:
            retriever: A CaseRetriever for finding similar past cases.
            prompt_builder: Formats retrieved cases into prompt context.
            llm_callable: A function that takes a list of message dicts
                (each with "role" and "content") and returns the LLM
                response string.
            case_limit: Maximum number of cases to retrieve.
        """
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_callable = llm_callable
        self._case_limit = case_limit

    async def plan(self, query: str) -> TaskPlan:
        """Generate a task plan for the given query.

        Args:
            query: The user's problem or request.

        Returns:
            A TaskPlan containing the decomposed steps.
        """
        cases = await self._retriever.retrieve(query, limit=self._case_limit)
        context = self._prompt_builder.build_context(cases)

        user_content = query
        if context:
            user_content = (
                f"## Past Experience Context\n\n{context}\n\n"
                f"## Current Query\n\n{query}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_response = await self._call_llm(messages)
        return _parse_plan(raw_response)

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM, handling both sync and async callables."""
        import inspect
        import asyncio
        from functools import partial

        fn = self._llm_callable
        is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )

        if is_async:
            result = fn(messages)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, partial(fn, messages))

        if inspect.isawaitable(result):
            result = await result

        if not isinstance(result, str):
            raise TypeError(f"llm_callable returned {type(result).__name__}, expected str")

        return result


def _parse_plan(response: str) -> TaskPlan:
    """Parse an LLM response into a TaskPlan."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse plan JSON, returning raw response")
        return TaskPlan(
            steps=[TaskStep(id=1, description=cleaned)],
            raw_response=response,
        )

    steps: list[TaskStep] = []
    plan_list = data.get("plan", []) if isinstance(data, dict) else []

    for item in plan_list:
        if isinstance(item, dict):
            step_id = item.get("id", len(steps) + 1)
            description = item.get("description", str(item))
            steps.append(TaskStep(id=step_id, description=description))

    if not steps:
        steps = [TaskStep(id=1, description=cleaned)]

    return TaskPlan(steps=steps, raw_response=response)
