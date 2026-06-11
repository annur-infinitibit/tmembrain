"""LLM-agnostic task executor.

Executes individual steps from a task plan, optionally using registered
tool functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .planner import LLMCallable, TaskStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskResult:
    """Result of executing a single task step."""

    task: TaskStep
    output: str
    success: bool
    error: str | None = None


class MembrainExecutor:
    """Executes individual tasks from a plan.

    Supports optional tool functions that the LLM can request.

    Usage::

        executor = MembrainExecutor(llm_callable=my_llm_function)
        executor.register_tool("search_web", search_web_fn)

        result = executor.execute(step)
        print(result.output)
    """

    def __init__(self, llm_callable: LLMCallable) -> None:
        """Initialize the executor.

        Args:
            llm_callable: A function that takes a list of message dicts
                and returns the LLM response string.
        """
        self._llm_callable = llm_callable
        self._tools: dict[str, Callable[..., str]] = {}

    def register_tool(
        self, name: str, function: Callable[..., str]
    ) -> None:
        """Register a tool function the executor can use.

        Args:
            name: The tool name referenced by the LLM.
            function: A callable that takes keyword arguments and returns
                a string result.
        """
        self._tools[name] = function

    @property
    def available_tools(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    async def execute(self, task: TaskStep) -> TaskResult:
        """Execute a single task step.

        Args:
            task: The task step to execute.

        Returns:
            TaskResult with the output, success status, and any error.
        """
        tool_descriptions = ""
        if self._tools:
            tool_list = ", ".join(self._tools.keys())
            tool_descriptions = (
                f"\n\nAvailable tools: {tool_list}\n"
                "To use a tool, respond with: TOOL_CALL: tool_name(arg1, arg2)\n"
                "Otherwise, respond with the task output directly."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task executor. Complete the given task and "
                    "return the result." + tool_descriptions
                ),
            },
            {"role": "user", "content": task.description},
        ]

        try:
            response = await self._call_llm(messages)
            output = self._handle_tool_calls(response)
            return TaskResult(
                task=task,
                output=output,
                success=True,
            )
        except Exception as error:
            logger.error("Task %d failed: %s", task.id, error)
            return TaskResult(
                task=task,
                output="",
                success=False,
                error=str(error),
            )

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

    def _handle_tool_calls(self, response: str) -> str:
        """Parse and execute tool calls from the LLM response."""
        if not response.strip().startswith("TOOL_CALL:"):
            return response

        tool_line = response.strip().removeprefix("TOOL_CALL:").strip()
        paren_index = tool_line.find("(")
        if paren_index == -1:
            return response

        tool_name = tool_line[:paren_index].strip()
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}. Raw response: {response}"

        args_str = tool_line[paren_index + 1 :].rstrip(")")
        args = [
            arg.strip().strip("'\"")
            for arg in args_str.split(",")
            if arg.strip()
        ]

        tool_function = self._tools[tool_name]
        result = tool_function(*args)
        return result
