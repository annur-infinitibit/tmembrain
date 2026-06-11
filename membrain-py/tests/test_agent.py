"""Tests for the Membrain agent framework.

The planner, executor, and top-level agent are exercised against the real
MembrainClient (no FFI mocks). The LLM layer is a user-supplied callable, so
tests use deterministic in-process callables for core logic and reserve the
live OpenAI path behind `OPENAI_API_KEY` for a single smoke test.
"""

from __future__ import annotations

import json
import os

import pytest

from membrain import MembrainClient
from membrain.agent import (
    MembrainAgent,
    MembrainExecutor,
    MembrainPlanner,
    TaskPlan,
    TaskStep,
)
from membrain.cbr import CasePromptBuilder, NonParametricRetriever


@pytest.fixture
def client(tmp_path):
    config = {"storage": {"backend": "memscaledb", "path": str(tmp_path / "db")}}
    instance = MembrainClient(config=config)
    try:
        yield instance
    finally:
        instance.close()


def _plan_json(steps: list[str]) -> str:
    payload = {
        "plan": [
            {"id": index + 1, "description": text}
            for index, text in enumerate(steps)
        ]
    }
    return json.dumps(payload)


def test_planner_returns_structured_plan(client):
    recorded: list[list[dict[str, str]]] = []

    def llm(messages: list[dict[str, str]]) -> str:
        recorded.append(messages)
        return _plan_json(["Gather requirements", "Write code", "Deploy"])

    planner = MembrainPlanner(
        retriever=NonParametricRetriever(client),
        prompt_builder=CasePromptBuilder(),
        llm_callable=llm,
    )

    plan = planner.plan("Ship a new microservice")

    assert isinstance(plan, TaskPlan)
    assert [step.description for step in plan.steps] == [
        "Gather requirements",
        "Write code",
        "Deploy",
    ]
    assert plan.steps[0].id == 1
    assert recorded[0][0]["role"] == "system"
    assert "task planner" in recorded[0][0]["content"].lower()


def test_planner_strips_code_fences():
    def llm(_: list[dict[str, str]]) -> str:
        return "```json\n" + _plan_json(["Step one"]) + "\n```"

    planner = MembrainPlanner(
        retriever=_EmptyRetriever(),
        prompt_builder=CasePromptBuilder(),
        llm_callable=llm,
    )
    plan = planner.plan("Query")
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Step one"


def test_planner_fallback_on_bad_json():
    def llm(_: list[dict[str, str]]) -> str:
        return "not-valid-json"

    planner = MembrainPlanner(
        retriever=_EmptyRetriever(),
        prompt_builder=CasePromptBuilder(),
        llm_callable=llm,
    )
    plan = planner.plan("Query")
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "not-valid-json"


def test_executor_dispatches_registered_tool():
    def llm(_: list[dict[str, str]]) -> str:
        return "TOOL_CALL: echo('hello world')"

    executor = MembrainExecutor(llm_callable=llm)
    executor.register_tool("echo", lambda value: f"echoed: {value}")
    result = executor.execute(TaskStep(id=1, description="Echo 'hello world'"))

    assert result.success is True
    assert result.output == "echoed: hello world"
    assert "echo" in executor.available_tools


def test_executor_passes_through_plain_response():
    def llm(_: list[dict[str, str]]) -> str:
        return "direct answer"

    executor = MembrainExecutor(llm_callable=llm)
    result = executor.execute(TaskStep(id=1, description="Task"))
    assert result.output == "direct answer"
    assert result.success is True


def test_executor_captures_failures():
    def llm(_: list[dict[str, str]]) -> str:
        raise RuntimeError("LLM upstream failure")

    executor = MembrainExecutor(llm_callable=llm)
    result = executor.execute(TaskStep(id=1, description="Task"))
    assert result.success is False
    assert result.error is not None
    assert "LLM upstream failure" in result.error


def test_agent_run_end_to_end(client):
    call_count = {"value": 0}

    def llm(messages: list[dict[str, str]]) -> str:
        call_count["value"] += 1
        system = messages[0]["content"] if messages else ""
        if "task planner" in system.lower():
            return _plan_json(["Do the task"])
        return "completed"

    agent = MembrainAgent(client=client, llm_callable=llm)
    output = agent.run("Diagnose the production outage")

    assert output == "completed"
    assert call_count["value"] >= 2


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_agent_run_with_live_openai(client):
    from openai import AuthenticationError, OpenAI

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def llm(messages: list[dict[str, str]]) -> str:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    agent = MembrainAgent(
        client=client, llm_callable=llm, max_cycles=1
    )
    try:
        output = agent.run(
            "Summarise the three steps to deploy a python web service"
        )
    except AuthenticationError:
        pytest.skip("OpenAI credentials rejected (401)")
    assert output
    assert len(output) > 10


class _EmptyRetriever(NonParametricRetriever):
    def __init__(self) -> None:
        pass

    def retrieve(self, query, limit=5, positive_reward_threshold=0.5):
        from membrain.types import CaseSearchResults

        return CaseSearchResults(
            positive_cases=[], negative_cases=[], duration_ms=0.0
        )
