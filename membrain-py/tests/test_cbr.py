"""Tests for the case-based reasoning helpers.

Covers `NonParametricRetriever`, `CasePromptBuilder`, `TrainingDataCollector`,
and `ExperienceReplay` against the real MembrainClient. Cross-language JSONL
parity is checked via a shared fixture file at
``tests/fixtures/cbr/training_data.jsonl`` that is also read by the Node
counterpart in `membrain-node/tests/test_cbr.mjs`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membrain import MembrainClient
from membrain.cbr import (
    CasePromptBuilder,
    NonParametricRetriever,
    TrainingDataCollector,
    TrainingPair,
)
from membrain.cbr.experience_replay import ExperienceReplay
from membrain.types import CaseEntry, CaseSearchResults


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "cbr"
)
GOLDEN_PROMPT = FIXTURE_DIR / "prompt_golden.txt"
SHARED_JSONL = FIXTURE_DIR / "training_data.jsonl"


@pytest.fixture
def client(tmp_path):
    config = {"storage": {"backend": "memscaledb", "path": str(tmp_path / "db")}}
    instance = MembrainClient(config=config)
    try:
        yield instance
    finally:
        instance.close()


def test_non_parametric_retriever_round_trip(client):
    client.store_case(
        problem="Flaky integration test",
        plan="Increase timeout and add retries",
        outcome="Tests pass on first run",
        reward=1.0,
    )
    client.store_case(
        problem="Database deadlock",
        plan="Reorder locks alphabetically",
        outcome="Still deadlocks intermittently",
        reward=0.0,
    )

    retriever = NonParametricRetriever(client)
    result = retriever.retrieve("flaky test", limit=5)

    assert isinstance(result, CaseSearchResults)
    assert len(result.positive_cases) + len(result.negative_cases) >= 1


def test_prompt_builder_matches_golden():
    cases = CaseSearchResults(
        positive_cases=[
            CaseEntry(
                id="c1",
                problem="Deploy the payment service",
                plan=json.dumps(
                    {
                        "plan": [
                            {"id": 1, "description": "Build image"},
                            {"id": 2, "description": "Push to registry"},
                        ]
                    }
                ),
                outcome="Service went live without downtime",
                reward=1.0,
                score=0.92,
            )
        ],
        negative_cases=[
            CaseEntry(
                id="c2",
                problem="Deploy the payment service",
                plan="Restart production database first",
                outcome="Caused a 20-minute outage",
                reward=0.0,
                score=0.41,
            )
        ],
        duration_ms=5,
    )

    builder = CasePromptBuilder()
    context = builder.build_context(cases)

    GOLDEN_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    if not GOLDEN_PROMPT.exists():
        GOLDEN_PROMPT.write_text(context, encoding="utf-8")

    golden = GOLDEN_PROMPT.read_text(encoding="utf-8")
    assert context == golden


def test_training_data_collector_flush_and_load(tmp_path):
    collector = TrainingDataCollector()
    cases = [
        CaseEntry(
            id="a",
            problem="Flaky test",
            plan="Add retry",
            outcome="Green",
            reward=1.0,
            score=0.9,
        ),
        CaseEntry(
            id="b",
            problem="Deadlock",
            plan="Reorder locks",
            outcome="Still flaky",
            reward=0.0,
            score=0.2,
        ),
    ]
    collector.record(query="fix flaky", retrieved_cases=cases, is_correct=True)

    path = tmp_path / "training.jsonl"
    count = collector.flush(path)
    assert count == 2
    assert collector.size == 0

    pairs = TrainingDataCollector.load(path)
    assert len(pairs) == 2
    assert pairs[0].query == "fix flaky"
    assert pairs[0].case_label == "positive"
    assert pairs[1].case_label == "negative"
    assert pairs[0].truth_label is True


def test_training_data_cross_language_fixture(tmp_path):
    """Write a shared fixture that the Node test reads back."""
    collector = TrainingDataCollector()
    cases = [
        CaseEntry(
            id="fix-1",
            problem="Memory leak in service",
            plan="Use weak references",
            outcome="Stable memory usage",
            reward=1.0,
            score=0.88,
        ),
        CaseEntry(
            id="fix-2",
            problem="Memory leak in service",
            plan="Restart every hour",
            outcome="Masked the issue, caused user-visible hiccups",
            reward=0.0,
            score=0.31,
        ),
    ]
    collector.record(
        query="memory leak", retrieved_cases=cases, is_correct=True
    )

    SHARED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if SHARED_JSONL.exists():
        SHARED_JSONL.unlink()
    count = collector.flush(SHARED_JSONL)
    assert count == 2

    pairs = TrainingDataCollector.load(SHARED_JSONL)
    assert [pair.case_label for pair in pairs] == ["positive", "negative"]
    assert pairs[0].plan == "Use weak references"


def test_experience_replay_records_and_stores(client, tmp_path):
    replay = ExperienceReplay(
        client=client,
        training_data_path=tmp_path / "replay.jsonl",
    )
    case_id = replay.record_execution(
        problem="Fix CI cache",
        plan="Bump cache key",
        outcome="Faster CI",
        reward=1.0,
        query="ci cache",
        retrieved_cases=[],
        is_correct=True,
    )
    assert case_id is not None
    assert replay.execution_count == 1
    written = replay.flush_training_data()
    assert written == 0  # no retrieved cases means no training pairs
    assert replay.execution_count == 0


def test_training_pair_is_frozen():
    pair = TrainingPair(
        query="q",
        case_text="[CASE]",
        case_label="positive",
        plan="p",
        truth_label=True,
    )
    with pytest.raises(Exception):
        pair.query = "mutated"  # type: ignore[misc]
