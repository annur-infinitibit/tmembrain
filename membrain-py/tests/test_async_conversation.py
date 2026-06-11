"""Unit tests for AsyncConversation."""

from __future__ import annotations

import asyncio
import functools
import logging
import uuid

import pytest

import membrain.conversation as conversation_module
from membrain import MembrainClient, MembrainError
from membrain.conversation import AsyncConversation


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _test_config(tmp_path) -> dict:
    """Config with novelty gating disabled so tests are deterministic."""
    return {
        "storage_path": str(tmp_path),
        "write": {
            "novelty": {"enabled": False},
            "salience": {"enabled": False},
        },
    }


# -----------------------------------------------------------------------
# Mock LLMs — no real API calls
# -----------------------------------------------------------------------


def _is_extraction_call(messages: list[dict]) -> bool:
    """Detect if the LLM is being called for memory extraction."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    return "extract" in system.lower() and "memorable" in system.lower()


async def async_mock_llm(messages: list[dict]) -> str:
    """Async LLM mock that returns a fixed response."""
    if _is_extraction_call(messages):
        return "[]"
    return "Mock async response."


def sync_mock_llm(messages: list[dict]) -> str:
    """Sync LLM mock that returns a fixed response."""
    if _is_extraction_call(messages):
        return "[]"
    return "Mock sync response."


def sync_returning_awaitable(messages: list[dict]):
    """Sync callable that returns an awaitable instead of a final string."""
    return async_mock_llm(messages)


def bad_sync_llm(messages: list[dict]):
    """Sync LLM mock that violates the expected return type."""
    return {"bad": "type"}


class AsyncCallableLLM:
    """Callable object with an async __call__ method."""

    async def __call__(self, messages: list[dict]) -> str:
        if _is_extraction_call(messages):
            return "[]"
        return "Mock callable response."


# -----------------------------------------------------------------------
# Basic reply tests
# -----------------------------------------------------------------------


async def test_reply_with_async_llm(tmp_path):
    """Async LLM is awaited directly."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=async_mock_llm, client=client
        ) as conv:
            response = await conv.reply("hello")
            assert response == "Mock async response."


async def test_reply_with_sync_llm(tmp_path):
    """Sync LLM runs in executor, does not block loop."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=sync_mock_llm, client=client
        ) as conv:
            response = await conv.reply("hello")
            assert response == "Mock sync response."


async def test_reply_with_partial_llm(tmp_path):
    """functools.partial(async_fn, ...) detected correctly as async."""
    partial_llm = functools.partial(async_mock_llm)
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=partial_llm, client=client
        ) as conv:
            response = await conv.reply("hello")
            assert response == "Mock async response."


async def test_reply_with_callable_object(tmp_path):
    """Object with async __call__ detected correctly."""
    callable_llm = AsyncCallableLLM()
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=callable_llm, client=client
        ) as conv:
            response = await conv.reply("hello")
            assert response == "Mock callable response."


async def test_reply_with_sync_callable_returning_awaitable(tmp_path):
    """Sync callables that return awaitables are fully awaited."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=sync_returning_awaitable,
            client=client,
            auto_extract=False,
        ) as conv:
            response = await conv.reply("hello")
            assert response == "Mock async response."


async def test_reply_with_non_string_llm_result_raises(tmp_path):
    """LLM outputs must resolve to strings before they enter conversation state."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=bad_sync_llm, client=client, auto_extract=False
        ) as conv:
            with pytest.raises(TypeError, match="return a string"):
                await conv.reply("hello")


# -----------------------------------------------------------------------
# History and session
# -----------------------------------------------------------------------


async def test_history_grows_per_turn(tmp_path):
    """3 turns -> 6 history entries."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=async_mock_llm, client=client, auto_extract=False
        ) as conv:
            for _ in range(3):
                await conv.reply("turn")
            assert len(conv.history) == 6


async def test_session_id_stable(tmp_path):
    """session_id unchanged across turns."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=async_mock_llm, client=client, auto_extract=False
        ) as conv:
            sid = conv.session_id
            await conv.reply("first")
            await conv.reply("second")
            assert conv.session_id == sid


async def test_history_limit_zero_injects_no_prior_turns(tmp_path):
    """history_limit=0 means only the current user turn is sent."""
    captured: list[list[dict[str, str]]] = []

    async def capture_llm(messages: list[dict[str, str]]) -> str:
        captured.append(messages)
        return "Captured response."

    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=capture_llm,
            client=client,
            auto_extract=False,
            history_limit=0,
            memory_limit=0,
        ) as conv:
            await conv.reply("first")
            await conv.reply("second")

    assert [m["role"] for m in captured[-1]] == ["system", "user"]
    assert captured[-1][-1]["content"] == "second"


# -----------------------------------------------------------------------
# end() and case storage
# -----------------------------------------------------------------------


async def test_end_stores_case(tmp_path):
    """await conv.end() does not raise."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=async_mock_llm, client=client, auto_extract=False
        ) as conv:
            await conv.reply("hello")
            await conv.end(outcome="helpful", reward=1.0)


# -----------------------------------------------------------------------
# auto_extract
# -----------------------------------------------------------------------


async def test_auto_extract_false(tmp_path):
    """No extra LLM calls when auto_extract=False."""
    call_count = 0
    original = async_mock_llm

    async def counting_llm(messages):
        nonlocal call_count
        call_count += 1
        return await original(messages)

    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=counting_llm, client=client, auto_extract=False
        ) as conv:
            await conv.reply("hello")
            # With auto_extract=False, only one LLM call per reply (no extraction)
            assert call_count == 1


async def test_store_event_failure_is_logged_and_reply_still_returns(
    tmp_path, caplog
):
    """Post-response persistence is best-effort once the LLM succeeded."""
    with MembrainClient(_test_config(tmp_path)) as client:
        original = client.store_event

        async def fail_store_event(*args, **kwargs):
            raise RuntimeError("store_event failed")

        client.store_event = fail_store_event
        try:
            conv = AsyncConversation(
                llm_callable=async_mock_llm, client=client, auto_extract=False
            )
            with caplog.at_level(
                logging.WARNING, logger=conversation_module.logger.name
            ):
                response = await conv.reply("hello")
        finally:
            client.store_event = original

    assert response == "Mock async response."
    assert "Failed to store conversation event" in caplog.text


# -----------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------


async def test_context_manager(tmp_path):
    """async with closes client on exit."""
    with MembrainClient(_test_config(tmp_path)) as client:
        async with AsyncConversation(
            llm_callable=async_mock_llm, client=client
        ) as conv:
            await conv.reply("hello")
        # conv is now closed, client should still be open (we created it externally)
        result = await client.store_fact(f"still works {_uid()}", confidence=0.9)
        assert result.success


async def test_external_client_not_closed(tmp_path):
    """Client passed in is not closed by conv.close()."""
    with MembrainClient(_test_config(tmp_path)) as client:
        conv = AsyncConversation(
            llm_callable=async_mock_llm, client=client, auto_extract=False
        )
        await conv.reply("hello")
        await conv.close()
        # Client should still be usable
        result = await client.store_fact(f"after conv close {_uid()}", confidence=0.9)
        assert result.success


async def test_close_waits_for_in_flight_reply_on_owned_client(
    tmp_path, monkeypatch
):
    """close() waits for the active turn before closing an owned client."""
    client = MembrainClient(_test_config(tmp_path))
    monkeypatch.setattr(
        conversation_module, "MembrainClient", lambda: client
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_llm(messages: list[dict[str, str]]) -> str:
        started.set()
        await release.wait()
        return "Slow response."

    conv = AsyncConversation(llm_callable=slow_llm, auto_extract=False)
    reply_task = asyncio.create_task(conv.reply("hello"))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    close_task = asyncio.create_task(conv.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    release.set()
    assert await reply_task == "Slow response."
    await close_task

    with pytest.raises(MembrainError, match="conversation is already closed"):
        await conv.reply("after close")


async def test_reply_and_end_raise_after_close(tmp_path):
    """A closed conversation rejects future operations immediately."""
    with MembrainClient(_test_config(tmp_path)) as client:
        conv = AsyncConversation(
            llm_callable=async_mock_llm, client=client, auto_extract=False
        )
        await conv.close()

        with pytest.raises(MembrainError, match="conversation is already closed"):
            await conv.reply("hello")

        with pytest.raises(MembrainError, match="conversation is already closed"):
            await conv.end(outcome="done")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"memory_limit": -1}, "memory_limit"),
        ({"history_limit": -1}, "history_limit"),
    ],
)
def test_negative_limits_raise_before_internal_client_creation(
    monkeypatch, kwargs, match
):
    """Validation should fail before constructing an internal client."""
    constructed = False

    def fail_if_called(*args, **inner_kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("MembrainClient should not be constructed")

    monkeypatch.setattr(
        conversation_module, "MembrainClient", fail_if_called
    )

    with pytest.raises(ValueError, match=match):
        AsyncConversation(llm_callable=async_mock_llm, **kwargs)

    assert constructed is False
