"""Smoke tests for AsyncConversation.

These exercise the full async loop (retrieve -> LLM -> store -> extract)
against the real FFI layer, using a fake async LLM callable. The LLM is a
deterministic stub so the tests are offline.
"""

from __future__ import annotations

import pytest

from membrain import AsyncConversation, MembrainClient


class FakeAsyncLLM:
    """Deterministic async LLM for testing.

    - First call returns a canned assistant reply.
    - Second call (the extraction prompt) returns a JSON array that
      Membrain's extractor understands, so we can assert a memory was
      stored.
    """

    def __init__(self, reply: str, extraction_json: str) -> None:
        self.reply = reply
        self.extraction_json = extraction_json
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        system = messages[0]["content"] if messages else ""
        if "Analyze the following conversation turn" in system:
            return self.extraction_json
        return self.reply


@pytest.mark.asyncio
async def test_async_reply_roundtrip(tmp_path):
    client = MembrainClient(config={"storage": {"path": str(tmp_path / "db.redb")}})
    try:
        llm = FakeAsyncLLM(
            reply="I have noted your preference for dark mode.",
            extraction_json=(
                '[{"type": "preference", "holder": "user", "subject": "theme",'
                ' "preference": "dark mode", "strength": "strong"}]'
            ),
        )

        conv = AsyncConversation(llm_callable=llm, client=client, auto_extract=True)
        reply = await conv.reply("I prefer dark mode.")

        assert reply == "I have noted your preference for dark mode."
        # Two LLM calls: reply + extraction.
        assert len(llm.calls) == 2
        # Extraction wrote a memory.
        results = client.search("dark mode", limit=5)
        assert any("dark" in m.content.lower() for m in results.memories)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_extraction_error_callback_fires(tmp_path):
    """When the extraction LLM raises, the callback must be invoked."""
    client = MembrainClient(config={"storage": {"path": str(tmp_path / "db.redb")}})
    try:
        captured: list[BaseException] = []

        async def llm(messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"] if messages else ""
            if "Analyze the following conversation turn" in system:
                raise RuntimeError("extraction upstream failed")
            return "ok"

        conv = AsyncConversation(
            llm_callable=llm,
            client=client,
            auto_extract=True,
            on_extraction_error=captured.append,
        )
        result = await conv.reply("hello")

        assert result == "ok"
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "extraction upstream failed" in str(captured[0])
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_end_stores_case(tmp_path):
    client = MembrainClient(config={"storage": {"path": str(tmp_path / "db.redb")}})
    try:
        async def llm(messages):
            return "assistant reply"

        conv = AsyncConversation(
            llm_callable=llm, client=client, auto_extract=False
        )
        await conv.reply("hi there")
        await conv.end(outcome="positive conclusion", reward=1.0)

        cases = client.search_cases("hi there", limit=5)
        assert cases.positive_cases, "expected at least one positive case"
    finally:
        client.close()
