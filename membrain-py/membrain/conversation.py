"""Automatic conversation management for Membrain (true async).

Provides a Conversation class that wraps MembrainClient to automatically
track sessions, extract and store memorable information from each turn,
retrieve relevant memories before generating responses, and learn from
conversation outcomes via case-based reasoning.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from functools import partial
from typing import Any, Callable

from .client import MembrainClient
from .errors import MembrainError
from .types import MemoryEntry

logger = logging.getLogger(__name__)

LLMCallable = Callable[[list[dict[str, str]]], str]

EXTRACTION_PROMPT = """\
Analyze the following conversation turn and extract any memorable information.

Return a JSON array of objects. Each object must have:
- "type": one of "fact", "preference", "observation", "entity", "concept"
- Plus type-specific fields:

For "fact":
  {"type": "fact", "statement": "...", "confidence": 0.0-1.0}

For "preference":
  {"type": "preference", "holder": "user", "subject": "...", "preference": "...", "strength": "weak"|"moderate"|"strong"|"absolute"}

For "observation":
  {"type": "observation", "content": "..."}

For "entity":
  {"type": "entity", "name": "...", "entity_type": "person"|"organization"|"place"|"product"|"other"}

For "concept":
  {"type": "concept", "name": "...", "definition": "..."}

Rules:
- Only extract information that is worth remembering across conversations.
- Do not extract trivial greetings or filler.
- Return [] if nothing is worth remembering.
- Return valid JSON only, no markdown or explanation.\
"""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to long-term memory. \
Use the relevant memories provided to personalize your responses \
and maintain continuity across conversations.\
"""


def _format_memories_for_prompt(memories: list[MemoryEntry]) -> str:
    """Format retrieved memories into a prompt section."""
    if not memories:
        return ""

    lines = ["## Relevant Memories", ""]
    for memory in memories:
        lines.append(f"- [{memory.memory_type}] {memory.content}")
    return "\n".join(lines)


def _parse_extraction_response(response: str) -> list[dict[str, Any]]:
    """Parse the LLM extraction response into a list of memory dicts."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        inner_lines = cleaned.split("\n")
        inner_lines = [
            line for line in inner_lines
            if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(inner_lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction response as JSON")
        return []

    if isinstance(result, list):
        return result
    return []


class Conversation:
    """Automatic conversation manager backed by Membrain.

    Handles the full conversation loop asynchronously: retrieve relevant
    memories, build the prompt with context, call the LLM, extract memorable
    information from the turn, and store it for future retrieval.

    Accepts both sync and async LLM callables. Sync callables are
    automatically run in the default executor to avoid blocking the event loop.

    Usage::

        from membrain import MembrainClient, Conversation

        async def my_llm(messages):
            return await call_my_llm(messages)

        with MembrainClient() as client:
            async with Conversation(llm_callable=my_llm, client=client) as conv:
                response = await conv.reply("I prefer dark mode")
                response = await conv.reply("What do you know about me?")
                await conv.end(outcome="positive")
    """

    def __init__(
        self,
        llm_callable: Callable,
        client: MembrainClient | None = None,
        system_prompt: str | None = None,
        memory_limit: int = 10,
        auto_extract: bool = True,
        history_limit: int = 50,
    ) -> None:
        """Initialize a conversation.

        Args:
            llm_callable: Function that takes a list of message dicts
                (each with "role" and "content") and returns the LLM
                response string. Can be sync or async.
            client: Optional MembrainClient instance. Creates one if not
                provided.
            system_prompt: Custom system prompt. Uses a default if not provided.
            memory_limit: Maximum number of memories to inject per turn.
            auto_extract: Whether to automatically extract and store memorable
                information from each turn.
            history_limit: Maximum number of turns to keep in the context window.
        """
        if memory_limit < 0:
            raise ValueError("memory_limit must be >= 0")
        if history_limit < 0:
            raise ValueError("history_limit must be >= 0")

        self._llm = llm_callable
        self._owns_client = client is None
        self._client = client or MembrainClient()
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._memory_limit = memory_limit
        self._auto_extract = auto_extract
        self._history_limit = history_limit
        self._session_id = str(uuid.uuid4())
        self._history: list[dict[str, str]] = []
        self._closed = False
        self._turn_lock = asyncio.Lock()

    @property
    def session_id(self) -> str:
        """The current session ID."""
        return self._session_id

    @property
    def history(self) -> list[dict[str, str]]:
        """The full conversation history as {role, content} dicts."""
        return list(self._history)

    async def reply(self, user_message: str) -> str:
        """Send a user message and get a response.

        Automatically retrieves relevant memories, builds the prompt,
        calls the LLM, and extracts memorable information from the turn.
        Serialized by an internal lock to protect shared conversation state.

        Args:
            user_message: The user's message text.

        Returns:
            The assistant's response text.
        """
        async with self._turn_lock:
            self._ensure_open_locked()

            # Retrieve relevant memories and case context concurrently
            memories, case_context = await asyncio.gather(
                self._retrieve_memories(user_message),
                self._retrieve_case_context(user_message),
            )

            # Build the message list for the LLM (pure Python, no I/O)
            messages = self._build_messages(user_message, memories, case_context)

            # Call the LLM
            assistant_response = await self._call_llm(messages)

            # Track the turn
            self._history.append({"role": "user", "content": user_message})
            self._history.append(
                {"role": "assistant", "content": assistant_response}
            )

            # Post-response persistence is best-effort: the user already has
            # the LLM response, so memory writes should not turn success into
            # a visible failure.
            try:
                await self._client.store_event(
                    event_type="conversation_turn",
                    description=f"User: {user_message}\nAssistant: {assistant_response}",
                )
            except Exception:
                logger.warning("Failed to store conversation event", exc_info=True)

            # Extract and store memorable information
            if self._auto_extract:
                await self._extract_and_store(user_message, assistant_response)

            return assistant_response

    async def end(self, outcome: str = "", reward: float = 1.0) -> None:
        """End the conversation and store it as a case for future learning.

        Args:
            outcome: Description of the conversation outcome.
            reward: Reward signal (1.0 = success, -1.0 = failure).
        """
        async with self._turn_lock:
            self._ensure_open_locked()

            if not self._history:
                return

            # Build a summary of the conversation
            first_user_message = ""
            for turn in self._history:
                if turn["role"] == "user":
                    first_user_message = turn["content"]
                    break

            turn_count = len(self._history) // 2
            last_exchange = ""
            if len(self._history) >= 2:
                last_exchange = (
                    f"User: {self._history[-2]['content']}\n"
                    f"Assistant: {self._history[-1]['content']}"
                )

            problem = first_user_message
            plan = f"Conversation with {turn_count} exchanges"
            final_outcome = outcome or last_exchange

            await self._client.store_case(
                problem=problem,
                plan=plan,
                outcome=final_outcome,
                reward=reward,
            )

            logger.info(
                "Conversation %s ended: %d turns, reward=%.1f",
                self._session_id,
                turn_count,
                reward,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retrieve_memories(self, query: str) -> list[MemoryEntry]:
        """Retrieve memories relevant to the user's message."""
        if self._memory_limit == 0:
            return []
        results = await self._client.search(query, limit=self._memory_limit)
        return results.memories

    async def _retrieve_case_context(self, query: str) -> str:
        """Retrieve and format relevant past conversation cases."""
        cases = await self._client.search_cases(query, limit=3)

        if not cases.positive_cases and not cases.negative_cases:
            return ""

        sections: list[str] = ["## Past Conversation Experience", ""]

        for case in cases.positive_cases[:2]:
            sections.append(
                f"- Previously when asked about '{case.problem[:60]}', "
                f"a successful approach was: {case.outcome[:100]}"
            )

        for case in cases.negative_cases[:1]:
            sections.append(
                f"- Avoid: when asked about '{case.problem[:60]}', "
                f"this approach failed: {case.outcome[:100]}"
            )

        return "\n".join(sections)

    def _build_messages(
        self,
        user_message: str,
        memories: list[MemoryEntry],
        case_context: str,
    ) -> list[dict[str, str]]:
        """Build the full message list for the LLM call. Pure Python, no I/O."""
        system_content = self._system_prompt

        # Inject retrieved memories into the system prompt
        memory_section = _format_memories_for_prompt(memories)
        if memory_section:
            system_content = f"{system_content}\n\n{memory_section}"

        # Inject relevant past cases
        if case_context:
            system_content = f"{system_content}\n\n{case_context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Add conversation history (trimmed to history_limit)
        if self._history_limit == 0:
            recent_history = []
        else:
            recent_history = self._history[-(self._history_limit * 2) :]
        messages.extend(recent_history)

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM, handling both sync and async callables."""
        fn = self._llm

        is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )

        if is_async:
            result = fn(messages)
        else:
            # Sync callable — run in default executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, partial(fn, messages))

        if inspect.isawaitable(result):
            result = await result

        if not isinstance(result, str):
            raise TypeError(
                "llm_callable must return a string or an awaitable that resolves "
                f"to a string, got {type(result).__name__}"
            )

        return result

    async def _extract_and_store(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Extract memorable information from the turn and store it."""
        turn_text = f"User: {user_message}\nAssistant: {assistant_response}"

        extraction_messages: list[dict[str, str]] = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": turn_text},
        ]

        try:
            extraction_response = await self._call_llm(extraction_messages)
        except Exception:
            logger.warning("Memory extraction LLM call failed", exc_info=True)
            return

        extracted = _parse_extraction_response(extraction_response)

        for item in extracted:
            memory_type = item.get("type", "")
            try:
                await self._store_extracted_memory(item, memory_type)
            except Exception:
                logger.warning(
                    "Failed to store extracted %s memory",
                    memory_type,
                    exc_info=True,
                )

    async def _store_extracted_memory(
        self, item: dict[str, Any], memory_type: str
    ) -> None:
        """Store a single extracted memory item."""
        if memory_type == "fact":
            await self._client.store_fact(
                statement=item.get("statement", ""),
                confidence=item.get("confidence", 0.8),
            )
        elif memory_type == "preference":
            await self._client.store_preference(
                holder=item.get("holder", "user"),
                subject=item.get("subject", ""),
                preference=item.get("preference", ""),
                strength=item.get("strength", "moderate"),
            )
        elif memory_type == "observation":
            await self._client.store_observation(
                content=item.get("content", ""),
            )
        elif memory_type == "entity":
            await self._client.store_entity(
                name=item.get("name", ""),
                entity_type=item.get("entity_type", "other"),
            )
        elif memory_type == "concept":
            await self._client.store_concept(
                name=item.get("name", ""),
                definition=item.get("definition", ""),
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise MembrainError("conversation is already closed")

    async def close(self) -> None:
        """Release resources. If the client was created internally, close it."""
        async with self._turn_lock:
            if self._closed:
                return

            self._closed = True
            if self._owns_client:
                # The PyO3 client close() is synchronous. It does drop the underlying Arc.
                self._client.close()

    async def __aenter__(self) -> Conversation:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

AsyncConversation = Conversation
