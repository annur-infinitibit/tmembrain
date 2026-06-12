import asyncio
import json
import os
import sys
from pathlib import Path

from graphbit import Executor, LlmConfig, Node, Workflow


PROJECT_ROOT = Path(__file__).resolve().parent
MEMBRAIN_PY_PATH = PROJECT_ROOT / "membrain-py"
if str(MEMBRAIN_PY_PATH) not in sys.path:
    sys.path.insert(0, str(MEMBRAIN_PY_PATH))

from membrain import MembrainClient


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()

EXTRACTION_PROMPT = """\
Analyze the following conversation turn and extract any memorable information.

Return a JSON array of objects. Each object must have:
- "type": one of "event", "case", "fact", "preference", "observation", "entity", "concept"
- Plus type-specific fields:

For "event":
  {"type": "event", "event_type": "conversation_turn", "description": "..."}

For "case":
  {"type": "case", "problem": "...", "plan": "...", "outcome": "...", "reward": -1.0 to 1.0}

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
- Always include exactly one "event" capturing the full turn.
- Include a "case" only when the user's message implies clear success or failure
  (e.g. explicit satisfaction/unsatisfaction, acceptance/rejection, or a clear outcome).
- Return valid JSON only, no markdown or explanation.\
"""


def _run_graphbit_agent(
    executor: Executor,
    *,
    name: str,
    prompt: str,
    system_prompt: str,
) -> str:
    workflow = Workflow(f"{name} Workflow")
    node = Node.agent(name=name, prompt=prompt, system_prompt=system_prompt)
    workflow.add_node(node)
    result = executor.execute(workflow)
    if not result.is_success():
        raise RuntimeError(f"Workflow failed for node '{name}'")
    output = result.get_node_output(name)
    return output if isinstance(output, str) else str(output)


def _run_graphbit_agent_streaming(
    executor: Executor,
    *,
    name: str,
    prompt: str,
    system_prompt: str,
) -> str:
    workflow = Workflow(f"{name} Workflow")
    node = Node.agent(name=name, prompt=prompt, system_prompt=system_prompt)
    workflow.add_node(node)

    chunks: list[str] = []
    node_output: str = ""

    for event in executor.execute_streaming(workflow, stream_mode="all"):
        event_type = event.get("event")
        node_name = event.get("node_name")

        if event_type == "token" and node_name == name:
            token = event.get("content", "")
            if token:
                chunks.append(token)
                print(token, end="", flush=True)

        elif event_type == "node_completed" and node_name == name:
            # Always capture node output as a reliable fallback for when
            # token events are absent (e.g. cached responses, non-streaming
            # model endpoints, or graphbit version differences).
            node_output = event.get("output", "")

        elif event_type == "node_failed" and node_name == name:
            error = event.get("error", "unknown error")
            raise RuntimeError(f"Node '{name}' failed: {error}")

        elif event_type == "workflow_failed":
            error = event.get("error", "unknown error")
            raise RuntimeError(f"Workflow failed: {error}")

    response_text = "".join(chunks).strip()

    # If token streaming produced nothing, fall back to the node_completed output.
    if not response_text:
        response_text = node_output.strip()
        if response_text:
            print(response_text, end="", flush=True)

    if not response_text:
        raise RuntimeError(f"Workflow streaming produced no output for node '{name}'")

    return response_text


def _messages_to_graphbit_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = ""
    prompt_lines: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system" and not system_prompt:
            system_prompt = content
            continue
        prompt_lines.append(f"{role.upper()}: {content}")
    return system_prompt, "\n\n".join(prompt_lines).strip()


def _format_memories_for_prompt(memories: list[object]) -> str:
    # MemoryEntry-like objects have `.memory_type` and `.content`
    if not memories:
        return ""
    lines = ["## Relevant Memories", ""]
    for m in memories:
        memory_type = getattr(m, "memory_type", "unknown")
        content = getattr(m, "content", "")
        lines.append(f"- [{memory_type}] {content}")
    return "\n".join(lines)


def _format_cases_for_prompt(cases: object) -> str:
    positive = getattr(cases, "positive_cases", []) or []
    negative = getattr(cases, "negative_cases", []) or []
    if not positive and not negative:
        return ""
    sections: list[str] = ["## Past Conversation Experience", ""]
    for case in positive[:2]:
        problem = getattr(case, "problem", "")
        outcome = getattr(case, "outcome", "")
        sections.append(
            f"- Previously when asked about '{str(problem)[:60]}', "
            f"a successful approach was: {str(outcome)[:100]}"
        )
    for case in negative[:1]:
        problem = getattr(case, "problem", "")
        outcome = getattr(case, "outcome", "")
        sections.append(
            f"- Avoid: when asked about '{str(problem)[:60]}', "
            f"this approach failed: {str(outcome)[:100]}"
        )
    return "\n".join(sections)


def _parse_extracted_memories(response_text: str) -> list[dict]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


async def _store_extracted_item(client: MembrainClient, item: dict) -> None:
    memory_type = str(item.get("type", "")).strip()
    if memory_type == "event":
        event_type = str(item.get("event_type", "conversation_turn")).strip() or "conversation_turn"
        description = str(item.get("description", "")).strip()
        if description:
            await client.store_event(event_type=event_type, description=description)
    elif memory_type == "case":
        problem = str(item.get("problem", "")).strip()
        plan = str(item.get("plan", "")).strip()
        outcome = str(item.get("outcome", "")).strip()
        reward_raw = item.get("reward", 0.0)
        try:
            reward = float(reward_raw)
        except (TypeError, ValueError):
            reward = 0.0
        if problem and (outcome or plan):
            await client.store_case(problem=problem, plan=plan, outcome=outcome, reward=reward)
    elif memory_type == "fact":
        await client.store_fact(
            statement=str(item.get("statement", "")).strip(),
            confidence=float(item.get("confidence", 0.8) or 0.8),
        )
    elif memory_type == "preference":
        await client.store_preference(
            holder=str(item.get("holder", "user")).strip() or "user",
            subject=str(item.get("subject", "")).strip(),
            preference=str(item.get("preference", "")).strip(),
            strength=str(item.get("strength", "moderate")).strip() or "moderate",
        )
    elif memory_type == "observation":
        await client.store_observation(content=str(item.get("content", "")).strip())
    elif memory_type == "entity":
        await client.store_entity(
            name=str(item.get("name", "")).strip(),
            entity_type=str(item.get("entity_type", "other")).strip() or "other",
        )
    elif memory_type == "concept":
        await client.store_concept(
            name=str(item.get("name", "")).strip(),
            definition=str(item.get("definition", "")).strip(),
        )


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY.")
        return

    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm_config = LlmConfig.openai(os.environ["OPENAI_API_KEY"], llm_model)
    executor = Executor(llm_config)
    # Use a dedicated executor for background extraction to avoid any thread-safety
    # issues if Graphbit's Executor is not safe to call concurrently.
    extraction_executor = Executor(llm_config)

    print("Enterprise Membrain Chatbot Initialized")
    user_id = "user_2"
    print(f"\nWelcome {user_id}! ")

    # Hard isolation: one storage directory per user.
    # This avoids relying on metadata scoping for tenant separation.
    user_storage_path = PROJECT_ROOT / "users" / user_id
    user_storage_path.mkdir(parents=True, exist_ok=True)
    user_client = MembrainClient(
        config={
            "storage": {
                "backend": "memscaledb",
                "path": str(user_storage_path),
            }
        }
    )

    user_client_lock = asyncio.Lock()

    # Higher-level "thread" state lives here, not in Membrain.
    chat_history: list[dict[str, str]] = []
    history_limit = 40  # exchanges => 80 messages (user+assistant)

    bg_tasks: set[asyncio.Task[None]] = set()

    async def extract_and_store_memories(user_msg: str, bot_msg: str) -> None:
        turn_text = f"User: {user_msg}\nAssistant: {bot_msg}"
        try:
            extracted_text = await asyncio.to_thread(
                _run_graphbit_agent,
                extraction_executor,
                name="Memory Extractor",
                prompt=turn_text,
                system_prompt=EXTRACTION_PROMPT,
            )
            for item in _parse_extracted_memories(extracted_text):
                async with user_client_lock:
                    await _store_extracted_item(user_client, item)
        except Exception as err:
            print(f"[Extraction Failed] {err}")

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "> ")
                if user_input.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                if not user_input.strip():
                    continue

                # Maintain thread context in Python.
                chat_history.append({"role": "user", "content": user_input})

                # Retrieve relevant memories + past cases.
                async with user_client_lock:
                    memory_results = await user_client.search(user_input, limit=10)
                    case_results = await user_client.search_cases(user_input, limit=3)

                system_content = "\n".join(
                    [
                        "You are a helpful AI assistant with access to long-term memory.",
                    ]
                )
                memories_section = _format_memories_for_prompt(memory_results.memories)
                if memories_section:
                    system_content = f"{system_content}\n\n{memories_section}"
                cases_section = _format_cases_for_prompt(case_results)
                if cases_section:
                    system_content = f"{system_content}\n\n{cases_section}"

                # Build the message list the way `Conversation` does, but with
                # thread context controlled by our Python `chat_history`.
                messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
                messages.extend(chat_history[-(history_limit * 2):])

                # print("--------------SYSTEM PROMPT START------------------")
                # print(system_content)
                # print("---------------SYSTEM PROMPT END-----------------")

                print("> ", end="", flush=True)
                system_prompt, prompt = _messages_to_graphbit_prompt(messages)
                response_text = await asyncio.to_thread(
                    _run_graphbit_agent_streaming,
                    executor,
                    name="Chatbot",
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                print("\n")

                chat_history.append({"role": "assistant", "content": response_text})

                # Kick off extraction + store in the background so the next
                # prompt isn't blocked by a second LLM call.
                task = asyncio.create_task(extract_and_store_memories(user_input, response_text))
                bg_tasks.add(task)
                task.add_done_callback(lambda t: bg_tasks.discard(t))

            except KeyboardInterrupt:
                break
    finally:
        try:
            if bg_tasks:
                await asyncio.wait(bg_tasks, timeout=3.0)
        except Exception:
            pass
        user_client.close()


if __name__ == "__main__":
    asyncio.run(main())
