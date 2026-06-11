"""Automatic conversation management with memory extraction and retrieval.

Requires: OPENAI_API_KEY environment variable.
"""

import os

from membrain import Conversation


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    from openai import OpenAI
    openai_client = OpenAI()

    def llm(messages: list[dict[str, str]]) -> str:
        response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    with Conversation(llm_callable=llm) as conv:
        print(f"Session: {conv.session_id}\n")

        response = conv.reply(
            "I prefer dark mode for all my tools, and I work at Acme Corp as a backend engineer."
        )
        print(f"Assistant: {response}\n")

        response = conv.reply("Can you recommend a code editor setup for me?")
        print(f"Assistant: {response}\n")

        response = conv.reply("What do you remember about me?")
        print(f"Assistant: {response}\n")

        conv.end(outcome="User was satisfied with recommendations", reward=1.0)
        print(f"Conversation ended. {len(conv.history)} messages tracked.")


if __name__ == "__main__":
    main()
