"""Plan-execute-learn agent driven by an OpenAI LLM.

Requires: OPENAI_API_KEY environment variable.
"""

import os

from membrain import MembrainClient
from membrain.agent import MembrainAgent


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    from openai import OpenAI

    openai_client = OpenAI()

    def llm(messages: list[dict[str, str]]) -> str:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    client = MembrainClient()
    try:
        # Seed one past case so the planner has context to learn from.
        client.store_case(
            problem="Deploy a Python web service",
            plan='{"plan": [{"id": 1, "description": "Build the wheel"},'
            ' {"id": 2, "description": "Run tests"},'
            ' {"id": 3, "description": "Promote to production"}]}',
            outcome="Deployment completed without downtime",
            reward=1.0,
        )

        agent = MembrainAgent(client=client, llm_callable=llm, max_cycles=1)
        # Register a trivial tool the executor can call when useful.
        agent.executor.register_tool(
            "lookup_health", lambda service: f"{service}: healthy"
        )

        output = agent.run("Deploy the billing microservice to production")
        print("--- Agent output ---")
        print(output)
    finally:
        client.close()


if __name__ == "__main__":
    main()
