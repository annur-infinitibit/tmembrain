"""Case-based reasoning: retrieve past cases, build a prompt, record training data.

Requires: OPENAI_API_KEY environment variable.
"""

import os
from pathlib import Path

from membrain import MembrainClient
from membrain.cbr import (
    CasePromptBuilder,
    NonParametricRetriever,
    TrainingDataCollector,
)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    from openai import OpenAI

    openai_client = OpenAI()

    client = MembrainClient()
    try:
        client.store_case(
            problem="Scale the API to handle 10k rps",
            plan="Add a connection pool and horizontal autoscaling",
            outcome="Latency stayed under 200ms at peak load",
            reward=1.0,
        )
        client.store_case(
            problem="Scale the API to handle 10k rps",
            plan="Rewrite the hot path in Rust only",
            outcome="Weeks of work, no measurable gain under load",
            reward=0.0,
        )

        retriever = NonParametricRetriever(client)
        prompt_builder = CasePromptBuilder(
            max_positive_examples=2, max_negative_examples=1
        )

        query = "Plan capacity for the checkout service going into Black Friday"
        cases = retriever.retrieve(query, limit=5)
        context = prompt_builder.build_context(cases)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an SRE assistant. Reply in three steps.",
                },
                {
                    "role": "user",
                    "content": f"{context}\n\n## Current Query\n\n{query}"
                    if context
                    else query,
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        print("--- Assistant answer ---")
        print(answer)

        collector = TrainingDataCollector()
        collector.record(
            query=query,
            retrieved_cases=cases.positive_cases + cases.negative_cases,
            is_correct=True,
        )
        output_path = Path("./training_data.jsonl")
        written = collector.flush(output_path)
        print(f"\nWrote {written} training pairs to {output_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
