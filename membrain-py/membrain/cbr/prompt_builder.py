"""Prompt builder for in-context learning with retrieved cases.

Formats positive and negative cases into sections that can be prepended
to an LLM prompt so the model learns from past experiences.
"""

from __future__ import annotations

import json

from ..types import CaseSearchResults


class CasePromptBuilder:
    """Builds in-context learning prompts from retrieved cases.

    Positive cases become "Successful Past Approaches" and negative cases
    become "Approaches to Avoid".
    """

    def __init__(
        self,
        max_positive_examples: int = 3,
        max_negative_examples: int = 2,
    ) -> None:
        self._max_positive = max_positive_examples
        self._max_negative = max_negative_examples

    def build_context(self, cases: CaseSearchResults) -> str:
        """Build a prompt context string from retrieved cases.

        Args:
            cases: The positive and negative cases from a retriever.

        Returns:
            A formatted string to include as context in the LLM prompt.
            Returns an empty string if no cases are available.
        """
        sections: list[str] = []

        positive = cases.positive_cases[: self._max_positive]
        if positive:
            lines = ["## Successful Past Approaches", ""]
            for index, case in enumerate(positive, start=1):
                lines.append(f"### Example {index}")
                lines.append(f"**Problem:** {case.problem}")
                lines.append(f"**Plan:**\n{_format_plan(case.plan)}")
                lines.append(f"**Outcome:** {case.outcome}")
                lines.append("")
            sections.append("\n".join(lines))

        negative = cases.negative_cases[: self._max_negative]
        if negative:
            lines = ["## Approaches to Avoid", ""]
            for index, case in enumerate(negative, start=1):
                lines.append(f"### Counter-Example {index}")
                lines.append(f"**Problem:** {case.problem}")
                lines.append(f"**Plan:**\n{_format_plan(case.plan)}")
                lines.append(f"**Outcome:** {case.outcome}")
                lines.append("")
            sections.append("\n".join(lines))

        return "\n".join(sections)


def _format_plan(plan: str) -> str:
    """Format a plan string for display.

    If the plan is valid JSON with a list of steps, formats them as
    a numbered list. Otherwise returns the plan text as-is.
    """
    try:
        parsed = json.loads(plan)
    except (json.JSONDecodeError, TypeError):
        return plan

    if isinstance(parsed, dict) and "plan" in parsed:
        parsed = parsed["plan"]

    if isinstance(parsed, list):
        lines = []
        for index, step in enumerate(parsed, start=1):
            if isinstance(step, dict):
                description = step.get("description", str(step))
            else:
                description = str(step)
            lines.append(f"{index}. {description}")
        return "\n".join(lines)

    return plan
