"""Training dataset and collator for the relevance classifier.

Loads JSONL training data produced by TrainingDataCollector and prepares
batches for the RelevanceClassifier.

Requires ``torch`` and ``transformers`` as optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .classifier import _check_dependencies, _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    import torch
    from torch.utils.data import Dataset
else:
    Dataset = object  # type: ignore[assignment,misc]


CASE_TOKEN = "[CASE]"
PLAN_TOKEN = "[PLAN]"


def _format_plan(plan: str, style: str = "pretty") -> str:
    """Format a plan string for the ICL text.

    Args:
        plan: Raw plan string (possibly JSON).
        style: "pretty" parses JSON into numbered steps, "raw" uses as-is.
    """
    if style == "raw":
        return plan
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


class CasePairDataset(Dataset):
    """Dataset that produces (ICL text, query text, label) triples.

    Each sample is loaded from a JSONL record:
        - ``icl``: ``"[CASE]\\n{case_text}\\n[PLAN]\\n{formatted_plan}"``
        - ``natural``: ``"{query}"``
        - ``label``: 1 if truth_label is true, 0 otherwise.
    """

    def __init__(
        self,
        path: str | Path,
        use_plan: bool = True,
        plan_style: str = "pretty",
    ) -> None:
        _check_dependencies()
        self._records: list[dict[str, Any]] = []
        self._use_plan = use_plan
        self._plan_style = plan_style

        with Path(path).open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]

        case_text = record["case"]
        plan = record.get("plan", "")
        icl_parts = [f"{CASE_TOKEN}\n{case_text}"]
        if self._use_plan and plan:
            formatted_plan = _format_plan(plan, self._plan_style)
            icl_parts.append(f"{PLAN_TOKEN}\n{formatted_plan}")
        icl = "\n".join(icl_parts)

        label = 1 if record["truth_label"] else 0

        return {
            "icl": icl,
            "natural": record["query"],
            "label": label,
        }


class CasePairCollator:
    """Collator that tokenises and pads a batch of CasePairDataset samples.

    Returns a tuple of ``(ids1, mask1, ids2, mask2, labels)`` tensors.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        _check_dependencies()
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        icl_texts = [sample["icl"] for sample in batch]
        natural_texts = [sample["natural"] for sample in batch]
        labels = [sample["label"] for sample in batch]

        tokens_icl = self._tokenizer(
            icl_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        tokens_natural = self._tokenizer(
            natural_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        return (
            tokens_icl["input_ids"],
            tokens_icl["attention_mask"],
            tokens_natural["input_ids"],
            tokens_natural["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
