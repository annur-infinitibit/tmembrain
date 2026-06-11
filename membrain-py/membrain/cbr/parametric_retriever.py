"""Parametric retriever that uses a trained RelevanceClassifier.

Over-fetches candidates via embedding similarity, then re-scores them with
the trained neural model to select the most relevant cases.

Requires ``torch`` and ``transformers`` as optional dependencies.
"""

from __future__ import annotations

from pathlib import Path

from ..client import MembrainClient
from ..types import CaseEntry, CaseSearchResults
from .classifier import _check_dependencies, _TORCH_AVAILABLE
from .dataset import CASE_TOKEN, PLAN_TOKEN, _format_plan
from .retriever import CaseRetriever

if _TORCH_AVAILABLE:
    import torch


class ParametricRetriever(CaseRetriever):
    """Retriever that re-ranks candidates with a trained classifier.

    Flow:
        1. Over-fetch candidates via MembrainClient.search_cases()
        2. Format each as ICL text
        3. Score all (query, candidate) pairs with the classifier
        4. Sort by score descending, take top-k
        5. Separate into positive/negative by case_label
    """

    def __init__(
        self,
        client: MembrainClient,
        checkpoint_path: str,
        backbone_model: str = "princeton-nlp/sup-simcse-roberta-base",
        overfetch_factor: int = 3,
        max_length: int = 256,
        use_plan: bool = True,
        plan_style: str = "pretty",
    ) -> None:
        """Initialize with a trained checkpoint.

        Args:
            client: MembrainClient for fetching candidate cases.
            checkpoint_path: Path to the checkpoint directory containing
                ``model.pt`` and ``tokenizer/``.
            backbone_model: HuggingFace model ID matching the training backbone.
            overfetch_factor: Fetch this many times more candidates than the
                requested limit for re-ranking.
            max_length: Max token sequence length for the tokeniser.
            use_plan: Whether to include plan text in ICL formatting.
            plan_style: "pretty" or "raw" for plan formatting.
        """
        _check_dependencies()
        from transformers import AutoModel, AutoTokenizer

        from .classifier import RelevanceClassifier

        self._client = client
        self._overfetch_factor = overfetch_factor
        self._max_length = max_length
        self._use_plan = use_plan
        self._plan_style = plan_style

        checkpoint_dir = Path(checkpoint_path)
        tokenizer_path = checkpoint_dir / "tokenizer"
        model_weights_path = checkpoint_dir / "model.pt"

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        backbone = AutoModel.from_pretrained(backbone_model)
        self._classifier = RelevanceClassifier(backbone)

        state_dict = torch.load(
            model_weights_path, map_location="cpu", weights_only=True
        )
        self._classifier.load_state_dict(state_dict)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._classifier.to(self._device)
        self._classifier.eval()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        positive_reward_threshold: float = 0.5,
    ) -> CaseSearchResults:
        """Retrieve and re-rank cases using the trained classifier."""
        overfetch_limit = limit * self._overfetch_factor
        raw_results = self._client.search_cases(
            query,
            limit=overfetch_limit,
            positive_reward_threshold=positive_reward_threshold,
        )

        all_cases = raw_results.positive_cases + raw_results.negative_cases
        if not all_cases:
            return CaseSearchResults()

        scored = self._score_candidates(query, all_cases)
        top_cases = scored[:limit]

        positive_cases = [
            case
            for case, _score in top_cases
            if case.reward >= positive_reward_threshold
        ]
        negative_cases = [
            case
            for case, _score in top_cases
            if case.reward < positive_reward_threshold
        ]

        return CaseSearchResults(
            positive_cases=positive_cases,
            negative_cases=negative_cases,
            duration_ms=raw_results.duration_ms,
        )

    def _format_icl(self, case: CaseEntry) -> str:
        """Format a case entry as ICL text for the classifier."""
        case_text = (
            f"Problem: {case.problem}\n"
            f"Outcome: {case.outcome}\n"
            f"Reward: {case.reward}"
        )
        parts = [f"{CASE_TOKEN}\n{case_text}"]
        if self._use_plan and case.plan:
            formatted_plan = _format_plan(case.plan, self._plan_style)
            parts.append(f"{PLAN_TOKEN}\n{formatted_plan}")
        return "\n".join(parts)

    @torch.inference_mode()
    def _score_candidates(
        self, query: str, candidates: list[CaseEntry]
    ) -> list[tuple[CaseEntry, float]]:
        """Score all candidates against the query with the classifier."""
        icl_texts = [self._format_icl(case) for case in candidates]
        queries = [query] * len(candidates)

        tokens_icl = self._tokenizer(
            icl_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        tokens_query = self._tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        logits = self._classifier(
            tokens_icl["input_ids"].to(self._device),
            tokens_icl["attention_mask"].to(self._device),
            tokens_query["input_ids"].to(self._device),
            tokens_query["attention_mask"].to(self._device),
        )
        scores = torch.softmax(logits, dim=1)[:, 1]
        paired = list(zip(candidates, scores.cpu().tolist()))
        paired.sort(key=lambda pair: pair[1], reverse=True)
        return paired
