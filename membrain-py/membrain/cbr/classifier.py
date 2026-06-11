"""Neural relevance classifier for case-based reasoning.

A sentence-transformer backbone encodes both the case (ICL) text and the
query text, then a small MLP head predicts whether the case is relevant
to the query (2-class classification).

Requires ``torch`` and ``transformers`` as optional dependencies.
"""

from __future__ import annotations

_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def _check_dependencies() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the neural classifier. "
            "Install it with: pip install torch"
        )
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "The transformers library is required for the neural classifier. "
            "Install it with: pip install transformers"
        )


DEFAULT_BACKBONE = "princeton-nlp/sup-simcse-roberta-base"


def build_classifier(
    backbone_model: str = DEFAULT_BACKBONE,
) -> "RelevanceClassifier":
    """Build a RelevanceClassifier with the given backbone.

    Args:
        backbone_model: HuggingFace model identifier for the backbone.

    Returns:
        An initialised RelevanceClassifier on CPU.
    """
    _check_dependencies()
    backbone = AutoModel.from_pretrained(backbone_model)
    return RelevanceClassifier(backbone)


if _TORCH_AVAILABLE:

    class RelevanceClassifier(nn.Module):
        """Two-tower classifier that scores (case, query) relevance.

        Architecture:
            1. Backbone encodes case text -> CLS embedding (o1)
            2. Backbone encodes query text -> CLS embedding (o2)
            3. Concatenate [o1, o2] -> hidden_size * 2
            4. MLP head: Linear -> ReLU -> Dropout -> Linear -> 2-class logits
        """

        def __init__(self, backbone: "AutoModel") -> None:
            super().__init__()
            hidden = backbone.config.hidden_size
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2),
            )

        def forward(
            self,
            ids1: "torch.Tensor",
            mask1: "torch.Tensor",
            ids2: "torch.Tensor",
            mask2: "torch.Tensor",
        ) -> "torch.Tensor":
            """Forward pass producing 2-class logits.

            Args:
                ids1: Token IDs for the case (ICL) text.
                mask1: Attention mask for the case text.
                ids2: Token IDs for the query text.
                mask2: Attention mask for the query text.

            Returns:
                Logits tensor of shape (batch_size, 2).
            """
            output_1 = self.backbone(ids1, attention_mask=mask1).last_hidden_state[
                :, 0
            ]
            output_2 = self.backbone(ids2, attention_mask=mask2).last_hidden_state[
                :, 0
            ]
            return self.head(torch.cat([output_1, output_2], dim=1))

else:
    # Placeholder so the module can be imported without torch for type hints.
    class RelevanceClassifier:  # type: ignore[no-redef]
        """Placeholder -- install torch to use the neural classifier."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            _check_dependencies()
