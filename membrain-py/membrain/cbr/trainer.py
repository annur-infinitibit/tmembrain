"""Training loop for the relevance classifier.

Provides ``RetrieverTrainer`` which handles data loading, stratified
train/validation splitting, optimizer and scheduler setup, mixed-precision
training, evaluation metrics, and checkpoint management.

Requires ``torch``, ``transformers``, and ``scikit-learn`` as optional deps.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classifier import DEFAULT_BACKBONE, _check_dependencies, _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    best_checkpoint_path: str
    final_checkpoint_path: str
    best_metric: float
    final_accuracy: float
    final_f1: float
    final_auc: float
    epochs_trained: int
    total_steps: int
    training_samples: int
    validation_samples: int


class RetrieverTrainer:
    """Trains a RelevanceClassifier on collected training data.

    Usage::

        trainer = RetrieverTrainer(
            training_data_path="training.jsonl",
            output_dir="./checkpoints",
        )
        result = trainer.train(epochs=3, batch_size=32)
        print(result.best_checkpoint_path)
    """

    def __init__(
        self,
        training_data_path: str,
        output_dir: str,
        backbone_model: str = DEFAULT_BACKBONE,
        validation_data_path: str | None = None,
        validation_ratio: float = 0.1,
        use_plan: bool = True,
        plan_style: str = "pretty",
    ) -> None:
        _check_dependencies()
        self._training_data_path = training_data_path
        self._output_dir = Path(output_dir)
        self._backbone_model = backbone_model
        self._validation_data_path = validation_data_path
        self._validation_ratio = validation_ratio
        self._use_plan = use_plan
        self._plan_style = plan_style

    def train(
        self,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        max_length: int = 256,
        gradient_clip: float = 1.0,
        mixed_precision: bool = False,
        eval_every_steps: int = 500,
        save_best: bool = True,
        class_weight_positive: float | None = None,
        seed: int = 42,
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Peak learning rate for AdamW.
            weight_decay: Weight decay coefficient (excludes bias/LayerNorm).
            warmup_ratio: Fraction of total steps used for linear warmup.
            max_length: Maximum token sequence length.
            gradient_clip: Maximum gradient norm.
            mixed_precision: Enable FP16 mixed precision.
            eval_every_steps: Run evaluation every N training steps.
            save_best: Whether to save best checkpoint by metric.
            class_weight_positive: Manual positive class weight. If None,
                automatically computed as negative_count / positive_count.
            seed: Random seed for reproducibility.

        Returns:
            TrainingResult with paths to checkpoints and final metrics.
        """
        _set_seed(seed)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Build model and tokeniser
        # ------------------------------------------------------------------
        from transformers import AutoModel, AutoTokenizer

        from .classifier import RelevanceClassifier
        from .dataset import CasePairCollator, CasePairDataset

        tokenizer = AutoTokenizer.from_pretrained(self._backbone_model)
        backbone = AutoModel.from_pretrained(self._backbone_model)
        model = RelevanceClassifier(backbone)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # ------------------------------------------------------------------
        # Datasets
        # ------------------------------------------------------------------
        full_dataset = CasePairDataset(
            self._training_data_path,
            use_plan=self._use_plan,
            plan_style=self._plan_style,
        )

        if self._validation_data_path:
            train_dataset: Any = full_dataset
            validation_dataset: Any = CasePairDataset(
                self._validation_data_path,
                use_plan=self._use_plan,
                plan_style=self._plan_style,
            )
        else:
            train_indices, validation_indices = _stratified_split(
                full_dataset, self._validation_ratio, seed
            )
            train_dataset = Subset(full_dataset, train_indices)
            validation_dataset = Subset(full_dataset, validation_indices)

        collator = CasePairCollator(tokenizer, max_length=max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
        )

        # ------------------------------------------------------------------
        # Class weights
        # ------------------------------------------------------------------
        if class_weight_positive is not None:
            weight_positive = class_weight_positive
        else:
            weight_positive = _compute_class_weight(train_dataset)

        class_weights = torch.tensor([1.0, weight_positive], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ------------------------------------------------------------------
        # Optimizer and scheduler
        # ------------------------------------------------------------------
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
        param_groups = [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not any(nd in name for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if any(nd in name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = _linear_warmup_decay_scheduler(
            optimizer, warmup_steps, total_steps
        )

        scaler: Any = None
        if mixed_precision and torch.cuda.is_available():
            scaler = torch.amp.GradScaler("cuda")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        best_metric = 0.0
        global_step = 0
        best_path = str(self._output_dir / "best.pt")
        final_path = str(self._output_dir / "last.pt")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for ids1, mask1, ids2, mask2, labels in train_loader:
                ids1 = ids1.to(device)
                mask1 = mask1.to(device)
                ids2 = ids2.to(device)
                mask2 = mask2.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(ids1, mask1, ids2, mask2)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(ids1, mask1, ids2, mask2)
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                if (
                    eval_every_steps > 0
                    and global_step % eval_every_steps == 0
                ):
                    metrics = _evaluate(model, validation_loader, device)
                    metric = metrics.get("auc", metrics.get("f1", 0.0))
                    logger.info(
                        "step %d  acc=%.4f  f1=%.4f  auc=%.4f",
                        global_step,
                        metrics["accuracy"],
                        metrics["f1"],
                        metrics.get("auc", 0.0),
                    )
                    if save_best and metric > best_metric:
                        best_metric = metric
                        _save_checkpoint(model, tokenizer, best_path)
                    model.train()

            avg_loss = epoch_loss / max(epoch_steps, 1)
            logger.info("epoch %d/%d  avg_loss=%.4f", epoch + 1, epochs, avg_loss)

        # ------------------------------------------------------------------
        # Final evaluation and save
        # ------------------------------------------------------------------
        final_metrics = _evaluate(model, validation_loader, device)
        _save_checkpoint(model, tokenizer, final_path)

        final_metric = final_metrics.get("auc", final_metrics.get("f1", 0.0))
        if save_best and final_metric > best_metric:
            best_metric = final_metric
            _save_checkpoint(model, tokenizer, best_path)

        if not Path(best_path).exists():
            _save_checkpoint(model, tokenizer, best_path)

        return TrainingResult(
            best_checkpoint_path=best_path,
            final_checkpoint_path=final_path,
            best_metric=best_metric,
            final_accuracy=final_metrics["accuracy"],
            final_f1=final_metrics["f1"],
            final_auc=final_metrics.get("auc", 0.0),
            epochs_trained=epochs,
            total_steps=global_step,
            training_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _stratified_split(
    dataset: Any, validation_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Perform a stratified train/validation split by label."""
    label_to_indices: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        label = sample["label"]
        label_to_indices.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        split_point = max(1, int(len(indices) * validation_ratio))
        validation_indices.extend(indices[:split_point])
        train_indices.extend(indices[split_point:])

    return train_indices, validation_indices


def _compute_class_weight(dataset: Any) -> float:
    """Compute positive class weight as negative_count / positive_count."""
    positive_count = 0
    negative_count = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["label"] == 1:
            positive_count += 1
        else:
            negative_count += 1
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def _linear_warmup_decay_scheduler(
    optimizer: Any, warmup_steps: int, total_steps: int
) -> Any:
    """Linear warmup then linear decay to zero."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        remaining = total_steps - current_step
        total_decay = total_steps - warmup_steps
        return max(0.0, remaining / max(total_decay, 1))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def _evaluate(
    model: Any, loader: Any, device: Any
) -> dict[str, float]:
    """Evaluate the model and return accuracy, F1, and AUC metrics."""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_scores: list[float] = []

    for ids1, mask1, ids2, mask2, labels in loader:
        ids1 = ids1.to(device)
        mask1 = mask1.to(device)
        ids2 = ids2.to(device)
        mask2 = mask2.to(device)

        logits = model(ids1, mask1, ids2, mask2)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_scores.extend(probabilities.cpu().tolist())

    if not all_labels:
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0}

    # Accuracy
    correct = sum(
        p == l for p, l in zip(all_predictions, all_labels)
    )
    accuracy = correct / len(all_labels)

    # F1 (binary, positive class = 1)
    true_positive = sum(
        p == 1 and l == 1 for p, l in zip(all_predictions, all_labels)
    )
    false_positive = sum(
        p == 1 and l == 0 for p, l in zip(all_predictions, all_labels)
    )
    false_negative = sum(
        p == 0 and l == 1 for p, l in zip(all_predictions, all_labels)
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ROC-AUC (simple trapezoidal implementation to avoid sklearn dep)
    metrics: dict[str, float] = {"accuracy": accuracy, "f1": f1}
    try:
        auc = _roc_auc(all_labels, all_scores)
        metrics["auc"] = auc
    except ValueError:
        pass

    return metrics


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC-AUC using the trapezoidal rule.

    Raises ValueError if only one class is present.
    """
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Only one class present in labels")

    paired = sorted(zip(scores, labels), reverse=True)
    true_positive_rate_prev = 0.0
    false_positive_rate_prev = 0.0
    true_positives = 0
    false_positives = 0
    auc = 0.0

    for _, label in paired:
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1
        true_positive_rate = true_positives / positive_count
        false_positive_rate = false_positives / negative_count
        auc += (false_positive_rate - false_positive_rate_prev) * (
            true_positive_rate + true_positive_rate_prev
        ) / 2.0
        true_positive_rate_prev = true_positive_rate
        false_positive_rate_prev = false_positive_rate

    return auc


def _save_checkpoint(model: Any, tokenizer: Any, path: str) -> None:
    """Save model weights and tokenizer to a directory."""
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))
    logger.info("Saved checkpoint to %s", path)
