"""Full-split optimization and evaluation for Phase 1 probe heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from nbv.eval import ndcg_at_k, normalized_regret, spearman_rank
from nbv.features import CachedFeatureDataset
from nbv.losses import masked_huber_loss, masked_pairwise_ranking_loss


TargetDirection = Literal["higher", "lower"]


@dataclass(frozen=True, slots=True)
class Phase1FitResult:
    """Best validation checkpoint information and per-epoch losses."""

    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: tuple[Mapping[str, float | int | None], ...]


@dataclass(frozen=True, slots=True)
class Phase1EvaluationResult:
    """Aggregate and per-sample metrics from one feature split."""

    summary: Mapping[str, float | int | None]
    per_sample: tuple[Mapping[str, Any], ...]
    predictions: Tensor


def combined_probe_loss(
    predictions: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None,
    *,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, Huber, and pairwise-ranking losses."""

    if not isinstance(ranking_weight, (int, float)) or isinstance(
        ranking_weight, bool
    ):
        raise TypeError("ranking_weight must be numeric")
    if float(ranking_weight) < 0.0:
        raise ValueError("ranking_weight must be non-negative")
    huber = masked_huber_loss(
        predictions,
        targets,
        valid_mask,
        delta=huber_delta,
    )
    if float(ranking_weight) == 0.0:
        ranking = predictions.sum() * 0.0
    else:
        ranking = masked_pairwise_ranking_loss(
            predictions,
            targets,
            valid_mask,
            minimum_target_margin=ranking_margin,
        )
    return huber + float(ranking_weight) * ranking, huber, ranking


def fit_phase1_probe(
    head: nn.Module,
    train: CachedFeatureDataset,
    validation: CachedFeatureDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    huber_delta: float = 1.0,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
    patience: int | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Phase1FitResult:
    """Fit only the lightweight head and restore its best validation state."""

    _validate_compatible_splits(train, validation)
    _positive_integer(epochs, "epochs")
    _positive_integer(batch_size, "batch_size")
    if patience is not None:
        _positive_integer(patience, "patience")
    _finite_number(learning_rate, "learning_rate", minimum=0.0, strict=True)
    _finite_number(weight_decay, "weight_decay", minimum=0.0)
    _finite_number(ranking_weight, "ranking_weight", minimum=0.0)
    _finite_number(ranking_margin, "ranking_margin", minimum=0.0)

    resolved_device = _resolve_device(device)
    head.to(resolved_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = _loader(train, batch_size=batch_size, shuffle=True, generator=generator)

    initial_validation = _mean_loss(
        head,
        validation,
        batch_size=batch_size,
        huber_delta=huber_delta,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        device=resolved_device,
    )
    best_validation_loss = initial_validation
    best_epoch = 0
    best_state = deepcopy(head.state_dict())
    history: list[Mapping[str, float | int | None]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": initial_validation,
        }
    ]
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_training_loss = 0.0
        sample_count = 0
        for features, targets, valid_mask in loader:
            features = features.to(
                resolved_device, dtype=torch.float32, non_blocking=True
            )
            targets = targets.to(
                resolved_device, dtype=torch.float32, non_blocking=True
            )
            valid_mask = valid_mask.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = head(features)
            loss, _, _ = combined_probe_loss(
                predictions,
                targets,
                valid_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            loss.backward()
            optimizer.step()
            total_training_loss += float(loss.item()) * features.shape[0]
            sample_count += features.shape[0]

        train_loss = total_training_loss / sample_count
        validation_loss = _mean_loss(
            head,
            validation,
            batch_size=batch_size,
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            ranking_margin=ranking_margin,
            device=resolved_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience is not None and epochs_without_improvement >= patience:
            break

    head.load_state_dict(best_state)
    return Phase1FitResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=epoch,
        history=tuple(history),
    )


def evaluate_phase1_probe(
    head: nn.Module,
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    target_direction: TargetDirection,
    huber_delta: float = 1.0,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
    ndcg_k: int = 5,
    device: str | torch.device = "cpu",
) -> Phase1EvaluationResult:
    """Predict a cached split and compute the common one-step metrics."""

    _positive_integer(batch_size, "batch_size")
    if target_direction not in ("higher", "lower"):
        raise ValueError("target_direction must be 'higher' or 'lower'")
    resolved_device = _resolve_device(device)
    head.to(resolved_device)
    head.eval()
    prediction_batches: list[Tensor] = []
    weighted_total = 0.0
    weighted_huber = 0.0
    weighted_ranking = 0.0
    sample_count = 0
    with torch.inference_mode():
        for features, targets, valid_mask in _loader(
            dataset, batch_size=batch_size, shuffle=False
        ):
            features = features.to(
                resolved_device, dtype=torch.float32, non_blocking=True
            )
            targets = targets.to(
                resolved_device, dtype=torch.float32, non_blocking=True
            )
            valid_mask = valid_mask.to(resolved_device, non_blocking=True)
            predictions = head(features)
            total, huber, ranking = combined_probe_loss(
                predictions,
                targets,
                valid_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            count = features.shape[0]
            weighted_total += float(total.item()) * count
            weighted_huber += float(huber.item()) * count
            weighted_ranking += float(ranking.item()) * count
            sample_count += count
            prediction_batches.append(predictions.detach().cpu().float())

    predictions = torch.cat(prediction_batches)
    targets_array = dataset.targets.cpu().float().numpy()
    predictions_array = predictions.numpy()
    masks_array = dataset.valid_mask.cpu().numpy()
    per_sample: list[Mapping[str, Any]] = []
    regrets: list[float] = []
    correlations: list[float] = []
    ndcgs: list[float] = []
    sign = 1.0 if target_direction == "higher" else -1.0
    for index, (prediction, target, mask) in enumerate(
        zip(predictions_array, targets_array, masks_array, strict=True)
    ):
        scores = sign * prediction
        relevance = _normalized_relevance(target, mask, target_direction)
        regret = normalized_regret(scores, relevance, mask)
        correlation = spearman_rank(scores, relevance, mask)
        ndcg = ndcg_at_k(scores, relevance, mask, k=ndcg_k)
        regrets.append(regret)
        correlations.append(correlation)
        ndcgs.append(ndcg)
        valid_indices = np.flatnonzero(mask)
        predicted_local_anchor = int(valid_indices[np.argmax(scores[mask])])
        target_local_anchor = int(valid_indices[np.argmax(relevance[mask])])
        per_sample.append(
            {
                "sample_id": dataset.sample_ids[index],
                "source_anchor_id": int(dataset.source_anchor_ids[index]),
                "predicted_local_anchor_id": predicted_local_anchor,
                "target_local_anchor_id": target_local_anchor,
                "normalized_regret": regret,
                "spearman": _finite_or_none(correlation),
                f"ndcg_at_{ndcg_k}": ndcg,
            }
        )

    summary: dict[str, float | int | None] = {
        "num_samples": len(dataset),
        "loss": weighted_total / sample_count,
        "huber_loss": weighted_huber / sample_count,
        "ranking_loss": weighted_ranking / sample_count,
        "normalized_regret_mean": _mean_finite(regrets),
        "spearman_mean": _mean_finite(correlations),
        f"ndcg_at_{ndcg_k}_mean": _mean_finite(ndcgs),
    }
    return Phase1EvaluationResult(
        summary=summary,
        per_sample=tuple(per_sample),
        predictions=predictions,
    )


def _mean_loss(
    head: nn.Module,
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
    device: torch.device,
) -> float:
    head.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for features, targets, valid_mask in _loader(
            dataset, batch_size=batch_size, shuffle=False
        ):
            features = features.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            predictions = head(features)
            loss, _, _ = combined_probe_loss(
                predictions,
                targets,
                valid_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            weighted_loss += float(loss.item()) * features.shape[0]
            sample_count += features.shape[0]
    return weighted_loss / sample_count


def _loader(
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    tensors = TensorDataset(dataset.features, dataset.targets, dataset.valid_mask)
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _normalized_relevance(
    target: np.ndarray,
    valid_mask: np.ndarray,
    direction: TargetDirection,
) -> np.ndarray:
    oriented = target.astype(np.float64, copy=True)
    if direction == "lower":
        oriented *= -1.0
    valid = oriented[valid_mask]
    minimum = float(np.min(valid))
    scale = float(np.max(valid)) - minimum
    if scale == 0.0:
        return np.zeros_like(oriented)
    return (oriented - minimum) / scale


def _validate_compatible_splits(
    train: CachedFeatureDataset,
    validation: CachedFeatureDataset,
) -> None:
    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError("train and validation feature widths must match")
    if train.targets.shape[1] != validation.targets.shape[1]:
        raise ValueError("train and validation anchor counts must match")


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    invalid = number <= minimum if strict else number < minimum
    if invalid:
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}")
    return number


def _mean_finite(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
