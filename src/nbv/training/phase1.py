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
    """Best validation checkpoint information and per-epoch diagnostics."""

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


def valid_target_mean(targets: Tensor, valid_mask: Tensor) -> Tensor:
    """Return the per-anchor training mean using only valid observations."""

    if targets.ndim != 2 or targets.shape[0] < 1:
        raise ValueError("targets must have non-empty shape [N, A]")
    if valid_mask.shape != targets.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with the target shape")
    if not targets.is_floating_point() or not torch.isfinite(targets).all():
        raise ValueError("targets must be finite floating-point values")
    valid_counts = valid_mask.sum(dim=0)
    valid_sums = torch.where(
        valid_mask, targets, torch.zeros_like(targets)
    ).sum(dim=0)
    means = valid_sums / valid_counts.clamp_min(1)
    return torch.where(valid_counts > 0, means, torch.zeros_like(means))


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
    target_direction: TargetDirection = "higher",
    ndcg_k: int = 5,
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
    if target_direction not in ("higher", "lower"):
        raise ValueError("target_direction must be 'higher' or 'lower'")
    _positive_integer(ndcg_k, "ndcg_k")

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

    evaluation_kwargs = {
        "batch_size": batch_size,
        "target_direction": target_direction,
        "huber_delta": huber_delta,
        "ranking_weight": ranking_weight,
        "ranking_margin": ranking_margin,
        "ndcg_k": ndcg_k,
        "device": resolved_device,
    }
    initial_train = _mean_loss_components(
        head,
        train,
        batch_size=batch_size,
        huber_delta=huber_delta,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        device=resolved_device,
    )
    initial_validation = evaluate_phase1_probe(
        head,
        validation,
        **evaluation_kwargs,
    )
    best_validation_loss = float(initial_validation.summary["loss"])
    best_epoch = 0
    best_state = deepcopy(head.state_dict())
    history: list[Mapping[str, float | int | None]] = [
        _history_row(
            epoch=0,
            optimization_train_loss=None,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            train=initial_train,
            validation=initial_validation,
            ndcg_k=ndcg_k,
        )
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

        optimization_train_loss = total_training_loss / sample_count
        train_result = _mean_loss_components(
            head,
            train,
            batch_size=batch_size,
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            ranking_margin=ranking_margin,
            device=resolved_device,
        )
        validation_result = evaluate_phase1_probe(
            head,
            validation,
            **evaluation_kwargs,
        )
        validation_loss = float(validation_result.summary["loss"])
        history.append(
            _history_row(
                epoch=epoch,
                optimization_train_loss=optimization_train_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                train=train_result,
                validation=validation_result,
                ndcg_k=ndcg_k,
            )
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


def _history_row(
    *,
    epoch: int,
    optimization_train_loss: float | None,
    learning_rate: float,
    train: Mapping[str, float],
    validation: Phase1EvaluationResult,
    ndcg_k: int,
) -> Mapping[str, float | int | None]:
    """Build one stable, JSON-ready training-history record."""

    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "optimization_train_loss": optimization_train_loss,
        "train_loss": train["loss"],
        "train_huber_loss": train["huber_loss"],
        "train_ranking_loss": train["ranking_loss"],
        "validation_loss": validation.summary["loss"],
        "validation_huber_loss": validation.summary["huber_loss"],
        "validation_ranking_loss": validation.summary["ranking_loss"],
        "validation_normalized_regret_mean": validation.summary[
            "normalized_regret_mean"
        ],
        "validation_spearman_mean": validation.summary["spearman_mean"],
        f"validation_ndcg_at_{ndcg_k}_mean": validation.summary[
            f"ndcg_at_{ndcg_k}_mean"
        ],
    }


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


def _mean_loss_components(
    head: nn.Module,
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
    device: torch.device,
) -> Mapping[str, float]:
    """Evaluate comparable total and component losses without ranking metrics."""

    head.eval()
    weighted_total = 0.0
    weighted_huber = 0.0
    weighted_ranking = 0.0
    sample_count = 0
    with torch.inference_mode():
        for features, targets, valid_mask in _loader(
            dataset, batch_size=batch_size, shuffle=False
        ):
            features = features.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
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
    return {
        "loss": weighted_total / sample_count,
        "huber_loss": weighted_huber / sample_count,
        "ranking_loss": weighted_ranking / sample_count,
    }


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
