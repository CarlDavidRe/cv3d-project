"""Training and one-step evaluation for Phase 3 history gain models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nbv.data.history_features import (
    HistoryFeatureBatch,
    HistoryFeatureDataset,
    collate_history_features,
)
from nbv.data.history_dataset import (
    HistoryBatch,
    HistoryDataset,
    collate_history_samples,
)
from nbv.eval.metrics import ndcg_at_k, normalized_regret, spearman_rank
from nbv.training.phase1 import combined_probe_loss


@dataclass(frozen=True, slots=True)
class HistoryFitResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: tuple[Mapping[str, float | int | None], ...]


@dataclass(frozen=True, slots=True)
class HistoryEvaluationResult:
    summary: Mapping[str, float | int | None]
    per_sample: tuple[Mapping[str, Any], ...]
    predictions: Tensor


def fit_history_model(
    model: nn.Module,
    train: HistoryFeatureDataset | HistoryDataset,
    validation: HistoryFeatureDataset | HistoryDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    huber_delta: float = 1.0,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
    ndcg_k: int = 5,
    patience: int | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    logger: logging.Logger | None = None,
    gradient_accumulation_steps: int = 1,
) -> HistoryFitResult:
    """Train on direct surface gain and restore the best validation state."""

    _validate_datasets(train, validation)
    _positive_integer(epochs, "epochs")
    _positive_integer(batch_size, "batch_size")
    if patience is not None:
        _positive_integer(patience, "patience")
    _finite_number(learning_rate, "learning_rate", minimum=0.0, strict=True)
    _finite_number(weight_decay, "weight_decay", minimum=0.0)
    _finite_number(huber_delta, "huber_delta", minimum=0.0, strict=True)
    _finite_number(ranking_weight, "ranking_weight", minimum=0.0)
    _finite_number(ranking_margin, "ranking_margin", minimum=0.0)
    _positive_integer(ndcg_k, "ndcg_k")
    _positive_integer(gradient_accumulation_steps, "gradient_accumulation_steps")
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = _loader(
        train, batch_size=batch_size, shuffle=True, generator=generator
    )
    evaluation_kwargs = dict(
        batch_size=batch_size,
        huber_delta=huber_delta,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        ndcg_k=ndcg_k,
        device=resolved_device,
    )
    initial_train = _mean_loss_components(model, train, **evaluation_kwargs)
    initial_validation = evaluate_history_model(model, validation, **evaluation_kwargs)
    best_validation_loss = float(initial_validation.summary["loss"])
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    history: list[Mapping[str, float | int | None]] = [
        _history_row(0, None, float(optimizer.param_groups[0]["lr"]), initial_train, initial_validation, ndcg_k)
    ]
    if logger is not None:
        logger.info(
            "Epoch 0/%d baseline: train loss %.6f, validation loss %.6f, "
            "regret %s, NDCG@%d %s",
            epochs,
            initial_train["loss"],
            best_validation_loss,
            _format_metric(initial_validation.summary["normalized_regret_mean"]),
            ndcg_k,
            _format_metric(initial_validation.summary[f"ndcg_at_{ndcg_k}_mean"]),
        )
    without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        sample_count = 0
        accumulated_batches = 0
        accumulated_samples = 0
        optimizer.zero_grad(set_to_none=True)
        for cpu_batch in train_loader:
            batch = cpu_batch.to(resolved_device)
            predictions = _predict(model, batch)
            total, _, _ = combined_probe_loss(
                predictions,
                batch.target_surface_gain,
                batch.valid_candidate_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            count = len(batch.sample_ids)
            (total * count).backward()
            accumulated_batches += 1
            accumulated_samples += count
            if accumulated_batches == gradient_accumulation_steps:
                _optimizer_step(optimizer, model, accumulated_samples)
                accumulated_batches = 0
                accumulated_samples = 0
            weighted_loss += float(total.item()) * count
            sample_count += count
        if accumulated_batches:
            _optimizer_step(optimizer, model, accumulated_samples)
        optimization_loss = weighted_loss / sample_count
        train_result = _mean_loss_components(model, train, **evaluation_kwargs)
        validation_result = evaluate_history_model(model, validation, **evaluation_kwargs)
        validation_loss = float(validation_result.summary["loss"])
        history.append(
            _history_row(
                epoch,
                optimization_loss,
                float(optimizer.param_groups[0]["lr"]),
                train_result,
                validation_result,
                ndcg_k,
            )
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            without_improvement = 0
            improved = True
        else:
            without_improvement += 1
            improved = False
        if logger is not None:
            logger.info(
                "Epoch %d/%d complete: optimization loss %.6f, train loss %.6f, "
                "validation loss %.6f, regret %s, NDCG@%d %s, "
                "best epoch %d%s%s",
                epoch,
                epochs,
                optimization_loss,
                train_result["loss"],
                validation_loss,
                _format_metric(validation_result.summary["normalized_regret_mean"]),
                ndcg_k,
                _format_metric(validation_result.summary[f"ndcg_at_{ndcg_k}_mean"]),
                best_epoch,
                " (improved)" if improved else "",
                (
                    ""
                    if patience is None
                    else f", early-stop wait {without_improvement}/{patience}"
                ),
            )
        if patience is not None and without_improvement >= patience:
            if logger is not None:
                logger.info(
                    "Early stopping after epoch %d; restoring epoch %d",
                    epoch,
                    best_epoch,
                )
            break

    model.load_state_dict(best_state)
    return HistoryFitResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=epoch,
        history=tuple(history),
    )


def evaluate_history_model(
    model: nn.Module,
    dataset: HistoryFeatureDataset | HistoryDataset,
    *,
    batch_size: int,
    huber_delta: float = 1.0,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
    ndcg_k: int = 5,
    device: str | torch.device = "cpu",
) -> HistoryEvaluationResult:
    """Evaluate raw higher-is-better predictions against direct gain labels."""

    _positive_integer(batch_size, "batch_size")
    _positive_integer(ndcg_k, "ndcg_k")
    resolved_device = _resolve_device(device)
    model.to(resolved_device).eval()
    prediction_batches: list[Tensor] = []
    totals = np.zeros(3, dtype=np.float64)
    sample_count = 0
    per_sample: list[Mapping[str, Any]] = []
    regrets: list[float] = []
    correlations: list[float] = []
    ndcgs: list[float] = []
    with torch.inference_mode():
        for cpu_batch in _loader(dataset, batch_size=batch_size, shuffle=False):
            batch = cpu_batch.to(resolved_device)
            predictions = _predict(model, batch)
            total, huber, ranking = combined_probe_loss(
                predictions,
                batch.target_surface_gain,
                batch.valid_candidate_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            count = len(batch.sample_ids)
            totals += np.asarray([total.item(), huber.item(), ranking.item()]) * count
            sample_count += count
            cpu_predictions = predictions.detach().cpu().float()
            prediction_batches.append(cpu_predictions)
            for row, sample_id in enumerate(cpu_batch.sample_ids):
                prediction = cpu_predictions[row].numpy()
                target = cpu_batch.target_surface_gain[row].numpy()
                valid = cpu_batch.valid_candidate_mask[row].numpy()
                regret = normalized_regret(prediction, target, valid)
                correlation = spearman_rank(prediction, target, valid)
                ndcg = ndcg_at_k(prediction, target, valid, k=ndcg_k)
                regrets.append(regret)
                correlations.append(correlation)
                ndcgs.append(ndcg)
                valid_ids = np.flatnonzero(valid)
                length = int(cpu_batch.history_lengths[row])
                per_sample.append({
                    "sample_id": sample_id,
                    "object_id": cpu_batch.object_ids[row],
                    "history_length": length,
                    "history_anchor_ids": cpu_batch.history_anchor_ids[row, :length].tolist(),
                    "predicted_anchor_id": int(valid_ids[np.argmax(prediction[valid])]),
                    "target_anchor_id": int(valid_ids[np.argmax(target[valid])]),
                    "normalized_regret": regret,
                    "spearman": _finite_or_none(correlation),
                    f"ndcg_at_{ndcg_k}": ndcg,
                })
    predictions = torch.cat(prediction_batches)
    summary: dict[str, float | int | None] = {
        "num_samples": sample_count,
        "loss": float(totals[0] / sample_count),
        "huber_loss": float(totals[1] / sample_count),
        "ranking_loss": float(totals[2] / sample_count),
        "normalized_regret_mean": _mean_finite(regrets),
        "normalized_regret_valid_count": len(regrets),
        "spearman_mean": _mean_finite(correlations),
        "spearman_valid_count": sum(math.isfinite(value) for value in correlations),
        f"ndcg_at_{ndcg_k}_mean": _mean_finite(ndcgs),
        f"ndcg_at_{ndcg_k}_valid_count": len(ndcgs),
    }
    return HistoryEvaluationResult(summary, tuple(per_sample), predictions)


def _mean_loss_components(
    model: nn.Module,
    dataset: HistoryFeatureDataset | HistoryDataset,
    *,
    batch_size: int,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
    ndcg_k: int,
    device: str | torch.device,
) -> Mapping[str, float]:
    del ndcg_k
    resolved = _resolve_device(device)
    model.to(resolved).eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for cpu_batch in _loader(dataset, batch_size=batch_size, shuffle=False):
            batch = cpu_batch.to(resolved)
            losses = combined_probe_loss(
                _predict(model, batch),
                batch.target_surface_gain,
                batch.valid_candidate_mask,
                huber_delta=huber_delta,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            size = len(batch.sample_ids)
            totals += np.asarray([loss.item() for loss in losses]) * size
            count += size
    return {
        "loss": float(totals[0] / count),
        "huber_loss": float(totals[1] / count),
        "ranking_loss": float(totals[2] / count),
    }


def _predict(model: nn.Module, batch: HistoryFeatureBatch | HistoryBatch) -> Tensor:
    if isinstance(batch, HistoryFeatureBatch):
        predictions = model(
            batch.history_features,
            batch.history_anchor_ids,
            batch.history_padding_mask,
        )
    else:
        if batch.history_images is None:
            raise ValueError("joint history training requires loaded RGB images")
        predictions = model(
            batch.history_images,
            batch.history_anchor_ids,
            batch.history_padding_mask,
        )
    expected = (len(batch.sample_ids), batch.target_surface_gain.shape[1])
    if predictions.shape != expected:
        raise ValueError(f"history model must return {expected}, got {tuple(predictions.shape)}")
    if not torch.isfinite(predictions).all():
        raise ValueError("history model returned non-finite predictions")
    return predictions


def _history_row(
    epoch: int,
    optimization_loss: float | None,
    learning_rate: float,
    train: Mapping[str, float],
    validation: HistoryEvaluationResult,
    ndcg_k: int,
) -> Mapping[str, float | int | None]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "optimization_train_loss": optimization_loss,
        "train_loss": train["loss"],
        "train_huber_loss": train["huber_loss"],
        "train_ranking_loss": train["ranking_loss"],
        "validation_loss": validation.summary["loss"],
        "validation_huber_loss": validation.summary["huber_loss"],
        "validation_ranking_loss": validation.summary["ranking_loss"],
        "validation_normalized_regret_mean": validation.summary["normalized_regret_mean"],
        "validation_spearman_mean": validation.summary["spearman_mean"],
        f"validation_ndcg_at_{ndcg_k}_mean": validation.summary[f"ndcg_at_{ndcg_k}_mean"],
    }


def _loader(
    dataset: HistoryFeatureDataset | HistoryDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    collate = (
        collate_history_features
        if isinstance(dataset, HistoryFeatureDataset)
        else collate_history_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )


def _validate_datasets(
    train: HistoryFeatureDataset | HistoryDataset,
    validation: HistoryFeatureDataset | HistoryDataset,
) -> None:
    if len(train) < 1 or len(validation) < 1:
        raise ValueError("training and validation histories must be non-empty")
    if type(train) is not type(validation):
        raise TypeError("training and validation must use the same history dataset type")
    if isinstance(train, HistoryFeatureDataset):
        if train.lookup.feature_dim != validation.lookup.feature_dim:
            raise ValueError("training and validation feature dimensions differ")
        train_histories = train.histories
        validation_histories = validation.histories
    else:
        if not train.load_images or not validation.load_images:
            raise ValueError("joint history datasets must load RGB images")
        train_histories = train
        validation_histories = validation
    if train_histories.manifest["dataset_id"] != validation_histories.manifest["dataset_id"]:
        raise ValueError("training and validation histories must share one dataset manifest")


def _optimizer_step(
    optimizer: torch.optim.Optimizer, model: nn.Module, sample_count: int
) -> None:
    if sample_count < 1:
        raise ValueError("optimizer accumulation must contain at least one sample")
    inverse = 1.0 / sample_count
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


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


def _finite_number(value: object, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {qualifier} {minimum}")
    return number


def _mean_finite(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _format_metric(value: float | int | None) -> str:
    return "undefined" if value is None else f"{float(value):.4f}"


__all__ = [
    "HistoryEvaluationResult",
    "HistoryFitResult",
    "evaluate_history_model",
    "fit_history_model",
]
