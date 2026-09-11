"""Training and one-step evaluation for Phase 3 history gain models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nbv.data.history_features import (
    HistoryFeatureBatch,
    HistoryFeatureDataset,
    MaterializedHistoryFeatureDataset,
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
    training_checkpoint_path: str | Path | None = None,
    checkpoint_every_batches: int | None = None,
    resume: bool = False,
    checkpoint_identity: Mapping[str, Any] | None = None,
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
    if checkpoint_every_batches is not None:
        _positive_integer(checkpoint_every_batches, "checkpoint_every_batches")
    if resume and training_checkpoint_path is None:
        raise ValueError("resume requires training_checkpoint_path")
    if checkpoint_every_batches is not None and training_checkpoint_path is None:
        raise ValueError(
            "checkpoint_every_batches requires training_checkpoint_path"
        )
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = _loader(
        train, batch_size=batch_size, shuffle=True, generator=generator
    )
    checkpoint_path = (
        None if training_checkpoint_path is None else Path(training_checkpoint_path)
    )
    training_configuration = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "huber_delta": float(huber_delta),
        "ranking_weight": float(ranking_weight),
        "ranking_margin": float(ranking_margin),
        "ndcg_k": ndcg_k,
        "patience": patience,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    evaluation_kwargs = dict(
        batch_size=batch_size,
        huber_delta=huber_delta,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        ndcg_k=ndcg_k,
        device=resolved_device,
    )
    if resume:
        assert checkpoint_path is not None
        resumed = _load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            generator=generator,
            training_configuration=training_configuration,
            checkpoint_identity=checkpoint_identity,
            device=resolved_device,
        )
        best_validation_loss = resumed["best_validation_loss"]
        best_epoch = resumed["best_epoch"]
        best_state = resumed["best_state"]
        history = resumed["history"]
        without_improvement = resumed["without_improvement"]
        epochs_completed = resumed["epochs_completed"]
        resume_epoch = resumed["epoch_in_progress"]
        resume_batches = resumed["batches_completed"]
        resume_weighted_loss = resumed["weighted_loss"]
        resume_sample_count = resumed["sample_count"]
        resume_epoch_generator_state = resumed["epoch_generator_state"]
        if logger is not None:
            location = (
                f"epoch {resume_epoch}, batch {resume_batches}/{len(train_loader)}"
                if resume_epoch is not None
                else f"after epoch {epochs_completed}"
            )
            logger.info("Resumed training checkpoint %s from %s", checkpoint_path, location)
    else:
        initial_train = _mean_loss_components(
            model,
            train,
            logger=logger,
            progress_label=f"Epoch 0/{epochs} train evaluation",
            **evaluation_kwargs,
        )
        initial_validation = evaluate_history_model(
            model,
            validation,
            logger=logger,
            progress_label=f"Epoch 0/{epochs} validation evaluation",
            **evaluation_kwargs,
        )
        best_validation_loss = float(initial_validation.summary["loss"])
        best_epoch = 0
        best_state = deepcopy(model.state_dict())
        history = [
            _history_row(
                0,
                None,
                float(optimizer.param_groups[0]["lr"]),
                initial_train,
                initial_validation,
                ndcg_k,
            )
        ]
        without_improvement = 0
        epochs_completed = 0
        resume_epoch = None
        resume_batches = 0
        resume_weighted_loss = 0.0
        resume_sample_count = 0
        resume_epoch_generator_state = None
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
        if checkpoint_path is not None:
            _save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                training_configuration=training_configuration,
                checkpoint_identity=checkpoint_identity,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                best_state=best_state,
                history=history,
                without_improvement=without_improvement,
                epochs_completed=epochs_completed,
                epoch_in_progress=None,
                batches_completed=0,
                weighted_loss=0.0,
                sample_count=0,
                epoch_generator_state=None,
            )

    start_epoch = resume_epoch if resume_epoch is not None else epochs_completed + 1
    if patience is not None and without_improvement >= patience:
        start_epoch = epochs + 1
        if logger is not None:
            logger.info(
                "Training checkpoint had already early-stopped after epoch %d",
                epochs_completed,
            )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        continuing_epoch = resume_epoch == epoch
        weighted_loss = resume_weighted_loss if continuing_epoch else 0.0
        sample_count = resume_sample_count if continuing_epoch else 0
        completed_batches = resume_batches if continuing_epoch else 0
        accumulated_batches = 0
        accumulated_samples = 0
        total_batches = len(train_loader)
        if continuing_epoch:
            if resume_epoch_generator_state is None:
                raise ValueError("resume checkpoint lacks the epoch shuffle state")
            generator.set_state(resume_epoch_generator_state)
            epoch_generator_state = resume_epoch_generator_state
        else:
            epoch_generator_state = generator.get_state().clone()
        last_progress_log = time.monotonic()
        last_checkpoint_batch = completed_batches
        optimizer.zero_grad(set_to_none=True)
        for batch_index, cpu_batch in enumerate(train_loader, start=1):
            if batch_index <= completed_batches:
                continue
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
            if logger is not None and _progress_due(
                batch_index, total_batches, last_progress_log
            ):
                logger.info(
                    "Epoch %d/%d optimization: batch %d/%d (%.1f%%), "
                    "%d/%d samples, running loss %.6f",
                    epoch,
                    epochs,
                    batch_index,
                    total_batches,
                    100.0 * batch_index / total_batches,
                    sample_count,
                    len(train),
                    weighted_loss / sample_count,
                )
                last_progress_log = time.monotonic()
            if (
                checkpoint_path is not None
                and checkpoint_every_batches is not None
                and accumulated_batches == 0
                and batch_index - last_checkpoint_batch >= checkpoint_every_batches
            ):
                _save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    training_configuration=training_configuration,
                    checkpoint_identity=checkpoint_identity,
                    best_validation_loss=best_validation_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    history=history,
                    without_improvement=without_improvement,
                    epochs_completed=epoch - 1,
                    epoch_in_progress=epoch,
                    batches_completed=batch_index,
                    weighted_loss=weighted_loss,
                    sample_count=sample_count,
                    epoch_generator_state=epoch_generator_state,
                )
                last_checkpoint_batch = batch_index
                if logger is not None:
                    logger.info(
                        "Saved resumable training checkpoint %s at epoch %d/%d, "
                        "batch %d/%d",
                        checkpoint_path,
                        epoch,
                        epochs,
                        batch_index,
                        total_batches,
                    )
        if accumulated_batches:
            _optimizer_step(optimizer, model, accumulated_samples)
        if checkpoint_path is not None and last_checkpoint_batch != total_batches:
            _save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                training_configuration=training_configuration,
                checkpoint_identity=checkpoint_identity,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                best_state=best_state,
                history=history,
                without_improvement=without_improvement,
                epochs_completed=epoch - 1,
                epoch_in_progress=epoch,
                batches_completed=total_batches,
                weighted_loss=weighted_loss,
                sample_count=sample_count,
                epoch_generator_state=epoch_generator_state,
            )
            if logger is not None:
                logger.info(
                    "Saved resumable training checkpoint %s after epoch %d optimization",
                    checkpoint_path,
                    epoch,
                )
        optimization_loss = weighted_loss / sample_count
        train_result = _mean_loss_components(
            model,
            train,
            logger=logger,
            progress_label=f"Epoch {epoch}/{epochs} train evaluation",
            **evaluation_kwargs,
        )
        validation_result = evaluate_history_model(
            model,
            validation,
            logger=logger,
            progress_label=f"Epoch {epoch}/{epochs} validation evaluation",
            **evaluation_kwargs,
        )
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
        epochs_completed = epoch
        if checkpoint_path is not None:
            _save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                training_configuration=training_configuration,
                checkpoint_identity=checkpoint_identity,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                best_state=best_state,
                history=history,
                without_improvement=without_improvement,
                epochs_completed=epochs_completed,
                epoch_in_progress=None,
                batches_completed=0,
                weighted_loss=0.0,
                sample_count=0,
                epoch_generator_state=None,
            )
            if logger is not None:
                logger.info(
                    "Saved resumable training checkpoint %s after epoch %d/%d",
                    checkpoint_path,
                    epoch,
                    epochs,
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
        epochs_completed=epochs_completed,
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
    logger: logging.Logger | None = None,
    progress_label: str | None = None,
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
    loader = _loader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)
    last_progress_log = time.monotonic()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader, start=1):
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
            if logger is not None and progress_label is not None and _progress_due(
                batch_index, total_batches, last_progress_log
            ):
                logger.info(
                    "%s: batch %d/%d (%.1f%%), %d/%d samples, running loss %.6f",
                    progress_label,
                    batch_index,
                    total_batches,
                    100.0 * batch_index / total_batches,
                    sample_count,
                    len(dataset),
                    totals[0] / sample_count,
                )
                last_progress_log = time.monotonic()
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
    logger: logging.Logger | None = None,
    progress_label: str | None = None,
) -> Mapping[str, float]:
    del ndcg_k
    resolved = _resolve_device(device)
    model.to(resolved).eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    loader = _loader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)
    last_progress_log = time.monotonic()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader, start=1):
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
            if logger is not None and progress_label is not None and _progress_due(
                batch_index, total_batches, last_progress_log
            ):
                logger.info(
                    "%s: batch %d/%d (%.1f%%), %d/%d samples, running loss %.6f",
                    progress_label,
                    batch_index,
                    total_batches,
                    100.0 * batch_index / total_batches,
                    count,
                    len(dataset),
                    totals[0] / count,
                )
                last_progress_log = time.monotonic()
    return {
        "loss": float(totals[0] / count),
        "huber_loss": float(totals[1] / count),
        "ranking_loss": float(totals[2] / count),
    }


def _predict(model: nn.Module, batch: HistoryFeatureBatch | HistoryBatch) -> Tensor:
    if isinstance(batch, HistoryFeatureBatch):
        forward_features = getattr(model, "forward_features", None)
        if callable(forward_features):
            predictions = forward_features(
                batch.history_features,
                batch.history_anchor_ids,
                batch.history_padding_mask,
            )
        else:
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
        train_shape = (
            train.feature_shape
            if isinstance(train, MaterializedHistoryFeatureDataset)
            else (train.lookup.feature_dim,)
        )
        validation_shape = (
            validation.feature_shape
            if isinstance(validation, MaterializedHistoryFeatureDataset)
            else (validation.lookup.feature_dim,)
        )
        if train_shape != validation_shape:
            raise ValueError("training and validation feature shapes differ")
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


def _save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    training_configuration: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any] | None,
    best_validation_loss: float,
    best_epoch: int,
    best_state: Mapping[str, Tensor],
    history: list[Mapping[str, float | int | None]],
    without_improvement: int,
    epochs_completed: int,
    epoch_in_progress: int | None,
    batches_completed: int,
    weighted_loss: float,
    sample_count: int,
    epoch_generator_state: Tensor | None,
) -> None:
    """Atomically persist everything needed to continue a training pass."""

    payload = {
        "schema_version": 1,
        "checkpoint_type": "phase3_history_training_state",
        "training_configuration": dict(training_configuration),
        "checkpoint_identity": dict(checkpoint_identity or {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "best_state_dict": dict(best_state),
        "history": [dict(row) for row in history],
        "without_improvement": without_improvement,
        "epochs_completed": epochs_completed,
        "epoch_in_progress": epoch_in_progress,
        "batches_completed": batches_completed,
        "weighted_loss": weighted_loss,
        "sample_count": sample_count,
        "generator_state": generator.get_state(),
        "epoch_generator_state": epoch_generator_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    training_configuration: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    """Validate and restore a resumable Phase 3 training state."""

    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    # Keep serialized RNG states on the CPU.  Loading the whole checkpoint
    # directly onto CUDA also moves ``cuda_rng_state_all`` there, but
    # ``torch.cuda.set_rng_state_all`` requires CPU ByteTensors.  Module and
    # optimizer state loading below moves their tensors to the parameter
    # device as needed.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("checkpoint_type") != "phase3_history_training_state"
    ):
        raise ValueError("unsupported Phase 3 training checkpoint")
    if payload.get("training_configuration") != dict(training_configuration):
        raise ValueError("resume checkpoint training configuration differs")
    if payload.get("checkpoint_identity") != dict(checkpoint_identity or {}):
        raise ValueError("resume checkpoint experiment identity differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload.get("cuda_rng_state_all", [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("resume checkpoint CUDA RNG state is incompatible")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    epoch_in_progress = payload["epoch_in_progress"]
    epoch_generator_state = payload["epoch_generator_state"]
    if epoch_in_progress is None:
        generator.set_state(payload["generator_state"].cpu())
    elif epoch_generator_state is None:
        raise ValueError("resume checkpoint lacks the active epoch shuffle state")
    return {
        "best_validation_loss": float(payload["best_validation_loss"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_state": payload["best_state_dict"],
        "history": [dict(row) for row in payload["history"]],
        "without_improvement": int(payload["without_improvement"]),
        "epochs_completed": int(payload["epochs_completed"]),
        "epoch_in_progress": (
            None if epoch_in_progress is None else int(epoch_in_progress)
        ),
        "batches_completed": int(payload["batches_completed"]),
        "weighted_loss": float(payload["weighted_loss"]),
        "sample_count": int(payload["sample_count"]),
        "epoch_generator_state": (
            None if epoch_generator_state is None else epoch_generator_state.cpu()
        ),
    }


def _progress_due(batch_index: int, total_batches: int, last_log: float) -> bool:
    """Report early, at roughly 10% increments, and at least once per minute."""

    interval = max(1, math.ceil(total_batches / 10))
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % interval == 0
        or time.monotonic() - last_log >= 60.0
    )


__all__ = [
    "HistoryEvaluationResult",
    "HistoryFitResult",
    "evaluate_history_model",
    "fit_history_model",
]
