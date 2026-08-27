"""Optimization loop for a lightweight head over pre-extracted features."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nbv.losses import masked_huber_loss


@dataclass(frozen=True, slots=True)
class ProbeFitResult:
    """Compact result of a tiny-subset probe fit."""

    initial_loss: float
    best_loss: float
    final_loss: float
    best_epoch: int
    epochs_completed: int

    @property
    def relative_loss_reduction(self) -> float:
        if self.initial_loss == 0.0:
            return 0.0
        return 1.0 - self.best_loss / self.initial_loss


def probe_loss(
    head: nn.Module,
    features: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
    *,
    huber_delta: float = 1.0,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate the full in-memory tiny subset without changing the head."""

    resolved_device = _resolve_device(device)
    head.eval()
    with torch.no_grad():
        predictions = head(features.to(resolved_device, dtype=torch.float32))
        loss = masked_huber_loss(
            predictions,
            targets.to(resolved_device, dtype=torch.float32),
            None if valid_mask is None else valid_mask.to(resolved_device),
            delta=huber_delta,
        )
    return float(loss.item())


def fit_probe(
    head: nn.Module,
    features: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    huber_delta: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> ProbeFitResult:
    """Fit only ``head`` to a small, already frozen feature tensor.

    Features remain on CPU between batches.  This lets the large frozen
    backbone be released before optimization and prevents accidental gradient
    flow or repeated backbone forwards during an overfit check.
    """

    _validate_training_inputs(features, targets, valid_mask)
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be greater than zero")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be non-negative")

    resolved_device = _resolve_device(device)
    head.to(resolved_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    initial_loss = probe_loss(
        head,
        features,
        targets,
        valid_mask,
        huber_delta=huber_delta,
        device=resolved_device,
    )
    best_loss = initial_loss
    best_epoch = 0
    best_state = deepcopy(head.state_dict())
    sample_count = features.shape[0]

    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            batch_features = features[indices].to(
                resolved_device, dtype=torch.float32
            )
            batch_targets = targets[indices].to(
                resolved_device, dtype=torch.float32
            )
            batch_mask = (
                None
                if valid_mask is None
                else valid_mask[indices].to(resolved_device)
            )

            optimizer.zero_grad(set_to_none=True)
            predictions = head(batch_features)
            loss = masked_huber_loss(
                predictions,
                batch_targets,
                batch_mask,
                delta=huber_delta,
            )
            loss.backward()
            optimizer.step()

        epoch_loss = probe_loss(
            head,
            features,
            targets,
            valid_mask,
            huber_delta=huber_delta,
            device=resolved_device,
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())

    final_loss = epoch_loss
    head.load_state_dict(best_state)
    return ProbeFitResult(
        initial_loss=initial_loss,
        best_loss=best_loss,
        final_loss=final_loss,
        best_epoch=best_epoch,
        epochs_completed=epochs,
    )


def _validate_training_inputs(
    features: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None,
) -> None:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("features must have non-empty shape [B, D]")
    if targets.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("targets must have shape [B, A] matching features")
    if not features.is_floating_point() or not targets.is_floating_point():
        raise TypeError("features and targets must be floating point")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if valid_mask is not None:
        if valid_mask.shape != targets.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with the target shape")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("every sample must contain a valid target")


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
