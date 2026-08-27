"""Huber regression for partially valid anchor maps."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Average Huber loss over valid entries of equally shaped maps.

    ``valid_mask=None`` includes every entry.  A supplied mask must have the
    same shape as the predictions and contain at least one valid value.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.ndim != 2:
        raise ValueError("predictions and targets must have shape [B, A]")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must be floating point")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        raise TypeError("delta must be numeric")
    if float(delta) <= 0.0:
        raise ValueError("delta must be greater than zero")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must be finite")

    losses = F.huber_loss(
        predictions,
        targets,
        reduction="none",
        delta=float(delta),
    )
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != predictions.shape:
        raise ValueError("valid_mask must have the same shape as predictions")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have boolean dtype")
    if not torch.any(valid_mask):
        raise ValueError("valid_mask must contain at least one valid entry")
    return losses.masked_select(valid_mask).mean()
