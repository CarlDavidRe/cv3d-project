"""Pairwise ranking loss for valid entries of Phase 1 target maps."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_pairwise_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
    *,
    minimum_target_margin: float = 0.0,
) -> Tensor:
    """Return logistic loss over valid, non-tied unordered anchor pairs.

    The raw target ordering is learned directly. Targets such as PSNR whose
    lower values mean higher uncertainty are converted only for evaluation.
    """

    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError(
            "predictions and targets must have the same shape [B, A]"
        )
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must be floating point")
    if not isinstance(minimum_target_margin, (int, float)) or isinstance(
        minimum_target_margin, bool
    ):
        raise TypeError("minimum_target_margin must be numeric")
    if float(minimum_target_margin) < 0.0:
        raise ValueError("minimum_target_margin must be non-negative")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must be finite")

    if valid_mask is None:
        mask = torch.ones_like(predictions, dtype=torch.bool)
    else:
        if valid_mask.shape != predictions.shape:
            raise ValueError(
                "valid_mask must have the same shape as predictions"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have boolean dtype")
        mask = valid_mask

    anchor_count = predictions.shape[1]
    pair_i, pair_j = torch.triu_indices(
        anchor_count,
        anchor_count,
        offset=1,
        device=predictions.device,
    )
    target_difference = targets[:, pair_i] - targets[:, pair_j]
    pair_mask = mask[:, pair_i] & mask[:, pair_j]
    pair_mask &= target_difference.abs() > float(minimum_target_margin)
    if not torch.any(pair_mask):
        return predictions.sum() * 0.0

    prediction_difference = predictions[:, pair_i] - predictions[:, pair_j]
    ordering = target_difference.sign()
    return F.softplus(
        -ordering[pair_mask] * prediction_difference[pair_mask]
    ).mean()
