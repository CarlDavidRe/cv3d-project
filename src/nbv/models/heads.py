"""Small heads used to probe frozen image representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LightweightProbeHead(nn.Module):
    """A deliberately small MLP mapping one image feature to anchor scores.

    The input is a single pooled frozen feature per image.  Keeping the head
    independent of the backbone makes its capacity and optimization behavior
    easy to compare across the Phase 1 representations.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_anchors: int = 48,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("num_anchors", num_anchors),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout must be numeric")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        self.dropout = float(dropout)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_anchors),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return one score for every anchor, with shape ``[B, A]``."""

        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [B, {self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        return self.network(features)


def count_trainable_parameters(module: nn.Module) -> int:
    """Count scalar parameters that an optimizer is allowed to update."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
