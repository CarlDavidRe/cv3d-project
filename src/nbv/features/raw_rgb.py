"""Backbone-free, fixed-size RGB features for the Phase 1 MLP baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nbv.features.base import FrozenFeatureExtractor, FrozenFeatures, detached


class RawRGBExtractor(FrozenFeatureExtractor):
    """Flatten a coarse RGB grid without applying a learned image backbone."""

    name = "raw_rgb"

    def __init__(
        self,
        *,
        output_size: int = 16,
        device: str | torch.device | None = None,
    ) -> None:
        if (
            isinstance(output_size, bool)
            or not isinstance(output_size, int)
            or output_size < 1
        ):
            raise ValueError("output_size must be a positive integer")
        self.output_size = output_size
        super().__init__(nn.Identity(), device=device)

    def _extract(self, images: Tensor, **kwargs: Any) -> FrozenFeatures:
        if kwargs:
            raise TypeError(f"Unexpected extraction options: {sorted(kwargs)}")
        pooled = F.adaptive_avg_pool2d(
            images, (self.output_size, self.output_size)
        )
        flattened = pooled.flatten(start_dim=1)
        # The common container requires a token axis. This single synthetic
        # token is the complete flattened RGB vector consumed by the MLP.
        tokens = flattened.unsqueeze(1)
        return FrozenFeatures(
            pooled_patch=detached(flattened),
            patch_tokens=detached(tokens),
            metadata={
                "backbone": self.name,
                "input_range": [0.0, 1.0],
                "pooling": "adaptive_average",
                "output_size": [self.output_size, self.output_size],
                "feature_dim": 3 * self.output_size * self.output_size,
            },
        )
