"""Trainable prediction modules for next-best-view experiments."""

from nbv.models.heads import (
    FixedMapHead,
    LightweightProbeHead,
    count_trainable_parameters,
)

__all__ = [
    "FixedMapHead",
    "LightweightProbeHead",
    "count_trainable_parameters",
]
