"""Trainable prediction modules for next-best-view experiments."""

from nbv.models.heads import LightweightProbeHead, count_trainable_parameters

__all__ = ["LightweightProbeHead", "count_trainable_parameters"]
