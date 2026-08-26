"""Frozen backbone interfaces for the Phase 1 representation probe."""

from __future__ import annotations

from typing import Any

from nbv.features.base import (
    FeatureExtractorError,
    FrozenFeatureExtractor,
    FrozenFeatures,
)
from nbv.features.dinov2 import DINOv2Extractor
from nbv.features.imagenet_vit import ImageNetViTExtractor
from nbv.features.vggt import VGGTExtractor


def create_feature_extractor(name: str, **kwargs: Any) -> FrozenFeatureExtractor:
    """Construct one of the three step-5 backbones by a stable short name."""

    extractors = {
        "imagenet_vit": ImageNetViTExtractor,
        "dinov2": DINOv2Extractor,
        "vggt": VGGTExtractor,
    }
    try:
        extractor_class = extractors[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature extractor {name!r}; choose one of {tuple(extractors)}"
        ) from exc
    return extractor_class(**kwargs)


__all__ = [
    "DINOv2Extractor",
    "FeatureExtractorError",
    "FrozenFeatureExtractor",
    "FrozenFeatures",
    "ImageNetViTExtractor",
    "VGGTExtractor",
    "create_feature_extractor",
]
