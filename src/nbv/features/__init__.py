"""Frozen backbone interfaces for the Phase 1 representation probe."""

from __future__ import annotations

from typing import Any

from nbv.features.base import (
    FeatureExtractorError,
    FrozenFeatureExtractor,
    FrozenFeatures,
)
from nbv.features.cache import (
    FEATURE_CACHE_SCHEMA_VERSION,
    CachedFeatureDataset,
    FeatureCacheError,
    cache_fingerprint,
    feature_cache_path,
    load_feature_cache,
    save_feature_cache,
)
from nbv.features.dinov2 import DINOv2Extractor
from nbv.features.imagenet_vit import ImageNetViTExtractor
from nbv.features.raw_rgb import RawRGBExtractor
from nbv.features.selection import (
    FEATURE_COMPONENTS_BY_BACKBONE,
    FeatureSelectionError,
    select_feature_components,
    validate_feature_selection,
)
from nbv.features.vggt import VGGTExtractor
from nbv.features.vggt_joint import JointVGGTFeatures, VGGTJointExtractor


def create_feature_extractor(name: str, **kwargs: Any) -> FrozenFeatureExtractor:
    """Construct one of the three frozen backbones by a stable short name."""

    extractors = {
        "raw_rgb": RawRGBExtractor,
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
    "CachedFeatureDataset",
    "DINOv2Extractor",
    "FEATURE_CACHE_SCHEMA_VERSION",
    "FeatureExtractorError",
    "FeatureCacheError",
    "FEATURE_COMPONENTS_BY_BACKBONE",
    "FeatureSelectionError",
    "FrozenFeatureExtractor",
    "FrozenFeatures",
    "ImageNetViTExtractor",
    "RawRGBExtractor",
    "VGGTExtractor",
    "JointVGGTFeatures",
    "VGGTJointExtractor",
    "cache_fingerprint",
    "create_feature_extractor",
    "feature_cache_path",
    "load_feature_cache",
    "save_feature_cache",
    "select_feature_components",
    "validate_feature_selection",
]
