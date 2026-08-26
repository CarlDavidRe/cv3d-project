"""Frozen DINOv2 feature extractor using the official torch.hub models."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from nbv.features.base import (
    FeatureExtractorError,
    FrozenFeatureExtractor,
    FrozenFeatures,
    detached,
)
from nbv.features.preprocessing import imagenet_normalize, resize_center_crop


class DINOv2Extractor(FrozenFeatureExtractor):
    """Extract CLS, register, and patch tokens from a frozen DINOv2 model."""

    name = "dinov2"

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vitb14",
        repo_or_dir: str = "facebookresearch/dinov2",
        source: str = "github",
        device: str | torch.device | None = None,
        pretrained: bool = True,
        autocast_dtype: torch.dtype | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            loader = model_loader or torch.hub.load
            try:
                model = loader(
                    repo_or_dir,
                    model_name,
                    source=source,
                    pretrained=pretrained,
                    trust_repo=True,
                )
            except Exception as exc:
                raise FeatureExtractorError(
                    "Could not load DINOv2. The first run needs internet access "
                    "for the official repository and checkpoint, or pass a local "
                    "repo_or_dir with source='local'."
                ) from exc
        if not hasattr(model, "forward_features"):
            raise FeatureExtractorError("DINOv2 model must expose forward_features")
        super().__init__(model, device=device, autocast_dtype=autocast_dtype)

    def _extract(self, images: Tensor, **kwargs: Any) -> FrozenFeatures:
        if kwargs:
            raise TypeError(f"Unexpected extraction options: {sorted(kwargs)}")
        inputs = imagenet_normalize(
            resize_center_crop(images, resize_short_side=256, crop_size=224)
        )
        raw = self.model.forward_features(inputs)
        if not isinstance(raw, Mapping):
            raise FeatureExtractorError(
                "DINOv2 forward_features must return a token mapping"
            )
        try:
            patch_tokens = raw["x_norm_patchtokens"]
            cls_token = raw["x_norm_clstoken"]
        except KeyError as exc:
            raise FeatureExtractorError(
                f"DINOv2 output is missing required token {exc.args[0]!r}"
            ) from exc
        register_tokens = raw.get("x_norm_regtokens")
        if register_tokens is not None and register_tokens.shape[1] == 0:
            register_tokens = None
        return FrozenFeatures(
            pooled_patch=detached(patch_tokens.mean(dim=1)),
            patch_tokens=detached(patch_tokens),
            cls_token=detached(cls_token),
            register_tokens=detached(register_tokens),
            metadata={
                "backbone": self.name,
                "model_name": self.model_name,
                "input_size": [224, 224],
                "layer": "final",
            },
        )
