"""Frozen supervised ImageNet ViT feature extractor."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from nbv.features.base import (
    FeatureExtractorError,
    FrozenFeatureExtractor,
    FrozenFeatures,
    detached,
)
from nbv.features.preprocessing import imagenet_normalize, resize_center_crop


class ImageNetViTExtractor(FrozenFeatureExtractor):
    """Extract final-layer tokens from torchvision's supervised ViT-B/16."""

    name = "imagenet_vit_b16"

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        pretrained: bool = True,
        autocast_dtype: torch.dtype | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[[bool], nn.Module] | None = None,
    ) -> None:
        if model is None:
            loader = model_loader or _load_torchvision_vit_b16
            model = loader(pretrained)
        _validate_torchvision_vit(model)
        super().__init__(model, device=device, autocast_dtype=autocast_dtype)

    def _extract(self, images: Tensor, **kwargs: Any) -> FrozenFeatures:
        if kwargs:
            raise TypeError(f"Unexpected extraction options: {sorted(kwargs)}")
        inputs = imagenet_normalize(
            resize_center_crop(images, resize_short_side=256, crop_size=224)
        )
        batch_size = inputs.shape[0]
        tokens = self.model._process_input(inputs)
        cls_token = self.model.class_token.expand(batch_size, -1, -1)
        tokens = self.model.encoder(torch.cat([cls_token, tokens], dim=1))
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return FrozenFeatures(
            pooled_patch=detached(patch_tokens.mean(dim=1)),
            patch_tokens=detached(patch_tokens),
            cls_token=detached(cls_token),
            metadata={
                "backbone": self.name,
                "input_size": [224, 224],
                "layer": "final",
            },
        )


def _load_torchvision_vit_b16(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:
        raise FeatureExtractorError(
            "ImageNetViTExtractor requires torchvision. Install the project "
            "requirements before constructing it."
        ) from exc
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    return vit_b_16(weights=weights)


def _validate_torchvision_vit(model: nn.Module) -> None:
    required = ("_process_input", "class_token", "encoder")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise FeatureExtractorError(
            f"ImageNet ViT model is missing required attributes: {missing}"
        )
