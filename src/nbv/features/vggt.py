"""Frozen VGGT aggregator feature extractor."""

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
from nbv.features.preprocessing import resize_square


class VGGTExtractor(FrozenFeatureExtractor):
    """Expose tokens from VGGT's frozen geometry-aware aggregator.

    Phase 1 processes each image independently, so this class deliberately
    inserts a one-frame sequence dimension before calling VGGT. Joint history
    processing belongs to Phase 3 and is not implemented here.
    """

    name = "vggt"

    def __init__(
        self,
        *,
        model_id: str = "facebook/VGGT-1B",
        image_size: int = 518,
        layer_index: int = -1,
        device: str | torch.device | None = None,
        autocast_dtype: torch.dtype | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[[str], nn.Module] | None = None,
    ) -> None:
        self.model_id = model_id
        self.image_size = image_size
        self.layer_index = layer_index
        if model is None:
            try:
                if model_loader is None:
                    from vggt.models.vggt import VGGT

                    model_loader = VGGT.from_pretrained
                full_model = model_loader(model_id)
                model = full_model.aggregator
            except (ImportError, ModuleNotFoundError) as exc:
                raise FeatureExtractorError(
                    "VGGTExtractor requires the official facebookresearch/vggt "
                    "package. Install it from its repository first."
                ) from exc
            except Exception as exc:
                raise FeatureExtractorError(
                    "Could not load the VGGT checkpoint. The first run needs "
                    "internet access and several GB of free disk space."
                ) from exc
        elif hasattr(model, "aggregator"):
            model = model.aggregator
        if not hasattr(model, "patch_start_idx"):
            raise FeatureExtractorError(
                "VGGT model must be an Aggregator or expose an .aggregator"
            )
        super().__init__(model, device=device, autocast_dtype=autocast_dtype)
        if autocast_dtype is None and self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16

    def _extract(self, images: Tensor, **kwargs: Any) -> FrozenFeatures:
        if kwargs:
            raise TypeError(f"Unexpected extraction options: {sorted(kwargs)}")
        inputs = resize_square(images, self.image_size).unsqueeze(1)
        token_layers, patch_start_idx = self.model(inputs)
        try:
            tokens = token_layers[self.layer_index]
        except IndexError as exc:
            raise FeatureExtractorError(
                f"VGGT layer_index {self.layer_index} is out of range"
            ) from exc
        if tokens is None:
            raise FeatureExtractorError(
                f"VGGT layer {self.layer_index} is not cached by this checkpoint"
            )
        if tokens.ndim != 4 or tokens.shape[1] != 1:
            raise FeatureExtractorError(
                "Single-image VGGT tokens must have shape [B, 1, N, D]"
            )
        tokens = tokens[:, 0]
        camera_tokens = tokens[:, :1]
        register_tokens = tokens[:, 1:patch_start_idx]
        patch_tokens = tokens[:, patch_start_idx:]
        if register_tokens.shape[1] == 0:
            register_tokens = None
        resolved_layer = self.layer_index % len(token_layers)
        return FrozenFeatures(
            pooled_patch=detached(patch_tokens.mean(dim=1)),
            patch_tokens=detached(patch_tokens),
            camera_tokens=detached(camera_tokens),
            register_tokens=detached(register_tokens),
            metadata={
                "backbone": self.name,
                "model_id": self.model_id,
                "input_size": [self.image_size, self.image_size],
                "layer": resolved_layer,
                "patch_start_idx": int(patch_start_idx),
                "history_mode": "single_image",
            },
        )
