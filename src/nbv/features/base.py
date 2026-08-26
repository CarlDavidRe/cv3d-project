"""Common contract for frozen image feature extractors.

The Phase 1 dataset supplies RGB arrays in ``[0, 1]`` with shape ``[3, H, W]``.
Extractors accept a batch of those arrays (or an equivalent torch tensor), do
their own backbone-specific resizing and normalization, and return the same
structured token representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


class FeatureExtractorError(RuntimeError):
    """Raised when a backbone cannot be loaded or returns invalid features."""


@dataclass(frozen=True, slots=True)
class FrozenFeatures:
    """Backbone features with a stable, model-independent layout.

    ``pooled_patch`` and ``patch_tokens`` are always present. Special tokens
    are optional because not every architecture defines them. All tensors use
    batch-first shapes and are detached from autograd.
    """

    pooled_patch: Tensor
    patch_tokens: Tensor
    cls_token: Tensor | None = None
    camera_tokens: Tensor | None = None
    register_tokens: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch, token_count, dimension = _shape3(
            self.patch_tokens, "patch_tokens"
        )
        if token_count < 1:
            raise FeatureExtractorError("patch_tokens must contain at least one token")
        if tuple(self.pooled_patch.shape) != (batch, dimension):
            raise FeatureExtractorError(
                "pooled_patch must have shape [B, D] matching patch_tokens"
            )
        for name in ("cls_token", "camera_tokens", "register_tokens"):
            value = getattr(self, name)
            if value is None:
                continue
            if value.ndim == 2 and name == "cls_token":
                if tuple(value.shape) != (batch, dimension):
                    raise FeatureExtractorError(
                        f"{name} must have shape [B, D] matching patch_tokens"
                    )
            elif (
                value.ndim != 3
                or value.shape[0] != batch
                or value.shape[2] != dimension
            ):
                raise FeatureExtractorError(
                    f"{name} must have shape [B, N, D] matching patch_tokens"
                )
        for tensor_name, tensor in self.tensors().items():
            if tensor.requires_grad:
                raise FeatureExtractorError(
                    f"Frozen feature {tensor_name!r} must be detached"
                )

    @property
    def batch_size(self) -> int:
        return int(self.patch_tokens.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.patch_tokens.shape[-1])

    def tensors(self) -> dict[str, Tensor]:
        """Return only the tensor-valued fields, useful for caching later."""

        values = {
            "pooled_patch": self.pooled_patch,
            "patch_tokens": self.patch_tokens,
            "cls_token": self.cls_token,
            "camera_tokens": self.camera_tokens,
            "register_tokens": self.register_tokens,
        }
        return {name: value for name, value in values.items() if value is not None}


class FrozenFeatureExtractor(ABC):
    """Inference-only base class shared by all Phase 1 backbones."""

    name: str

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.autocast_dtype = autocast_dtype
        self.model = freeze_module(model).to(self.device)

    def extract(self, images: Tensor | np.ndarray, **kwargs: Any) -> FrozenFeatures:
        """Extract detached features from RGB images in ``[0, 1]``.

        Accepted input shapes are ``[3, H, W]`` and ``[B, 3, H, W]``.
        Backbone parameters remain frozen even if external code changed the
        module's training mode between calls.
        """

        batch = as_rgb_batch(images).to(self.device, non_blocking=True)
        self.model.eval()
        with torch.inference_mode(), _autocast_context(
            self.device, self.autocast_dtype
        ):
            features = self._extract(batch, **kwargs)
        if not isinstance(features, FrozenFeatures):
            raise FeatureExtractorError(
                f"{type(self).__name__}._extract must return FrozenFeatures"
            )
        return features

    __call__ = extract

    @abstractmethod
    def _extract(self, images: Tensor, **kwargs: Any) -> FrozenFeatures:
        """Run the backbone after common input validation."""


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable every parameter gradient."""

    module.requires_grad_(False)
    module.eval()
    return module


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise FeatureExtractorError("CUDA was requested but is not available")
    return resolved


def as_rgb_batch(images: Tensor | np.ndarray) -> Tensor:
    """Convert an image batch to contiguous float32 ``[B, 3, H, W]``."""

    if isinstance(images, np.ndarray):
        batch = torch.from_numpy(np.ascontiguousarray(images))
    elif isinstance(images, Tensor):
        batch = images
    else:
        raise TypeError("images must be a torch.Tensor or numpy.ndarray")
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError("images must have shape [3, H, W] or [B, 3, H, W]")
    if batch.shape[0] < 1 or batch.shape[2] < 1 or batch.shape[3] < 1:
        raise ValueError("images must have non-empty batch and spatial dimensions")
    batch = batch.to(dtype=torch.float32).contiguous()
    if not torch.isfinite(batch).all():
        raise ValueError("images contain NaN or infinite values")
    minimum, maximum = batch.aminmax()
    if minimum.item() < 0.0 or maximum.item() > 1.0:
        raise ValueError("images must contain RGB values in [0, 1]")
    return batch


def detached(tensor: Tensor | None) -> Tensor | None:
    """Detach a feature while preserving its inference dtype and device."""

    return None if tensor is None else tensor.detach()


def _shape3(tensor: Tensor, name: str) -> tuple[int, int, int]:
    if tensor.ndim != 3:
        raise FeatureExtractorError(f"{name} must have shape [B, N, D]")
    return tuple(int(value) for value in tensor.shape)


def _autocast_context(
    device: torch.device, dtype: torch.dtype | None
) -> torch.autocast:
    enabled = dtype is not None
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
