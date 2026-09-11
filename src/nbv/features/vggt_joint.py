"""Frozen VGGT extraction for complete, variable-length image histories."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nbv.features.base import FeatureExtractorError, freeze_module, resolve_device
from nbv.features.model_cache import model_cache_directory
from nbv.features.selection import validate_feature_selection


@dataclass(frozen=True, slots=True)
class JointVGGTFeatures:
    """History-conditioned vectors or spatial tokens for every observation.

    ``view_features`` has shape ``[B, H, D]`` for pooled extraction or
    ``[B, H, K, D]`` for a reduced spatial token grid. Padded entries are
    exactly zero and are never passed through VGGT. Real entries come from a
    forward that contained every real image in that sample's history.
    """

    view_features: Tensor
    history_padding_mask: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.view_features.ndim not in (3, 4):
            raise FeatureExtractorError(
                "joint view_features must have shape [B, H, D] or [B, H, K, D]"
            )
        if (
            self.history_padding_mask.shape != self.view_features.shape[:2]
            or self.history_padding_mask.dtype != torch.bool
        ):
            raise FeatureExtractorError(
                "joint history_padding_mask must be boolean with shape [B, H]"
            )
        if self.view_features.requires_grad:
            raise FeatureExtractorError("frozen joint VGGT features must be detached")
        if not torch.isfinite(self.view_features).all():
            raise FeatureExtractorError("joint VGGT features must be finite")
        if torch.count_nonzero(self.view_features[self.history_padding_mask]).item():
            raise FeatureExtractorError("padded joint VGGT features must be zero")

    @property
    def feature_dim(self) -> int:
        return int(self.view_features.shape[-1])


class VGGTJointExtractor:
    """Run each complete history through one frozen VGGT aggregator forward.

    VGGT does not expose an observation-padding mask. Mixed-length minibatches
    are therefore partitioned by real history length. Each partition is sent
    to the backbone without padded frames, then restored to the padded batch
    layout. Histories of the same length remain batched together.
    """

    name = "vggt"
    history_mode = "joint_multiview"

    def __init__(
        self,
        *,
        feature_components: Sequence[str] = ("max_pooled_patch",),
        model_id: str = "facebook/VGGT-1B",
        image_size: int = 518,
        layer_index: int = -1,
        expected_feature_dim: int | None = None,
        spatial_token_grid_size: int | None = None,
        device: str | torch.device | None = None,
        autocast_dtype: torch.dtype | None = None,
        model_cache_root: str | Path | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
    ) -> None:
        self.feature_components = validate_feature_selection(
            "vggt", feature_components
        )
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError("image_size must be a positive integer")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if expected_feature_dim is not None and (
            isinstance(expected_feature_dim, bool)
            or not isinstance(expected_feature_dim, int)
            or expected_feature_dim < 1
        ):
            raise ValueError("expected_feature_dim must be a positive integer or null")
        self.model_id = model_id
        self.image_size = image_size
        self.layer_index = layer_index
        self.expected_feature_dim = expected_feature_dim
        if spatial_token_grid_size is not None and (
            isinstance(spatial_token_grid_size, bool)
            or not isinstance(spatial_token_grid_size, int)
            or spatial_token_grid_size < 1
        ):
            raise ValueError("spatial_token_grid_size must be a positive integer or null")
        self.spatial_token_grid_size = spatial_token_grid_size
        self.device = resolve_device(device)
        self.autocast_dtype = autocast_dtype
        if model is None:
            try:
                if model_loader is None:
                    from vggt.models.vggt import VGGT

                    model_loader = VGGT.from_pretrained
                if model_cache_root is None:
                    full_model = model_loader(model_id)
                else:
                    full_model = model_loader(
                        model_id,
                        cache_dir=model_cache_directory(
                            model_cache_root, "huggingface"
                        ),
                    )
                model = full_model.aggregator
            except (ImportError, ModuleNotFoundError) as exc:
                raise FeatureExtractorError(
                    "VGGTJointExtractor requires the official "
                    "facebookresearch/vggt package. Install it from its "
                    "repository first."
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
        self.model = freeze_module(model).to(self.device)
        if autocast_dtype is None and self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16

    def extract(
        self, history_images: Tensor, history_padding_mask: Tensor
    ) -> JointVGGTFeatures:
        """Extract joint features while excluding every padded observation."""

        images, padding = _validate_history_images(
            history_images, history_padding_mask
        )
        images = images.to(self.device, non_blocking=True)
        padding = padding.to(self.device, non_blocking=True)
        lengths = (~padding).sum(dim=1)
        output: Tensor | None = None
        resolved_layer: int | None = None
        patch_start_idx: int | None = None
        forward_lengths: list[int] = []
        self.model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            for length_tensor in torch.unique(lengths, sorted=True):
                length = int(length_tensor.item())
                rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
                real_images = images[rows, :length]
                flattened = real_images.flatten(0, 1)
                resized = F.interpolate(
                    flattened,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).reshape(rows.numel(), length, 3, self.image_size, self.image_size)
                token_layers, current_patch_start = self.model(resized)
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
                expected_prefix = (rows.numel(), length)
                if tokens.ndim != 4 or tuple(tokens.shape[:2]) != expected_prefix:
                    raise FeatureExtractorError(
                        "Joint VGGT tokens must have shape [B, H, N, D]"
                    )
                selected = (
                    _select_joint_spatial_tokens(
                        tokens,
                        int(current_patch_start),
                        self.spatial_token_grid_size,
                    )
                    if self.spatial_token_grid_size is not None
                    else _select_joint_components(
                        tokens, int(current_patch_start), self.feature_components
                    )
                )
                if output is None:
                    if (
                        self.expected_feature_dim is not None
                        and selected.shape[-1] != self.expected_feature_dim
                    ):
                        raise FeatureExtractorError(
                            "Joint VGGT selected feature dimension differs from "
                            f"expected_feature_dim={self.expected_feature_dim}: "
                            f"got {selected.shape[-1]}"
                        )
                    output = torch.zeros(
                        (*images.shape[:2], *selected.shape[2:]),
                        dtype=selected.dtype,
                        device=selected.device,
                    )
                    resolved_layer = self.layer_index % len(token_layers)
                    patch_start_idx = int(current_patch_start)
                elif (
                    tuple(selected.shape[2:]) != tuple(output.shape[2:])
                    or int(current_patch_start) != patch_start_idx
                    or self.layer_index % len(token_layers) != resolved_layer
                ):
                    raise FeatureExtractorError(
                        "VGGT token layout changed between history-length groups"
                    )
                output[rows, :length] = selected
                forward_lengths.extend([length] * int(rows.numel()))
        if output is None or resolved_layer is None or patch_start_idx is None:
            raise FeatureExtractorError("joint extraction produced no features")
        return JointVGGTFeatures(
            view_features=output.detach(),
            history_padding_mask=padding.detach(),
            metadata={
                "backbone": self.name,
                "model_id": self.model_id,
                "input_size": [self.image_size, self.image_size],
                "layer": resolved_layer,
                "patch_start_idx": patch_start_idx,
                "feature_components": list(self.feature_components),
                "representation": (
                    "spatial_patch_tokens"
                    if self.spatial_token_grid_size is not None
                    else "pooled_view_components"
                ),
                "spatial_token_grid_size": self.spatial_token_grid_size,
                "tokens_per_view": (
                    None
                    if self.spatial_token_grid_size is None
                    else self.spatial_token_grid_size ** 2
                ),
                "history_mode": self.history_mode,
                "padding_strategy": "group_by_real_history_length",
                "forward_history_lengths": sorted(forward_lengths),
                "backbone_cross_view_interaction": True,
            },
        )

    __call__ = extract


def _validate_history_images(
    history_images: Tensor, history_padding_mask: Tensor
) -> tuple[Tensor, Tensor]:
    if not isinstance(history_images, Tensor):
        raise TypeError("history_images must be a torch.Tensor")
    if history_images.ndim != 5 or history_images.shape[2] != 3:
        raise ValueError("history_images must have shape [B, H, 3, image_H, image_W]")
    if min(history_images.shape[0], history_images.shape[1], history_images.shape[3], history_images.shape[4]) < 1:
        raise ValueError("history_images must have non-empty dimensions")
    if (
        history_padding_mask.shape != history_images.shape[:2]
        or history_padding_mask.dtype != torch.bool
    ):
        raise ValueError("history_padding_mask must be boolean with shape [B, H]")
    if torch.any(history_padding_mask.all(dim=1)):
        raise ValueError("every history must contain at least one real observation")
    padding = history_padding_mask.to(history_images.device)
    lengths = (~padding).sum(dim=1)
    positions = torch.arange(history_images.shape[1], device=padding.device)
    expected_padding = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(padding, expected_padding):
        raise ValueError("real history observations must precede suffix padding")
    images = history_images.to(dtype=torch.float32).contiguous()
    real = images[~padding]
    if not torch.isfinite(real).all():
        raise ValueError("real history images contain NaN or infinite values")
    minimum, maximum = real.aminmax()
    if minimum.item() < 0.0 or maximum.item() > 1.0:
        raise ValueError("real history images must contain RGB values in [0, 1]")
    return images, padding.contiguous()


def _select_joint_components(
    tokens: Tensor, patch_start_idx: int, components: Sequence[str]
) -> Tensor:
    if patch_start_idx < 1 or patch_start_idx >= tokens.shape[2]:
        raise FeatureExtractorError("VGGT patch_start_idx is incompatible with tokens")
    camera = tokens[:, :, :1]
    registers = tokens[:, :, 1:patch_start_idx]
    patches = tokens[:, :, patch_start_idx:]
    selected: list[Tensor] = []
    for component in components:
        if component == "pooled_patch":
            value = patches.mean(dim=2)
        elif component == "max_pooled_patch":
            value = patches.amax(dim=2)
        elif component == "pooled_camera":
            value = camera.mean(dim=2)
        elif component == "pooled_register":
            if registers.shape[2] == 0:
                raise FeatureExtractorError(
                    "Selected pooled_register but VGGT exposes no register tokens"
                )
            value = registers.mean(dim=2)
        else:  # validated during construction
            raise FeatureExtractorError(f"Unknown joint VGGT component {component!r}")
        selected.append(value)
    result = torch.cat(selected, dim=-1)
    if result.ndim != 3 or tuple(result.shape[:2]) != tuple(tokens.shape[:2]):
        raise FeatureExtractorError("selected joint features must have shape [B, H, D]")
    return result


def _select_joint_spatial_tokens(
    tokens: Tensor,
    patch_start_idx: int,
    grid_size: int,
) -> Tensor:
    """Reduce the square patch grid while retaining multiple spatial tokens."""

    if patch_start_idx < 1 or patch_start_idx >= tokens.shape[2]:
        raise FeatureExtractorError("VGGT patch_start_idx is incompatible with tokens")
    patches = tokens[:, :, patch_start_idx:]
    patch_count = int(patches.shape[2])
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise FeatureExtractorError(
            "spatial token extraction requires a square VGGT patch grid"
        )
    if grid_size > side:
        raise FeatureExtractorError(
            "spatial_token_grid_size cannot exceed the VGGT patch-grid side"
        )
    batch, history, _, dimension = patches.shape
    spatial = patches.reshape(batch * history, side, side, dimension).permute(0, 3, 1, 2)
    reduced = F.adaptive_avg_pool2d(spatial, (grid_size, grid_size))
    return reduced.permute(0, 2, 3, 1).reshape(
        batch, history, grid_size * grid_size, dimension
    )


__all__ = ["JointVGGTFeatures", "VGGTJointExtractor"]
