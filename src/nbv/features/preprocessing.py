"""Tensor-only preprocessing shared by frozen feature extractors."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_center_crop(
    images: Tensor, *, resize_short_side: int, crop_size: int
) -> Tensor:
    """Resize the shorter side, then take a deterministic square center crop."""

    height, width = images.shape[-2:]
    scale = resize_short_side / min(height, width)
    resized_height = max(resize_short_side, round(height * scale))
    resized_width = max(resize_short_side, round(width * scale))
    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - crop_size) // 2
    left = (resized_width - crop_size) // 2
    return resized[..., top : top + crop_size, left : left + crop_size]


def resize_square(images: Tensor, size: int) -> Tensor:
    """Resize to a square; NUM renders are square so this preserves aspect ratio."""

    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def imagenet_normalize(images: Tensor) -> Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std
