"""Pixel-space diagnostics; these never alter visibility or camera geometry."""

from __future__ import annotations

import numpy as np


def silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate without wrapping; positive x/y mean right/down."""
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    if abs(dx) >= width or abs(dy) >= height:
        return shifted
    shifted[max(0, dy):min(height, height + dy), max(0, dx):min(width, width + dx)] = (
        mask[max(0, -dy):min(height, height - dy), max(0, -dx):min(width, width - dx)]
    )
    return shifted


def translation_scores(mask: np.ndarray, foreground: np.ndarray, radius: int = 3) -> list[dict]:
    """Post-hoc diagnostic search, excluding translations that crop foreground."""
    scores = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = shift_mask(mask, dx, dy)
            if shifted.sum() == mask.sum():
                scores.append({"dx": dx, "dy": dy, "iou": silhouette_iou(shifted, foreground)})
    return scores


def best_translation(scores: list[dict]) -> dict:
    return max(scores, key=lambda s: (s["iou"], -s["dx"] ** 2 - s["dy"] ** 2, -s["dy"], -s["dx"]))


def mask_extent(mask: np.ndarray) -> dict:
    y, x = np.nonzero(mask)
    return {
        "area_pixels": int(len(x)),
        "centroid_xy": [float(x.mean()), float(y.mean())] if len(x) else None,
        "bbox_xyxy_inclusive": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None,
    }


def boundary_distances(mask: np.ndarray, foreground: np.ndarray) -> dict:
    """Symmetric nearest-boundary distances, in RGB pixels (4-neighbor edge)."""
    def boundary(value):
        padded = np.pad(value, 1)
        interior = value & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        return np.argwhere(value & ~interior)

    mesh, rgb = boundary(mask), boundary(foreground)
    if not len(mesh) or not len(rgb):
        return {"mean_pixels": None, "p95_pixels": None, "within_one_pixel_fraction": None}
    # Bound temporary memory for larger images while retaining exact distances.
    def nearest(first, second):
        return np.concatenate([
            np.sqrt(((first[i:i + 128, None] - second[None]) ** 2).sum(axis=2).min(axis=1))
            for i in range(0, len(first), 128)
        ])
    distances = np.concatenate((nearest(mesh, rgb), nearest(rgb, mesh)))
    return {
        "mean_pixels": float(distances.mean()),
        "p95_pixels": float(np.percentile(distances, 95)),
        "within_one_pixel_fraction": float((distances <= 1).mean()),
    }
