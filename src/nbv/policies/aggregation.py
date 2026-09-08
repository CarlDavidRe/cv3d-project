"""PUN-compatible source-frame alignment and history aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


PUN_ALIGNMENT = "official_minimal_rotation_exp30_interpolation_v1"
PUN_AGGREGATION = "official_small_filter_then_raw_product_psnr_argmin_v1"


@dataclass(frozen=True, slots=True)
class PUNAggregationResult:
    """Intermediate raw quantities and final higher-is-better policy scores."""

    aligned_maps: np.ndarray
    normalized_maps: np.ndarray
    combined_raw_psnr: np.ndarray
    kept_candidate_mask: np.ndarray
    policy_scores: np.ndarray
    filter_fallback_used: bool


def pun_relative_directions(
    anchor_directions: np.ndarray, source_direction: np.ndarray
) -> np.ndarray:
    """Rotate anchor zero to a source direction like official PUN.

    This deliberately retains the official antipodal special case: when the
    cross-product magnitude is exactly zero, no rotation is applied.
    """

    directions = _directions(anchor_directions)
    target = np.asarray(source_direction, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("source_direction must be a finite [3] vector")
    norm = float(np.linalg.norm(target))
    if norm == 0:
        raise ValueError("source_direction must be nonzero")
    target = target / norm
    first = directions[0] / np.linalg.norm(directions[0])
    axis = np.cross(first, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(first, target))
    if sine == 0:
        return directions.copy()
    skew = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / sine**2)
    return directions @ rotation.T


def align_pun_relative_map(
    relative_map: np.ndarray,
    *,
    source_direction: np.ndarray,
    anchor_directions: np.ndarray,
    candidate_directions: np.ndarray | None = None,
    interpolation_degrees: float = 30.0,
) -> np.ndarray:
    """Interpolate one source-relative map into a common candidate frame."""

    values = np.asarray(relative_map, dtype=np.float64)
    directions = _directions(anchor_directions)
    candidates = directions if candidate_directions is None else _directions(
        candidate_directions
    )
    if values.shape != (directions.shape[0],) or not np.isfinite(values).all():
        raise ValueError("relative_map must be one finite value per anchor")
    if (
        not math.isfinite(interpolation_degrees)
        or interpolation_degrees <= 0
        or interpolation_degrees > 180
    ):
        raise ValueError("interpolation_degrees must be in (0, 180]")

    relative_directions = pun_relative_directions(directions, source_direction)
    cosine = np.clip(candidates @ relative_directions.T, -1.0, 1.0)
    angles = np.arccos(cosine)
    nearby = angles < math.radians(interpolation_degrees)
    if not nearby.any(axis=1).all():
        raise ValueError("PUN interpolation has a candidate with no nearby anchor")
    weights = np.exp(-angles) * nearby
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values


def align_pun_history(
    relative_maps: np.ndarray,
    source_anchor_ids: tuple[int, ...] | list[int] | np.ndarray,
    anchor_directions: np.ndarray,
    *,
    interpolation_degrees: float = 30.0,
) -> np.ndarray:
    """Align independently predicted maps in acquired-history order."""

    maps = np.asarray(relative_maps, dtype=np.float64)
    directions = _directions(anchor_directions)
    source_ids = np.asarray(source_anchor_ids)
    if maps.ndim != 2 or maps.shape[1] != directions.shape[0] or maps.shape[0] < 1:
        raise ValueError("relative_maps must have shape [history, anchors]")
    if source_ids.shape != (maps.shape[0],) or source_ids.dtype.kind not in "iu":
        raise ValueError("source_anchor_ids must contain one integer per map")
    if np.any(source_ids < 0) or np.any(source_ids >= directions.shape[0]):
        raise ValueError("source_anchor_ids contains an out-of-range anchor")
    return np.stack(
        [
            align_pun_relative_map(
                relative_map,
                source_direction=directions[int(source_id)],
                anchor_directions=directions,
                interpolation_degrees=interpolation_degrees,
            )
            for relative_map, source_id in zip(maps, source_ids, strict=True)
        ]
    )


def aggregate_pun_psnr(
    aligned_maps: np.ndarray,
    valid_candidate_mask: np.ndarray,
    *,
    suppression_threshold: float = 0.1,
) -> PUNAggregationResult:
    """Apply official ``small`` suppression and raw-product PSNR selection.

    Official PUN minimizes the product. The returned common-interface scores
    negate that product so the shared evaluator can continue to use argmax.
    """

    maps = np.asarray(aligned_maps, dtype=np.float64)
    valid = np.asarray(valid_candidate_mask)
    if maps.ndim != 2 or maps.shape[0] < 1 or maps.shape[1] != 48:
        raise ValueError("aligned_maps must have shape [history, 48]")
    if not np.isfinite(maps).all():
        raise ValueError("aligned_maps must be finite")
    if valid.shape != (48,) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("valid_candidate_mask must be boolean [48] with a candidate")
    if (
        not math.isfinite(suppression_threshold)
        or suppression_threshold < 0
        or suppression_threshold >= 1
    ):
        raise ValueError("suppression_threshold must be in [0, 1)")

    normalized = np.zeros_like(maps)
    for index, row in enumerate(maps):
        minimum = float(row[valid].min())
        maximum = float(row[valid].max())
        if maximum > minimum:
            normalized[index, valid] = (row[valid] - minimum) / (maximum - minimum)
    suppressed = valid & np.any(
        normalized >= (1.0 - suppression_threshold), axis=0
    )
    kept = valid & ~suppressed
    fallback = not kept.any()
    if fallback:
        kept = valid.copy()

    combined = np.prod(maps, axis=0)
    if not np.isfinite(combined).all():
        raise ValueError("PUN raw-map product is not finite")
    scores = -combined
    removed = valid & ~kept
    if removed.any():
        floor = np.nextafter(float(scores[kept].min()), -np.inf)
        if not math.isfinite(floor):
            raise ValueError("Could not construct finite PUN suppression scores")
        scores[removed] = floor
    return PUNAggregationResult(
        aligned_maps=maps.copy(),
        normalized_maps=normalized,
        combined_raw_psnr=combined,
        kept_candidate_mask=kept,
        policy_scores=scores,
        filter_fallback_used=fallback,
    )


def _directions(value: np.ndarray) -> np.ndarray:
    directions = np.asarray(value, dtype=np.float64)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or directions.shape[0] < 1
        or not np.isfinite(directions).all()
    ):
        raise ValueError("directions must have finite shape [N, 3]")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0):
        raise ValueError("directions must be nonzero")
    return directions / norms[:, None]
