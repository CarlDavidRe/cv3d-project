"""On-disk caches for fixed-size frozen Phase 1 feature vectors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from nbv.config import validate_artifact_path


FEATURE_CACHE_SCHEMA_VERSION = 1


class FeatureCacheError(ValueError):
    """Raised when a feature cache is malformed or does not match a request."""


@dataclass(frozen=True, slots=True)
class CachedFeatureDataset:
    """Frozen vectors and supervised targets for one deterministic split."""

    features: Tensor
    targets: Tensor
    valid_mask: Tensor
    sample_ids: tuple[str, ...]
    source_anchor_ids: Tensor
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[0] < 1:
            raise FeatureCacheError("features must have non-empty shape [N, D]")
        if (
            self.targets.ndim != 2
            or self.targets.shape[0] != self.features.shape[0]
        ):
            raise FeatureCacheError(
                "targets must have shape [N, A] matching features"
            )
        if self.valid_mask.shape != self.targets.shape:
            raise FeatureCacheError("valid_mask must have the target shape")
        if self.valid_mask.dtype != torch.bool:
            raise FeatureCacheError("valid_mask must have boolean dtype")
        if not torch.all(self.valid_mask.any(dim=1)):
            raise FeatureCacheError(
                "every sample must have at least one valid target"
            )
        if len(self.sample_ids) != self.features.shape[0]:
            raise FeatureCacheError("sample_ids must have one entry per feature")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise FeatureCacheError("sample_ids must be unique")
        if any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in self.sample_ids
        ):
            raise FeatureCacheError("sample_ids must be non-empty strings")
        if self.source_anchor_ids.shape != (self.features.shape[0],):
            raise FeatureCacheError("source_anchor_ids must have shape [N]")
        if self.source_anchor_ids.dtype != torch.int64:
            raise FeatureCacheError("source_anchor_ids must have int64 dtype")
        if (
            not self.features.is_floating_point()
            or not self.targets.is_floating_point()
        ):
            raise FeatureCacheError("features and targets must be floating point")
        if (
            not torch.isfinite(self.features).all()
            or not torch.isfinite(self.targets).all()
        ):
            raise FeatureCacheError("features and targets must be finite")
        if not isinstance(self.metadata, Mapping):
            raise FeatureCacheError("metadata must be a mapping")

    def __len__(self) -> int:
        return int(self.features.shape[0])


def cache_fingerprint(metadata: Mapping[str, Any], *, length: int = 16) -> str:
    """Return a stable short fingerprint for JSON-compatible cache metadata."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 8:
        raise ValueError("length must be an integer of at least 8")
    try:
        serialized = json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FeatureCacheError(
            "cache metadata must be JSON-compatible"
        ) from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def feature_cache_path(
    root: str | Path,
    *,
    backbone: str,
    variant: str,
    split: str,
    metadata: Mapping[str, Any],
) -> Path:
    """Build a collision-resistant path without trusting config fragments."""

    validate_artifact_path(root, "feature cache root")
    safe_backbone = _safe_component(backbone, "backbone")
    safe_variant = _safe_component(variant, "variant")
    safe_split = _safe_component(split, "split")
    fingerprint = cache_fingerprint(metadata)
    return (
        Path(root)
        / safe_backbone
        / safe_variant
        / f"{safe_split}_{fingerprint}.pt"
    )


def save_feature_cache(cache: CachedFeatureDataset, path: str | Path) -> None:
    """Atomically save one cache after validating its in-memory schema."""

    destination = Path(path)
    validate_artifact_path(destination.parent, "feature cache destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "features": cache.features.detach().cpu().contiguous(),
        "targets": cache.targets.detach().cpu().contiguous(),
        "valid_mask": cache.valid_mask.detach().cpu().contiguous(),
        "sample_ids": list(cache.sample_ids),
        "source_anchor_ids": cache.source_anchor_ids.detach().cpu().contiguous(),
        "metadata": dict(cache.metadata),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_feature_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> CachedFeatureDataset:
    """Load and validate one cache, optionally requiring exact metadata."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise FeatureCacheError(
            f"Feature cache does not exist: {source}"
        ) from exc
    except Exception as exc:
        raise FeatureCacheError(
            f"Could not load feature cache {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FeatureCacheError("Feature cache root must be a mapping")
    if payload.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise FeatureCacheError(
            f"Feature cache schema_version must be "
            f"{FEATURE_CACHE_SCHEMA_VERSION}"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise FeatureCacheError("Feature cache metadata must be a mapping")
    if expected_metadata is not None and dict(metadata) != dict(expected_metadata):
        raise FeatureCacheError(
            "Feature cache metadata does not match the request"
        )
    try:
        return CachedFeatureDataset(
            features=payload["features"],
            targets=payload["targets"],
            valid_mask=payload["valid_mask"],
            sample_ids=_string_tuple(payload["sample_ids"]),
            source_anchor_ids=payload["source_anchor_ids"],
            metadata=dict(metadata),
        )
    except KeyError as exc:
        raise FeatureCacheError(
            f"Feature cache is missing field {exc.args[0]!r}"
        ) from exc
    except (TypeError, AttributeError) as exc:
        raise FeatureCacheError(
            "Feature cache contains invalid field types"
        ) from exc


def _string_tuple(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FeatureCacheError("sample_ids must be a sequence")
    return tuple(values)


def _safe_component(value: object, name: str) -> str:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )
    if not isinstance(value, str) or not value:
        raise FeatureCacheError(f"{name} must be a non-empty string")
    if value in {".", ".."} or any(character not in allowed for character in value):
        raise FeatureCacheError(
            f"{name} may contain only letters, numbers, underscores, and hyphens"
        )
    return value
