"""Compact, validated, and atomically written surface-visibility caches."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, CANONICAL_ORDERING


VISIBILITY_CACHE_SCHEMA_VERSION = 1


class VisibilityCacheError(ValueError):
    """Raised when a visibility cache is malformed or cannot be read."""


class IncompatibleVisibilityCacheError(VisibilityCacheError):
    """Raised when cache metadata does not match the requested generation."""


@dataclass(frozen=True, slots=True)
class VisibilityCache:
    """Fixed surface samples and per-anchor masks for one object."""

    surface_points: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    anchor_ids: NDArray[np.int16]
    sample_face_indices: NDArray[np.int32]
    sample_barycentric: NDArray[np.float32]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if np.asarray(self.visibility).dtype != np.bool_:
            raise VisibilityCacheError("visibility dtype must be bool")
        if not np.issubdtype(np.asarray(self.anchor_ids).dtype, np.integer):
            raise VisibilityCacheError("anchor_ids dtype must be integer")
        if not np.issubdtype(
            np.asarray(self.sample_face_indices).dtype, np.integer
        ):
            raise VisibilityCacheError("sample_face_indices dtype must be integer")
        points = np.ascontiguousarray(self.surface_points, dtype=np.float32)
        visibility = np.ascontiguousarray(self.visibility, dtype=np.bool_)
        anchor_ids = np.ascontiguousarray(self.anchor_ids, dtype=np.int16)
        face_indices = np.ascontiguousarray(
            self.sample_face_indices, dtype=np.int32
        )
        barycentric = np.ascontiguousarray(
            self.sample_barycentric, dtype=np.float32
        )
        count = len(points)
        if points.shape != (count, 3):
            raise VisibilityCacheError("surface_points must have shape [N, 3]")
        if visibility.shape != (CANONICAL_ANCHOR_COUNT, count):
            raise VisibilityCacheError(
                f"visibility must have shape [{CANONICAL_ANCHOR_COUNT}, N]"
            )
        if anchor_ids.shape != (CANONICAL_ANCHOR_COUNT,):
            raise VisibilityCacheError(
                f"anchor_ids must have shape [{CANONICAL_ANCHOR_COUNT}]"
            )
        expected_ids = np.arange(CANONICAL_ANCHOR_COUNT, dtype=np.int16)
        if not np.array_equal(anchor_ids, expected_ids):
            raise VisibilityCacheError(
                "anchor_ids must equal canonical row order [0, ..., 47]"
            )
        if face_indices.shape != (count,):
            raise VisibilityCacheError("sample_face_indices must have shape [N]")
        if barycentric.shape != (count, 3):
            raise VisibilityCacheError("sample_barycentric must have shape [N, 3]")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(barycentric)):
            raise VisibilityCacheError("sample arrays must contain finite values")
        if not isinstance(self.metadata, dict):
            raise VisibilityCacheError("metadata must be a dictionary")
        if self.metadata.get("schema_version") != VISIBILITY_CACHE_SCHEMA_VERSION:
            raise VisibilityCacheError(
                f"metadata.schema_version must be {VISIBILITY_CACHE_SCHEMA_VERSION}"
            )
        if self.metadata.get("n_surface") != count:
            raise VisibilityCacheError("metadata.n_surface does not match arrays")
        required_metadata = {
            "object_id",
            "n_surface",
            "sampling_seed",
            "anchor_ordering",
            "render_resolution",
            "depth_tolerance",
        }
        missing_metadata = required_metadata - set(self.metadata)
        if missing_metadata:
            raise VisibilityCacheError(
                f"metadata is missing required fields: {sorted(missing_metadata)}"
            )
        if self.metadata["anchor_ordering"] != CANONICAL_ORDERING:
            raise VisibilityCacheError("metadata.anchor_ordering is not canonical")

        object.__setattr__(self, "surface_points", points)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "sample_face_indices", face_indices)
        object.__setattr__(self, "sample_barycentric", barycentric)


def visibility_cache_path(cache_root: str | Path, object_id: str) -> Path:
    """Resolve ``category/object`` to a safe cache path."""

    parts = object_id.split("/")
    if len(parts) != 2 or any(
        not part or part in {".", ".."} or "/" in part or "\\" in part
        for part in parts
    ):
        raise VisibilityCacheError(
            "object_id must have safe 'category_id/object_id' form"
        )
    return Path(cache_root) / parts[0] / f"{parts[1]}.npz"


def save_visibility_cache(cache: VisibilityCache, path: str | Path) -> None:
    """Atomically store compressed arrays and canonical JSON metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        cache.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                surface_points=cache.surface_points,
                visibility=cache.visibility,
                anchor_ids=cache.anchor_ids,
                sample_face_indices=cache.sample_face_indices,
                sample_barycentric=cache.sample_barycentric,
                metadata_json=np.asarray(metadata_json),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_visibility_cache(
    path_or_object_id: str | Path,
    *,
    cache_root: str | Path | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> VisibilityCache:
    """Load by path, or by ``object_id`` when ``cache_root`` is supplied."""

    path = (
        visibility_cache_path(cache_root, str(path_or_object_id))
        if cache_root is not None
        else Path(path_or_object_id)
    )
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "surface_points",
                "visibility",
                "anchor_ids",
                "sample_face_indices",
                "sample_barycentric",
                "metadata_json",
            }
            missing = required - set(payload.files)
            if missing:
                raise VisibilityCacheError(
                    f"Cache {path} is missing arrays: {sorted(missing)}"
                )
            metadata_raw = payload["metadata_json"]
            if metadata_raw.shape != ():
                raise VisibilityCacheError("metadata_json must be a scalar")
            metadata = json.loads(str(metadata_raw.item()))
            cache = VisibilityCache(
                surface_points=payload["surface_points"],
                visibility=payload["visibility"],
                anchor_ids=payload["anchor_ids"],
                sample_face_indices=payload["sample_face_indices"],
                sample_barycentric=payload["sample_barycentric"],
                metadata=metadata,
            )
    except FileNotFoundError as exc:
        raise VisibilityCacheError(f"Visibility cache does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, VisibilityCacheError):
            raise
        raise VisibilityCacheError(
            f"Could not load visibility cache {path}: {exc}"
        ) from exc

    if expected_metadata is not None:
        errors = visibility_cache_compatibility_errors(cache, expected_metadata)
        if errors:
            raise IncompatibleVisibilityCacheError(
                f"Incompatible visibility cache {path}: " + "; ".join(errors)
            )
    return cache


def visibility_cache_compatibility_errors(
    cache: VisibilityCache, expected_metadata: Mapping[str, Any]
) -> list[str]:
    """Return deterministic, human-readable metadata incompatibilities."""

    errors: list[str] = []
    for key in sorted(expected_metadata):
        expected = expected_metadata[key]
        if key not in cache.metadata:
            errors.append(f"missing metadata {key!r}")
        elif cache.metadata[key] != expected:
            errors.append(
                f"metadata {key!r}: cached={cache.metadata[key]!r}, "
                f"requested={expected!r}"
            )
    return errors
