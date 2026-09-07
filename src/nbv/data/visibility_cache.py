"""Validated, atomic caches for PUN mesh-face visibility."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, CANONICAL_ORDERING
from nbv.geometry.coverage import (
    VISIBILITY_TARGETS,
    candidate_visibility_gains,
    visibility_coverage,
    visibility_metrics,
)


VISIBILITY_CACHE_SCHEMA_VERSION = 2


class VisibilityCacheError(ValueError):
    """Raised when a visibility cache is malformed or cannot be read."""


class IncompatibleVisibilityCacheError(VisibilityCacheError):
    """Raised when cache metadata does not match the requested generation."""


@dataclass(frozen=True, slots=True)
class VisibilityCache:
    """Per-anchor visible-face masks and areas for one ground-truth mesh."""

    face_visibility: NDArray[np.bool_]
    face_areas: NDArray[np.float64]
    anchor_ids: NDArray[np.int16]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if np.asarray(self.face_visibility).dtype != np.bool_:
            raise VisibilityCacheError("face_visibility dtype must be bool")
        if not np.issubdtype(np.asarray(self.anchor_ids).dtype, np.integer):
            raise VisibilityCacheError("anchor_ids dtype must be integer")
        face_visibility = np.ascontiguousarray(
            self.face_visibility, dtype=np.bool_
        )
        face_areas = np.ascontiguousarray(self.face_areas, dtype=np.float64)
        anchor_ids = np.ascontiguousarray(self.anchor_ids, dtype=np.int16)
        if face_visibility.ndim != 2:
            raise VisibilityCacheError(
                "face_visibility must have shape [48, number_of_faces]"
            )
        face_count = face_visibility.shape[1]
        if face_count == 0 or face_visibility.shape[0] != CANONICAL_ANCHOR_COUNT:
            raise VisibilityCacheError(
                "face_visibility must have shape [48, number_of_faces > 0]"
            )
        if face_areas.shape != (face_count,):
            raise VisibilityCacheError(
                "face_areas must match the face_visibility face dimension"
            )
        if not np.all(np.isfinite(face_areas)) or np.any(face_areas < 0):
            raise VisibilityCacheError("face_areas must be finite and non-negative")
        if float(face_areas.sum(dtype=np.float64)) <= 0:
            raise VisibilityCacheError("face_areas must have positive total area")
        if anchor_ids.shape != (CANONICAL_ANCHOR_COUNT,):
            raise VisibilityCacheError(
                f"anchor_ids must have shape [{CANONICAL_ANCHOR_COUNT}]"
            )
        expected_ids = np.arange(CANONICAL_ANCHOR_COUNT, dtype=np.int16)
        if not np.array_equal(anchor_ids, expected_ids):
            raise VisibilityCacheError(
                "anchor_ids must equal canonical row order [0, ..., 47]"
            )
        if not isinstance(self.metadata, dict):
            raise VisibilityCacheError("metadata must be a dictionary")
        if self.metadata.get("schema_version") != VISIBILITY_CACHE_SCHEMA_VERSION:
            raise VisibilityCacheError(
                "metadata.schema_version must be "
                f"{VISIBILITY_CACHE_SCHEMA_VERSION}"
            )
        required_metadata = {
            "object_id",
            "n_faces",
            "anchor_ordering",
            "render_resolution",
            "visibility_target",
            "visibility_definition",
        }
        missing_metadata = required_metadata - set(self.metadata)
        if missing_metadata:
            raise VisibilityCacheError(
                f"metadata is missing required fields: {sorted(missing_metadata)}"
            )
        if self.metadata["n_faces"] != face_count:
            raise VisibilityCacheError("metadata.n_faces does not match arrays")
        if self.metadata["anchor_ordering"] != CANONICAL_ORDERING:
            raise VisibilityCacheError("metadata.anchor_ordering is not canonical")
        if self.metadata["visibility_target"] not in VISIBILITY_TARGETS:
            raise VisibilityCacheError(
                f"metadata.visibility_target must be one of {VISIBILITY_TARGETS}"
            )

        object.__setattr__(self, "face_visibility", face_visibility)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "anchor_ids", anchor_ids)

    @property
    def visibility_target(self) -> str:
        """Return the configured default target (``vis`` or ``vis_a``)."""

        return str(self.metadata["visibility_target"])

    def coverage(
        self, selected_anchor_ids: Iterable[int], *, target: str | None = None
    ) -> float:
        """Compute accumulated coverage with the configured or given target."""

        return visibility_coverage(
            self.face_visibility,
            self.face_areas,
            selected_anchor_ids,
            target=self.visibility_target if target is None else target,
        )

    def metrics(self, selected_anchor_ids: Iterable[int]) -> dict[str, float]:
        """Compute both paper metrics explicitly for the selected views."""

        return visibility_metrics(
            self.face_visibility, self.face_areas, selected_anchor_ids
        )

    def candidate_gains(
        self, selected_anchor_ids: Iterable[int], *, target: str | None = None
    ) -> NDArray[np.float64]:
        """Compute marginal coverage gain under the selected target."""

        return candidate_visibility_gains(
            self.face_visibility,
            self.face_areas,
            selected_anchor_ids,
            target=self.visibility_target if target is None else target,
        )


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
                face_visibility=cache.face_visibility,
                face_areas=cache.face_areas,
                anchor_ids=cache.anchor_ids,
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
                "face_visibility",
                "face_areas",
                "anchor_ids",
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
                face_visibility=payload["face_visibility"],
                face_areas=payload["face_areas"],
                anchor_ids=payload["anchor_ids"],
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
