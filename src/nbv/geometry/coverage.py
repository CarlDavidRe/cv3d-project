"""PUN ``Vis`` and ``VisA`` aggregation over visible mesh faces."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


VIS_TARGET = "vis"
VIS_A_TARGET = "vis_a"
VISIBILITY_TARGETS = (VIS_TARGET, VIS_A_TARGET)


def triangle_areas(
    vertices: NDArray[np.floating], faces: NDArray[np.integer]
) -> NDArray[np.float64]:
    """Return the non-negative area of every triangular mesh face."""

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise ValueError("vertices must have shape [N, 3]")
    if face_array.ndim != 2 or face_array.shape[1:] != (3,):
        raise ValueError("faces must have shape [F, 3]")
    if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
        raise ValueError("face index is outside the vertex array")
    triangles = vertex_array[face_array]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def visibility_weights(
    face_areas: NDArray[np.floating], target: str
) -> NDArray[np.float64]:
    """Return normalized face weights for paper target ``vis`` or ``vis_a``.

    ``vis`` weights every ground-truth face equally. ``vis_a`` weights each
    face by its full triangle area, matching the paper's Visible Area metric.
    """

    if target not in VISIBILITY_TARGETS:
        raise ValueError(
            f"visibility target must be one of {VISIBILITY_TARGETS}, got {target!r}"
        )
    areas = np.asarray(face_areas, dtype=np.float64)
    if areas.ndim != 1 or len(areas) == 0:
        raise ValueError("face_areas must be a non-empty [F] array")
    if not np.all(np.isfinite(areas)) or np.any(areas < 0):
        raise ValueError("face_areas must be finite and non-negative")
    if target == VIS_TARGET:
        return np.full(len(areas), 1.0 / len(areas), dtype=np.float64)
    total_area = float(areas.sum(dtype=np.float64))
    if total_area <= 0:
        raise ValueError("vis_a requires positive total mesh area")
    return areas / total_area


def visible_face_union(
    face_visibility: NDArray[np.bool_], selected_anchor_ids: Iterable[int]
) -> NDArray[np.bool_]:
    """Union per-anchor face masks for the selected views."""

    visibility = _validated_visibility(face_visibility)
    selected = _validated_anchor_ids(selected_anchor_ids, len(visibility))
    if selected.size == 0:
        return np.zeros(visibility.shape[1], dtype=np.bool_)
    return visibility[selected].any(axis=0)


def visibility_coverage(
    face_visibility: NDArray[np.bool_],
    face_areas: NDArray[np.floating],
    selected_anchor_ids: Iterable[int],
    *,
    target: str,
) -> float:
    """Compute accumulated PUN ``Vis`` or ``VisA`` for selected views."""

    visible = visible_face_union(face_visibility, selected_anchor_ids)
    weights = visibility_weights(face_areas, target)
    if len(visible) != len(weights):
        raise ValueError("face_visibility and face_areas disagree on face count")
    return float(weights[visible].sum(dtype=np.float64))


def visibility_metrics(
    face_visibility: NDArray[np.bool_],
    face_areas: NDArray[np.floating],
    selected_anchor_ids: Iterable[int],
) -> dict[str, float]:
    """Return both paper visibility metrics explicitly."""

    selected = tuple(selected_anchor_ids)
    return {
        VIS_TARGET: visibility_coverage(
            face_visibility, face_areas, selected, target=VIS_TARGET
        ),
        VIS_A_TARGET: visibility_coverage(
            face_visibility, face_areas, selected, target=VIS_A_TARGET
        ),
    }


def candidate_visibility_gains(
    face_visibility: NDArray[np.bool_],
    face_areas: NDArray[np.floating],
    selected_anchor_ids: Iterable[int],
    *,
    target: str,
) -> NDArray[np.float64]:
    """Return each anchor's marginal ``Vis`` or ``VisA`` coverage gain."""

    visibility = _validated_visibility(face_visibility)
    seen = visible_face_union(visibility, selected_anchor_ids)
    weights = visibility_weights(face_areas, target)
    if visibility.shape[1] != len(weights):
        raise ValueError("face_visibility and face_areas disagree on face count")
    return ((visibility & ~seen) * weights[None, :]).sum(axis=1)


def _validated_visibility(
    face_visibility: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    visibility = np.asarray(face_visibility)
    if visibility.dtype != np.bool_:
        raise ValueError("face_visibility dtype must be bool")
    if visibility.ndim != 2 or visibility.shape[1] == 0:
        raise ValueError("face_visibility must have shape [anchors, F] with F > 0")
    return visibility


def _validated_anchor_ids(
    selected_anchor_ids: Iterable[int], anchor_count: int
) -> NDArray[np.int64]:
    selected = np.asarray(tuple(selected_anchor_ids))
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected_anchor_ids must be a one-dimensional integer list")
    selected = selected.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= anchor_count):
        raise ValueError("selected anchor ID is outside the visibility rows")
    return selected
