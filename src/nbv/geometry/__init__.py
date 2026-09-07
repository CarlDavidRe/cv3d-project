"""Canonical camera-anchor geometry."""

from nbv.geometry.anchors import (
    CANONICAL_ANCHOR_COUNT,
    CANONICAL_ORDERING,
    Anchor,
    AnchorSet,
    AnchorValidationError,
    canonical_anchors,
    load_anchors,
)
from nbv.geometry.mesh import (
    MeshError,
    TriangleMesh,
    load_mesh,
    load_obj_mesh,
    load_ply_mesh,
)
from nbv.geometry.coverage import (
    VIS_A_TARGET,
    VIS_TARGET,
    VISIBILITY_TARGETS,
    candidate_visibility_gains,
    triangle_areas,
    visibility_coverage,
    visibility_metrics,
    visibility_weights,
    visible_face_union,
)
from nbv.geometry.visibility import (
    CAMERA_CONVENTION,
    FACE_VISIBILITY_RENDERER,
    PerspectiveCamera,
    compute_anchor_visibility,
    render_face_index_map,
)

__all__ = [
    "CANONICAL_ANCHOR_COUNT",
    "CANONICAL_ORDERING",
    "Anchor",
    "AnchorSet",
    "AnchorValidationError",
    "canonical_anchors",
    "load_anchors",
    "MeshError",
    "TriangleMesh",
    "load_mesh",
    "load_obj_mesh",
    "load_ply_mesh",
    "VIS_TARGET",
    "VIS_A_TARGET",
    "VISIBILITY_TARGETS",
    "candidate_visibility_gains",
    "triangle_areas",
    "visibility_coverage",
    "visibility_metrics",
    "visibility_weights",
    "visible_face_union",
    "CAMERA_CONVENTION",
    "FACE_VISIBILITY_RENDERER",
    "PerspectiveCamera",
    "compute_anchor_visibility",
    "render_face_index_map",
]
