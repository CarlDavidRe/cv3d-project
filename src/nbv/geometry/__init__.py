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
from nbv.geometry.mesh_sampling import (
    SURFACE_SAMPLING_ALGORITHM,
    MeshError,
    SurfaceSample,
    TriangleMesh,
    load_obj_mesh,
    sample_mesh_surface,
    sample_obj_surface,
)
from nbv.geometry.visibility import (
    CAMERA_CONVENTION,
    DEPTH_RENDERER,
    PerspectiveCamera,
    compute_anchor_visibility,
    point_visibility_from_depth,
    render_depth_map,
)

__all__ = [
    "CANONICAL_ANCHOR_COUNT",
    "CANONICAL_ORDERING",
    "Anchor",
    "AnchorSet",
    "AnchorValidationError",
    "canonical_anchors",
    "load_anchors",
    "SURFACE_SAMPLING_ALGORITHM",
    "MeshError",
    "SurfaceSample",
    "TriangleMesh",
    "load_obj_mesh",
    "sample_mesh_surface",
    "sample_obj_surface",
    "CAMERA_CONVENTION",
    "DEPTH_RENDERER",
    "PerspectiveCamera",
    "compute_anchor_visibility",
    "point_visibility_from_depth",
    "render_depth_map",
]
