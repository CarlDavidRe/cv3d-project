"""Dependency-free SVG diagnostic for one visibility-cache anchor."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from nbv.data.visibility_cache import VisibilityCache
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.mesh import TriangleMesh
from nbv.geometry.num_camera import anchor_camera_to_world
from nbv.geometry.visibility import (
    PerspectiveCamera,
    project_camera_points,
    world_to_camera,
)


def write_visibility_debug_svg(
    cache: VisibilityCache,
    mesh: TriangleMesh,
    anchor_id: int,
    output_path: str | Path,
) -> Path:
    """Plot projected face centroids and highlight raster-visible faces."""

    anchors = canonical_anchors()
    anchor = anchors.by_id(anchor_id)
    metadata = cache.metadata
    resolution = metadata["render_resolution"]
    camera = PerspectiveCamera(
        height=int(resolution[0]),
        width=int(resolution[1]),
        horizontal_fov_degrees=float(metadata["horizontal_fov_degrees"]),
        near=float(metadata["near"]),
        far=float(metadata["far"]),
    )
    camera_to_world = anchor_camera_to_world(
        anchor, float(metadata["camera_radius"]),
        convention=metadata["camera_convention"],
    )
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    camera_points = world_to_camera(face_centroids, camera_to_world)
    u, v, depth = project_camera_points(camera_points, camera)
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(depth)
        & (depth >= camera.near)
        & (depth <= camera.far)
        & (u >= 0)
        & (u < camera.width)
        & (v >= 0)
        & (v < camera.height)
    )
    visible = cache.face_visibility[anchor_id] & inside

    canvas = 640
    margin = 55
    plot_size = canvas - 2 * margin
    x = margin + u / camera.width * plot_size
    y = margin + v / camera.height * plot_size
    all_circles = "".join(
        f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="1.15"/>'
        for index in np.flatnonzero(inside)
    )
    visible_circles = "".join(
        f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="1.6"/>'
        for index in np.flatnonzero(visible)
    )
    object_id = html.escape(str(metadata["object_id"]))
    metrics = cache.metrics([anchor_id])
    title = (
        f"{object_id} — anchor {anchor_id}: "
        f"Vis {100 * metrics['vis']:.2f}%, VisA {100 * metrics['vis_a']:.2f}%"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
 width="{canvas}" height="{canvas + 45}" viewBox="0 0 {canvas} {canvas + 45}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="28" font-family="sans-serif" font-size="16">{title}</text>
<rect x="{margin}" y="{margin}" width="{plot_size}" height="{plot_size}"
 fill="#fafafa" stroke="#333"/>
<g fill="#8b95a1" fill-opacity="0.35">{all_circles}</g>
<g fill="#d62728" fill-opacity="0.85">{visible_circles}</g>
<circle cx="{margin}" cy="{canvas + 20}" r="4" fill="#8b95a1"/>
<text x="{margin + 10}" y="{canvas + 25}" font-family="sans-serif"
 font-size="13">projected face centroids</text>
<circle cx="{margin + 220}" cy="{canvas + 20}" r="4" fill="#d62728"/>
<text x="{margin + 230}" y="{canvas + 25}" font-family="sans-serif"
 font-size="13">raster-visible mesh faces</text>
</svg>
"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return destination
