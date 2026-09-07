"""Rasterized mesh-face visibility matching the PUN evaluation metric.

The renderer is a small deterministic CPU z-buffer. It uses OpenGL/Blender
look-at poses (camera looks along local -Z), perspective-correct depth, pixel
centres, and a top-left image origin. Faces are two-sided by default, matching
the PUN Blender generation path's lack of explicit back-face culling.

PUN defines a face as visible when it appears without occlusion in at least one
selected view. The face-index buffer implemented here records exactly which
triangle wins the z-buffer at every pixel. ``Vis`` and ``VisA`` differ only in
how those visible faces are aggregated; see :mod:`nbv.geometry.coverage`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from nbv.geometry.anchors import AnchorSet, canonical_anchors
from nbv.geometry.mesh import TriangleMesh


FACE_VISIBILITY_RENDERER = "numpy_cpu_triangle_id_zbuffer_perspective_v1"
CAMERA_CONVENTION = "opengl_camera_minus_z_y_up_image_y_down_pixel_centers_v1"


@dataclass(frozen=True, slots=True)
class PerspectiveCamera:
    """Square-pixel pinhole camera matching the official PUN NMR setup."""

    height: int = 64
    width: int = 64
    horizontal_fov_degrees: float = 30.0
    near: float = 1.2
    far: float = 4.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("camera resolution must be positive")
        if not 0 < self.horizontal_fov_degrees < 180:
            raise ValueError("horizontal_fov_degrees must be in (0, 180)")
        if not math.isfinite(self.near) or not math.isfinite(self.far):
            raise ValueError("near and far must be finite")
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("camera requires 0 < near < far")

    @property
    def focal_x(self) -> float:
        half_fov = math.radians(self.horizontal_fov_degrees) / 2
        return 0.5 * self.width / math.tan(half_fov)

    @property
    def focal_y(self) -> float:
        # PUN uses fx == fy. A square render is the official/default setup.
        return self.focal_x

    @property
    def center_x(self) -> float:
        return self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.height / 2.0


def world_to_camera(
    points: NDArray[np.floating], camera_to_world: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Transform row-vector world points into an OpenGL camera frame."""

    points_array = np.asarray(points, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1:] != (3,):
        raise ValueError("points must have shape [N, 3]")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("camera_to_world must be a finite [4, 4] matrix")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points_array - translation) @ rotation


def project_camera_points(
    camera_points: NDArray[np.floating], camera: PerspectiveCamera
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Project camera-frame points, returning pixel ``u``, pixel ``v``, depth."""

    points = np.asarray(camera_points, dtype=np.float64)
    depth = -points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.focal_x * points[:, 0] / depth + camera.center_x
        v = camera.center_y - camera.focal_y * points[:, 1] / depth
    return u, v, depth


def render_face_index_map(
    mesh: TriangleMesh,
    camera_to_world: NDArray[np.floating],
    camera: PerspectiveCamera,
    *,
    cull_backfaces: bool = False,
) -> NDArray[np.int32]:
    """Rasterize the nearest, unoccluded mesh-face index at every pixel.

    Pixels not covered by the mesh contain ``-1``. A face is directly visible
    from the view iff its index occurs at least once in this map.
    """

    _, face_index_buffer = _render_mesh_buffers(
        mesh,
        camera_to_world,
        camera,
        cull_backfaces=cull_backfaces,
    )
    return face_index_buffer


def _render_mesh_buffers(
    mesh: TriangleMesh,
    camera_to_world: NDArray[np.floating],
    camera: PerspectiveCamera,
    *,
    cull_backfaces: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    camera_vertices = world_to_camera(mesh.vertices, camera_to_world)
    depth_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    face_index_buffer = np.full(
        (camera.height, camera.width), -1, dtype=np.int32
    )

    for face_index, face in enumerate(mesh.faces):
        polygon = camera_vertices[face]
        polygon = _clip_depth_polygon(polygon, camera.near, keep_greater=True)
        polygon = _clip_depth_polygon(polygon, camera.far, keep_greater=False)
        if len(polygon) < 3:
            continue
        for index in range(1, len(polygon) - 1):
            triangle = np.stack((polygon[0], polygon[index], polygon[index + 1]))
            _rasterize_camera_triangle(
                triangle,
                depth_buffer,
                camera,
                cull_backfaces=cull_backfaces,
                face_index_buffer=face_index_buffer,
                face_index=face_index,
            )
    return depth_buffer, face_index_buffer


def compute_anchor_visibility(
    mesh: TriangleMesh,
    *,
    camera: PerspectiveCamera,
    camera_radius: float,
    cull_backfaces: bool = False,
    anchors: AnchorSet | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> NDArray[np.bool_]:
    """Return a PUN-style ``[anchors, faces]`` direct-visibility mask.

    A face is visible in a row when at least one pixel-centre ray reaches that
    face as the nearest mesh intersection. Rows follow the exact supplied or
    canonical anchor sequence.
    """

    if not math.isfinite(camera_radius) or camera_radius <= 0:
        raise ValueError("camera_radius must be finite and greater than zero")
    anchor_set = anchors if anchors is not None else canonical_anchors()
    visibility = np.zeros((len(anchor_set), len(mesh.faces)), dtype=np.bool_)
    for row_index, anchor in enumerate(anchor_set):
        if row_index != anchor.anchor_id:
            raise ValueError("anchor sequence index and anchor_id must be identical")
        camera_to_world = anchor.camera_to_world(camera_radius)
        face_index_map = render_face_index_map(
            mesh, camera_to_world, camera, cull_backfaces=cull_backfaces
        )
        visible_face_indices = np.unique(face_index_map[face_index_map >= 0])
        visibility[row_index, visible_face_indices] = True
        if progress is not None:
            progress(
                anchor.anchor_id,
                int(visibility[row_index].sum()),
                len(mesh.faces),
            )
    return visibility


def _clip_depth_polygon(
    polygon: NDArray[np.float64], threshold: float, *, keep_greater: bool
) -> NDArray[np.float64]:
    if len(polygon) == 0:
        return polygon

    def inside(vertex: NDArray[np.float64]) -> bool:
        depth = -float(vertex[2])
        return depth >= threshold if keep_greater else depth <= threshold

    output: list[NDArray[np.float64]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            previous_depth = -float(previous[2])
            current_depth = -float(current[2])
            denominator = current_depth - previous_depth
            fraction = (threshold - previous_depth) / denominator
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _rasterize_camera_triangle(
    triangle: NDArray[np.float64],
    depth_buffer: NDArray[np.float64],
    camera: PerspectiveCamera,
    *,
    cull_backfaces: bool,
    face_index_buffer: NDArray[np.int32],
    face_index: int = -1,
) -> None:
    if cull_backfaces:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.dot(normal, -triangle.mean(axis=0))) <= 0:
            return

    u, v, depth = project_camera_points(triangle, camera)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        return
    min_x = max(0, int(math.ceil(float(u.min()) - 0.5)))
    max_x = min(camera.width - 1, int(math.floor(float(u.max()) - 0.5)))
    min_y = max(0, int(math.ceil(float(v.min()) - 0.5)))
    max_y = min(camera.height - 1, int(math.floor(float(v.max()) - 0.5)))
    if min_x > max_x or min_y > max_y:
        return

    x0, x1, x2 = u
    y0, y1, y2 = v
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) < 1e-14:
        return
    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
        np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
    )
    weight0 = (
        (y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)
    ) / denominator
    weight1 = (
        (y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)
    ) / denominator
    weight2 = 1.0 - weight0 - weight1
    epsilon = 1e-10
    inside = (weight0 >= -epsilon) & (weight1 >= -epsilon) & (weight2 >= -epsilon)
    if not np.any(inside):
        return

    inverse_depth = weight0 / depth[0] + weight1 / depth[1] + weight2 / depth[2]
    candidate_depth = np.where(inside, 1.0 / inverse_depth, np.inf)
    target = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
    nearer = candidate_depth < target
    target[nearer] = candidate_depth[nearer]
    face_target = face_index_buffer[min_y : max_y + 1, min_x : max_x + 1]
    face_target[nearer] = face_index
