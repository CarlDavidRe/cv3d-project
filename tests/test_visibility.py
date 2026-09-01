from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbv.data import (
    IncompatibleVisibilityCacheError,
    VisibilityCache,
    load_visibility_cache,
    save_visibility_cache,
)
from nbv.geometry import (
    CANONICAL_ORDERING,
    PerspectiveCamera,
    TriangleMesh,
    canonical_anchors,
    compute_anchor_visibility,
    load_obj_mesh,
    point_visibility_from_depth,
    render_depth_map,
    sample_mesh_surface,
)


def cube_mesh() -> TriangleMesh:
    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces)


class MeshSamplingTests(unittest.TestCase):
    def test_obj_loader_triangulates_and_resolves_negative_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_normalized.obj"
            path.write_text(
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "f -4 -3 -2 -1 # quad with negative indices\n",
                encoding="utf-8",
            )
            mesh = load_obj_mesh(path)

        self.assertEqual(mesh.vertices.shape, (4, 3))
        self.assertEqual(mesh.faces.shape, (2, 3))
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])

    def test_surface_sampling_is_deterministic_for_mesh_and_seed(self) -> None:
        mesh = cube_mesh()
        first = sample_mesh_surface(mesh, n_surface=1000, seed=73)
        second = sample_mesh_surface(mesh, n_surface=1000, seed=73)

        np.testing.assert_allclose(first.points, second.points, atol=0, rtol=0)
        np.testing.assert_array_equal(first.face_indices, second.face_indices)
        np.testing.assert_allclose(
            first.barycentric, second.barycentric, atol=0, rtol=0
        )
        np.testing.assert_allclose(first.barycentric.sum(axis=1), 1.0, atol=1e-6)

    def test_different_seed_changes_surface_sample(self) -> None:
        first = sample_mesh_surface(cube_mesh(), n_surface=64, seed=1)
        second = sample_mesh_surface(cube_mesh(), n_surface=64, seed=2)
        self.assertFalse(np.array_equal(first.points, second.points))


class VisibilityTests(unittest.TestCase):
    def test_cache_shape_dtype_and_canonical_anchor_order(self) -> None:
        mesh = cube_mesh()
        sample = sample_mesh_surface(mesh, n_surface=128, seed=5)
        visibility = compute_anchor_visibility(
            mesh,
            sample.points,
            camera=PerspectiveCamera(
                height=32,
                width=32,
                horizontal_fov_degrees=60,
                near=0.1,
                far=10,
            ),
            camera_radius=3,
            depth_tolerance=0.1,
            depth_neighborhood_radius=1,
        )
        cache = self._cache(sample, visibility)

        self.assertEqual(cache.surface_points.shape, (128, 3))
        self.assertEqual(cache.visibility.shape, (48, 128))
        self.assertEqual(cache.visibility.dtype, np.bool_)
        np.testing.assert_array_equal(cache.anchor_ids, np.arange(48))
        self.assertEqual(cache.metadata["anchor_ordering"], CANONICAL_ORDERING)
        self.assertEqual(
            [anchor.anchor_id for anchor in canonical_anchors()], list(cache.anchor_ids)
        )
        self.assertTrue(cache.visibility.any())

    def test_depth_consistency_rejects_occluded_behind_and_outside_points(self) -> None:
        vertices = np.asarray(
            [
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64
        )
        mesh = TriangleMesh(vertices, faces)
        camera = PerspectiveCamera(
            height=64,
            width=64,
            horizontal_fov_degrees=60,
            near=0.1,
            far=10,
        )
        camera_to_world = canonical_anchors().by_id(0).camera_to_world(radius=3)
        depth = render_depth_map(mesh, camera_to_world, camera)
        points = np.asarray(
            [
                [0, 0, 0.5],   # front plate
                [0, 0, -0.5],  # hidden by front plate
                [0, 0, 4.0],   # behind the camera
                [10, 0, 0],    # outside the image
            ],
            dtype=np.float64,
        )
        visible = point_visibility_from_depth(
            points,
            camera_to_world,
            camera,
            depth,
            depth_tolerance=0.02,
            depth_neighborhood_radius=0,
        )
        np.testing.assert_array_equal(visible, [True, False, False, False])

    def test_cache_round_trip_preserves_arrays_and_metadata(self) -> None:
        sample = sample_mesh_surface(cube_mesh(), n_surface=37, seed=9)
        generator = np.random.default_rng(2)
        visibility = generator.random((48, 37)) > 0.5
        cache = self._cache(sample, visibility)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "category" / "object.npz"
            save_visibility_cache(cache, path)
            loaded = load_visibility_cache(path)

        np.testing.assert_array_equal(loaded.surface_points, cache.surface_points)
        np.testing.assert_array_equal(loaded.visibility, cache.visibility)
        np.testing.assert_array_equal(loaded.anchor_ids, cache.anchor_ids)
        np.testing.assert_array_equal(
            loaded.sample_face_indices, cache.sample_face_indices
        )
        np.testing.assert_array_equal(
            loaded.sample_barycentric, cache.sample_barycentric
        )
        self.assertEqual(loaded.metadata, cache.metadata)

    def test_incompatible_metadata_is_detected(self) -> None:
        sample = sample_mesh_surface(cube_mesh(), n_surface=8, seed=9)
        cache = self._cache(sample, np.zeros((48, 8), dtype=np.bool_))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_visibility_cache(cache, path)
            with self.assertRaisesRegex(
                IncompatibleVisibilityCacheError, "depth_tolerance"
            ):
                load_visibility_cache(
                    path, expected_metadata={"depth_tolerance": 0.25}
                )

    @staticmethod
    def _cache(sample, visibility: np.ndarray) -> VisibilityCache:
        metadata = {
            "schema_version": 1,
            "object_id": "synthetic/cube",
            "n_surface": len(sample.points),
            "sampling_seed": sample.metadata["sampling_seed"],
            "anchor_ordering": CANONICAL_ORDERING,
            "render_resolution": [32, 32],
            "depth_tolerance": 0.1,
            "nested": {"camera": "test"},
        }
        return VisibilityCache(
            surface_points=sample.points,
            visibility=visibility,
            anchor_ids=np.arange(48, dtype=np.int16),
            sample_face_indices=sample.face_indices,
            sample_barycentric=sample.barycentric,
            metadata=metadata,
        )


if __name__ == "__main__":
    unittest.main()
