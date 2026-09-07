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
    candidate_visibility_gains,
    canonical_anchors,
    compute_anchor_visibility,
    load_mesh,
    load_obj_mesh,
    load_ply_mesh,
    render_face_index_map,
    triangle_areas,
    visibility_metrics,
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


class MeshLoadingTests(unittest.TestCase):
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

    def test_ascii_ply_loader_uses_named_coordinates_and_triangulates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_normalized.ply"
            path.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "comment synthetic fixture\n"
                "element vertex 4\n"
                "property uchar red\n"
                "property float z\n"
                "property float x\n"
                "property float y\n"
                "element face 1\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
                "255 0 0 0\n"
                "255 0 1 0\n"
                "255 0 1 1\n"
                "255 0 0 1\n"
                "4 0 1 2 3\n",
                encoding="ascii",
            )
            mesh = load_ply_mesh(path)
            generic = load_mesh(path)

        np.testing.assert_array_equal(
            mesh.vertices,
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        )
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(generic.vertices, mesh.vertices)


class VisibilityTests(unittest.TestCase):
    def test_cache_shape_dtype_and_canonical_anchor_order(self) -> None:
        mesh = cube_mesh()
        visibility = compute_anchor_visibility(
            mesh,
            camera=PerspectiveCamera(
                height=32,
                width=32,
                horizontal_fov_degrees=60,
                near=0.1,
                far=10,
            ),
            camera_radius=3,
        )
        cache = self._cache(mesh, visibility)

        self.assertEqual(cache.face_visibility.shape, (48, 12))
        self.assertEqual(cache.face_visibility.dtype, np.bool_)
        self.assertEqual(cache.face_areas.shape, (12,))
        np.testing.assert_array_equal(cache.anchor_ids, np.arange(48))
        self.assertEqual(cache.metadata["anchor_ordering"], CANONICAL_ORDERING)
        self.assertEqual(
            [anchor.anchor_id for anchor in canonical_anchors()], list(cache.anchor_ids)
        )
        self.assertTrue(cache.face_visibility.any())

    def test_face_index_rasterizer_records_nearest_unoccluded_faces(self) -> None:
        vertices = np.asarray(
            [
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            ],
            dtype=np.float64,
        )
        mesh = TriangleMesh(
            vertices,
            np.asarray(
                [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
                dtype=np.int64,
            ),
        )
        camera = PerspectiveCamera(
            height=64,
            width=64,
            horizontal_fov_degrees=60,
            near=0.1,
            far=10,
        )
        pose = canonical_anchors().by_id(0).camera_to_world(radius=3)

        face_ids = set(np.unique(render_face_index_map(mesh, pose, camera)))

        self.assertTrue(face_ids.issuperset({-1, 0, 1}))
        self.assertTrue(face_ids.isdisjoint({2, 3}))

    def test_cache_round_trip_preserves_arrays_and_metadata(self) -> None:
        mesh = cube_mesh()
        generator = np.random.default_rng(2)
        visibility = generator.random((48, len(mesh.faces))) > 0.5
        cache = self._cache(mesh, visibility)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "category" / "object.npz"
            save_visibility_cache(cache, path)
            loaded = load_visibility_cache(path)

        np.testing.assert_array_equal(
            loaded.face_visibility, cache.face_visibility
        )
        np.testing.assert_array_equal(loaded.face_areas, cache.face_areas)
        np.testing.assert_array_equal(loaded.anchor_ids, cache.anchor_ids)
        self.assertEqual(loaded.metadata, cache.metadata)

    def test_incompatible_metadata_is_detected(self) -> None:
        mesh = cube_mesh()
        cache = self._cache(
            mesh, np.zeros((48, len(mesh.faces)), dtype=np.bool_)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_visibility_cache(cache, path)
            with self.assertRaisesRegex(
                IncompatibleVisibilityCacheError, "visibility_target"
            ):
                load_visibility_cache(
                    path, expected_metadata={"visibility_target": "vis"}
                )

    def test_vis_and_vis_a_are_explicit_and_target_selects_default(self) -> None:
        mesh = TriangleMesh(
            np.asarray(
                [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 3, 0]],
                dtype=np.float64,
            ),
            np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
        )
        face_visibility = np.zeros((48, 2), dtype=np.bool_)
        face_visibility[0, 0] = True
        face_visibility[1, 1] = True
        cache = self._cache(mesh, face_visibility, target="vis_a")

        self.assertEqual(cache.metrics([0]), {"vis": 0.5, "vis_a": 0.25})
        self.assertAlmostEqual(cache.coverage([0]), 0.25)
        self.assertAlmostEqual(cache.coverage([0], target="vis"), 0.5)
        np.testing.assert_allclose(cache.candidate_gains([0])[:2], [0.0, 0.75])
        np.testing.assert_allclose(
            candidate_visibility_gains(
                face_visibility, cache.face_areas, [0], target="vis"
            )[:2],
            [0.0, 0.5],
        )
        self.assertEqual(
            visibility_metrics(face_visibility, cache.face_areas, [0, 1]),
            {"vis": 1.0, "vis_a": 1.0},
        )

    @staticmethod
    def _cache(
        mesh: TriangleMesh, visibility: np.ndarray, target: str = "vis_a"
    ) -> VisibilityCache:
        metadata = {
            "schema_version": 2,
            "object_id": "synthetic/cube",
            "n_faces": len(mesh.faces),
            "anchor_ordering": CANONICAL_ORDERING,
            "render_resolution": [32, 32],
            "visibility_target": target,
            "visibility_definition": "pun_unoccluded_rasterized_mesh_faces_v1",
            "nested": {"camera": "test"},
        }
        return VisibilityCache(
            face_visibility=visibility,
            face_areas=triangle_areas(mesh.vertices, mesh.faces),
            anchor_ids=np.arange(48, dtype=np.int16),
            metadata=metadata,
        )


if __name__ == "__main__":
    unittest.main()
