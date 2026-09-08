from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbv.data.visibility_cache import (
    IncompatibleVisibilityCacheError, VisibilityCache,
    load_visibility_cache, save_visibility_cache,
)
from nbv.geometry import CANONICAL_ORDERING, canonical_anchors
from nbv.geometry.num_camera import (
    CAMERA_CONVENTION,
    NUM_FOCAL_LENGTH_PER_PIXEL, NUM_HORIZONTAL_FOV_DEGREES,
    anchor_camera_to_world,
)
from nbv.geometry.visibility import PerspectiveCamera, project_camera_points, world_to_camera


class NUMCameraTests(unittest.TestCase):
    def test_actual_blender_focal_length(self):
        for resolution in (64, 256):
            camera = PerspectiveCamera(height=resolution, width=resolution)
            self.assertAlmostEqual(camera.focal_x, resolution * 525 / 512)
            self.assertAlmostEqual(camera.focal_x, camera.focal_y)
        self.assertAlmostEqual(NUM_FOCAL_LENGTH_PER_PIXEL, 1.025390625)
        self.assertAlmostEqual(NUM_HORIZONTAL_FOV_DEGREES, 51.98948897809546)

    def test_nonpole_poses_match_independent_official_blender_reference(self):
        path = Path(__file__).parent / "fixtures/num_camera_reference.json"
        reference = np.asarray(json.loads(path.read_text())["camera_to_world"])
        for anchor in canonical_anchors():
            if anchor.anchor_id in (0, 46):
                continue
            with self.subTest(anchor=anchor.anchor_id):
                actual = anchor_camera_to_world(anchor, 2.73)
                np.testing.assert_allclose(actual, reference[anchor.anchor_id], atol=1e-6, rtol=0)
                # Camera Y must be world-Z's projection onto the image plane.
                z = np.asarray(anchor.direction)
                up = np.array([0., 0., 1.]) - z[2] * z
                up /= np.linalg.norm(up)
                np.testing.assert_allclose(actual[:3, 1], up, atol=1e-12)

    def test_released_num_pole_rolls_are_explicit_and_stable(self):
        for anchor_id, rotation in ((0, np.eye(3)), (46, np.diag([-1., 1., -1.]))):
            pose = anchor_camera_to_world(canonical_anchors()[anchor_id], 2.73)
            np.testing.assert_array_equal(pose[:3, :3], rotation)
            # World +Y projects upward in both released pole RGB images.
            camera = PerspectiveCamera(height=64, width=64)
            _, v, _ = project_camera_points(world_to_camera(np.array([[0., .2, 0.]]), pose), camera)
            self.assertLess(v[0], 32)

    def test_all_poses_preserve_anchor_centers_and_look_at_origin(self):
        for anchor in canonical_anchors():
            pose = anchor_camera_to_world(anchor, 2.73)
            np.testing.assert_allclose(pose[:3, 3], anchor.camera_position(2.73), atol=1e-12)
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)
            np.testing.assert_allclose(world_to_camera(np.zeros((1, 3)), pose), [[0, 0, -2.73]], atol=1e-12)

    def test_incompatible_camera_conventions_are_rejected(self):
        anchor = canonical_anchors()[12]
        np.testing.assert_array_equal(
            anchor_camera_to_world(anchor, 2.73), anchor.camera_to_world(2.73)
        )
        cache = VisibilityCache(np.ones((48, 1), bool), np.ones(1), np.arange(48), {
            "schema_version": 2, "object_id": "synthetic/incompatible", "n_faces": 1,
            "anchor_ordering": CANONICAL_ORDERING, "render_resolution": [256, 256],
            "visibility_target": "vis_a", "visibility_definition": "pun_unoccluded_rasterized_mesh_faces_v1",
            "camera_convention": "unsupported_camera",
            "horizontal_fov_degrees": NUM_HORIZONTAL_FOV_DEGREES,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_visibility_cache(cache, path)
            with self.assertRaisesRegex(IncompatibleVisibilityCacheError, "camera_convention"):
                load_visibility_cache(path, expected_metadata={
                    "camera_convention": CAMERA_CONVENTION,
                    "horizontal_fov_degrees": NUM_HORIZONTAL_FOV_DEGREES,
                })
        with self.assertRaisesRegex(ValueError, "Unsupported camera convention"):
            anchor_camera_to_world(anchor, 2.73, convention="unknown")


if __name__ == "__main__":
    unittest.main()
