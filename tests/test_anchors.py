from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbv.geometry import (
    CANONICAL_ANCHOR_COUNT,
    CANONICAL_ORDERING,
    AnchorValidationError,
    canonical_anchors,
    load_anchors,
)


class CanonicalAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchors = canonical_anchors()

    def test_count_ids_and_sequence_indices_are_identical(self) -> None:
        self.assertEqual(len(self.anchors), CANONICAL_ANCHOR_COUNT)
        self.assertEqual(
            [anchor.anchor_id for anchor in self.anchors], list(range(48))
        )
        for anchor_id in range(48):
            self.assertIs(self.anchors[anchor_id], self.anchors.by_id(anchor_id))

    def test_ordering_name_and_sentinel_directions_pin_the_convention(self) -> None:
        self.assertEqual(self.anchors.ordering, CANONICAL_ORDERING)
        expected = {
            0: (0.0, 0.0, 1.0),
            12: (0.8093263039727374, -0.13348273760932583, 0.57199064018404),
            24: (-0.869439408974771, -0.328243308828574, -0.3692308821467541),
            36: (0.836439928681756, 0.43305617232019583, -0.3359026604869926),
            46: (0.0, 0.0, -1.0),
            47: (0.5416444083745875, -0.023549756885851758, -0.8402777777777777),
        }
        for anchor_id, direction in expected.items():
            np.testing.assert_allclose(
                self.anchors.by_id(anchor_id).direction,
                direction,
                atol=1e-14,
                rtol=0,
            )

    def test_directions_are_normalized_unique_and_read_only(self) -> None:
        directions = self.anchors.directions
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1),
            np.ones(48),
            atol=1e-12,
            rtol=0,
        )
        self.assertEqual(len(np.unique(np.round(directions, 12), axis=0)), 48)
        self.assertFalse(directions.flags.writeable)

    def test_angles_match_directions(self) -> None:
        for anchor in self.anchors:
            self.assertAlmostEqual(
                anchor.elevation_rad, math.asin(anchor.direction[2]), places=12
            )
            self.assertAlmostEqual(
                anchor.polar_angle_rad,
                math.pi / 2 - anchor.elevation_rad,
                places=12,
            )

    def test_angular_distances_support_farthest_view_selection(self) -> None:
        distances = self.anchors.angular_distance_matrix
        np.testing.assert_allclose(distances, distances.T, atol=1e-14)
        np.testing.assert_array_equal(np.diag(distances), np.zeros(48))
        self.assertAlmostEqual(self.anchors.angular_distance(0, 46), math.pi)
        self.assertEqual(int(np.argmax(distances[0])), 46)

    def test_candidate_mask_excludes_acquired_ids(self) -> None:
        mask = self.anchors.valid_candidate_mask([0, 17, 47])
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (48,))
        self.assertEqual(int(mask.sum()), 45)
        self.assertFalse(mask[[0, 17, 47]].any())

    def test_camera_pose_looks_from_anchor_toward_origin(self) -> None:
        anchor = self.anchors.by_id(12)
        transform = anchor.camera_to_world(radius=2.5)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(
            transform[:3, 3], np.asarray(anchor.direction) * 2.5, atol=1e-12
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        np.testing.assert_allclose(
            rotation @ np.array([0.0, 0.0, -1.0]),
            anchor.view_direction,
            atol=1e-12,
        )

    def test_invalid_ids_and_radii_are_rejected(self) -> None:
        with self.assertRaises(IndexError):
            self.anchors.by_id(48)
        with self.assertRaises(TypeError):
            self.anchors.by_id(True)
        with self.assertRaises(ValueError):
            self.anchors.by_id(0).camera_position(0)

    def test_loader_rejects_rows_that_are_not_in_id_order(self) -> None:
        csv_text = (
            "anchor_id,azimuth_rad,elevation_rad,direction_x,direction_y,"
            "direction_z\n"
            "1,0,0,1,0,0\n"
            "0,1.5707963267948966,0,0,1,0\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "anchors.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(
                AnchorValidationError, "ordered with contiguous IDs"
            ):
                load_anchors(path)


if __name__ == "__main__":
    unittest.main()
