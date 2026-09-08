from __future__ import annotations

import unittest

import numpy as np

from nbv.visualization.alignment import (
    best_translation, boundary_distances, mask_extent, shift_mask,
    silhouette_iou, translation_scores,
)


class AlignmentDiagnosticsTests(unittest.TestCase):
    def test_recovers_known_translation_with_explicit_axis_direction(self):
        mesh = np.zeros((16, 16), bool)
        mesh[5:8, 5:10] = True
        rgb = np.zeros_like(mesh)
        rgb[7:10, 4:9] = True
        best = best_translation(translation_scores(mesh, rgb))
        self.assertEqual(best, {"dx": -1, "dy": 2, "iou": 1.0})
        np.testing.assert_array_equal(shift_mask(mesh, -1, 2), rgb)
        self.assertEqual(mask_extent(rgb)["centroid_xy"], [6., 8.])
        self.assertEqual(mask_extent(rgb)["bbox_xyxy_inclusive"], [4, 7, 8, 9])

    def test_translation_never_wraps_and_search_excludes_cropping(self):
        mask = np.zeros((8, 8), bool)
        mask[0, 0] = True
        self.assertFalse(shift_mask(mask, -1, 0).any())
        self.assertFalse(shift_mask(mask, 0, -1).any())
        scores = translation_scores(mask, mask)
        self.assertTrue(all(s["dx"] >= 0 and s["dy"] >= 0 for s in scores))
        self.assertEqual(best_translation(scores), {"dx": 0, "dy": 0, "iou": 1.0})

    def test_known_boundary_distance_and_empty_masks(self):
        first = np.zeros((8, 8), bool)
        second = first.copy()
        first[1, 1] = True
        second[5, 4] = True
        self.assertEqual(boundary_distances(first, second), {
            "mean_pixels": 5., "p95_pixels": 5., "within_one_pixel_fraction": 0.,
        })
        self.assertEqual(boundary_distances(first, first)["mean_pixels"], 0.)
        empty = np.zeros_like(first)
        self.assertIsNone(boundary_distances(first, empty)["mean_pixels"])
        self.assertEqual(silhouette_iou(empty, empty), 1.)
        self.assertEqual(silhouette_iou(first, empty), 0.)


if __name__ == "__main__":
    unittest.main()
