from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.visualize_reconstruction import (
    apply_transform,
    deterministic_subset,
    estimate_similarity,
    select_row,
    similarity_icp,
    write_interactive_html,
    write_ply,
    write_preview,
)


class ReconstructionVisualizationTests(unittest.TestCase):
    def test_select_row_defaults_to_largest_view_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("object_id", "policy", "acquired_view_count")
                )
                writer.writeheader()
                writer.writerows([
                    {"object_id": "cat/object", "policy": "vggt", "acquired_view_count": 2},
                    {"object_id": "cat/object", "policy": "vggt", "acquired_view_count": 10},
                    {"object_id": "cat/object", "policy": "pun", "acquired_view_count": 10},
                ])
            row = select_row(path, "cat/object", "vggt", None)
            self.assertEqual(row["acquired_view_count"], "10")

    def test_writes_colored_ply_and_projection_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = np.asarray([
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
            ])
            predicted = target + 0.01
            write_ply(root / "cloud.ply", target, (1, 2, 3))
            clipped = write_preview(root / "preview.png", target, predicted, max_points=3)
            text = (root / "cloud.ply").read_text(encoding="ascii")
            self.assertIn("element vertex 4", text)
            self.assertTrue(text.rstrip().endswith("1 1 1 1 2 3"))
            self.assertEqual(clipped, 0)
            with Image.open(root / "preview.png") as image:
                self.assertEqual(image.size, (1290, 484))

    def test_deterministic_subset_includes_endpoints(self) -> None:
        points = np.arange(30).reshape(10, 3)
        selected = deterministic_subset(points, 4)
        np.testing.assert_array_equal(selected[[0, -1]], points[[0, -1]])

    def test_similarity_estimation_recovers_known_transform(self) -> None:
        generator = np.random.default_rng(7)
        source = generator.normal(size=(50, 3))
        angle = 0.3
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        target = 1.7 * (source @ rotation.T) + np.asarray([0.2, -0.4, 0.7])
        transform = estimate_similarity(source, target)
        np.testing.assert_allclose(apply_transform(source, transform), target, atol=1e-10)

    def test_oracle_icp_corrects_global_scale_and_translation(self) -> None:
        generator = np.random.default_rng(11)
        target = generator.normal(size=(100, 3)) * np.asarray([2.0, 1.0, 0.5])
        prediction = 0.6 * target + np.asarray([1.0, -0.5, 0.3])
        aligned, _, info = similarity_icp(
            prediction, target, iterations=10, sample_count=100
        )
        np.testing.assert_allclose(aligned, target, atol=1e-8)
        self.assertEqual(info["method"], "symmetric_trimmed_similarity_icp_v1")

    def test_writes_self_contained_interactive_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "viewer.html"
            points = np.asarray([
                [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
            ])
            write_interactive_html(
                path, points, points + 0.1, max_points=2, title="test <cloud>"
            )
            document = path.read_text(encoding="utf-8")
            self.assertIn("Drag to rotate", document)
            self.assertIn("Auto-rotate", document)
            self.assertIn('id="showGt"', document)
            self.assertIn('value="side-by-side"', document)
            self.assertIn("Overlay (on top)", document)
            self.assertIn('id="showBox"', document)
            self.assertIn("GT world dimensions", document)
            self.assertIn('"dimensions":[2.0,1.0,0.0]', document)
            self.assertIn("test &lt;cloud&gt;", document)
            self.assertNotIn("https://", document)
            self.assertNotIn("__PAYLOAD__", document)


if __name__ == "__main__":
    unittest.main()
