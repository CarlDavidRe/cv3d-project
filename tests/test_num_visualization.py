from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from nbv.config import ConfigError
from scripts.inspect_num_sample import (
    normalize_uncertainty,
    write_interactive_sphere_html,
    write_target_svg,
)


class NUMVisualizationTests(unittest.TestCase):
    def test_step_numbered_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "step5" / "umap.svg"
            Image.new("RGB", (8, 8)).save(source)

            with self.assertRaisesRegex(ConfigError, "step-numbered"):
                write_target_svg(
                    np.arange(48, dtype=np.float32),
                    source,
                    output,
                    title="test sample",
                    target_name="MSE",
                    source_global_anchor_id=0,
                )
            self.assertFalse(output.exists())

    def test_quality_metrics_are_inverted_into_uncertainty(self) -> None:
        target = np.linspace(30.0, 10.0, 48, dtype=np.float32)

        uncertainty, direction = normalize_uncertainty(target, "PSNR")

        self.assertEqual(direction, "lower")
        self.assertAlmostEqual(float(uncertainty[0]), 0.0)
        self.assertAlmostEqual(float(uncertainty[-1]), 1.0)

    def test_error_metrics_increase_with_uncertainty(self) -> None:
        target = np.linspace(0.0, 4.0, 48, dtype=np.float32)

        uncertainty, direction = normalize_uncertainty(target, "LPIPS")

        self.assertEqual(direction, "higher")
        self.assertAlmostEqual(float(uncertainty[0]), 0.0)
        self.assertAlmostEqual(float(uncertainty[-1]), 1.0)

    def test_unknown_metric_requires_an_explicit_direction(self) -> None:
        target = np.arange(48, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "uncertainty direction"):
            normalize_uncertainty(target, "custom")

        uncertainty, direction = normalize_uncertainty(
            target, "custom", "lower-is-uncertain"
        )
        self.assertEqual(direction, "lower")
        self.assertEqual(int(np.argmax(uncertainty)), 0)

    def test_svg_embeds_input_and_uses_local_anchor_zero_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "umap.svg"
            Image.new("RGB", (8, 8), color=(240, 240, 240)).save(source)
            target = np.full(48, 15.0, dtype=np.float32)
            target[0] = 20.0
            target[18] = 10.0

            write_target_svg(
                target,
                source,
                output,
                title="test sample",
                target_name="PSNR",
                source_global_anchor_id=17,
            )

            ET.parse(output)
            svg = output.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", svg)
            self.assertIn('data-role="source" data-anchor-id="0"', svg)
            self.assertNotIn('data-role="source" data-anchor-id="17"', svg)
            self.assertIn(
                'data-role="most-uncertain" data-anchor-id="18"', svg
            )
            self.assertIn("global source anchor: 17", svg)
            self.assertIn("low PSNR → high uncertainty", svg)

    def test_interactive_html_is_self_contained_and_source_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "umap_3d.html"
            Image.new("RGB", (8, 8), color=(240, 240, 240)).save(source)
            target = np.full(48, 15.0, dtype=np.float32)
            target[0] = 20.0
            target[18] = 10.0

            write_interactive_sphere_html(
                target,
                source,
                output,
                title="test </script> sample",
                target_name="PSNR",
                source_global_anchor_id=17,
            )

            document = output.read_text(encoding="utf-8")
            self.assertIn('<canvas id="sphere"', document)
            self.assertIn("data:image/png;base64,", document)
            self.assertIn('"mostUncertainId":18', document)
            self.assertIn('"sourceGlobalAnchorId":17', document)
            self.assertIn('"id":0,"direction":[0.0,0.0,1.0]', document)
            self.assertIn("test \\u003c/script> sample", document)
            self.assertNotIn("https://", document)


if __name__ == "__main__":
    unittest.main()
