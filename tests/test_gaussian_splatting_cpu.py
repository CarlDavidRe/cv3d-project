from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from nbv.eval.gaussian_splatting import (
    GaussianParameters, GaussianSplatSettings, SplatCameras,
    save_gaussian_checkpoint, write_point_ply,
)
from nbv.eval.gaussian_splatting_cpu import render_depth_cpu
from scripts.reevaluate_gaussian_splatting_cpu import read_point_ply, recover


class CpuDepthTests(unittest.TestCase):
    def render(self, z, opacity, *, blue=0.1, camera_z=0.0):
        import torch
        model = GaussianParameters(
            np.array([[0., 0., d] for d in z]), np.full((len(z), 3), blue), "2dgs",
        )
        with torch.no_grad():
            model.log_scales.fill_(0)
            model.opacity_logits.copy_(torch.logit(torch.tensor(opacity)))
        c2w = np.eye(4)
        c2w[2, 3] = camera_z
        cameras = SplatCameras(c2w[None], np.linalg.inv(c2w)[None],
                              np.array([[[2., 0., 0.5], [0., 2., 0.5], [0., 0., 1.]]]))
        return render_depth_cpu(model, cameras, 1)

    def test_median_uses_depth_and_front_to_back_opacity(self):
        # Near layer is transparent: far layer crosses accumulated alpha 0.5.
        alpha, depth = self.render([4., 2.], [0.8, 0.2])
        np.testing.assert_allclose(alpha, 0.84, atol=1e-6)
        np.testing.assert_allclose(depth, 4.)
        _, different_color = self.render([4., 2.], [0.8, 0.2], blue=0.9)
        np.testing.assert_array_equal(depth, different_color)

    def test_opaque_near_layer_sets_median(self):
        alpha, depth = self.render([4., 2.], [0.8, 0.8])
        np.testing.assert_allclose(alpha, 0.96, atol=1e-6)
        np.testing.assert_allclose(depth, 2.)

    def test_camera_translation_and_near_clipping(self):
        alpha, depth = self.render([0., 3.], [0.9, 0.9], camera_z=1.)
        np.testing.assert_allclose(alpha, 0.9, atol=1e-6)
        np.testing.assert_allclose(depth, 2.)

    def test_no_visible_splats(self):
        alpha, depth = self.render([-2., -1.], [0.9, 0.9])
        np.testing.assert_array_equal(alpha, 0.)
        np.testing.assert_array_equal(depth, 0.)

    def test_tilted_surfel_uses_perspective_intersection(self):
        import torch
        model = GaussianParameters(np.array([[0., 0., 2.]]), np.full((1, 3), 0.2), "2dgs")
        with torch.no_grad():
            model.log_scales.fill_(0)
            model.quaternions.copy_(torch.tensor([[np.cos(np.pi/8), 0., np.sin(np.pi/8), 0.]]))
            model.opacity_logits.fill_(np.log(4.))
        cameras = SplatCameras(np.eye(4)[None], np.eye(4)[None],
            np.array([[[2., 0., 1.5], [0., 2., 1.5], [0., 0., 1.]]]))
        alpha, depth = render_depth_cpu(model, cameras, 3)
        self.assertAlmostEqual(float(alpha[0, 1, 2]), 0.8*np.exp(-4/9), places=5)
        self.assertEqual(depth[0, 1, 2], 2.)

    def test_recovery_publishes_and_resumes_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "input", root / "output"
            source.mkdir()
            visibility = root / "visibility/a/b.npz"
            visibility.parent.mkdir(parents=True)
            visibility.write_bytes(b"fixture")
            (source / "summary.json").write_text(json.dumps({
                "backend": "2dgs", "object_id": "a/b", "policy": "vggt", "acquired_view_count": 2,
            }))
            model = GaussianParameters(np.array([[0., 0., 0.], [0.1, 0., 0.]]),
                                       np.full((2, 3), 0.5), "2dgs")
            save_gaussian_checkpoint(source / "checkpoint.pt", model,
                                     GaussianSplatSettings("2dgs", resolution=1), [])
            write_point_ply(source / "ground_truth.ply", np.eye(3), (0, 0, 255))
            original = {p.name: p.read_bytes() for p in source.iterdir()}
            metadata = {"camera_radius": 2.73, "horizontal_fov_degrees": 60, "near": 0.1, "far": 10.}
            prefix = "scripts.reevaluate_gaussian_splatting_cpu."
            with patch(prefix + "load_visibility_cache", return_value=SimpleNamespace(metadata=metadata)), \
                 patch(prefix + "render_depth_cpu", return_value=(np.ones((48, 1, 1)), np.full((48, 1, 1), 2.73))) as renderer:
                self.assertIn("ready", recover(source, output, root / "visibility", dry_run=True))
                renderer.assert_not_called()
                self.assertFalse(output.exists())
                self.assertIn("complete", recover(source, output, root / "visibility"))
                self.assertIn("skipped", recover(source, output, root / "visibility"))
                self.assertEqual(renderer.call_count, 1)
                (output / "surface.ply").unlink()
                self.assertIn("complete", recover(source, output, root / "visibility"))
                self.assertEqual(renderer.call_count, 2)
            self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_recovery_requires_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps({"backend": "2dgs", "object_id": "a/b"}))
            with self.assertRaises(FileNotFoundError):
                recover(root, root / "output", root / "visibility", dry_run=True)
            self.assertFalse((root / "output").exists())

    def test_saved_ply_count_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cloud.ply"
            path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nend_header\n1 2 3\n")
            with self.assertRaisesRegex(ValueError, "invalid point cloud"):
                read_point_ply(path)


if __name__ == "__main__":
    unittest.main()
