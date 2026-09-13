from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

import numpy as np
import torch

from nbv.eval.gaussian_splatting import known_num_splat_cameras, GaussianParameters, SplatCameras
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.visibility import PerspectiveCamera
from scripts.repair_gaussian_alignment_cpu import (
    apply_checkpoint_transform, audit_cameras, fit_similarity, main,
    quaternion_rotation, render_rgb_3dgs_cpu, repair, digest,
)


class AlignmentRepairTests(unittest.TestCase):
    def test_3d_repair_publishes_gallery_without_mesh_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            source.mkdir()
            model = GaussianParameters(np.array([[0., 0., 0.], [.1, 0., 0.]]), np.full((2, 3), .5), "3dgs")
            torch.save({"schema_version": 1, "settings": {"backend": "3dgs", "resolution": 8},
                        "state_dict": model.state_dict()}, source / "checkpoint.pt")
            summary = {"backend": "3dgs", "object_id": "a/b", "history_anchor_ids": [0, 12],
                       "acquired_view_count": 2, "policy": "pun"}
            (source / "summary.json").write_text(json.dumps(summary))
            images = root / "rgb/a/b/images"
            images.mkdir(parents=True)
            for i in range(48):
                Image.new("RGB", (8, 8), (128, 128, 128)).save(images / f"viewpoint_{i}_offset_phi_0.png")
            visibility = root / "visibility/a/b.npz"
            visibility.parent.mkdir(parents=True)
            visibility.write_bytes(b"fixture")
            args = SimpleNamespace(data_root=root / "rgb", visibility_cache_root=root / "visibility",
                prediction_cache_root=root / "predictions", steps=2, starts=2, seed=0, dry_run=False, force=False)
            fit = {"scale": 1., "quaternion_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.],
                   "accepted": False, "silhouette_proxy_before": .1, "silhouette_proxy_after": .1}
            before = digest(source / "checkpoint.pt")
            with patch("scripts.repair_gaussian_alignment_cpu.load_visibility_cache", return_value=SimpleNamespace(
                metadata={"camera_radius": 2.73, "horizontal_fov_degrees": 52.})), patch(
                    "scripts.repair_gaussian_alignment_cpu.fit_similarity", return_value=fit) as fitting:
                self.assertIn("48 repaired", repair(source, output, args))
                self.assertEqual(len(fitting.call_args.args[2]), 2)
                self.assertIn("skipped", repair(source, output, args))
                self.assertEqual(fitting.call_count, 1)
                (output / "preview.png").unlink()
                self.assertIn("48 repaired", repair(source, output, args))
                self.assertEqual(fitting.call_count, 2)
            self.assertEqual(digest(source / "checkpoint.pt"), before)
            self.assertFalse((output / "metrics.csv").exists())
            result = json.loads((output / "summary.json").read_text())
            self.assertFalse(result["cpu_rendering"]["cuda_parity_verified"])
            self.assertEqual(len(result["ground_truth_comparison"]["held_out_view_ids"]), 46)
            self.assertTrue(Path(result["ground_truth_comparison"]["interactive"]).is_file())
            self.assertTrue(Path(result["turntable"]).is_file())
            self.assertNotIn(".alignment-repair-", json.dumps(result))

    def rgb_fixture(self, points, colors, opacity, *, view=None, resolution=1):
        model = GaussianParameters(np.asarray(points), np.asarray(colors), "3dgs")
        with torch.no_grad():
            model.opacity_logits.copy_(torch.logit(torch.tensor(opacity)))
            model.log_scales.fill_(math.log(.2))
        pose = np.eye(4) if view is None else view
        cameras = SplatCameras(np.linalg.inv(pose)[None], pose[None],
            np.array([[[2., 0., resolution/2], [0., 2., resolution/2], [0., 0., 1.]]]))
        return model, cameras

    def test_rgb_renderer_alpha_order_and_white_background(self):
        model, cameras = self.rgb_fixture([[0., 0., 4.], [0., 0., 2.]],
            [[.1, .2, .9], [.9, .2, .1]], [.8, .2])
        image = render_rgb_3dgs_cpu(model, cameras, 1)
        near_alpha, far_alpha = .2*.04/.34, .8*.01/.31
        expected = near_alpha*np.array([.9, .2, .1]) + (1-near_alpha)*far_alpha*np.array([.1, .2, .9]) + (1-near_alpha)*(1-far_alpha)
        np.testing.assert_allclose(image[0, 0, 0], expected, atol=1e-6)

    def test_rgb_renderer_camera_translation_and_clipping(self):
        view = np.eye(4)
        view[2, 3] = -1.
        model, cameras = self.rgb_fixture([[0., 0., .5], [0., 0., 3.]],
            [[.9, .1, .1], [.1, .9, .1]], [.9, .5], view=view)
        image = render_rgb_3dgs_cpu(model, cameras, 1)
        alpha = .5*.04/.34
        np.testing.assert_allclose(image[0, 0, 0], alpha*np.array([.1, .9, .1])+1-alpha, atol=1e-6)
        with torch.no_grad():
            model.means[:, 2].fill_(-1)
        np.testing.assert_allclose(render_rgb_3dgs_cpu(model, cameras, 1), 1.)

    def test_rgb_renderer_projects_anisotropic_covariance(self):
        model, cameras = self.rgb_fixture([[0., 0., 2.]], [[.1, .1, .1]], [.8], resolution=5)
        with torch.no_grad():
            model.log_scales.copy_(torch.log(torch.tensor([[1., .1, .1]])))
        image = render_rgb_3dgs_cpu(model, cameras, 5)[0]
        self.assertLess(image[2, 3, 0], image[3, 2, 0])
        with torch.no_grad():
            model.quaternions.copy_(torch.tensor([[2**-.5, 0., 0., 2**-.5]]))
        rotated = render_rgb_3dgs_cpu(model, cameras, 5)[0]
        np.testing.assert_allclose(rotated, image.transpose(1, 0, 2), atol=1e-6)

    def test_similarity_transforms_full_covariance_and_preserves_source(self):
        state = {
            "means": torch.tensor([[1., 2., 3.], [-1., .2, .5]]),
            "quaternions": torch.tensor([[.8, .2, .3, .4], [1., 0., 0., 0.]]),
            "log_scales": torch.log(torch.tensor([[.1, .2, .01], [.3, .1, .02]])),
            "opacity_logits": torch.tensor([1., 2.]),
            "color_logits": torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
        }
        original = {key: value.clone() for key, value in state.items()}
        checkpoint = {"schema_version": 1, "state_dict": state}
        q = np.array([np.cos(.4), 0., np.sin(.4), 0.])
        t = np.array([.2, -.5, .8])
        updated = apply_checkpoint_transform(checkpoint, 2.3, q, t)["state_dict"]
        rotation = quaternion_rotation(torch.tensor(q, dtype=torch.float32))

        def covariance(parameters):
            axes = quaternion_rotation(parameters["quaternions"])
            return axes @ torch.diag_embed(parameters["log_scales"].exp().square()) @ axes.transpose(-1, -2)

        torch.testing.assert_close(covariance(updated), 2.3**2 * rotation @ covariance(state) @ rotation.T)
        torch.testing.assert_close(updated["means"], 2.3*state["means"]@rotation.T+torch.tensor(t, dtype=torch.float32))
        for key, value in original.items():
            torch.testing.assert_close(state[key], value)
        for key in ("opacity_logits", "color_logits"):
            torch.testing.assert_close(updated[key], original[key])
        inverse_q = q * np.array([1., -1., -1., -1.])
        inverse_t = -(rotation.T.numpy() @ t)/2.3
        restored = apply_checkpoint_transform({"state_dict": updated}, 1/2.3, inverse_q, inverse_t)["state_dict"]
        torch.testing.assert_close(restored["means"], original["means"])
        torch.testing.assert_close(covariance(restored), covariance(original))

    def test_camera_audit_is_invariant_to_global_similarity(self):
        anchors = canonical_anchors()
        known = np.stack([anchors.by_id(i).camera_to_world(2.73) for i in (0, 12, 24)])
        known_cv = known @ np.diag([1., -1., -1., 1.])
        rotation = quaternion_rotation(torch.tensor([.8, .2, .1, .4], dtype=torch.float64)).numpy()
        predicted = known_cv.copy()
        predicted[:, :3, :3] = rotation @ predicted[:, :3, :3]
        predicted[:, :3, 3] = 3*predicted[:, :3, 3]@rotation.T + [1., 2., 3.]
        summary = {"object_id": "a/b", "history_anchor_ids": [0, 12, 24], "prediction_cache": "/old/location/cache.npz"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a/b").mkdir(parents=True)

            def save():
                np.savez(root / "a/b/cache.npz", camera_to_world=predicted,
                         identity_json=json.dumps({"history_anchor_ids": [0, 12, 24]}))

            save()
            audit = audit_cameras(summary, root, known)
            self.assertLess(max(audit["relative_rotation_errors_degrees"]), 1e-5)
            predicted[1, :3, :3] = predicted[1, :3, :3] @ np.diag([-1., -1., 1.])
            save()
            audit = audit_cameras(summary, root, known)
            self.assertGreater(audit["median_relative_rotation_error_degrees"], 179.)

    def test_cpu_fit_repairs_synthetic_placement_without_ground_truth_input(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        points = rng.normal(size=(500, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= [.6, .18, .8]
        anchors = canonical_anchors()
        poses = np.stack([anchors.by_id(i).camera_to_world(2.73) for i in (0, 12, 24, 36)])
        cameras = known_num_splat_cameras(poses, PerspectiveCamera(height=64, width=64))
        masks = []
        for view, intrinsic in zip(cameras.world_to_camera, cameras.intrinsics):
            cp = points @ view[:3, :3].T + view[:3, 3]
            xy = cp[:, :2]/cp[:, 2:] * intrinsic[[0, 1], [0, 1]] + intrinsic[:2, 2]
            mask = np.zeros((64, 64), dtype=bool)
            for x, y in xy.astype(int):
                mask[max(0, y-1):min(64, y+2), max(0, x-1):min(64, x+2)] = True
            masks.append(mask)
        displaced = points*.6 + [1.1, .8, -.4]
        fitted = fit_similarity(displaced, cameras, masks, resolution=64,
                                radius=2.73, steps=100, starts=4, seed=0)
        self.assertTrue(fitted["accepted"])
        self.assertFalse(fitted["ground_truth_used_for_fit"])
        self.assertLess(fitted["silhouette_proxy_after"], fitted["silhouette_proxy_before"]*.1)
        transform = np.array(fitted["transform"])
        repaired = displaced @ transform[:3, :3].T + transform[:3, 3]
        self.assertLess(np.linalg.norm(repaired.mean(0)), .2)
        self.assertGreater(fitted["scale"], 1.2)
        self.assertLess(fitted["scale"], 2.2)
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_similarity(displaced, cameras, masks[:1], resolution=64, radius=2.73)

    def test_output_cannot_overwrite_input_or_caches(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr"):
            root = Path(tmp)
            for output in (root / "input", root / "input/nested", root / "cache/repair"):
                with self.assertRaises(SystemExit) as caught:
                    main(["--input-root", str(root / "input"), "--output-root", str(output),
                          "--visibility-cache-root", str(root / "cache")])
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
