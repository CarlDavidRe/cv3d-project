from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nbv.eval.gaussian_splatting import known_num_splat_cameras
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.visibility import PerspectiveCamera
from scripts.repair_gaussian_alignment_cpu import (
    apply_checkpoint_transform, audit_cameras, fit_similarity, main,
    quaternion_rotation,
)


class AlignmentRepairTests(unittest.TestCase):
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
