from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.evaluate_gaussian_splatting import (
    resolve_metric_cache,
    total_training_iterations,
)
from scripts.visualize_reconstruction import mapping_sha256

from nbv.eval.gaussian_splatting import (
    GaussianParameters,
    GaussianSplatSettings,
    SplatCameras,
    _rasterize_2dgs_training,
    _rasterize_3dgs,
    evaluate_2dgs_geometry,
    fuse_depth_surfaces,
    initialize_point_colors,
    known_num_splat_cameras,
    orbit_camera_poses,
    render_splats,
    require_gsplat,
    write_image_comparison_gallery,
    write_render_gallery,
)
from nbv.geometry.visibility import PerspectiveCamera


class GaussianSplattingTests(unittest.TestCase):
    def test_training_iterations_scale_with_acquired_views(self) -> None:
        self.assertEqual(total_training_iterations(1_500, 2), 3_000)
        self.assertEqual(total_training_iterations(1_500, 10), 15_000)
        with self.assertRaisesRegex(ValueError, "positive"):
            total_training_iterations(0, 10)

    def test_stale_metric_cache_path_resolves_by_prediction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            object_id = "category/object"
            identity = {"object_id": object_id, "history_anchor_ids": [0]}
            current = root / "metrics" / object_id / "current.json"
            current.parent.mkdir(parents=True)
            current.write_text(json.dumps({
                "identity": {
                    "prediction_id": mapping_sha256(identity),
                    "target_id": "target-hash",
                },
                "metrics": {},
            }), encoding="utf-8")

            path, payload = resolve_metric_cache(
                root, object_id, "/old/location/stale.json", identity
            )

            self.assertEqual(path, current)
            self.assertEqual(payload["identity"]["target_id"], "target-hash")

    def test_2dgs_distortion_loss_requests_depth_rendering(self) -> None:
        captured: dict[str, object] = {}
        expected = (object(),) * 7

        def rasterizer(*args: object, **kwargs: object) -> tuple[object, ...]:
            captured.update(kwargs)
            return expected

        actual = _rasterize_2dgs_training(
            rasterizer,
            (
                torch.zeros((2, 3)),
                torch.zeros((2, 4)),
                torch.zeros((2, 3)),
                torch.zeros(2),
                torch.tensor(
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True
                ),
            ),
            torch.zeros((3, 4, 4)),
            torch.zeros((3, 3, 3)),
            16,
            torch.ones((3, 3)),
        )

        self.assertIs(actual, expected)
        self.assertEqual(captured["render_mode"], "RGB+ED")
        self.assertIs(captured["distloss"], True)

    def test_2dgs_training_broadcasts_colors_across_cameras(self) -> None:
        captured: dict[str, object] = {}
        colors = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True
        )

        def rasterizer(*args: object, **kwargs: object) -> tuple[object, ...]:
            captured["colors"] = args[4]
            return (object(),) * 7

        _rasterize_2dgs_training(
            rasterizer,
            (
                torch.zeros((2, 3)),
                torch.zeros((2, 4)),
                torch.zeros((2, 3)),
                torch.zeros(2),
                colors,
            ),
            torch.zeros((3, 4, 4)),
            torch.zeros((3, 3, 3)),
            16,
            torch.ones((3, 3)),
        )

        expanded = captured["colors"]
        self.assertIsInstance(expanded, torch.Tensor)
        self.assertEqual(expanded.shape, (3, 2, 3))
        torch.testing.assert_close(expanded[0], colors)
        expanded.sum().backward()
        torch.testing.assert_close(colors.grad, torch.full_like(colors, 3.0))

    def test_3dgs_disables_packed_mode_for_camera_backgrounds(self) -> None:
        captured: dict[str, object] = {}
        expected = (object(),) * 3

        def rasterizer(*args: object, **kwargs: object) -> tuple[object, ...]:
            captured.update(kwargs)
            return expected

        actual = _rasterize_3dgs(
            rasterizer,
            (torch.zeros(1),) * 5,
            torch.zeros((3, 4, 4)),
            torch.zeros((3, 3, 3)),
            16,
            torch.ones((3, 3)),
            "RGB+ED",
        )

        self.assertIs(actual, expected)
        self.assertIs(captured["packed"], False)
        self.assertEqual(captured["render_mode"], "RGB+ED")
        self.assertEqual(captured["rasterize_mode"], "antialiased")

    def test_num_camera_conversion_flips_opengl_axes(self) -> None:
        pose = np.eye(4)[None]
        cameras = known_num_splat_cameras(
            pose, PerspectiveCamera(height=8, width=8, near=0.1, far=10)
        )
        np.testing.assert_array_equal(
            cameras.camera_to_world[0], np.diag([1.0, -1.0, -1.0, 1.0])
        )
        self.assertEqual(cameras.intrinsics[0, 0, 2], 4.0)
        self.assertEqual(cameras.intrinsics[0, 1, 2], 4.0)

    def test_projected_foreground_colors_initialize_points(self) -> None:
        intrinsic = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        cameras = SplatCameras(
            np.eye(4)[None], np.eye(4)[None], intrinsic[None]
        )
        image = np.zeros((1, 3, 3, 3), dtype=np.float32)
        image[0, 1, 1] = (0.2, 0.4, 0.6)
        masks = np.zeros((1, 3, 3), dtype=bool)
        masks[0, 1, 1] = True
        colors = initialize_point_colors(
            np.asarray([[0.0, 0.0, 2.0]]), image, masks, cameras
        )
        np.testing.assert_allclose(colors[0], (0.2, 0.4, 0.6))

    def test_orbit_camera_poses_are_closed_and_look_at_origin(self) -> None:
        poses = orbit_camera_poses(8, 2.73, elevation_degrees=20)
        self.assertEqual(poses.shape, (8, 4, 4))
        np.testing.assert_allclose(np.linalg.norm(poses[:, :3, 3], axis=1), 2.73)
        directions = poses[:, :3, 3] / 2.73
        np.testing.assert_allclose(poses[:, :3, 2], directions)

    def test_depth_fusion_backprojects_opencv_pixels(self) -> None:
        intrinsic = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        cameras = SplatCameras(
            np.eye(4)[None], np.eye(4)[None], intrinsic[None]
        )
        depth = np.zeros((1, 2, 2), dtype=np.float32)
        alpha = np.zeros_like(depth)
        depth[0, 0, 0] = 2.0
        alpha[0, 0, 0] = 1.0
        points = fuse_depth_surfaces(
            depth, alpha, cameras, alpha_threshold=0.5, point_count=10, seed=0
        )
        np.testing.assert_allclose(points, [[-0.5, -0.5, 2.0]])

    def test_2dgs_surface_render_uses_depth_feature_not_blue(self) -> None:
        model = GaussianParameters(
            np.asarray([[0.0, 0.0, 2.5], [0.1, 0.0, 2.5]]),
            np.full((2, 3), 0.25), "2dgs",
        )
        cameras = SplatCameras(
            np.repeat(np.eye(4)[None], 3, axis=0),
            np.repeat(np.eye(4)[None], 3, axis=0),
            np.repeat(np.eye(3)[None], 3, axis=0),
        )
        settings = GaussianSplatSettings("2dgs", resolution=1, render_batch_size=2)

        def rasterizer(*args: object, **kwargs: object) -> tuple[object, ...]:
            count = len(args[5])
            rgb = torch.full((count, 1, 1, 3), 0.25)
            alpha = torch.ones((count, 1, 1, 1))
            features = (
                torch.cat((rgb, torch.full_like(alpha, 2.5)), dim=-1)
                if kwargs["render_mode"] == "RGB+ED" else rgb
            )
            # Reproduce the upstream kernel's last-feature median contract.
            return features, alpha, None, None, None, features[..., -1:], {}

        with patch("nbv.eval.gaussian_splatting.require_gsplat", return_value=rasterizer):
            rgb, alpha, depth = render_splats(model, cameras, settings, include_depth=True)
            gallery, _, gallery_depth = render_splats(model, cameras, settings)

        self.assertEqual(rgb.shape, (3, 1, 1, 3))
        np.testing.assert_allclose(rgb, 0.25)
        np.testing.assert_allclose(gallery, rgb)
        self.assertIsNone(gallery_depth)
        np.testing.assert_allclose(depth, 2.5)
        surface = fuse_depth_surfaces(
            depth, alpha, cameras, alpha_threshold=0.5, point_count=10, seed=0,
        )
        np.testing.assert_allclose(surface, [[1.25, 1.25, 2.5]])

    def test_geometry_metrics_reject_3dgs_by_contract(self) -> None:
        model = GaussianParameters(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.full((3, 3), 0.5),
            "3dgs",
        )
        camera = SplatCameras(np.eye(4)[None], np.eye(4)[None], np.eye(3)[None])
        with self.assertRaisesRegex(ValueError, "restricted to 2dgs"):
            evaluate_2dgs_geometry(
                model, camera, np.ones((3, 3)), GaussianSplatSettings("3dgs")
            )

    def test_missing_cuda_has_actionable_error(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA-capable"):
                require_gsplat("2dgs")

    def test_turntable_is_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "turntable.html"
            write_render_gallery(path, np.zeros((2, 4, 4, 3)), "3DGS fixture")
            document = path.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", document)
            self.assertIn("3DGS fixture", document)
            self.assertNotIn("https://", document)

    def test_rgb_comparison_marks_training_and_held_out_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.html"
            references = np.zeros((2, 4, 4, 3))
            predictions = np.ones_like(references)
            write_image_comparison_gallery(
                path,
                references,
                predictions,
                "3DGS comparison fixture",
                training_view_ids=(0,),
            )
            document = path.read_text(encoding="utf-8")
            self.assertIn("Side by side", document)
            self.assertIn("Overlay opacity", document)
            self.assertIn("held-out view", document)
            self.assertIn('"training":[true,false]', document)
            self.assertEqual(document.count("data:image/png;base64,"), 4)
            self.assertNotIn("https://", document)


if __name__ == "__main__":
    unittest.main()
