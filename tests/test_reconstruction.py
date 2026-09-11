from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from nbv.eval.reconstruction import (
    ReconstructionSettings,
    align_vggt_to_num,
    evaluate_rollout_reconstruction,
    point_cloud_metrics,
    sample_mesh_surface,
)
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.mesh import TriangleMesh
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world


class _FakeReconstructor:
    provenance = {"backend": "fixture"}

    def __init__(self, points: np.ndarray, cameras: np.ndarray) -> None:
        self.points = points
        self.cameras = cameras
        self.calls = 0

    def reconstruct(self, image_paths):
        self.calls += 1
        return self.points.copy(), self.cameras[: len(image_paths)].copy()


class ReconstructionTests(unittest.TestCase):
    def test_point_metrics_are_zero_for_identical_clouds(self) -> None:
        points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        metrics = point_cloud_metrics(
            points, points, chunk_size=2, fscore_thresholds=(0.01, 0.02)
        )
        self.assertEqual(metrics["chamfer_l1_normalized"], 0.0)
        self.assertEqual(metrics["fscore_1pct"], 1.0)
        self.assertEqual(metrics["fscore_2pct"], 1.0)

    def test_camera_sim3_recovers_known_transform(self) -> None:
        anchors = canonical_anchors()
        known = np.stack([
            anchor_camera_to_world(anchors.by_id(index), 2.73)
            for index in (0, 7, 19)
        ])
        # Convert known OpenGL cameras to a synthetic VGGT/OpenCV frame.
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        predicted = known @ flip
        points = np.asarray([
            [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
        ])
        transform, metadata = align_vggt_to_num(points, predicted, known, points)
        np.testing.assert_allclose(transform, np.eye(4), atol=1e-8)
        self.assertEqual(metadata["mode"], "camera_pose_sim3")

    def test_surface_sampling_and_rollout_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            object_id = "cat/object"
            mesh_dir = root / "meshes" / object_id / "models"
            mesh_dir.mkdir(parents=True)
            mesh_dir.joinpath("model_normalized.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "element face 1\nproperty list uchar int vertex_indices\n"
                "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
            )
            mesh = TriangleMesh(
                np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                np.asarray([[0, 1, 2]]),
            )
            sampled = sample_mesh_surface(mesh, 32, 7)
            self.assertEqual(sampled.shape, (32, 3))
            self.assertTrue(np.allclose(sampled[:, 2], 0))
            image_paths = []
            for anchor in (0, 7):
                path = root / f"{anchor}.png"
                Image.new("RGB", (4, 4), (10, 20, 30)).save(path)
                image_paths.append(str(path))
            anchors = canonical_anchors()
            known = np.stack([
                anchor_camera_to_world(anchors.by_id(index), 2.73)
                for index in (0, 7)
            ])
            predicted = known @ np.diag([1.0, -1.0, -1.0, 1.0])
            reconstructor = _FakeReconstructor(sampled, predicted)
            rollout = SimpleNamespace(
                metadata={
                    "object_id": object_id,
                    "policy": "pun",
                    "visibility_cache_metadata": {
                        "camera_radius": 2.73,
                        "camera_convention": CAMERA_CONVENTION,
                        "mesh_relative_path": "models/model_normalized.ply",
                        "mesh_scale": 1,
                        "mesh_centering": "bounding_box",
                    },
                },
                acquired_view_counts=np.asarray([1, 2]),
                acquired_anchor_ids=np.asarray([0, 7]),
                image_paths=tuple(image_paths),
            )
            settings = ReconstructionSettings(
                enabled=True, policies=("pun",), view_counts=(2,), backend="vggt",
                model_id="fixture", image_size=4, point_source="point_map",
                device="cpu", confidence_percentile=0,
                foreground_threshold=245 / 255, predicted_point_count=32,
                ground_truth_point_count=32, chamfer_chunk_size=8,
                fscore_thresholds=(0.01, 0.02), mesh_root=root / "meshes",
                mesh_relative_path="models/model_normalized.ply", mesh_scale=1,
                mesh_centering="bounding_box", cache_root=root / "cache",
                model_cache_root=root / "models-cache",
            )
            first, _, first_meta = evaluate_rollout_reconstruction(
                [rollout], settings, reconstructor=reconstructor
            )
            second, _, second_meta = evaluate_rollout_reconstruction(
                [rollout], settings, reconstructor=reconstructor
            )
            self.assertEqual(reconstructor.calls, 1)
            self.assertFalse(first[0]["prediction_cache_hit"])
            self.assertTrue(second[0]["prediction_cache_hit"])
            self.assertEqual(first_meta["prediction_cache_misses"], 1)
            self.assertEqual(second_meta["prediction_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
