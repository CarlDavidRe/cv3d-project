import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from nbv.config import load_config
from nbv.data import build_history_dataset, relocated_history_manifest_identity
from nbv.data.visibility_cache import (
    VisibilityCache,
    save_visibility_cache,
    visibility_cache_path,
)
from nbv.experiments.phase3_controlled import (
    _validate_checkpoint_history_identity,
    parse_phase3_controlled_settings,
    run_phase3_controlled,
)
from nbv.features import CachedFeatureDataset, VGGTJointExtractor, save_feature_cache
from nbv.geometry.anchors import CANONICAL_ORDERING
from nbv.models import (
    IndependentHistoryGainModel,
    JointHistoryGainModel,
    TokenCandidateAttentionHistoryGainModel,
)


class _JointAwareAggregator(torch.nn.Module):
    patch_start_idx = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.input_shapes = []

    def forward(self, images):
        self.input_shapes.append(tuple(images.shape))
        means = images.mean(dim=(2, 3, 4))
        context = means.mean(dim=1, keepdim=True)
        values = torch.stack(
            (means, context.expand_as(means), means + context, means - context),
            dim=-1,
        )
        tokens = torch.stack((values, values + 0.1, values + 0.2), dim=2)
        return [tokens], self.patch_start_idx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class Phase3HistoryIdentityTests(unittest.TestCase):
    def test_accepts_checkpoints_saved_at_different_repository_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "data/processed/histories/example/manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest = {
                "dataset_id": "pending",
                "provenance": {
                    "history_config": str(root / "configs/history.yaml"),
                    "visibility_config": str(root / "configs/visibility.yaml"),
                    "evaluation_config": str(root / "configs/evaluation.yaml"),
                    "split_manifest": str(root / "data/splits/example.json"),
                },
                "split_manifest_sha256": "split",
                "coverage_target": "vis_a",
                "visibility_definition": {"kind": "fixture"},
                "sampling": {"seed": 0},
                "rotation_metadata": {"convention": "fixture"},
                "invalid_anchor_ids": [],
                "visibility_cache_ids": {"fixture": "cache"},
                "splits": {
                    split: {"sha256": f"{split}-sha"}
                    for split in ("train", "val", "test")
                },
            }
            dataset_id, _ = relocated_history_manifest_identity(
                manifest,
                current_repository_root=root,
                artifact_repository_root=root,
            )
            manifest["dataset_id"] = dataset_id
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            local_sha = _sha256(manifest_path)
            other_root = Path("/content/cv3d-project")
            relocated_id, relocated_sha = relocated_history_manifest_identity(
                manifest,
                current_repository_root=root,
                artifact_repository_root=other_root,
            )
            relative = manifest_path.relative_to(root)
            payloads = (
                {
                    "model_type": "phase3_independent_history_gain",
                    "supervision": {
                        "history_dataset_id": dataset_id,
                        "history_manifest_sha256": local_sha,
                        "history_manifest": str(manifest_path),
                    },
                },
                {
                    "model_type": "phase3_joint_history_gain",
                    "supervision": {
                        "history_dataset_id": relocated_id,
                        "history_manifest_sha256": relocated_sha,
                        "history_manifest": str(other_root / relative),
                    },
                },
            )

            identity = _validate_checkpoint_history_identity(
                manifest, manifest_path, root, payloads
            )

            self.assertTrue(identity["match"])
            self.assertEqual(
                identity["mode"], "mixed_exact_and_repository_root_relocation"
            )
            self.assertEqual(len(identity["checkpoint_identities"]), 2)


class Phase3ControlledTests(unittest.TestCase):
    def test_checked_in_config_pins_retained_control_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiments/phase3_controlled.yaml")
        settings = parse_phase3_controlled_settings(config, root)
        self.assertEqual(settings.ndcg_k, 5)
        self.assertEqual(settings.max_acquired_views, 10)
        self.assertEqual(settings.independent_checkpoint_sha256, _sha256(
            settings.independent_checkpoint
        ))
        self.assertEqual(settings.joint_checkpoint_sha256, _sha256(
            settings.joint_checkpoint
        ))

    def test_relocated_identity_reproduces_artifact_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data/processed/histories/random_unique_v1/manifest.json"
        manifest = json.loads(path.read_text())
        dataset_id, digest = relocated_history_manifest_identity(
            manifest,
            current_repository_root=root,
            artifact_repository_root="/content/cv3d-project",
        )
        self.assertEqual(
            dataset_id,
            "1e03c79186ae3a091368a058721cdb8617fb0ca6b8140b8013bbecfd09ce625a",
        )
        self.assertEqual(
            digest,
            "35ca3213c8453e662c17e03a8534618afaa41a12950f5d081f6a33845db4a8f3",
        )

    def test_complete_synthetic_step18_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "NUM"
            visibility_root = root / "visibility"
            objects = {
                "train": "cat/train_object",
                "val": "cat/val_object",
                "test": "cat/test_object",
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps({
                "schema_version": 1,
                "splits": {split: [object_id] for split, object_id in objects.items()},
            }))
            for offset, object_id in enumerate(objects.values()):
                image_root = data_root / object_id / "images"
                image_root.mkdir(parents=True)
                for anchor in range(48):
                    Image.new("RGB", (4, 4), color=(anchor + offset,) * 3).save(
                        image_root / f"viewpoint_{anchor}_offset_phi_0.png"
                    )
                visibility = np.zeros((48, 9), dtype=np.bool_)
                for anchor in range(48):
                    visibility[anchor, anchor % 9] = True
                save_visibility_cache(
                    VisibilityCache(
                        face_visibility=visibility,
                        face_areas=np.arange(1, 10, dtype=np.float64),
                        anchor_ids=np.arange(48, dtype=np.int16),
                        metadata={
                            "schema_version": 2,
                            "object_id": object_id,
                            "n_faces": 9,
                            "anchor_ordering": CANONICAL_ORDERING,
                            "visibility_definition": "fixture_faces",
                            "visibility_target": "vis_a",
                            "render_resolution": [8, 8],
                            "camera_radius": 2.73,
                        },
                    ),
                    visibility_cache_path(visibility_root, object_id),
                )
            manifest_path = build_history_dataset(
                root / "histories",
                data_root=data_root,
                visibility_cache_root=visibility_root,
                split_manifest=split_path,
                splits=["train", "val", "test"],
                history_lengths=[1, 2],
                histories_per_object_per_length=1,
                seed=3,
                coverage_target="vis_a",
            )
            manifest = json.loads(manifest_path.read_text())
            feature_path = root / "test_features.pt"
            test_pixels = torch.arange(48, dtype=torch.float32).add(2).div(255)
            independent_features = torch.stack((
                test_pixels + 0.2,
                test_pixels + 0.2,
                test_pixels.mul(2) + 0.2,
                torch.full_like(test_pixels, 0.2),
            ), dim=1)
            save_feature_cache(
                CachedFeatureDataset(
                    features=independent_features,
                    targets=torch.zeros(48, 48),
                    valid_mask=torch.ones(48, 48, dtype=torch.bool),
                    sample_ids=tuple(f"{objects['test']}/{anchor}" for anchor in range(48)),
                    source_anchor_ids=torch.arange(48),
                    metadata={
                        "split": "test",
                        "backbone": "vggt",
                        "variant": "fixture",
                        "feature_components": ["max_pooled_patch"],
                        "history_mode": "single_image",
                        "split_manifest_sha256": manifest["split_manifest_sha256"],
                    },
                ),
                feature_path,
            )
            extractor = VGGTJointExtractor(
                model=_JointAwareAggregator(), image_size=4, device="cpu",
                expected_feature_dim=4,
            )
            independent_model = IndependentHistoryGainModel(4, hidden_dim=8, dropout=0.0)
            joint_model = JointHistoryGainModel(extractor, 4, hidden_dim=8, dropout=0.0)
            supervision = {
                "target": "target_surface_gain",
                "coverage_target": "vis_a",
                "target_direction": "higher",
                "history_dataset_id": manifest["dataset_id"],
                "history_manifest": str(manifest_path),
                "history_manifest_sha256": _sha256(manifest_path),
            }
            common_model = {
                "feature_dim": 4,
                "hidden_dim": 8,
                "num_anchors": 48,
                "dropout": 0.0,
                "include_anchor_directions": True,
                "aggregation": "masked_mean",
                "anchor_ordering": CANONICAL_ORDERING,
            }
            independent_checkpoint = root / "independent.pt"
            torch.save({
                "schema_version": 1,
                "model_type": "phase3_independent_history_gain",
                "model": {**common_model, "backbone_history_mode": "independent_single_image"},
                "state_dict": independent_model.state_dict(),
                "feature": {
                    "components": ["max_pooled_patch"],
                    "cache_sha256": {"test": _sha256(feature_path)},
                    "metadata": {"test": {"backbone_configuration": {
                        "model_id": "fixture/VGGT",
                        "image_size": 4,
                        "layer_index": -1,
                        "preprocessing": "square_resize_rgb_0_1",
                    }}},
                },
                "supervision": supervision,
                "checkpoint_selection": {"split": "val", "best_epoch": 2},
            }, independent_checkpoint)
            joint_checkpoint = root / "joint.pt"
            torch.save({
                "schema_version": 1,
                "model_type": "phase3_joint_history_gain",
                "model": {**common_model, "backbone_history_mode": "joint_multiview"},
                "backbone": {
                    "name": "vggt",
                    "model_id": "fixture/VGGT",
                    "image_size": 4,
                    "layer_index": -1,
                    "feature_components": ["max_pooled_patch"],
                    "history_mode": "joint_multiview",
                    "padding_strategy": "group_by_real_history_length",
                    "frozen": True,
                },
                "state_dict": joint_model.state_dict(),
                "supervision": supervision,
                "checkpoint_selection": {"split": "val", "best_epoch": 2},
            }, joint_checkpoint)
            phase2_summary = root / "phase2.json"
            phase2_summary.write_text(json.dumps({
                "policies": [
                    {"policy": "pun"},
                    {"policy": "oracle"},
                    {"policy": "random"},
                    {"policy": "vggt"},
                    {"policy": "farthest"},
                ]
            }))
            config = {
                "schema_version": 1,
                "experiment": {
                    "phase": "phase3", "name": "controlled", "seed": 3,
                    "deterministic": True,
                },
                "paths": {
                    "data_root": str(data_root),
                    "visibility_cache_root": str(visibility_root),
                    "output_root": str(root / "outputs"),
                    "model_cache_root": str(root / "models"),
                },
                "phase3": {"controlled": {
                    "history_manifest": str(manifest_path),
                    "coverage_target": "vis_a",
                    "device": "auto",
                    "independent": {
                        "checkpoint": str(independent_checkpoint),
                        "checkpoint_sha256": _sha256(independent_checkpoint),
                        "test_feature_cache": str(feature_path),
                    },
                    "joint": {
                        "checkpoint": str(joint_checkpoint),
                        "checkpoint_sha256": _sha256(joint_checkpoint),
                        "test_feature_cache": {
                            "batch_size": 1,
                            "num_workers": 0,
                            "pin_memory": False,
                            "cache_every_batches": 1,
                        },
                    },
                    "one_step": {
                        "batch_size": 2,
                        "huber_delta": 0.01,
                        "ranking_weight": 0.1,
                        "ranking_margin": 0.0,
                        "ndcg_k": 5,
                    },
                    "closed_loop": {
                        "split": "test",
                        "object_ids": [],
                        "limit": None,
                        "skip_missing_caches": False,
                        "initial_anchor_ids": [0],
                        "max_acquired_views": 3,
                    },
                    "external_references": {"phase2_summary": str(phase2_summary)},
                }},
            }
            settings = parse_phase3_controlled_settings(config, root)
            self.assertEqual(settings.max_acquired_views, 3)
            with mock.patch(
                "nbv.experiments.phase3_controlled.torch.cuda.is_available",
                return_value=False,
            ):
                run = run_phase3_controlled(config, root, joint_extractor=extractor)
            completion = json.loads((run / "metrics/phase3_completion.json").read_text())
            summary = json.loads((run / "metrics/summary.json").read_text())
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(len(summary["one_step"]), 2)
            self.assertGreaterEqual(
                summary["length_one_feature_equivalence"]["cosine_similarity_min"],
                0.999,
            )
            self.assertEqual(
                summary["profiling"]["parameters"]["independent_trainable"], 510
            )
            self.assertEqual(summary["profiling"]["parameters"]["joint_trainable"], 510)
            self.assertEqual(summary["cohort"]["evaluated_object_count"], 1)
            self.assertEqual(
                [row["policy"] for row in summary["external_references"]["policies"]],
                ["random", "farthest", "pun", "vggt", "oracle"],
            )
            self.assertTrue((run / "metrics/report.md").is_file())
            self.assertEqual(len(list((run / "rollouts").glob("*/*/*.npz"))), 2)

            with mock.patch(
                "nbv.experiments.phase3_controlled.torch.cuda.is_available",
                return_value=False,
            ):
                resumed = run_phase3_controlled(
                    config, root, joint_extractor=extractor, resume=True
                )
            self.assertEqual(resumed, run)

            token_extractor = VGGTJointExtractor(
                model=_JointAwareAggregator(),
                image_size=4,
                device="cpu",
                expected_feature_dim=4,
                spatial_token_grid_size=1,
            )
            token_model = TokenCandidateAttentionHistoryGainModel(
                token_extractor,
                4,
                attention_dim=8,
                attention_heads=2,
                score_hidden_dim=8,
                dropout=0.0,
            )
            token_checkpoint = root / "token_joint.pt"
            torch.save({
                "schema_version": 1,
                "model_type": "phase3_joint_history_gain",
                "model": {
                    **common_model,
                    "architecture": "token_candidate_attention",
                    "aggregation": "candidate_cross_attention",
                    "backbone_history_mode": "joint_multiview_spatial_tokens",
                    "attention_dim": 8,
                    "attention_heads": 2,
                    "score_hidden_dim": 8,
                    "token_grid_size": 1,
                },
                "backbone": {
                    "name": "vggt",
                    "model_id": "fixture/VGGT",
                    "image_size": 4,
                    "layer_index": -1,
                    "feature_components": ["max_pooled_patch"],
                    "representation": "spatial_patch_tokens",
                    "spatial_token_grid_size": 1,
                    "history_mode": "joint_multiview",
                    "padding_strategy": "group_by_real_history_length",
                    "frozen": True,
                },
                "state_dict": token_model.state_dict(),
                "supervision": supervision,
                "checkpoint_selection": {"split": "val", "best_epoch": 2},
            }, token_checkpoint)
            expressive_config = json.loads(json.dumps(config))
            expressive_config["experiment"]["name"] = "controlled_token"
            expressive = expressive_config["phase3"]["controlled"]
            expressive["comparison_mode"] = "expressive_joint_variant"
            expressive["joint"]["checkpoint"] = str(token_checkpoint)
            expressive["joint"]["checkpoint_sha256"] = _sha256(token_checkpoint)

            expressive_run = run_phase3_controlled(
                expressive_config,
                root,
                joint_extractor=token_extractor,
            )
            expressive_summary = json.loads(
                (expressive_run / "metrics/summary.json").read_text()
            )
            expressive_completion = json.loads(
                (expressive_run / "metrics/phase3_completion.json").read_text()
            )
            self.assertEqual(
                expressive_summary["comparison_mode"], "expressive_joint_variant"
            )
            self.assertEqual(
                expressive_summary["length_one_feature_equivalence"]["status"],
                "not_shape_comparable",
            )
            self.assertIn(
                "vggt_joint_token_attention", expressive_summary["policies"]
            )
            self.assertNotEqual(
                expressive_summary["profiling"]["parameters"]["independent_trainable"],
                expressive_summary["profiling"]["parameters"]["joint_trainable"],
            )
            self.assertEqual(expressive_completion["status"], "complete")


if __name__ == "__main__":
    unittest.main()
