from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from nbv.config import load_config
from nbv.data import (
    HistoryFeatureSample,
    HistorySample,
    MaterializedHistoryFeatureDataset,
    collate_history_features,
)
from nbv.data import (
    VisibilityCache,
    build_history_dataset,
    save_visibility_cache,
    visibility_cache_path,
)
from nbv.experiments.phase3_joint import (
    load_joint_history_checkpoint,
    parse_phase3_joint_settings,
    run_phase3_joint,
)
from nbv.features import CachedFeatureDataset, VGGTJointExtractor, save_feature_cache
from nbv.geometry import CANONICAL_ORDERING
from nbv.models import (
    IndependentHistoryGainModel,
    JointHistoryGainModel,
    PoseConditionedDeepSetsHistoryGainModel,
    TokenCandidateAttentionHistoryGainModel,
    count_trainable_parameters,
)
from nbv.training import evaluate_history_model, fit_history_model


class _JointAwareAggregator(nn.Module):
    patch_start_idx = 3

    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.dimension = dimension
        self.frozen_marker = nn.Parameter(torch.ones(()))
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        self.input_shapes.append(tuple(images.shape))
        marker = images.mean(dim=(2, 3, 4))
        # Every output frame explicitly depends on every input frame.
        context = marker.mean(dim=1, keepdim=True)
        values = marker + context
        offsets = torch.arange(7, device=images.device).view(1, 1, 7, 1)
        channels = torch.arange(self.dimension, device=images.device).view(
            1, 1, 1, self.dimension
        )
        tokens = values[:, :, None, None] + offsets + channels / 10
        return [tokens * self.frozen_marker], self.patch_start_idx


class JointExtractorTests(unittest.TestCase):
    def test_complete_histories_share_one_forward_and_padding_never_enters(self) -> None:
        aggregator = _JointAwareAggregator()
        extractor = VGGTJointExtractor(
            model=aggregator,
            image_size=8,
            device="cpu",
            feature_components=["max_pooled_patch"],
            expected_feature_dim=4,
        )
        images = torch.rand(2, 3, 3, 5, 6)
        padding = torch.tensor([[False, True, True], [False, False, False]])
        changed = images.clone()
        changed[padding] = 1.0

        first = extractor.extract(images, padding)
        second = extractor.extract(changed, padding)

        self.assertEqual(
            aggregator.input_shapes,
            [(1, 1, 3, 8, 8), (1, 3, 3, 8, 8)] * 2,
        )
        self.assertEqual(first.view_features.shape, (2, 3, 4))
        self.assertEqual(torch.count_nonzero(first.view_features[padding]).item(), 0)
        torch.testing.assert_close(first.view_features, second.view_features)
        self.assertEqual(first.metadata["history_mode"], "joint_multiview")

    def test_another_view_changes_a_frames_joint_feature(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(),
            image_size=6,
            device="cpu",
            expected_feature_dim=4,
        )
        images = torch.zeros(1, 2, 3, 6, 6)
        padding = torch.zeros(1, 2, dtype=torch.bool)
        baseline = extractor.extract(images, padding).view_features[:, 0]
        images[:, 1] = 1.0
        changed = extractor.extract(images, padding).view_features[:, 0]

        self.assertFalse(torch.equal(baseline, changed))

    def test_rejects_non_suffix_padding(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(), image_size=6, device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "suffix padding"):
            extractor.extract(
                torch.zeros(1, 3, 3, 6, 6),
                torch.tensor([[False, True, False]]),
            )

    def test_spatial_token_grid_retains_multiple_tokens_per_view(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(),
            image_size=6,
            device="cpu",
            expected_feature_dim=4,
            spatial_token_grid_size=2,
        )
        images = torch.rand(2, 2, 3, 6, 6)
        padding = torch.tensor([[False, True], [False, False]])

        result = extractor.extract(images, padding)

        self.assertEqual(result.view_features.shape, (2, 2, 4, 4))
        self.assertEqual(torch.count_nonzero(result.view_features[padding]).item(), 0)
        self.assertEqual(result.metadata["representation"], "spatial_patch_tokens")
        self.assertEqual(result.metadata["tokens_per_view"], 4)


class JointHistoryModelTests(unittest.TestCase):
    def test_joint_head_capacity_exactly_matches_independent_control(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(),
            image_size=6,
            device="cpu",
            expected_feature_dim=4,
        )
        joint = JointHistoryGainModel(extractor, 4, hidden_dim=9, dropout=0.1)
        independent = IndependentHistoryGainModel(4, hidden_dim=9, dropout=0.1)

        self.assertEqual(
            count_trainable_parameters(joint),
            count_trainable_parameters(independent),
        )
        self.assertTrue(all(not p.requires_grad for p in extractor.model.parameters()))

    def test_padding_is_ignored_by_extractor_and_pooling(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(),
            image_size=6,
            device="cpu",
            expected_feature_dim=4,
        )
        model = JointHistoryGainModel(extractor, 4, hidden_dim=7, dropout=0.0).eval()
        images = torch.rand(2, 3, 3, 6, 6)
        anchors = torch.tensor([[2, -1, -1], [4, 1, 9]])
        padding = torch.tensor([[False, True, True], [False, False, False]])
        changed = images.clone()
        changed[padding] = 1.0

        first = model(images, anchors, padding)
        second = model(changed, anchors, padding)

        self.assertEqual(first.shape, (2, 48))
        torch.testing.assert_close(first, second)

    def test_pose_deepsets_preserves_feature_direction_pairing(self) -> None:
        torch.manual_seed(11)
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(), image_size=6, device="cpu",
            expected_feature_dim=4,
        )
        model = PoseConditionedDeepSetsHistoryGainModel(
            extractor, 4, element_dim=8, hidden_dim=7, dropout=0.0
        ).eval()
        features = torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]])
        swapped = features.flip(1)
        anchors = torch.tensor([[0, 13]])
        padding = torch.zeros((1, 2), dtype=torch.bool)

        first = model.forward_features(features, anchors, padding)
        second = model.forward_features(swapped, anchors, padding)

        self.assertFalse(torch.allclose(first, second))

    def test_token_candidate_attention_returns_one_score_per_anchor(self) -> None:
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(), image_size=6, device="cpu",
            expected_feature_dim=4, spatial_token_grid_size=2,
        )
        model = TokenCandidateAttentionHistoryGainModel(
            extractor,
            4,
            attention_dim=8,
            attention_heads=2,
            score_hidden_dim=6,
            dropout=0.0,
        ).eval()
        images = torch.rand(2, 3, 3, 6, 6)
        anchors = torch.tensor([[2, -1, -1], [4, 1, 9]])
        padding = torch.tensor([[False, True, True], [False, False, False]])
        changed = images.clone()
        changed[padding] = 1.0

        first = model(images, anchors, padding)
        second = model(changed, anchors, padding)

        self.assertEqual(first.shape, (2, 48))
        torch.testing.assert_close(first, second)


class _SyntheticImageHistories:
    def __init__(self, split: str, samples: list[HistorySample]) -> None:
        self.split = split
        self.load_images = True
        self.samples = samples
        self.manifest = {"dataset_id": "joint-synthetic"}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HistorySample:
        return self.samples[index]


def _image_histories(split: str) -> _SyntheticImageHistories:
    samples = []
    for index, anchors in enumerate(((0,), (1, 2), (3,), (4, 5))):
        images = tuple(
            torch.full((3, 4, 4), (anchor + 1) / 10) for anchor in anchors
        )
        value = sum(float(image.mean()) for image in images) / len(images)
        target = np.linspace(value, value + 0.4, 48, dtype=np.float32)
        valid = np.ones(48, dtype=np.bool_)
        valid[list(anchors)] = False
        target[list(anchors)] = 0
        samples.append(HistorySample(
            sample_id=f"{split}-{index}",
            object_id=f"category/{split}_object",
            history_image_paths=tuple(f"{anchor}.png" for anchor in anchors),
            history_anchor_ids=anchors,
            history_length=len(anchors),
            target_surface_gain=target,
            valid_candidate_mask=valid,
            rotation_metadata={"enabled": False},
            split=split,
            visibility_cache_id="b" * 64,
            coverage_target="vis_a",
            sampling_metadata={},
            history_images=images,
        ))
    return _SyntheticImageHistories(split, samples)


class JointHistoryTrainingTests(unittest.TestCase):
    def test_spatial_token_feature_samples_collate_with_history_padding(self) -> None:
        samples = []
        for index, length in enumerate((1, 2)):
            samples.append(HistoryFeatureSample(
                sample_id=f"token-{index}",
                object_id="category/object",
                history_features=torch.full((length, 4, 6), float(index + 1)),
                history_anchor_ids=torch.arange(length),
                target_surface_gain=torch.zeros(48),
                valid_candidate_mask=torch.ones(48, dtype=torch.bool),
                visibility_cache_id="c" * 64,
            ))

        batch = collate_history_features(samples)

        self.assertEqual(batch.history_features.shape, (2, 2, 4, 6))
        self.assertTrue(batch.history_padding_mask[0, 1])
        self.assertEqual(torch.count_nonzero(batch.history_features[0, 1]).item(), 0)

    def test_token_attention_trains_from_materialized_spatial_features(self) -> None:
        histories = _image_histories("train")
        histories._arrays = {
            "sample_ids": np.asarray(
                [sample.sample_id for sample in histories.samples], dtype=np.str_
            )
        }
        samples = []
        for index, sample in enumerate(histories.samples):
            length = sample.history_length
            samples.append(HistoryFeatureSample(
                sample_id=sample.sample_id,
                object_id=sample.object_id,
                history_features=torch.rand(length, 4, 6),
                history_anchor_ids=torch.as_tensor(sample.history_anchor_ids),
                target_surface_gain=torch.from_numpy(sample.target_surface_gain.copy()),
                valid_candidate_mask=torch.from_numpy(sample.valid_candidate_mask.copy()),
                visibility_cache_id=sample.visibility_cache_id,
            ))
        materialized = MaterializedHistoryFeatureDataset(histories, samples)
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(dimension=6),
            image_size=4,
            device="cpu",
            expected_feature_dim=6,
            spatial_token_grid_size=2,
        )
        model = TokenCandidateAttentionHistoryGainModel(
            extractor,
            6,
            attention_dim=8,
            attention_heads=2,
            score_hidden_dim=8,
            dropout=0.0,
        )

        fit = fit_history_model(
            model,
            materialized,
            materialized,
            epochs=2,
            batch_size=2,
            learning_rate=0.01,
            weight_decay=0.0,
            ranking_weight=0.0,
            patience=None,
            device="cpu",
            seed=4,
        )
        result = evaluate_history_model(model, materialized, batch_size=2)

        self.assertGreater(fit.best_epoch, 0)
        self.assertEqual(result.predictions.shape, (4, 48))

    def test_joint_rgb_histories_use_shared_loss_and_validation_selection(self) -> None:
        train = _image_histories("train")
        extractor = VGGTJointExtractor(
            model=_JointAwareAggregator(),
            image_size=4,
            device="cpu",
            expected_feature_dim=4,
        )
        model = JointHistoryGainModel(extractor, 4, hidden_dim=64, dropout=0.0)
        initial = evaluate_history_model(model, train, batch_size=2).summary["huber_loss"]

        fit = fit_history_model(
            model,
            train,
            train,
            epochs=120,
            batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=0.03,
            weight_decay=0.0,
            ranking_weight=0.0,
            patience=None,
            device="cpu",
            seed=5,
        )
        final = evaluate_history_model(model, train, batch_size=2)

        self.assertLess(final.summary["huber_loss"], initial * 0.08)
        self.assertGreater(fit.best_epoch, 0)
        self.assertEqual(final.predictions.shape, (4, 48))

    def test_mid_epoch_checkpoint_resumes_to_identical_result(self) -> None:
        import nbv.training.history as history_training

        train = _image_histories("train")
        torch.manual_seed(23)
        reference = JointHistoryGainModel(
            VGGTJointExtractor(
                model=_JointAwareAggregator(), image_size=4, device="cpu",
                expected_feature_dim=4,
            ),
            4,
            hidden_dim=12,
            dropout=0.0,
        )
        initial_state = {
            key: value.detach().clone() for key, value in reference.state_dict().items()
        }
        fit_kwargs = {
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "ranking_weight": 0.0,
            "patience": None,
            "device": "cpu",
            "seed": 9,
        }
        reference_fit = fit_history_model(reference, train, train, **fit_kwargs)

        interrupted = JointHistoryGainModel(
            VGGTJointExtractor(
                model=_JointAwareAggregator(), image_size=4, device="cpu",
                expected_feature_dim=4,
            ),
            4,
            hidden_dim=12,
            dropout=0.0,
        )
        interrupted.load_state_dict(initial_state)
        original_save = history_training._save_training_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "training_state.pt"

            def save_then_interrupt(*args: object, **kwargs: object) -> None:
                original_save(*args, **kwargs)
                if (
                    kwargs["epoch_in_progress"] == 1
                    and kwargs["batches_completed"] == 1
                ):
                    raise RuntimeError("simulated interruption")

            with patch.object(
                history_training,
                "_save_training_checkpoint",
                side_effect=save_then_interrupt,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    fit_history_model(
                        interrupted,
                        train,
                        train,
                        training_checkpoint_path=checkpoint,
                        checkpoint_every_batches=1,
                        checkpoint_identity={"fixture": "joint"},
                        **fit_kwargs,
                    )

            resumed = JointHistoryGainModel(
                VGGTJointExtractor(
                    model=_JointAwareAggregator(), image_size=4, device="cpu",
                    expected_feature_dim=4,
                ),
                4,
                hidden_dim=12,
                dropout=0.0,
            )
            with patch.object(
                history_training.torch,
                "load",
                wraps=history_training.torch.load,
            ) as load_checkpoint:
                resumed_fit = fit_history_model(
                    resumed,
                    train,
                    train,
                    training_checkpoint_path=checkpoint,
                    checkpoint_every_batches=1,
                    resume=True,
                    checkpoint_identity={"fixture": "joint"},
                    **fit_kwargs,
                )
            self.assertEqual(load_checkpoint.call_args.kwargs["map_location"], "cpu")

        self.assertEqual(resumed_fit.history, reference_fit.history)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


class JointConfigTests(unittest.TestCase):
    def test_checked_in_joint_config_matches_step16_control(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiments/phase3_joint.yaml")
        settings = parse_phase3_joint_settings(config, root)

        self.assertEqual(settings.feature_components, ("max_pooled_patch",))
        self.assertEqual(settings.expected_feature_dim, 2048)
        self.assertEqual(
            settings.training["batch_size"]
            * settings.training["gradient_accumulation_steps"],
            64,
        )
        self.assertEqual(settings.training["batch_size"], 64)
        self.assertEqual(settings.training["gradient_accumulation_steps"], 1)
        self.assertTrue(settings.feature_precompute_enabled)
        self.assertEqual(settings.feature_precompute_batch_size, 4)
        self.assertEqual(settings.feature_precompute_num_workers, 4)
        self.assertTrue(settings.feature_precompute_pin_memory)
        self.assertEqual(settings.evaluation_batch_size, 256)
        self.assertEqual(settings.preflight_history_lengths, (1, 2))

    def test_follow_up_variant_configs_parse(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pose = parse_phase3_joint_settings(
            load_config(root / "configs/experiments/phase3_joint_pose_deepsets.yaml"),
            root,
        )
        token = parse_phase3_joint_settings(
            load_config(root / "configs/experiments/phase3_joint_token_attention.yaml"),
            root,
        )

        self.assertEqual(pose.architecture, "pose_deepsets")
        self.assertEqual(pose.element_dim, 128)
        self.assertIsNone(pose.token_grid_size)
        self.assertEqual(token.architecture, "token_candidate_attention")
        self.assertEqual(token.backbone_representation, "spatial_patch_tokens")
        self.assertEqual(token.token_grid_size, 2)
        self.assertEqual(token.attention_heads, 4)
        self.assertEqual(
            token.training["batch_size"]
            * token.training["gradient_accumulation_steps"],
            64,
        )

    def test_runner_saves_joint_checkpoint_match_and_memory_preflight(self) -> None:
        from PIL import Image

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
                visibility = np.zeros((48, 7), dtype=np.bool_)
                for anchor in range(48):
                    visibility[anchor, anchor % 7] = True
                save_visibility_cache(
                    VisibilityCache(
                        face_visibility=visibility,
                        face_areas=np.arange(1, 8, dtype=np.float64),
                        anchor_ids=np.arange(48, dtype=np.int16),
                        metadata={
                            "schema_version": 2,
                            "object_id": object_id,
                            "n_faces": 7,
                            "anchor_ordering": CANONICAL_ORDERING,
                            "visibility_definition": "fixture_faces",
                            "visibility_target": "vis_a",
                            "render_resolution": [8, 8],
                            "camera_radius": 2.73,
                        },
                    ),
                    visibility_cache_path(visibility_root, object_id),
                )
            manifest = build_history_dataset(
                root / "histories",
                data_root=data_root,
                visibility_cache_root=visibility_root,
                split_manifest=split_path,
                splits=["train", "val", "test"],
                history_lengths=[1, 2],
                histories_per_object_per_length=1,
                seed=7,
                coverage_target="vis_a",
            )
            manifest_data = json.loads(manifest.read_text())
            feature_path = root / "train_features.pt"
            save_feature_cache(
                CachedFeatureDataset(
                    features=torch.zeros(48, 4),
                    targets=torch.zeros(48, 48),
                    valid_mask=torch.ones(48, 48, dtype=torch.bool),
                    sample_ids=tuple(
                        f"{objects['train']}/{anchor}" for anchor in range(48)
                    ),
                    source_anchor_ids=torch.arange(48),
                    metadata={
                        "split": "train",
                        "backbone": "vggt",
                        "variant": "fixture",
                        "feature_components": ["max_pooled_patch"],
                        "history_mode": "single_image",
                        "split_manifest_sha256": manifest_data["split_manifest_sha256"],
                        "backbone_configuration": {
                            "model_id": "fixture/VGGT",
                            "image_size": 4,
                            "layer_index": -1,
                            "input_resolution": [4, 4],
                            "preprocessing": "square_resize_rgb_0_1",
                            "history_mode": "single_image",
                        },
                    },
                ),
                feature_path,
            )
            independent = {
                "schema_version": 1,
                "experiment": {
                    "phase": "phase3", "name": "independent_fixture", "seed": 7,
                    "deterministic": True,
                },
                "paths": {
                    "data_root": str(data_root), "visibility_cache_root": str(visibility_root),
                    "output_root": str(root / "outputs"), "model_cache_root": str(root / "models"),
                },
                "phase3": {"independent_control": {
                    "history_manifest": str(manifest), "coverage_target": "vis_a", "device": "cpu",
                    "features": {
                        "variant": "fixture", "components": ["max_pooled_patch"],
                        "cache_paths": {key: str(feature_path) for key in ("train", "val", "test")},
                    },
                    "model": {
                        "aggregation": "masked_mean", "include_anchor_directions": True,
                        "hidden_dim": 8, "dropout": 0.0,
                    },
                    "training": {
                        "epochs": 2, "batch_size": 2, "learning_rate": 0.01,
                        "weight_decay": 0.0, "huber_delta": 0.1,
                        "ranking_weight": 0.0, "ranking_margin": 0.0, "patience": 2,
                    },
                    "evaluation": {"batch_size": 2, "ndcg_k": 5},
                    "closed_loop_smoke": {
                        "enabled": False, "split": "val", "object_id": None,
                        "initial_anchor_ids": [0], "max_acquired_views": 2,
                    },
                }},
            }
            independent_path = root / "independent.json"
            independent_path.write_text(json.dumps(independent))
            joint = {
                "schema_version": 1,
                "experiment": {
                    "phase": "phase3", "name": "joint_fixture", "seed": 7,
                    "deterministic": True,
                },
                "paths": independent["paths"],
                "phase3": {"joint_control": {
                    "history_manifest": str(manifest), "coverage_target": "vis_a", "device": "cpu",
                    "matched_independent_config": str(independent_path),
                    "backbone": {
                        "model_id": "fixture/VGGT", "image_size": 4, "layer_index": -1,
                        "components": ["max_pooled_patch"], "expected_feature_dim": 4,
                    },
                    "model": {
                        "aggregation": "masked_mean", "include_anchor_directions": True,
                        "hidden_dim": 8, "dropout": 0.0,
                    },
                    "training": {
                        "epochs": 2, "batch_size": 1, "gradient_accumulation_steps": 2,
                        "learning_rate": 0.01, "weight_decay": 0.0, "huber_delta": 0.1,
                        "ranking_weight": 0.0, "ranking_margin": 0.0, "patience": 2,
                        "checkpoint_every_batches": 2,
                    },
                    "evaluation": {"batch_size": 1, "ndcg_k": 5},
                    "memory_preflight": {"enabled": True, "history_lengths": [1, 2]},
                }},
            }
            extractor = VGGTJointExtractor(
                model=_JointAwareAggregator(), image_size=4, device="cpu",
                expected_feature_dim=4,
            )

            run_dir = run_phase3_joint(joint, root, extractor=extractor)

            summary = json.loads((run_dir / "metrics/summary.json").read_text())
            self.assertEqual(summary["step"], 17)
            self.assertTrue(summary["control_match"]["all_required_fields_match"])
            self.assertEqual(
                summary["trainable_parameters"],
                summary["matched_independent_trainable_parameters"],
            )
            self.assertEqual(
                [row["history_length"] for row in summary["memory_preflight"]["measurements"]],
                [1, 2],
            )
            self.assertTrue(summary["frozen_feature_precompute"]["enabled"])
            self.assertEqual(
                summary["frozen_feature_precompute"]["backbone_forwards_per_history"],
                1,
            )
            loaded, payload = load_joint_history_checkpoint(
                run_dir / "checkpoints/best.pt",
                extractor=VGGTJointExtractor(
                    model=_JointAwareAggregator(), image_size=4, device="cpu",
                    expected_feature_dim=4,
                ),
            )
            self.assertEqual(loaded.feature_dim, 4)
            self.assertEqual(payload["supervision"]["target"], "target_surface_gain")
            self.assertNotIn("test", summary)
            self.assertTrue((run_dir / "checkpoints/training_state.pt").is_file())
            self.assertTrue(
                (run_dir / "joint_feature_cache/train/shard_00000.pt").is_file()
            )
            self.assertTrue(
                (run_dir / "joint_feature_cache/val/shard_00000.pt").is_file()
            )
            with self.assertRaisesRegex(ValueError, "use --resume"):
                run_phase3_joint(joint, root)

            (run_dir / "checkpoints/training_state.pt").unlink()
            resumed_aggregator = _JointAwareAggregator()
            resumed_dir = run_phase3_joint(
                joint,
                root,
                extractor=VGGTJointExtractor(
                    model=resumed_aggregator, image_size=4, device="cpu",
                    expected_feature_dim=4,
                ),
                resume=True,
            )
            resumed_summary = json.loads(
                (resumed_dir / "metrics/summary.json").read_text()
            )
            self.assertTrue(resumed_summary["resumed"])
            self.assertFalse(resumed_summary["training_resumed"])
            self.assertEqual(resumed_summary["epochs_completed"], 2)
            self.assertEqual(len(resumed_aggregator.input_shapes), 2)

            training_resumed_dir = run_phase3_joint(
                joint,
                root,
                extractor=VGGTJointExtractor(
                    model=_JointAwareAggregator(), image_size=4, device="cpu",
                    expected_feature_dim=4,
                ),
                resume=True,
            )
            training_resumed_summary = json.loads(
                (training_resumed_dir / "metrics/summary.json").read_text()
            )
            self.assertTrue(training_resumed_summary["training_resumed"])


if __name__ == "__main__":
    unittest.main()
