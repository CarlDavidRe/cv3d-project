from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nbv.data import (
    FrozenFeatureLookup,
    HistoryFeatureDataset,
    HistorySample,
    VisibilityCache,
    build_history_dataset,
    collate_history_features,
    save_visibility_cache,
    visibility_cache_path,
)
from nbv.data.observation_store import AcquiredObservation
from nbv.eval.result_schema import load_rollout
from nbv.experiments.phase3_independent import (
    load_independent_history_checkpoint,
    parse_phase3_independent_settings,
    run_phase3_independent,
)
from nbv.features import (
    CachedFeatureDataset,
    FrozenFeatures,
    save_feature_cache,
)
from nbv.geometry import CANONICAL_ORDERING, canonical_anchors
from nbv.models import (
    IndependentHistoryGainModel,
    independent_feature_batch,
)
from nbv.policies import IndependentHistoryPolicy, ObservationState
from nbv.training import evaluate_history_model, fit_history_model


class _FakeExtractor:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def extract(self, images: torch.Tensor) -> FrozenFeatures:
        self.input_shapes.append(tuple(images.shape))
        marker = images.mean(dim=(1, 2, 3))
        patches = torch.stack((marker, marker + 1), dim=1).unsqueeze(-1)
        return FrozenFeatures(
            pooled_patch=patches.mean(dim=1),
            patch_tokens=patches,
        )


class IndependentHistoryModelTests(unittest.TestCase):
    def test_padding_is_ignored_and_multiple_lengths_work(self) -> None:
        torch.manual_seed(1)
        model = IndependentHistoryGainModel(4, hidden_dim=8, dropout=0.0)
        features = torch.randn(2, 3, 4)
        anchors = torch.tensor([[2, -1, -1], [4, 1, 9]])
        padding = torch.tensor([[False, True, True], [False, False, False]])
        changed = features.clone()
        changed[padding] = 10000

        first = model(features, anchors, padding)
        second = model(changed, anchors, padding)

        self.assertEqual(first.shape, (2, 48))
        torch.testing.assert_close(first, second)

    def test_history_permutation_leaves_aggregate_and_prediction_unchanged(self) -> None:
        torch.manual_seed(2)
        model = IndependentHistoryGainModel(5, hidden_dim=9, dropout=0.0).eval()
        features = torch.randn(1, 4, 5)
        anchors = torch.tensor([[7, 2, 31, 4]])
        padding = torch.zeros((1, 4), dtype=torch.bool)
        order = torch.tensor([2, 0, 3, 1])

        expected = model(features, anchors, padding)
        actual = model(features[:, order], anchors[:, order], padding[:, order])

        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    def test_independent_extraction_never_passes_a_history_axis_to_vggt(self) -> None:
        extractor = _FakeExtractor()
        images = torch.rand(2, 3, 3, 4, 5)
        padding = torch.tensor([[False, True, True], [False, False, False]])

        features = independent_feature_batch(
            images, padding, extractor, ["max_pooled_patch"]
        )

        self.assertEqual(extractor.input_shapes, [(4, 3, 4, 5)])
        self.assertEqual(features.shape, (2, 3, 1))
        self.assertEqual(torch.count_nonzero(features[padding]).item(), 0)

    def test_rejects_invalid_padding_contract(self) -> None:
        model = IndependentHistoryGainModel(2)
        with self.assertRaisesRegex(ValueError, "padded history anchor IDs"):
            model(
                torch.zeros(1, 2, 2),
                torch.tensor([[0, 1]]),
                torch.tensor([[False, True]]),
            )


class _SyntheticHistories:
    def __init__(self, split: str, samples: list[HistorySample]) -> None:
        self.split = split
        self.load_images = False
        self.samples = samples
        self.manifest = {"dataset_id": "synthetic-history-dataset"}
        max_length = max(sample.history_length for sample in samples)
        anchors = np.full((len(samples), max_length), -1, dtype=np.int64)
        for row, sample in enumerate(samples):
            anchors[row, :sample.history_length] = sample.history_anchor_ids
        self._arrays = {
            "object_ids": np.asarray([sample.object_id for sample in samples]),
            "history_lengths": np.asarray([sample.history_length for sample in samples]),
            "history_anchor_ids": anchors,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HistorySample:
        return self.samples[index]


def _feature_dataset(split: str) -> HistoryFeatureDataset:
    torch.manual_seed(8)
    object_id = f"category/{split}_object"
    vectors = torch.randn(6, 4)
    targets = torch.zeros(6, 48)
    cache = CachedFeatureDataset(
        features=vectors,
        targets=targets,
        valid_mask=torch.ones(6, 48, dtype=torch.bool),
        sample_ids=tuple(f"{object_id}/{anchor}" for anchor in range(6)),
        source_anchor_ids=torch.arange(6, dtype=torch.int64),
        metadata={"split": split, "backbone": "vggt", "history_mode": "single_image"},
    )
    samples = []
    histories = ((0,), (1, 2), (3,), (4, 5))
    for index, anchor_ids in enumerate(histories):
        mean = vectors[list(anchor_ids)].mean(dim=0)
        weights = torch.linspace(-1, 1, 4 * 48).reshape(4, 48)
        target = torch.sigmoid(
            mean @ weights + torch.linspace(-1, 1, 48)
        ).numpy().astype(np.float32)
        valid = np.ones(48, dtype=np.bool_)
        valid[list(anchor_ids)] = False
        target[list(anchor_ids)] = 0
        samples.append(HistorySample(
            sample_id=f"{split}-{index}",
            object_id=object_id,
            history_image_paths=tuple(f"unused/{anchor}.png" for anchor in anchor_ids),
            history_anchor_ids=anchor_ids,
            history_length=len(anchor_ids),
            target_surface_gain=target,
            valid_candidate_mask=valid,
            rotation_metadata={"enabled": False},
            split=split,
            visibility_cache_id="a" * 64,
            coverage_target="vis_a",
            sampling_metadata={},
        ))
    return HistoryFeatureDataset(
        _SyntheticHistories(split, samples),
        FrozenFeatureLookup(cache, split=split),
    )


class HistoryTrainingTests(unittest.TestCase):
    def test_cached_history_features_collate_variable_lengths(self) -> None:
        dataset = _feature_dataset("train")
        batch = collate_history_features([dataset[0], dataset[1]])

        self.assertEqual(batch.history_features.shape, (2, 2, 4))
        self.assertEqual(batch.history_lengths.tolist(), [1, 2])
        self.assertEqual(batch.history_padding_mask.tolist(), [[False, True], [False, False]])
        self.assertEqual(batch.history_anchor_ids.tolist(), [[0, -1], [1, 2]])
        self.assertEqual(batch.target_surface_gain.shape, (2, 48))

    def test_tiny_direct_gain_history_set_can_be_overfit_and_evaluated(self) -> None:
        train = _feature_dataset("train")
        model = IndependentHistoryGainModel(
            4, hidden_dim=96, dropout=0.0, include_anchor_directions=True
        )
        initial = evaluate_history_model(model, train, batch_size=4).summary["huber_loss"]

        result = fit_history_model(
            model,
            train,
            train,
            epochs=250,
            batch_size=4,
            learning_rate=0.02,
            weight_decay=0.0,
            huber_delta=1.0,
            ranking_weight=0.0,
            patience=None,
            device="cpu",
            seed=3,
        )
        evaluated = evaluate_history_model(model, train, batch_size=4)

        self.assertLess(evaluated.summary["huber_loss"], initial * 0.02)
        self.assertGreater(result.best_epoch, 0)
        self.assertEqual(evaluated.predictions.shape, (4, 48))
        self.assertEqual(len(evaluated.per_sample), 4)


class HistoryPolicyTests(unittest.TestCase):
    def _state(self, anchor_ids: tuple[int, ...], step: int) -> ObservationState:
        observations = []
        for anchor in anchor_ids:
            rgb = np.full((3, 4, 3), anchor, dtype=np.uint8)
            rgb.setflags(write=False)
            observations.append(AcquiredObservation(anchor, f"/{anchor}.png", rgb))
        return ObservationState(
            object_id="category/object",
            acquired_observations=tuple(observations),
            anchor_directions=canonical_anchors().directions.copy(),
            camera_to_world=np.zeros((48, 4, 4)),
            valid_candidate_mask=canonical_anchors().valid_candidate_mask(anchor_ids),
            step_index=step,
            seed=0,
            anchor_ordering=CANONICAL_ORDERING,
        )

    def test_policy_uses_cached_history_and_returns_direct_scores(self) -> None:
        model = IndependentHistoryGainModel(2, hidden_dim=5, dropout=0.0)
        cache = {
            "category/object/0": torch.tensor([1.0, 2.0]),
            "category/object/3": torch.tensor([4.0, 5.0]),
        }
        policy = IndependentHistoryPolicy(
            model,
            ["max_pooled_patch"],
            cached_features=cache,
            provenance={"coverage_target": "vis_a"},
        )

        scores = policy.score(self._state((0, 3), 1))

        self.assertEqual(scores.shape, (48,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(
            policy.provenance["prediction_source_counts"],
            {"feature_cache": 2, "live_vggt": 0},
        )
        self.assertFalse(policy.provenance["backbone_cross_view_interaction"])

    def test_live_policy_extraction_keeps_views_in_the_batch_dimension(self) -> None:
        extractor = _FakeExtractor()
        model = IndependentHistoryGainModel(1, hidden_dim=4, dropout=0.0)
        policy = IndependentHistoryPolicy(
            model, ["max_pooled_patch"], extractor=extractor
        )

        policy.score(self._state((0, 3), 1))

        self.assertEqual(extractor.input_shapes, [(2, 3, 3, 4)])


class Phase3IndependentExperimentTests(unittest.TestCase):
    def test_checked_in_config_parses_before_histories_are_generated(self) -> None:
        from nbv.config import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiments/phase3_independent.yaml")
        settings = parse_phase3_independent_settings(config, root)

        self.assertEqual(settings.feature_components, ("max_pooled_patch",))
        self.assertEqual(settings.coverage_target, "vis_a")
        self.assertTrue(settings.include_anchor_directions)

    def test_end_to_end_runner_saves_selected_checkpoint_metrics_and_rollout(self) -> None:
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
            }), encoding="utf-8")
            for offset, object_id in enumerate(objects.values()):
                image_root = data_root / object_id / "images"
                image_root.mkdir(parents=True)
                for anchor in range(48):
                    from PIL import Image

                    Image.new("RGB", (4, 3), color=(20 + offset,) * 3).save(
                        image_root / f"viewpoint_{anchor}_offset_phi_0.png"
                    )
                visibility = np.zeros((48, 7), dtype=np.bool_)
                for anchor in range(48):
                    visibility[anchor, anchor % 7] = True
                    visibility[anchor, (anchor + offset + 1) % 7] = True
                cache = VisibilityCache(
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
                )
                save_visibility_cache(cache, visibility_cache_path(visibility_root, object_id))
            manifest = build_history_dataset(
                root / "histories",
                data_root=data_root,
                visibility_cache_root=visibility_root,
                split_manifest=split_path,
                splits=["train", "val", "test"],
                history_lengths=[1, 2],
                histories_per_object_per_length=1,
                seed=4,
                coverage_target="vis_a",
            )
            manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
            feature_paths = {}
            for split, object_id in objects.items():
                feature_path = root / "features" / f"{split}.pt"
                feature_paths[split] = feature_path
                save_feature_cache(CachedFeatureDataset(
                    features=torch.arange(48 * 4, dtype=torch.float32).reshape(48, 4) / 100,
                    targets=torch.zeros(48, 48),
                    valid_mask=torch.ones(48, 48, dtype=torch.bool),
                    sample_ids=tuple(f"{object_id}/{anchor}" for anchor in range(48)),
                    source_anchor_ids=torch.arange(48, dtype=torch.int64),
                    metadata={
                        "split": split,
                        "backbone": "vggt",
                        "variant": "fixture_max_patch",
                        "feature_components": ["max_pooled_patch"],
                        "history_mode": "single_image",
                        "split_manifest_sha256": manifest_json["split_manifest_sha256"],
                    },
                ), feature_path)
            config = {
                "schema_version": 1,
                "experiment": {
                    "phase": "phase3", "name": "independent_fixture", "seed": 4,
                    "deterministic": True,
                },
                "paths": {
                    "data_root": str(data_root),
                    "visibility_cache_root": str(visibility_root),
                    "output_root": str(root / "outputs"),
                    "model_cache_root": str(root / "models"),
                },
                "phase3": {"independent_control": {
                    "history_manifest": str(manifest),
                    "coverage_target": "vis_a",
                    "device": "cpu",
                    "features": {
                        "variant": "fixture_max_patch",
                        "components": ["max_pooled_patch"],
                        "cache_paths": {key: str(value) for key, value in feature_paths.items()},
                    },
                    "model": {
                        "aggregation": "masked_mean",
                        "include_anchor_directions": True,
                        "hidden_dim": 8,
                        "dropout": 0.0,
                    },
                    "training": {
                        "epochs": 2, "batch_size": 2, "learning_rate": 0.01,
                        "weight_decay": 0.0, "huber_delta": 0.1,
                        "ranking_weight": 0.0, "ranking_margin": 0.0, "patience": 2,
                    },
                    "evaluation": {"batch_size": 2, "ndcg_k": 5},
                    "closed_loop_smoke": {
                        "enabled": True, "split": "val", "object_id": None,
                        "initial_anchor_ids": [0], "max_acquired_views": 3,
                    },
                }},
            }

            run_dir = run_phase3_independent(config, root)

            summary = json.loads((run_dir / "metrics/summary.json").read_text())
            self.assertEqual(summary["train_samples"], 2)
            self.assertEqual(summary["test"]["num_samples"], 2)
            self.assertTrue(summary["closed_loop_smoke"]["replay_verified"])
            self.assertEqual(summary["model"]["backbone_history_mode"], "independent_single_image")
            loaded, payload = load_independent_history_checkpoint(
                run_dir / "checkpoints/best.pt"
            )
            self.assertEqual(loaded.feature_dim, 4)
            self.assertEqual(payload["supervision"]["target"], "target_surface_gain")
            self.assertTrue((run_dir / "metrics/comparison.csv").is_file())
            comparison = json.loads(
                (run_dir / "metrics/comparison.json").read_text()
            )
            self.assertEqual(
                comparison["results"][0]["variant"], "independent_fixture"
            )
            self.assertEqual(comparison["results"][0]["input_dim"], 7)
            self.assertTrue((run_dir / "metrics/comparison.md").is_file())
            self.assertTrue(
                (
                    run_dir / "figures/training/independent_fixture_losses.svg"
                ).is_file()
            )
            self.assertTrue(
                (
                    run_dir
                    / "figures/training/independent_fixture_validation_metrics.svg"
                ).is_file()
            )
            self.assertTrue(
                (run_dir / "figures/training/validation_loss_comparison.svg").is_file()
            )
            rollout = load_rollout(run_dir / "closed_loop_smoke/rollout.npz")
            self.assertEqual(rollout.metadata["training_target_semantics"], "phase3_history_dependent_surface_gain")


if __name__ == "__main__":
    unittest.main()
