from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from torch import nn

from nbv.config import ConfigError, load_config
from nbv.experiments.phase1 import (
    BaselineVariant,
    ProbeVariant,
    _evaluate_baseline,
    _load_completed_entry,
    _train_and_evaluate_variant,
    extract_variant_caches,
    parse_phase1_sweep_settings,
)
from nbv.features import CachedFeatureDataset, FrozenFeatures
from nbv.models import LightweightProbeHead
from nbv.reproducibility import RunContext
from nbv.training import evaluate_phase1_probe, fit_phase1_probe
from nbv.visualization import (
    write_validation_loss_comparison,
    write_variant_training_curves,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _FakeDataset:
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(
                image=np.full((3, 2, 2), index / 4, dtype=np.float32),
                target_map=np.arange(4, dtype=np.float32) + index,
                source_anchor_id=index,
                record=SimpleNamespace(sample_id=f"object/{index}"),
            )
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> object:
        return self.samples[index]


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, images: np.ndarray) -> FrozenFeatures:
        self.calls += 1
        pooled = torch.from_numpy(images.mean(axis=(2, 3))).float()
        patches = torch.stack((pooled, pooled + 1.0), dim=1)
        camera = (pooled + 2.0).unsqueeze(1)
        return FrozenFeatures(
            pooled_patch=patches.mean(dim=1),
            patch_tokens=patches,
            camera_tokens=camera,
        )


def _cached(features: torch.Tensor, targets: torch.Tensor) -> CachedFeatureDataset:
    sample_count, anchor_count = targets.shape
    return CachedFeatureDataset(
        features=features,
        targets=targets,
        valid_mask=torch.ones(sample_count, anchor_count, dtype=torch.bool),
        sample_ids=tuple(f"sample/{index}" for index in range(sample_count)),
        source_anchor_ids=torch.arange(sample_count, dtype=torch.int64),
        metadata={"fixture": True},
    )


class Phase1ExperimentTests(unittest.TestCase):
    def test_complete_saved_variant_is_loaded_for_resume(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["probe.device=cpu"],
        )
        settings = parse_phase1_sweep_settings(config, REPOSITORY_ROOT)
        variant = settings.variants[0]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            context = RunContext(
                run_id="test",
                run_dir=run_dir,
                checkpoint_dir=run_dir / "checkpoints",
                metrics_dir=run_dir / "metrics",
                figure_dir=run_dir / "figures",
                log_path=run_dir / "run.log",
            )
            variant_dir = run_dir / "variants" / variant.name
            variant_dir.mkdir(parents=True)
            (variant_dir / "best.pt").write_bytes(b"saved checkpoint")
            (variant_dir / "test_per_sample.csv").write_text(
                "sample_id\nfixture\n", encoding="utf-8"
            )
            (variant_dir / "training_history.json").write_text(
                '[{"epoch": 0}]\n', encoding="utf-8"
            )
            (variant_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant": variant.name,
                        "backbone": variant.backbone,
                        "feature": "+".join(variant.components),
                        "feature_components": list(variant.components),
                        "input_dim": 12,
                        "trainable_parameters": 34,
                        "best_epoch": 2,
                        "target_name": settings.target_name,
                        "target_direction": settings.target_direction,
                        "test": {
                            "huber_loss": 1.0,
                            "normalized_regret_mean": 0.2,
                            "spearman_mean": 0.3,
                            f"ndcg_at_{settings.ndcg_k}_mean": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = _load_completed_entry(variant, context, settings)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            row, history = loaded
            self.assertEqual(row["variant"], variant.name)
            self.assertEqual(row["best_epoch"], 2)
            self.assertEqual(history, [{"epoch": 0}])

            (variant_dir / "test_per_sample.csv").unlink()
            self.assertIsNone(
                _load_completed_entry(variant, context, settings)
            )

    def test_sweep_config_resolves_variants_and_psnr_direction(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml"
        )
        settings = parse_phase1_sweep_settings(config, REPOSITORY_ROOT)

        self.assertEqual(
            tuple(variant.name for variant in settings.variants),
            (
                "raw_rgb_16x16_mlp",
                "imagenet_vit_pooled_patch",
                "imagenet_vit_cls_token",
                "dinov2_pooled_patch",
                "dinov2_cls_token",
                "vggt_pooled_patch",
                "vggt_max_pooled_patch",
                "vggt_camera_token",
                "vggt_pooled_register",
                "vggt_camera_patch",
            ),
        )
        self.assertEqual(
            tuple(
                (baseline.name, baseline.baseline_type)
                for baseline in settings.baselines
            ),
            (
                ("train_mean_map", "train_mean_map"),
                ("pun_upnet", "pun"),
            ),
        )
        self.assertEqual(settings.target_direction, "lower")
        self.assertEqual(settings.extraction_batch_sizes["raw_rgb"], 256)
        self.assertEqual(settings.extraction_batch_sizes["vggt"], 32)
        self.assertEqual(settings.cache_dtype, torch.float16)
        self.assertEqual(
            settings.model_cache_root,
            REPOSITORY_ROOT / "data" / "cache" / "models",
        )
        self.assertFalse(settings.rebuild_cache)
        self.assertIsNotNone(settings.pun)
        assert settings.pun is not None
        self.assertEqual(settings.pun["model_name"], "vit_small_patch16_224")
        self.assertEqual(
            settings.pun["release_name"],
            "vit_small_patch16_224_PSNR_250425172703",
        )
        self.assertTrue(settings.pun["download_if_missing"])
        self.assertEqual(
            settings.pun["checkpoint_sha256"],
            "91b2065f7652aac0c84386d4af10cd0c1ae723c049c91e45ccc78722f70907ae",
        )

    def test_step_numbered_feature_cache_path_is_rejected(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["probe.feature_cache.root=data/step5/features"],
        )
        with self.assertRaisesRegex(ConfigError, "step-numbered"):
            parse_phase1_sweep_settings(config, REPOSITORY_ROOT)

    def test_pretrained_pun_target_must_match_the_experiment(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["phase1.num_dataset.target_name=MSE"],
        )
        with self.assertRaisesRegex(ValueError, "pun.target_name must match"):
            parse_phase1_sweep_settings(config, REPOSITORY_ROOT)

    def test_multiple_variants_share_one_backbone_forward_per_batch(self) -> None:
        extractor = _FakeExtractor()
        variants = (
            ProbeVariant("patch", "vggt", ("pooled_patch",)),
            ProbeVariant(
                "camera_patch",
                "vggt",
                ("pooled_camera", "pooled_patch"),
            ),
        )
        metadata = {variant.name: {"variant": variant.name} for variant in variants}

        caches = extract_variant_caches(
            _FakeDataset(),
            extractor,
            variants,
            metadata,
            extraction_batch_size=2,
            num_anchors=4,
            mask_source_view=True,
            cache_dtype=torch.float16,
        )

        self.assertEqual(extractor.calls, 2)
        self.assertEqual(caches["patch"].features.shape, (3, 3))
        self.assertEqual(caches["camera_patch"].features.shape, (3, 6))
        self.assertTrue(torch.all(~caches["patch"].valid_mask[:, 0]))
        self.assertEqual(caches["patch"].sample_ids[-1], "object/2")

    def test_train_mean_baseline_saves_common_evaluation_outputs(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["probe.device=cpu"],
        )
        settings = parse_phase1_sweep_settings(config, REPOSITORY_ROOT)
        train = _cached(
            torch.zeros(2, 1),
            torch.tensor([[1.0, 4.0, 9.0], [3.0, 8.0, 5.0]]),
        )
        evaluation = _cached(
            torch.zeros(1, 1), torch.tensor([[2.0, 6.0, 7.0]])
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            context = RunContext(
                run_id="test",
                run_dir=run_dir,
                checkpoint_dir=run_dir / "checkpoints",
                metrics_dir=run_dir / "metrics",
                figure_dir=run_dir / "figures",
                log_path=run_dir / "run.log",
            )

            row = _evaluate_baseline(
                BaselineVariant("train_mean_map", "train_mean_map"),
                {"train": train, "val": evaluation, "test": evaluation},
                settings,
                context,
            )
            checkpoint = torch.load(
                run_dir / "variants/train_mean_map/best.pt",
                map_location="cpu",
                weights_only=True,
            )

        torch.testing.assert_close(
            checkpoint["prediction_map"], torch.tensor([2.0, 6.0, 7.0])
        )
        self.assertEqual(row["trainable_parameters"], 0)
        self.assertEqual(row["normalized_regret_mean"], 0.0)

    def test_full_split_training_restores_best_validation_head(self) -> None:
        torch.manual_seed(3)
        features = torch.randn(12, 4)
        weights = torch.tensor(
            [[1.0, -0.3, 0.5], [-0.2, 0.4, 0.8], [0.7, 0.2, -0.4], [0.1, 0.9, 0.3]]
        )
        targets = features @ weights
        train = _cached(features[:8], targets[:8])
        validation = _cached(features[8:], targets[8:])
        head = LightweightProbeHead(4, hidden_dim=24, num_anchors=3)

        result = fit_phase1_probe(
            head,
            train,
            validation,
            epochs=150,
            batch_size=4,
            learning_rate=0.02,
            ranking_weight=0.1,
            patience=30,
            device="cpu",
            seed=5,
        )

        self.assertGreater(result.best_epoch, 0)
        initial_validation_loss = result.history[0]["validation_loss"]
        self.assertIsInstance(initial_validation_loss, float)
        self.assertLess(result.best_validation_loss, initial_validation_loss)
        self.assertLessEqual(result.epochs_completed, 150)
        first_epoch = result.history[1]
        self.assertIsInstance(first_epoch["optimization_train_loss"], float)
        self.assertIsInstance(first_epoch["train_loss"], float)
        self.assertIsInstance(first_epoch["train_huber_loss"], float)
        self.assertIsInstance(first_epoch["train_ranking_loss"], float)
        self.assertIsInstance(first_epoch["validation_huber_loss"], float)
        self.assertIsInstance(first_epoch["validation_ranking_loss"], float)
        self.assertIn("validation_normalized_regret_mean", first_epoch)
        self.assertIn("validation_spearman_mean", first_epoch)
        self.assertIn("validation_ndcg_at_5_mean", first_epoch)
        self.assertNotIn("test_loss", first_epoch)

    def test_training_histories_produce_svg_diagnostics_without_test_curves(
        self,
    ) -> None:
        history = (
            {
                "epoch": 0,
                "optimization_train_loss": None,
                "train_loss": 2.0,
                "train_huber_loss": 1.5,
                "train_ranking_loss": 5.0,
                "validation_loss": 2.2,
                "validation_huber_loss": 1.7,
                "validation_ranking_loss": 5.0,
                "validation_normalized_regret_mean": 0.4,
                "validation_spearman_mean": 0.1,
                "validation_ndcg_at_5_mean": 0.6,
            },
            {
                "epoch": 1,
                "optimization_train_loss": 1.4,
                "train_loss": 1.1,
                "train_huber_loss": 0.8,
                "train_ranking_loss": 3.0,
                "validation_loss": 1.3,
                "validation_huber_loss": 1.0,
                "validation_ranking_loss": 3.0,
                "validation_normalized_regret_mean": 0.2,
                "validation_spearman_mean": 0.5,
                "validation_ndcg_at_5_mean": 0.8,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            loss_path, metric_path = write_variant_training_curves(
                history,
                output_dir,
                variant_name="fixture_variant",
                best_epoch=1,
                epochs_completed=1,
                stopped_early=True,
                ndcg_k=5,
            )
            comparison_path = write_validation_loss_comparison(
                {"fixture_variant": history},
                output_dir / "validation_loss_comparison.svg",
                best_epochs={"fixture_variant": 1},
            )
            loss_svg = loss_path.read_text(encoding="utf-8")
            metric_svg = metric_path.read_text(encoding="utf-8")
            comparison_svg = comparison_path.read_text(encoding="utf-8")
            ElementTree.parse(loss_path)
            ElementTree.parse(metric_path)
            ElementTree.parse(comparison_path)

        self.assertIn('data-series="train_loss"', loss_svg)
        self.assertIn('data-series="validation_ranking_loss"', loss_svg)
        self.assertIn('data-best-epoch="1"', loss_svg)
        self.assertIn('data-last-epoch="1"', loss_svg)
        self.assertIn('data-stopped-early="true"', loss_svg)
        self.assertIn("validation_ndcg_at_5_mean", metric_svg)
        self.assertIn("fixture_variant", comparison_svg)
        self.assertNotIn('data-series="test_loss"', loss_svg)
        self.assertNotIn('data-series="test_loss"', metric_svg)

    def test_learned_variant_saves_history_and_training_figures(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["probe.device=cpu"],
        )
        parsed = parse_phase1_sweep_settings(config, REPOSITORY_ROOT)
        settings = replace(
            parsed,
            num_anchors=3,
            hidden_dim=8,
            training={
                **parsed.training,
                "epochs": 2,
                "batch_size": 2,
                "patience": None,
            },
            evaluation_batch_size=2,
        )
        features = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.2]]
        )
        targets = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ]
        )
        splits = {
            "train": _cached(features, targets),
            "val": _cached(features[:2], targets[:2]),
            "test": _cached(features[2:], targets[2:]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            context = RunContext(
                run_id="test",
                run_dir=run_dir,
                checkpoint_dir=run_dir / "checkpoints",
                metrics_dir=run_dir / "metrics",
                figure_dir=run_dir / "figures",
                log_path=run_dir / "run.log",
            )
            row = _train_and_evaluate_variant(
                ProbeVariant("fixture", "raw_rgb", ("flattened_rgb",)),
                splits,
                settings,
                context,
                seed=0,
            )
            history = (
                run_dir / "variants/fixture/training_history.json"
            ).read_text(encoding="utf-8")
            loss_figure = run_dir / "figures/training/fixture_losses.svg"
            metric_figure = (
                run_dir / "figures/training/fixture_validation_metrics.svg"
            )

            self.assertEqual(row["variant"], "fixture")
            self.assertIn('"optimization_train_loss"', history)
            self.assertNotIn('"test_loss"', history)
            self.assertTrue(loss_figure.is_file())
            self.assertTrue(metric_figure.is_file())

    def test_evaluation_orients_lower_is_better_targets(self) -> None:
        targets = torch.tensor(
            [[3.0, 1.0, 2.0], [8.0, 7.0, 9.0]], dtype=torch.float32
        )
        dataset = _cached(targets.clone(), targets)
        result = evaluate_phase1_probe(
            nn.Identity(),
            dataset,
            batch_size=2,
            target_direction="lower",
            device="cpu",
        )

        self.assertAlmostEqual(result.summary["normalized_regret_mean"], 0.0)
        self.assertAlmostEqual(result.summary["spearman_mean"], 1.0)
        self.assertAlmostEqual(result.summary["ndcg_at_5_mean"], 1.0)
        self.assertEqual(result.per_sample[0]["predicted_local_anchor_id"], 1)


if __name__ == "__main__":
    unittest.main()
