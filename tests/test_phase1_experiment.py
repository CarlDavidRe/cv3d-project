from __future__ import annotations

from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nbv.config import ConfigError, load_config
from nbv.experiments.phase1 import (
    ProbeVariant,
    extract_variant_caches,
    parse_phase1_sweep_settings,
)
from nbv.features import CachedFeatureDataset, FrozenFeatures
from nbv.models import LightweightProbeHead
from nbv.training import evaluate_phase1_probe, fit_phase1_probe


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
    def test_sweep_config_resolves_variants_and_psnr_direction(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml"
        )
        settings = parse_phase1_sweep_settings(config, REPOSITORY_ROOT)

        self.assertEqual(
            tuple(variant.name for variant in settings.variants),
            (
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
        self.assertEqual(settings.target_direction, "lower")
        self.assertEqual(settings.extraction_batch_sizes["vggt"], 32)
        self.assertEqual(settings.cache_dtype, torch.float16)
        self.assertEqual(
            settings.model_cache_root,
            REPOSITORY_ROOT / "data" / "cache" / "models",
        )
        self.assertFalse(settings.rebuild_cache)

    def test_step_numbered_feature_cache_path_is_rejected(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/experiments/phase1_sweep.yaml",
            ["probe.feature_cache.root=data/step5/features"],
        )
        with self.assertRaisesRegex(ConfigError, "step-numbered"):
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
