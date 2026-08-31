from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from nbv.features import (
    DINOv2Extractor,
    FeatureExtractorError,
    FeatureSelectionError,
    FrozenFeatures,
    ImageNetViTExtractor,
    VGGTExtractor,
    create_feature_extractor,
    select_feature_components,
    validate_feature_selection,
)


class _FakeImageNetViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, 3))
        self.encoder = nn.Identity()

    def _process_input(self, images: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
            .flatten(2)
            .transpose(1, 2)
        )


class _FakeDINOv2(nn.Module):
    def __init__(self, *, with_registers: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.with_registers = with_registers

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        patches = torch.ones(batch_size, 4, 5, device=images.device) * self.scale
        registers = torch.ones(
            batch_size,
            2 if self.with_registers else 0,
            5,
            device=images.device,
        )
        return {
            "x_norm_clstoken": torch.zeros(batch_size, 5, device=images.device),
            "x_norm_patchtokens": patches,
            "x_norm_regtokens": registers,
        }


class _FakeVGGTAggregator(nn.Module):
    patch_start_idx = 3

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.ones(()))

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        batch_size = images.shape[0]
        tokens = (
            torch.ones(batch_size, 1, 7, 6, device=images.device) * self.offset
        )
        return [tokens], self.patch_start_idx


class FrozenFeatureTests(unittest.TestCase):
    def test_torch_backbones_use_and_restore_the_common_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "models"
            original_hub_dir = torch.hub.get_dir()
            observed: dict[str, Path] = {}

            def imagenet_loader(pretrained: bool) -> nn.Module:
                self.assertTrue(pretrained)
                observed["imagenet"] = Path(torch.hub.get_dir())
                return _FakeImageNetViT()

            def dinov2_loader(*args: object, **kwargs: object) -> nn.Module:
                observed["dinov2"] = Path(torch.hub.get_dir())
                return _FakeDINOv2()

            ImageNetViTExtractor(
                model_loader=imagenet_loader,
                model_cache_root=cache_root,
                device="cpu",
            )
            DINOv2Extractor(
                model_loader=dinov2_loader,
                model_cache_root=cache_root,
                device="cpu",
            )

            expected = cache_root / "torch"
            self.assertEqual(
                observed, {"imagenet": expected, "dinov2": expected}
            )
            self.assertTrue(expected.is_dir())
            self.assertEqual(torch.hub.get_dir(), original_hub_dir)

    def test_vggt_uses_the_common_huggingface_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "models"
            observed: dict[str, object] = {}

            def loader(model_id: str, **kwargs: object) -> object:
                observed["model_id"] = model_id
                observed.update(kwargs)
                return SimpleNamespace(aggregator=_FakeVGGTAggregator())

            VGGTExtractor(
                model_loader=loader,
                model_cache_root=cache_root,
                image_size=28,
                device="cpu",
            )

            self.assertEqual(observed["model_id"], "facebook/VGGT-1B")
            self.assertEqual(
                observed["cache_dir"], cache_root / "huggingface"
            )
            self.assertTrue((cache_root / "huggingface").is_dir())

    def test_imagenet_vit_returns_common_layout_and_stays_frozen(self) -> None:
        model = _FakeImageNetViT()
        extractor = ImageNetViTExtractor(model=model, device="cpu")
        model.train()

        features = extractor.extract(np.zeros((2, 3, 32, 40), dtype=np.float32))

        self.assertEqual(features.patch_tokens.shape, (2, 4, 3))
        self.assertEqual(features.pooled_patch.shape, (2, 3))
        self.assertEqual(features.cls_token.shape, (2, 3))
        self.assertFalse(model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )
        self.assertTrue(
            all(not tensor.requires_grad for tensor in features.tensors().values())
        )

    def test_dinov2_exposes_patch_cls_and_register_tokens(self) -> None:
        extractor = DINOv2Extractor(model=_FakeDINOv2(), device="cpu")
        features = extractor.extract(torch.zeros(1, 3, 28, 28))

        self.assertEqual(features.patch_tokens.shape, (1, 4, 5))
        self.assertEqual(features.cls_token.shape, (1, 5))
        self.assertEqual(features.register_tokens.shape, (1, 2, 5))
        torch.testing.assert_close(features.pooled_patch, torch.ones(1, 5))

    def test_empty_dinov2_register_tensor_becomes_none(self) -> None:
        extractor = DINOv2Extractor(
            model=_FakeDINOv2(with_registers=False), device="cpu"
        )
        self.assertIsNone(
            extractor.extract(torch.zeros(3, 16, 16)).register_tokens
        )

    def test_vggt_splits_camera_register_and_patch_tokens(self) -> None:
        extractor = VGGTExtractor(
            model=_FakeVGGTAggregator(), image_size=28, device="cpu"
        )
        features = extractor.extract(torch.zeros(2, 3, 28, 28))

        self.assertEqual(features.camera_tokens.shape, (2, 1, 6))
        self.assertEqual(features.register_tokens.shape, (2, 2, 6))
        self.assertEqual(features.patch_tokens.shape, (2, 4, 6))
        self.assertEqual(features.metadata["history_mode"], "single_image")

    def test_input_validation_rejects_bad_shape_and_range(self) -> None:
        extractor = ImageNetViTExtractor(model=_FakeImageNetViT(), device="cpu")
        with self.assertRaisesRegex(ValueError, "shape"):
            extractor.extract(torch.zeros(1, 4, 8, 8))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            extractor.extract(torch.full((1, 3, 8, 8), 2.0))

    def test_vggt_reports_uncached_layer(self) -> None:
        class SparseAggregator(_FakeVGGTAggregator):
            def forward(
                self, images: torch.Tensor
            ) -> tuple[list[torch.Tensor | None], int]:
                tokens, start = super().forward(images)
                return [tokens[0], None], start

        extractor = VGGTExtractor(
            model=SparseAggregator(), layer_index=1, device="cpu"
        )
        with self.assertRaisesRegex(FeatureExtractorError, "not cached"):
            extractor.extract(torch.zeros(1, 3, 28, 28))

    def test_factory_rejects_unknown_backbone(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown feature extractor"):
            create_feature_extractor("unknown")

    def test_probe_components_can_select_and_concatenate_token_features(self) -> None:
        features = FrozenFeatures(
            pooled_patch=torch.tensor([[2.0, 1.0]]),
            patch_tokens=torch.tensor([[[1.0, 2.0], [3.0, 0.0]]]),
            cls_token=torch.tensor([[10.0, 11.0]]),
            camera_tokens=torch.tensor([[[20.0, 21.0]]]),
            register_tokens=torch.tensor(
                [[[30.0, 31.0], [32.0, 33.0]]]
            ),
        )

        selected = select_feature_components(
            features,
            [
                "pooled_patch",
                "max_pooled_patch",
                "cls_token",
                "pooled_camera",
                "pooled_register",
            ],
        )

        torch.testing.assert_close(
            selected,
            torch.tensor(
                [[2.0, 1.0, 3.0, 2.0, 10.0, 11.0, 20.0, 21.0, 31.0, 32.0]]
            ),
        )
        self.assertFalse(selected.requires_grad)

    def test_feature_selection_rejects_unsupported_or_missing_tokens(self) -> None:
        with self.assertRaisesRegex(FeatureSelectionError, "Unsupported"):
            validate_feature_selection("vggt", ["cls_token"])

        features = FrozenFeatures(
            pooled_patch=torch.ones(1, 2),
            patch_tokens=torch.ones(1, 3, 2),
        )
        with self.assertRaisesRegex(FeatureSelectionError, "not exposed"):
            select_feature_components(features, ["pooled_register"])

    def test_feature_selection_rejects_empty_or_duplicate_components(self) -> None:
        with self.assertRaisesRegex(FeatureSelectionError, "at least one"):
            validate_feature_selection("dinov2", [])
        with self.assertRaisesRegex(FeatureSelectionError, "duplicates"):
            validate_feature_selection(
                "imagenet_vit", ["pooled_patch", "pooled_patch"]
            )


if __name__ == "__main__":
    unittest.main()
