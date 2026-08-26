from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from nbv.features import (
    DINOv2Extractor,
    FeatureExtractorError,
    ImageNetViTExtractor,
    VGGTExtractor,
    create_feature_extractor,
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


if __name__ == "__main__":
    unittest.main()
