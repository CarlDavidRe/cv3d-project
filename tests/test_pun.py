from __future__ import annotations

from types import SimpleNamespace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from nbv.models import (
    PUNModelError,
    PUNUPNet,
    count_trainable_parameters,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
)
from nbv.training import PUNImageDataset, evaluate_pun_upnet


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 2)

    def reset_classifier(self, output_dim: int) -> None:
        if output_dim != 0:
            raise ValueError("fixture only supports classifier removal")
        self.head = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(images.mean(dim=(2, 3)))


class _TinyNUM:
    def __init__(self, count: int, *, offset: int = 0) -> None:
        self.samples = []
        self.records = []
        weights = torch.tensor(
            [[0.2, -0.3, 0.4, 0.1], [0.5, 0.1, -0.2, 0.3], [-0.4, 0.2, 0.1, 0.6]]
        )
        for index in range(count):
            rgb = torch.tensor(
                [0.1 + index / 20, 0.3 + index / 30, 0.6 - index / 40]
            )
            target = rgb @ weights
            sample_id = f"fixture/object/{offset + index}"
            record = SimpleNamespace(sample_id=sample_id)
            self.records.append(record)
            self.samples.append(
                SimpleNamespace(
                    image=rgb[:, None, None].expand(3, 2, 2).numpy().astype(np.float32),
                    target_map=target.numpy().astype(np.float32),
                    source_anchor_id=offset + index,
                    record=record,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> object:
        return self.samples[index]


class PUNTests(unittest.TestCase):
    def test_upnet_matches_official_backbone_plus_linear_regressor(self) -> None:
        backbone = _TinyBackbone()
        model = PUNUPNet(
            model_name="fixture_vit",
            num_anchors=4,
            pretrained=False,
            backbone=backbone,
        )

        prediction = model(torch.ones(2, 3, 5, 5))

        self.assertIsInstance(backbone.head, nn.Identity)
        self.assertEqual(model.feature_dim, 3)
        self.assertEqual(prediction.shape, (2, 4))
        self.assertEqual(count_trainable_parameters(model), 16)

    def test_pretrained_inference_reports_official_and_common_metrics(self) -> None:
        torch.manual_seed(4)
        validation = PUNImageDataset(
            _TinyNUM(3, offset=8), num_anchors=4, mask_source_view=True
        )
        model = PUNUPNet(
            model_name="fixture_vit",
            num_anchors=4,
            pretrained=False,
            backbone=_TinyBackbone(),
        )

        result = evaluate_pun_upnet(
            model,
            validation,
            batch_size=3,
            target_direction="higher",
            huber_delta=1.0,
            ranking_weight=0.1,
            ranking_margin=0.0,
            ndcg_k=3,
            device="cpu",
        )

        self.assertIn("official_unmasked_mse_loss", result.summary)
        self.assertIn("ndcg_at_3_mean", result.summary)
        self.assertEqual(result.predictions.shape, (3, 4))
        self.assertTrue(
            all(row["predicted_local_anchor_id"] != 0 for row in result.per_sample)
        )

    def test_checkpoint_loader_requires_the_pinned_checksum(self) -> None:
        source_model = PUNUPNet(
            model_name="fixture_vit",
            num_anchors=4,
            pretrained=False,
            backbone=_TinyBackbone(),
        )
        destination_model = PUNUPNet(
            model_name="fixture_vit",
            num_anchors=4,
            pretrained=False,
            backbone=_TinyBackbone(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "official.pth"
            torch.save(source_model.state_dict(), checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            resolved = ensure_pun_checkpoint(
                checkpoint,
                expected_sha256=digest,
                download_if_missing=False,
            )
            downloaded = Path(temporary) / "cache" / "official.pth"
            self.assertEqual(
                ensure_pun_checkpoint(
                    downloaded,
                    expected_sha256=digest,
                    download_url=checkpoint.as_uri(),
                    download_if_missing=True,
                ),
                downloaded,
            )
            self.assertEqual(downloaded.read_bytes(), checkpoint.read_bytes())
            load_pun_checkpoint(
                destination_model, resolved, expected_sha256=digest
            )
            with self.assertRaisesRegex(PUNModelError, "SHA-256 mismatch"):
                ensure_pun_checkpoint(
                    checkpoint,
                    expected_sha256="0" * 64,
                    download_if_missing=False,
                )

        for source, destination in zip(
            source_model.parameters(), destination_model.parameters(), strict=True
        ):
            torch.testing.assert_close(source, destination)


if __name__ == "__main__":
    unittest.main()
