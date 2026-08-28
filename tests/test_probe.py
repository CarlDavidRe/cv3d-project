from __future__ import annotations

import unittest

import torch

from nbv.losses import masked_huber_loss, masked_pairwise_ranking_loss
from nbv.models import LightweightProbeHead, count_trainable_parameters
from nbv.training import fit_probe


class ProbeTests(unittest.TestCase):
    def test_head_maps_pooled_features_to_anchor_map(self) -> None:
        head = LightweightProbeHead(6, hidden_dim=4, num_anchors=3)

        predictions = head(torch.randn(2, 6))

        self.assertEqual(predictions.shape, (2, 3))
        self.assertEqual(count_trainable_parameters(head), 55)

    def test_head_rejects_wrong_feature_shape(self) -> None:
        head = LightweightProbeHead(6)
        with self.assertRaisesRegex(ValueError, r"\[B, 6\]"):
            head(torch.randn(2, 5))

    def test_masked_huber_ignores_invalid_anchors(self) -> None:
        predictions = torch.tensor([[0.0, 100.0, 2.0]])
        targets = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, True]])

        loss = masked_huber_loss(predictions, targets, mask, delta=1.0)

        self.assertAlmostEqual(loss.item(), 0.75)

    def test_masked_huber_rejects_empty_mask(self) -> None:
        values = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_huber_loss(values, values, torch.zeros(1, 3, dtype=torch.bool))

    def test_pairwise_ranking_prefers_the_target_order(self) -> None:
        targets = torch.tensor([[3.0, 2.0, 1.0]])
        correct = masked_pairwise_ranking_loss(
            torch.tensor([[4.0, 2.0, 0.0]]), targets
        )
        reversed_order = masked_pairwise_ranking_loss(
            torch.tensor([[0.0, 2.0, 4.0]]), targets
        )

        self.assertLess(correct, reversed_order)

    def test_pairwise_ranking_ignores_masked_pairs_and_target_ties(self) -> None:
        predictions = torch.tensor([[0.0, 100.0, 1.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 2.0, 0.0]])
        mask = torch.tensor([[True, False, True]])

        loss = masked_pairwise_ranking_loss(predictions, targets, mask)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(predictions.grad)

    def test_tiny_detached_feature_set_can_be_overfit(self) -> None:
        torch.manual_seed(4)
        features = torch.randn(4, 5).detach()
        targets = torch.randn(4, 3)
        mask = torch.ones_like(targets, dtype=torch.bool)
        head = LightweightProbeHead(5, hidden_dim=24, num_anchors=3)

        result = fit_probe(
            head,
            features,
            targets,
            mask,
            epochs=300,
            batch_size=4,
            learning_rate=0.02,
            weight_decay=0.0,
            device="cpu",
            seed=9,
        )

        self.assertLess(result.best_loss, result.initial_loss * 0.01)
        self.assertGreater(result.relative_loss_reduction, 0.99)
        self.assertFalse(features.requires_grad)


if __name__ == "__main__":
    unittest.main()
