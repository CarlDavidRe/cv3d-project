from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from nbv.data.observation_store import AcquiredObservation
from nbv.geometry import CANONICAL_ORDERING, canonical_anchors
from nbv.features import FrozenFeatures
from nbv.policies import ObservationState, PUNPolicy, VGGTPolicy
from nbv.policies.aggregation import (
    aggregate_pun_psnr,
    align_pun_history,
    align_pun_relative_map,
    pun_relative_directions,
)


def _official_reference_alignment(
    relative_map: np.ndarray, source: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    target = source / np.linalg.norm(source)
    first = directions[0] / np.linalg.norm(directions[0])
    axis = np.cross(first, target)
    sine = np.linalg.norm(axis)
    cosine = np.dot(first, target)
    rotated = directions.copy()
    if sine != 0:
        skew = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        rotation = np.eye(3) + skew + skew @ skew * ((1 - cosine) / sine**2)
        rotated = directions @ rotation.T
    result = []
    for candidate in directions:
        angles = np.arccos(np.clip(rotated @ candidate, -1.0, 1.0))
        nearby = np.where(angles < np.radians(30))[0]
        weights = np.exp(-angles[nearby])
        weights /= weights.sum()
        result.append(np.dot(weights, relative_map[nearby]))
    return np.asarray(result)


class _FakeUPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []
        self.marker = nn.Parameter(torch.ones(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append(images.shape[0])
        base = images[:, 0, 0, 0]
        anchors = torch.linspace(10.0, 20.0, 48, device=images.device)
        return base[:, None] + anchors[None, :] * self.marker


def _transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


class _FakeVGGTExtractor:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def extract(self, images: torch.Tensor | np.ndarray) -> FrozenFeatures:
        batch = torch.as_tensor(images)
        self.calls.append(batch.shape[0])
        marker = batch[:, 0, 0, 0]
        patches = torch.stack([
            torch.stack([marker, marker + 1], dim=1),
            torch.stack([marker + 2, marker + 3], dim=1),
        ], dim=1)
        return FrozenFeatures(
            pooled_patch=patches.mean(dim=1),
            patch_tokens=patches,
        )


class _FakeNUMHead(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        anchors = torch.linspace(10.0, 20.0, 48, device=features.device)
        return features.sum(dim=1, keepdim=True) + anchors


class PUNAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directions = canonical_anchors().directions
        self.raw = np.linspace(7.5, 31.0, 48)

    def test_alignment_matches_pinned_official_formula(self) -> None:
        for source_id in (0, 12, 31, 46):
            expected = _official_reference_alignment(
                self.raw, self.directions[source_id], self.directions
            )
            actual = align_pun_relative_map(
                self.raw,
                source_direction=self.directions[source_id],
                anchor_directions=self.directions,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)

        # The exact-antipode no-op is a behavior of official rotate_to_target.
        np.testing.assert_allclose(
            pun_relative_directions(self.directions, self.directions[46]),
            self.directions,
        )

    def test_history_alignment_retains_source_and_history_order(self) -> None:
        raw = np.stack([self.raw, self.raw[::-1]])
        actual = align_pun_history(raw, [12, 0], self.directions)
        np.testing.assert_allclose(
            actual[0],
            _official_reference_alignment(raw[0], self.directions[12], self.directions),
        )
        np.testing.assert_allclose(
            actual[1],
            _official_reference_alignment(raw[1], self.directions[0], self.directions),
        )

    def test_small_filter_precedes_raw_product_and_psnr_argmin(self) -> None:
        aligned = np.stack([
            np.linspace(1.0, 48.0, 48),
            np.linspace(48.0, 1.0, 48),
        ])
        valid = np.ones(48, dtype=bool)
        result = aggregate_pun_psnr(aligned, valid, suppression_threshold=0.1)
        np.testing.assert_allclose(result.combined_raw_psnr, np.prod(aligned, axis=0))
        expected_suppressed = np.any(result.normalized_maps >= 0.9, axis=0)
        np.testing.assert_array_equal(result.kept_candidate_mask, ~expected_suppressed)
        selected = int(np.argmax(result.policy_scores))
        expected = int(np.flatnonzero(~expected_suppressed)[
            np.argmin(result.combined_raw_psnr[~expected_suppressed])
        ])
        self.assertEqual(selected, expected)
        self.assertLess(
            result.policy_scores[expected_suppressed].max(),
            result.policy_scores[result.kept_candidate_mask].min(),
        )

    def test_empty_small_filter_uses_documented_finite_fallback(self) -> None:
        aligned = np.eye(48) * 10.0
        result = aggregate_pun_psnr(aligned, np.ones(48, bool))
        self.assertTrue(result.filter_fallback_used)
        self.assertTrue(result.kept_candidate_mask.all())
        self.assertTrue(np.isfinite(result.policy_scores).all())


class _PolicyStateMixin:
    def observation(self, anchor_id: int) -> AcquiredObservation:
        rgb = np.full((3, 4, 3), anchor_id, dtype=np.uint8)
        rgb.setflags(write=False)
        return AcquiredObservation(anchor_id, f"/images/{anchor_id}.png", rgb)

    def state(self, ids: tuple[int, ...], step: int) -> ObservationState:
        valid = canonical_anchors().valid_candidate_mask(ids)
        return ObservationState(
            object_id="category/object",
            acquired_observations=tuple(self.observation(i) for i in ids),
            anchor_directions=canonical_anchors().directions.copy(),
            camera_to_world=np.zeros((48, 4, 4)),
            valid_candidate_mask=valid,
            step_index=step,
            seed=0,
            anchor_ordering=CANONICAL_ORDERING,
        )


class PUNPolicyTests(_PolicyStateMixin, unittest.TestCase):

    def test_policy_infers_only_new_images_and_saves_raw_maps_separately(self) -> None:
        model = _FakeUPNet()
        policy = PUNPolicy(
            model,
            _transform,
            provenance={"checkpoint_sha256": "fixture"},
        )
        first = policy.score(self.state((0,), 0))
        second = policy.score(self.state((0, 12), 1))
        self.assertEqual(model.calls, [1, 1])
        self.assertEqual(first.shape, (48,))
        self.assertEqual(second.shape, (48,))
        self.assertTrue(np.isfinite(second).all())
        self.assertFalse(next(model.parameters()).requires_grad)

        with tempfile.TemporaryDirectory() as temporary:
            path = policy.save_prediction_cache(Path(temporary) / "maps.npz")
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload["anchor_ids"].tolist(), [0, 12])
                self.assertEqual(payload["raw_prediction_maps"].shape, (2, 48))
                np.testing.assert_allclose(
                    payload["raw_prediction_maps"][1],
                    np.linspace(22.0, 32.0, 48),
                    atol=1e-6,
                )
                metadata = str(payload["metadata_json"].item())
                self.assertIn("official_small_filter_then_raw_product", metadata)
                self.assertIn("fixture", metadata)

    def test_policy_rejects_empty_history(self) -> None:
        policy = PUNPolicy(_FakeUPNet(), _transform)
        state = self.state((0,), 0)
        empty = ObservationState(
            object_id=state.object_id,
            acquired_observations=(),
            anchor_directions=state.anchor_directions,
            camera_to_world=state.camera_to_world,
            valid_candidate_mask=np.ones(48, bool),
            step_index=0,
            seed=0,
            anchor_ordering=state.anchor_ordering,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            policy.score(empty)


class VGGTPolicyTests(_PolicyStateMixin, unittest.TestCase):
    def test_cached_and_live_predictions_match_and_history_is_incremental(self) -> None:
        extractor = _FakeVGGTExtractor()
        # RGB value 0 produces max-pooled [2, 3]. Anchor 12 is extracted live.
        cached = {"category/object/0": torch.tensor([2.0, 3.0])}
        mixed = VGGTPolicy(
            _FakeNUMHead(), ["max_pooled_patch"],
            cached_features=cached, extractor=extractor,
            provenance={"checkpoint_sha256": "fixture"},
        )
        mixed.score(self.state((0,), 0))
        mixed_scores = mixed.score(self.state((0, 12), 1))
        self.assertEqual(extractor.calls, [1])
        self.assertEqual(
            mixed.provenance["prediction_source_counts"],
            {"feature_cache": 1, "live_vggt": 1},
        )

        live_extractor = _FakeVGGTExtractor()
        live = VGGTPolicy(
            _FakeNUMHead(), ["max_pooled_patch"], extractor=live_extractor
        )
        live_scores = live.score(self.state((0, 12), 1))
        np.testing.assert_allclose(mixed_scores, live_scores, atol=1e-10)
        self.assertEqual(live_extractor.calls, [2])

        with tempfile.TemporaryDirectory() as temporary:
            path = mixed.save_prediction_cache(Path(temporary) / "maps.npz")
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload["anchor_ids"].tolist(), [0, 12])
                self.assertEqual(
                    payload["prediction_sources"].tolist(),
                    ["feature_cache", "live_vggt"],
                )
                self.assertEqual(payload["raw_prediction_maps"].shape, (2, 48))

    def test_cache_miss_requires_explicit_live_extraction(self) -> None:
        policy = VGGTPolicy(_FakeNUMHead(), ["max_pooled_patch"])
        with self.assertRaisesRegex(ValueError, "live VGGT extraction is disabled"):
            policy.score(self.state((0,), 0))

    def test_pun_and_vggt_consume_the_identical_supplied_history(self) -> None:
        pun_model = _FakeUPNet()
        pun = PUNPolicy(pun_model, _transform)
        extractor = _FakeVGGTExtractor()
        vggt = VGGTPolicy(
            _FakeNUMHead(), ["max_pooled_patch"], extractor=extractor
        )
        for state in (self.state((0,), 0), self.state((0, 12), 1)):
            pun.score(state)
            vggt.score(state)
        self.assertEqual(pun_model.calls, [1, 1])
        self.assertEqual(extractor.calls, [1, 1])

        with tempfile.TemporaryDirectory() as temporary:
            pun_path = pun.save_prediction_cache(Path(temporary) / "pun.npz")
            vggt_path = vggt.save_prediction_cache(Path(temporary) / "vggt.npz")
            with (
                np.load(pun_path, allow_pickle=False) as pun_cache,
                np.load(vggt_path, allow_pickle=False) as vggt_cache,
            ):
                np.testing.assert_array_equal(
                    pun_cache["anchor_ids"], vggt_cache["anchor_ids"]
                )
                np.testing.assert_array_equal(
                    pun_cache["image_paths"], vggt_cache["image_paths"]
                )


if __name__ == "__main__":
    unittest.main()
