from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from nbv.config import ConfigError
from nbv.features import (
    CachedFeatureDataset,
    FeatureCacheError,
    cache_fingerprint,
    feature_cache_path,
    load_feature_cache,
    save_feature_cache,
)


class FeatureCacheTests(unittest.TestCase):
    def _cache(self, metadata: dict[str, object]) -> CachedFeatureDataset:
        return CachedFeatureDataset(
            features=torch.arange(12, dtype=torch.float16).reshape(3, 4),
            targets=torch.arange(9, dtype=torch.float32).reshape(3, 3),
            valid_mask=torch.tensor(
                [[False, True, True], [True, True, True], [True, False, True]]
            ),
            sample_ids=("a/one/0", "a/one/1", "b/two/0"),
            source_anchor_ids=torch.tensor([0, 1, 0], dtype=torch.int64),
            metadata=metadata,
        )

    def test_round_trip_preserves_tensors_and_metadata(self) -> None:
        metadata = {"backbone": "fake", "components": ["pooled_patch"]}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "cache.pt"
            save_feature_cache(self._cache(metadata), destination)

            loaded = load_feature_cache(
                destination, expected_metadata=metadata
            )

        torch.testing.assert_close(
            loaded.features, self._cache(metadata).features
        )
        torch.testing.assert_close(loaded.targets, self._cache(metadata).targets)
        self.assertEqual(loaded.sample_ids, self._cache(metadata).sample_ids)
        self.assertEqual(dict(loaded.metadata), metadata)

    def test_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "cache.pt"
            save_feature_cache(self._cache({"version": 1}), destination)
            with self.assertRaisesRegex(FeatureCacheError, "does not match"):
                load_feature_cache(
                    destination, expected_metadata={"version": 2}
                )

    def test_fingerprint_is_stable_and_path_components_are_safe(self) -> None:
        left = cache_fingerprint({"b": 2, "a": [1]})
        right = cache_fingerprint({"a": [1], "b": 2})
        self.assertEqual(left, right)
        path = feature_cache_path(
            "cache",
            backbone="fake",
            variant="probe_one",
            split="train",
            metadata={"a": 1},
        )
        self.assertEqual(path.parent, Path("cache/fake/probe_one"))
        with self.assertRaisesRegex(FeatureCacheError, "may contain"):
            feature_cache_path(
                "cache",
                backbone="../fake",
                variant="probe",
                split="train",
                metadata={},
            )
        with self.assertRaisesRegex(ConfigError, "step-numbered"):
            feature_cache_path(
                "data/step5/features",
                backbone="fake",
                variant="probe",
                split="train",
                metadata={},
            )


if __name__ == "__main__":
    unittest.main()
