from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nbv.data import (
    HistoryDataset,
    HistoryDatasetError,
    VisibilityCache,
    assert_surface_gain_matches_coverage,
    build_history_dataset,
    collate_history_samples,
    save_visibility_cache,
    visibility_cache_fingerprint,
    visibility_cache_path,
)
from nbv.eval.result_schema import visibility_fingerprint
from nbv.geometry import CANONICAL_ORDERING


class HistoryDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_root = self.root / "NUM"
        self.cache_root = self.root / "visibility"
        self.split_path = self.root / "split.json"
        self.object_splits = {
            "train": ["category_a/object_1"],
            "val": ["category_b/object_2"],
            "test": [],
        }
        self.split_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "splits": self.object_splits,
                }
            ),
            encoding="utf-8",
        )
        self.caches = {}
        for offset, object_id in enumerate(
            ("category_a/object_1", "category_b/object_2")
        ):
            self._write_images(object_id, 32 + offset)
            cache = self._cache(object_id, offset)
            save_visibility_cache(
                cache, visibility_cache_path(self.cache_root, object_id)
            )
            self.caches[object_id] = cache

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_generation_is_deterministic_object_disjoint_and_direct(self) -> None:
        first = self._build(self.root / "generated_a")
        second = self._build(self.root / "generated_b")
        first_manifest = json.loads(first.read_text(encoding="utf-8"))
        second_manifest = json.loads(second.read_text(encoding="utf-8"))

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(
            (first.parent / "train.npz").read_bytes(),
            (second.parent / "train.npz").read_bytes(),
        )
        self.assertEqual(first_manifest["coverage_target"], "vis_a")
        self.assertTrue(first_manifest["object_disjoint"])
        self.assertEqual(first_manifest["rotation_metadata"]["kind"], "identity")
        self.assertEqual(first_manifest["splits"]["train"]["num_samples"], 4)
        self.assertEqual(first_manifest["splits"]["val"]["num_samples"], 4)
        self.assertEqual(
            set(first_manifest["splits"]["train"]["object_ids"])
            & set(first_manifest["splits"]["val"]["object_ids"]),
            set(),
        )

        dataset = HistoryDataset(first, split="train")
        self.assertEqual(len(dataset), 4)
        for sample in dataset:
            self.assertEqual(len(sample.history_anchor_ids), sample.history_length)
            self.assertEqual(
                len(set(sample.history_anchor_ids)), sample.history_length
            )
            self.assertFalse(
                sample.valid_candidate_mask[list(sample.history_anchor_ids)].any()
            )
            self.assertFalse(sample.valid_candidate_mask[47])
            self.assertEqual(
                sample.visibility_cache_id,
                visibility_cache_fingerprint(self.caches[sample.object_id]),
            )
            self.assertEqual(
                sample.visibility_cache_id,
                visibility_fingerprint(self.caches[sample.object_id]),
            )
            assert_surface_gain_matches_coverage(
                self.caches[sample.object_id],
                sample.history_anchor_ids,
                sample.target_surface_gain,
                target="vis_a",
            )

    def test_variable_length_collation_keeps_masks_separate(self) -> None:
        manifest = self._build(self.root / "generated")
        dataset = HistoryDataset(manifest, split="train")
        batch = collate_history_samples([dataset[0], dataset[2]])

        self.assertEqual(batch.history_anchor_ids.shape, (2, 3))
        self.assertEqual(batch.history_lengths.tolist(), [1, 3])
        self.assertEqual(
            batch.history_padding_mask.tolist(),
            [[False, True, True], [False, False, False]],
        )
        self.assertEqual(batch.history_anchor_ids[0, 1:].tolist(), [-1, -1])
        self.assertEqual(batch.valid_candidate_mask.shape, (2, 48))
        self.assertEqual(batch.target_surface_gain.shape, (2, 48))
        self.assertEqual(batch.target_surface_gain.dtype, torch.float32)
        self.assertIsNone(batch.history_images)

    def test_images_load_in_history_order_and_are_padded(self) -> None:
        manifest = self._build(self.root / "generated")
        dataset = HistoryDataset(
            manifest,
            split="val",
            load_images=True,
        )
        short = dataset[0]
        long = dataset[2]
        batch = collate_history_samples([short, long])

        self.assertIsNotNone(batch.history_images)
        assert batch.history_images is not None
        self.assertEqual(batch.history_images.shape, (2, 3, 3, 3, 4))
        self.assertTrue(
            torch.allclose(
                batch.history_images[0, 0],
                torch.full((3, 3, 4), 33 / 255),
            )
        )
        self.assertEqual(
            torch.count_nonzero(batch.history_images[0, 1:]).item(), 0
        )

    def test_missing_visibility_cache_fails_before_output_is_created(self) -> None:
        missing_path = visibility_cache_path(
            self.cache_root, "category_b/object_2"
        )
        missing_path.unlink()
        output = self.root / "missing"
        with self.assertRaisesRegex(
            HistoryDatasetError, "missing visibility caches"
        ):
            self._build(output)
        self.assertFalse(output.exists())

    def test_manifest_integrity_detects_modification(self) -> None:
        manifest = self._build(self.root / "generated")
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["coverage_target"] = "vis"
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(HistoryDatasetError, "dataset_id"):
            HistoryDataset(manifest, split="train")

    def _build(self, output: Path) -> Path:
        return build_history_dataset(
            output,
            data_root=self.data_root,
            visibility_cache_root=self.cache_root,
            split_manifest=self.split_path,
            splits=["train", "val"],
            history_lengths=[1, 3],
            histories_per_object_per_length=2,
            seed=7,
            coverage_target="vis_a",
            invalid_anchor_ids=[47],
            expected_visibility_metadata={
                "schema_version": 2,
                "anchor_ordering": CANONICAL_ORDERING,
                "visibility_definition": (
                    "pun_unoccluded_rasterized_mesh_faces_v1"
                ),
            },
        )

    def _write_images(self, object_id: str, pixel_value: int) -> None:
        images = self.data_root / object_id / "images"
        images.mkdir(parents=True)
        for anchor_id in range(48):
            Image.new(
                "RGB", (4, 3), color=(pixel_value,) * 3
            ).save(
                images
                / f"viewpoint_{anchor_id}_offset_phi_0.png"
            )

    @staticmethod
    def _cache(object_id: str, offset: int) -> VisibilityCache:
        visibility = np.zeros((48, 6), dtype=np.bool_)
        for anchor_id in range(48):
            visibility[anchor_id, (anchor_id + offset) % 6] = True
            visibility[anchor_id, (anchor_id + offset + 1) % 6] = True
        areas = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)
        return VisibilityCache(
            face_visibility=visibility,
            face_areas=areas,
            anchor_ids=np.arange(48, dtype=np.int16),
            metadata={
                "schema_version": 2,
                "object_id": object_id,
                "n_faces": 6,
                "anchor_ordering": CANONICAL_ORDERING,
                "render_resolution": [32, 32],
                "visibility_target": "vis_a",
                "visibility_definition": (
                    "pun_unoccluded_rasterized_mesh_faces_v1"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
