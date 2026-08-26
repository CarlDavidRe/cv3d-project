from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nbv.data import NUMDataset, NUMDatasetError, load_object_split


class NUMDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "NUM"
        self.split_path = Path(self.temporary_directory.name) / "split.json"
        self._write_object("category_b", "object_2", [10, 2], pixel_value=64)
        self._write_object("category_a", "object_1", [1, 0], pixel_value=128)
        self._write_object("category_c", "object_3", [0], pixel_value=255)
        self._write_manifest(
            train=["category_b/object_2"],
            val=["category_a/object_1"],
            test=["category_c/object_3"],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_sample_has_preprocessed_rgb_target_and_source_anchor(self) -> None:
        dataset = NUMDataset(
            self.root,
            split="val",
            split_manifest=self.split_path,
            target_name="PSNR",
            require_complete_objects=False,
        )

        sample = dataset[0]
        self.assertEqual(sample.record.sample_id, "category_a/object_1/0")
        self.assertEqual(sample.source_anchor_id, 0)
        self.assertEqual(sample.split, "val")
        self.assertEqual(sample.image.shape, (3, 3, 4))
        self.assertEqual(sample.image.dtype, np.float32)
        np.testing.assert_allclose(sample.image, np.float32(128 / 255))
        self.assertEqual(sample.target_map.shape, (48,))
        self.assertEqual(sample.target_map.dtype, np.float32)
        np.testing.assert_array_equal(sample.target_map, np.arange(48))

    def test_discovery_is_sorted_by_category_object_then_numeric_anchor(self) -> None:
        dataset = NUMDataset(
            self.root,
            split="all",
            require_complete_objects=False,
        )
        self.assertEqual(
            [record.sample_id for record in dataset.records],
            [
                "category_a/object_1/0",
                "category_a/object_1/1",
                "category_b/object_2/2",
                "category_b/object_2/10",
                "category_c/object_3/0",
            ],
        )

    def test_transform_receives_rgb_pil_image(self) -> None:
        seen: list[tuple[str, tuple[int, int]]] = []

        def transform(image: Image.Image) -> str:
            seen.append((image.mode, image.size))
            return "transformed"

        dataset = NUMDataset(
            self.root,
            split="test",
            split_manifest=self.split_path,
            transform=transform,
            require_complete_objects=False,
        )
        self.assertEqual(dataset[0].image, "transformed")
        self.assertEqual(seen, [("RGB", (4, 3))])

    def test_example_images_are_not_treated_as_source_views(self) -> None:
        object_path = self.root / "category_c" / "object_3"
        Image.new("RGB", (4, 3)).save(
            object_path / "images" / "viewpoint_example_1_offset_phi_0.png"
        )
        dataset = NUMDataset(
            self.root,
            split="test",
            split_manifest=self.split_path,
            require_complete_objects=False,
        )
        self.assertEqual(len(dataset), 1)

    def test_object_overlap_in_manifest_is_rejected(self) -> None:
        self._write_manifest(
            train=["category_a/object_1"],
            val=["category_a/object_1"],
            test=[],
        )
        with self.assertRaisesRegex(NUMDatasetError, "appears in both"):
            load_object_split(self.split_path)

    def test_split_is_required_for_training_partitions(self) -> None:
        with self.assertRaisesRegex(NUMDatasetError, "split_manifest is required"):
            NUMDataset(
                self.root,
                split="train",
                require_complete_objects=False,
            )

    def test_invalid_target_shape_is_rejected_when_sample_is_loaded(self) -> None:
        target_path = (
            self.root
            / "category_c"
            / "object_3"
            / "uncertainties"
            / "viewpoint_0_offset_phi_0.json"
        )
        target_path.write_text(json.dumps({"PSNR": [1.0, 2.0]}), encoding="utf-8")
        dataset = NUMDataset(
            self.root,
            split="test",
            split_manifest=self.split_path,
            require_complete_objects=False,
        )
        with self.assertRaisesRegex(NUMDatasetError, r"shape \[48\]"):
            _ = dataset[0]

    def test_complete_object_check_reports_missing_anchors(self) -> None:
        with self.assertRaisesRegex(NUMDatasetError, "does not have exactly anchors"):
            NUMDataset(
                self.root,
                split="test",
                split_manifest=self.split_path,
            )

    def test_non_strict_discovery_skips_images_without_targets(self) -> None:
        missing_target_image = (
            self.root
            / "category_c"
            / "object_3"
            / "images"
            / "viewpoint_1_offset_phi_0.png"
        )
        Image.new("RGB", (4, 3)).save(missing_target_image)
        dataset = NUMDataset(
            self.root,
            split="all",
            require_complete_objects=False,
        )
        self.assertNotIn(
            "category_c/object_3/1",
            [record.sample_id for record in dataset.records],
        )

    def _write_object(
        self,
        category_id: str,
        object_id: str,
        anchor_ids: list[int],
        *,
        pixel_value: int,
    ) -> None:
        object_path = self.root / category_id / object_id
        images = object_path / "images"
        targets = object_path / "uncertainties"
        images.mkdir(parents=True)
        targets.mkdir()
        target = {
            "PSNR": list(range(48)),
            "SSIM": [index / 48 for index in range(48)],
        }
        for anchor_id in anchor_ids:
            stem = f"viewpoint_{anchor_id}_offset_phi_0"
            Image.new("L", (4, 3), color=pixel_value).save(images / f"{stem}.png")
            (targets / f"{stem}.json").write_text(
                json.dumps(target), encoding="utf-8"
            )

    def _write_manifest(
        self, *, train: list[str], val: list[str], test: list[str]
    ) -> None:
        self.split_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "splits": {"train": train, "val": val, "test": test},
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
