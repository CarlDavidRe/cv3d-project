from __future__ import annotations

import json
import unittest
from pathlib import Path

from nbv.data import (
    NUMDatasetError,
    PUN_HELD_OUT_CATEGORIES,
    load_object_split,
    pun_compatible_object_split,
)


class NUMSplitTests(unittest.TestCase):
    def test_checked_in_split_is_frozen_disjoint_and_pun_compatible(self) -> None:
        split_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "splits"
            / "num_v1.json"
        )
        splits = load_object_split(split_path)
        self.assertEqual(
            {name: len(entries) for name, entries in splits.items()},
            {"train": 870, "val": 109, "test": 300},
        )
        held_out = set(PUN_HELD_OUT_CATEGORIES)
        self.assertFalse(
            any(category_id in held_out for category_id, _ in splits["train"])
        )
        self.assertFalse(
            any(category_id in held_out for category_id, _ in splits["val"])
        )

        raw = json.loads(split_path.read_text(encoding="utf-8"))
        assigned = set().union(*splits.values())
        excluded = {
            tuple(entry.split("/"))
            for entry in raw["excluded_incomplete_objects"]
        }
        self.assertEqual(len(assigned), 1279)
        self.assertFalse(assigned & excluded)

    def test_split_matches_pun_boundaries_without_object_overlap(self) -> None:
        regular = [f"object_{index:03d}" for index in range(100)]
        short_category = [f"short_{index:03d}" for index in range(79)]
        held_out = [f"held_{index:03d}" for index in range(4)]

        splits = pun_compatible_object_split(
            {
                "02691156": regular,
                "02958343": held_out,
                "03636649": short_category,
            }
        )

        self.assertEqual(len(splits["train"]), 150)
        self.assertEqual(len(splits["val"]), 19)
        self.assertEqual(len(splits["test"]), 14)
        self.assertIn("02691156/object_079", splits["train"])
        self.assertIn("02691156/object_080", splits["val"])
        self.assertIn("02691156/object_090", splits["test"])
        self.assertTrue(
            all(entry.startswith("02958343/") for entry in splits["test"][-4:])
        )

        split_sets = {name: set(entries) for name, entries in splits.items()}
        self.assertFalse(split_sets["train"] & split_sets["val"])
        self.assertFalse(split_sets["train"] & split_sets["test"])
        self.assertFalse(split_sets["val"] & split_sets["test"])

    def test_duplicate_object_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(NUMDatasetError, "duplicate object IDs"):
            pun_compatible_object_split({"category": ["same", "same"]})


if __name__ == "__main__":
    unittest.main()
