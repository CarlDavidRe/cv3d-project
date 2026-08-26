"""Frozen object-disjoint split construction for the Phase 1 NUM dataset."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from nbv.data.num_dataset import NUMDatasetError, NUMSampleRecord, ObjectKey
from nbv.geometry import CANONICAL_ANCHOR_COUNT


PUN_HELD_OUT_CATEGORIES = ("02958343", "03001627")
PUN_TRAIN_OBJECT_LIMIT = 90
PUN_TEST_OBJECT_END = 100
PUN_VALIDATION_FRACTION = 1 / 9


def complete_objects_by_category(
    records: Iterable[NUMSampleRecord],
) -> dict[str, tuple[str, ...]]:
    """Return objects having one paired sample for every canonical anchor."""

    anchors_by_object: dict[ObjectKey, set[int]] = defaultdict(set)
    for record in records:
        key = (record.category_id, record.object_id)
        if record.source_anchor_id in anchors_by_object[key]:
            raise NUMDatasetError(
                f"Duplicate source anchor {record.source_anchor_id} for "
                f"{record.category_id}/{record.object_id}"
            )
        anchors_by_object[key].add(record.source_anchor_id)

    expected = set(range(CANONICAL_ANCHOR_COUNT))
    by_category: dict[str, list[str]] = defaultdict(list)
    for (category_id, object_id), anchor_ids in anchors_by_object.items():
        if anchor_ids == expected:
            by_category[category_id].append(object_id)
    return {
        category_id: tuple(sorted(object_ids))
        for category_id, object_ids in sorted(by_category.items())
    }


def pun_compatible_object_split(
    objects_by_category: Mapping[str, Sequence[str]],
    *,
    held_out_categories: Iterable[str] = PUN_HELD_OUT_CATEGORIES,
) -> dict[str, list[str]]:
    """Create a deterministic object-level adaptation of the official split.

    PUN holds car and chair categories out of training, uses the first 90
    sorted objects of every other category for training, and uses objects
    90:100 for in-category testing.  Its validation split is image-level.  To
    satisfy this project's no-leakage rule, the final one ninth of each PUN
    training pool is instead reserved for validation at the object level.
    """

    held_out = frozenset(held_out_categories)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[ObjectKey] = set()

    for category_id in sorted(objects_by_category):
        raw_object_ids = objects_by_category[category_id]
        object_ids = sorted(raw_object_ids)
        if len(object_ids) != len(set(object_ids)):
            raise NUMDatasetError(
                f"Category {category_id!r} contains duplicate object IDs"
            )
        if any(not object_id or "/" in object_id for object_id in object_ids):
            raise NUMDatasetError(
                f"Category {category_id!r} contains an invalid object ID"
            )

        if category_id in held_out:
            assignments = {"test": object_ids}
        else:
            training_pool = object_ids[:PUN_TRAIN_OBJECT_LIMIT]
            test_pool = object_ids[PUN_TRAIN_OBJECT_LIMIT:PUN_TEST_OBJECT_END]
            validation_count = (
                max(1, round(len(training_pool) * PUN_VALIDATION_FRACTION))
                if len(training_pool) > 1
                else 0
            )
            split_at = len(training_pool) - validation_count
            assignments = {
                "train": training_pool[:split_at],
                "val": training_pool[split_at:],
                "test": test_pool,
            }

        for split_name, assigned_ids in assignments.items():
            for object_id in assigned_ids:
                key = (category_id, object_id)
                if key in seen:
                    raise NUMDatasetError(
                        f"Object {category_id}/{object_id} has multiple splits"
                    )
                seen.add(key)
                splits[split_name].append(f"{category_id}/{object_id}")

    for entries in splits.values():
        entries.sort()
    return splits


def build_num_v1_manifest(
    objects_by_category: Mapping[str, Sequence[str]],
    *,
    excluded_objects: Iterable[str] = (),
) -> dict[str, object]:
    """Build the checked-in NUMv2/PUN object-disjoint manifest payload."""

    splits = pun_compatible_object_split(objects_by_category)
    return {
        "schema_version": 1,
        "split_id": "num_v2_pun_object_disjoint_v1",
        "dataset_release": "NUMv2.zip",
        "pun_commit": "aa6f8f4f12154854a4c1867209725c80475af102",
        "protocol": {
            "held_out_test_categories": list(PUN_HELD_OUT_CATEGORIES),
            "non_held_out_training_slice": [0, PUN_TRAIN_OBJECT_LIMIT],
            "non_held_out_test_slice": [
                PUN_TRAIN_OBJECT_LIMIT,
                PUN_TEST_OBJECT_END,
            ],
            "validation_rule": (
                "last round(N/9) complete objects of each non-held-out PUN "
                "training pool; replaces PUN's image-level random split"
            ),
        },
        "excluded_incomplete_objects": sorted(excluded_objects),
        "object_counts": {
            split_name: len(entries) for split_name, entries in splits.items()
        },
        "splits": splits,
    }
