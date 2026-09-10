"""Deterministic Phase 3 histories with direct mesh-surface-gain targets."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from nbv.data.num_dataset import load_object_split, rgb_to_float_chw
from nbv.data.visibility_cache import (
    VisibilityCache,
    VisibilityCacheError,
    load_visibility_cache,
    visibility_cache_fingerprint,
    visibility_cache_path,
)
from nbv.geometry.anchors import (
    CANONICAL_ANCHOR_COUNT,
    CANONICAL_ORDERING,
    canonical_anchors,
)
from nbv.geometry.coverage import VISIBILITY_TARGETS


HISTORY_DATASET_SCHEMA_VERSION = 1
HISTORY_SAMPLING_STRATEGY = "seeded_unique_anchor_subsets_v1"
NO_ROTATION_METADATA = {
    "enabled": False,
    "kind": "identity",
    "description": "non_rotated_canonical_num_observations",
}
_SPLITS = ("train", "val", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_NAME = "viewpoint_{anchor_id}_offset_phi_0.png"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class HistoryDatasetError(ValueError):
    """Raised when Phase 3 history data or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class HistorySample:
    """One logical direct-gain training sample."""

    sample_id: str
    object_id: str
    history_image_paths: tuple[str, ...]
    history_anchor_ids: tuple[int, ...]
    history_length: int
    target_surface_gain: NDArray[np.float32]
    valid_candidate_mask: NDArray[np.bool_]
    rotation_metadata: Mapping[str, Any]
    split: str
    visibility_cache_id: str
    coverage_target: str
    sampling_metadata: Mapping[str, Any]
    history_images: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise HistoryDatasetError("sample_id must be a non-empty string")
        _validate_object_id(self.object_id)
        if self.split not in _SPLITS:
            raise HistoryDatasetError(f"split must be one of {_SPLITS}")
        if self.coverage_target not in VISIBILITY_TARGETS:
            raise HistoryDatasetError(
                f"coverage_target must be one of {VISIBILITY_TARGETS}"
            )
        if not _SHA256.fullmatch(self.visibility_cache_id):
            raise HistoryDatasetError("visibility_cache_id must be a SHA-256 digest")
        anchors = _validate_history(self.history_anchor_ids)
        if self.history_length != len(anchors):
            raise HistoryDatasetError(
                "history_length must equal the number of history anchors"
            )
        if len(self.history_image_paths) != self.history_length:
            raise HistoryDatasetError(
                "history_image_paths must have one path per history anchor"
            )
        if any(
            not isinstance(path, str) or not path
            for path in self.history_image_paths
        ):
            raise HistoryDatasetError("history image paths must be non-empty strings")
        target = np.ascontiguousarray(self.target_surface_gain, dtype=np.float32)
        valid = np.ascontiguousarray(self.valid_candidate_mask)
        if target.shape != (CANONICAL_ANCHOR_COUNT,):
            raise HistoryDatasetError("target_surface_gain must have shape [48]")
        if not np.isfinite(target).all() or np.any(target < -1e-7):
            raise HistoryDatasetError(
                "target_surface_gain must be finite and non-negative"
            )
        target = np.maximum(target, np.float32(0))
        if valid.shape != (CANONICAL_ANCHOR_COUNT,) or valid.dtype != np.bool_:
            raise HistoryDatasetError(
                "valid_candidate_mask must be a boolean [48] array"
            )
        if not valid.any():
            raise HistoryDatasetError("each history must retain a valid candidate")
        if valid[list(anchors)].any():
            raise HistoryDatasetError("acquired history anchors must be masked")
        if not np.allclose(target[list(anchors)], 0.0, atol=1e-7, rtol=0):
            raise HistoryDatasetError(
                "acquired history anchors must have zero direct marginal gain"
            )
        if not isinstance(self.rotation_metadata, Mapping):
            raise HistoryDatasetError("rotation_metadata must be a mapping")
        if not isinstance(self.sampling_metadata, Mapping):
            raise HistoryDatasetError("sampling_metadata must be a mapping")
        if (
            self.history_images is not None
            and len(self.history_images) != len(anchors)
        ):
            raise HistoryDatasetError(
                "history_images must have one item per history anchor"
            )
        target.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "history_anchor_ids", anchors)
        object.__setattr__(self, "target_surface_gain", target)
        object.__setattr__(self, "valid_candidate_mask", valid)
        object.__setattr__(self, "rotation_metadata", dict(self.rotation_metadata))
        object.__setattr__(self, "sampling_metadata", dict(self.sampling_metadata))


@dataclass(frozen=True, slots=True)
class HistoryBatch:
    """Variable-length batch; true padding-mask entries are ignored."""

    sample_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    history_image_paths: tuple[tuple[str, ...], ...]
    history_anchor_ids: Tensor
    history_lengths: Tensor
    history_padding_mask: Tensor
    target_surface_gain: Tensor
    valid_candidate_mask: Tensor
    visibility_cache_ids: tuple[str, ...]
    rotation_metadata: tuple[Mapping[str, Any], ...]
    sampling_metadata: tuple[Mapping[str, Any], ...]
    history_images: Tensor | None

    def pin_memory(self) -> "HistoryBatch":
        """Pin tensor storage so CUDA transfers can overlap image loading."""

        return HistoryBatch(
            sample_ids=self.sample_ids,
            object_ids=self.object_ids,
            history_image_paths=self.history_image_paths,
            history_anchor_ids=self.history_anchor_ids.pin_memory(),
            history_lengths=self.history_lengths.pin_memory(),
            history_padding_mask=self.history_padding_mask.pin_memory(),
            target_surface_gain=self.target_surface_gain.pin_memory(),
            valid_candidate_mask=self.valid_candidate_mask.pin_memory(),
            visibility_cache_ids=self.visibility_cache_ids,
            rotation_metadata=self.rotation_metadata,
            sampling_metadata=self.sampling_metadata,
            history_images=(
                None
                if self.history_images is None
                else self.history_images.pin_memory()
            ),
        )

    def to(self, device: str | torch.device) -> "HistoryBatch":
        return HistoryBatch(
            sample_ids=self.sample_ids,
            object_ids=self.object_ids,
            history_image_paths=self.history_image_paths,
            history_anchor_ids=self.history_anchor_ids.to(device, non_blocking=True),
            history_lengths=self.history_lengths.to(device, non_blocking=True),
            history_padding_mask=self.history_padding_mask.to(
                device, non_blocking=True
            ),
            target_surface_gain=self.target_surface_gain.to(
                device, dtype=torch.float32, non_blocking=True
            ),
            valid_candidate_mask=self.valid_candidate_mask.to(
                device, non_blocking=True
            ),
            visibility_cache_ids=self.visibility_cache_ids,
            rotation_metadata=self.rotation_metadata,
            sampling_metadata=self.sampling_metadata,
            history_images=(
                None
                if self.history_images is None
                else self.history_images.to(
                    device, dtype=torch.float32, non_blocking=True
                )
            ),
        )


class HistoryDataset(Sequence[HistorySample]):
    """Read one deterministic split from a generated Phase 3 dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: Literal["train", "val", "test"],
        data_root: str | Path | None = None,
        load_images: bool = False,
        transform: Callable[[Image.Image], Any] | None = None,
        verify_integrity: bool = True,
    ) -> None:
        if split not in _SPLITS:
            raise HistoryDatasetError(f"split must be one of {_SPLITS}")
        if not isinstance(load_images, bool):
            raise HistoryDatasetError("load_images must be a boolean")
        if not isinstance(verify_integrity, bool):
            raise HistoryDatasetError("verify_integrity must be a boolean")
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = _load_manifest(self.manifest_path)
        split_info = self.manifest["splits"].get(split)
        if not isinstance(split_info, Mapping):
            raise HistoryDatasetError(
                f"Dataset does not contain requested split {split!r}"
            )
        shard_path = _safe_child(self.root, split_info.get("file"), "split file")
        if verify_integrity:
            actual = sha256_file(shard_path)
            if actual != split_info.get("sha256"):
                raise HistoryDatasetError(
                    f"History shard checksum mismatch for {shard_path}"
                )
        self._arrays, shard_metadata = _load_shard(shard_path)
        if shard_metadata.get("split") != split:
            raise HistoryDatasetError("History shard split metadata disagrees")
        for key, expected in (
            ("coverage_target", self.manifest["coverage_target"]),
            ("anchor_ordering", self.manifest["anchor_ordering"]),
            ("sampling", self.manifest["sampling"]),
            ("rotation_metadata", self.manifest["rotation_metadata"]),
        ):
            if shard_metadata.get(key) != expected:
                raise HistoryDatasetError(
                    f"History shard {key} metadata disagrees with manifest"
                )
        self.split = split
        self.load_images = bool(load_images)
        self.transform = transform if transform is not None else rgb_to_float_chw
        default_data_root = self.root / self.manifest["image_paths"][
            "data_root_relative_to_dataset"
        ]
        self.data_root = (
            Path(data_root).resolve()
            if data_root is not None
            else default_data_root.resolve()
        )
        self._validate_loaded_shard(split_info)

    def __len__(self) -> int:
        return int(self._arrays["history_lengths"].shape[0])

    def __getitem__(self, index: int | slice) -> HistorySample | list[HistorySample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        length = int(self._arrays["history_lengths"][index])
        anchor_ids = tuple(
            int(value)
            for value in self._arrays["history_anchor_ids"][index, :length]
        )
        relative_paths = tuple(
            str(value)
            for value in self._arrays["history_image_paths"][index, :length]
        )
        images = None
        if self.load_images:
            loaded: list[Any] = []
            for relative in relative_paths:
                path = _safe_child(self.data_root, relative, "history image path")
                try:
                    with Image.open(path) as image:
                        loaded.append(self.transform(image.convert("RGB")))
                except (OSError, ValueError) as exc:
                    raise HistoryDatasetError(
                        f"Could not read history image {path}: {exc}"
                    ) from exc
            images = tuple(loaded)
        draw_index = int(self._arrays["draw_indices"][index])
        sample_seed = str(self._arrays["sample_seeds"][index])
        return HistorySample(
            sample_id=str(self._arrays["sample_ids"][index]),
            object_id=str(self._arrays["object_ids"][index]),
            history_image_paths=relative_paths,
            history_anchor_ids=anchor_ids,
            history_length=length,
            target_surface_gain=self._arrays["target_surface_gain"][index].copy(),
            valid_candidate_mask=self._arrays["valid_candidate_mask"][index].copy(),
            rotation_metadata=dict(self.manifest["rotation_metadata"]),
            split=self.split,
            visibility_cache_id=str(self._arrays["visibility_cache_ids"][index]),
            coverage_target=str(self.manifest["coverage_target"]),
            sampling_metadata={
                **self.manifest["sampling"],
                "draw_index": draw_index,
                "sample_seed": sample_seed,
            },
            history_images=images,
        )

    def _validate_loaded_shard(self, split_info: Mapping[str, Any]) -> None:
        arrays = self._arrays
        required = {
            "sample_ids",
            "object_ids",
            "history_anchor_ids",
            "history_image_paths",
            "history_lengths",
            "target_surface_gain",
            "valid_candidate_mask",
            "visibility_cache_ids",
            "draw_indices",
            "sample_seeds",
        }
        missing = required - set(arrays)
        if missing:
            raise HistoryDatasetError(
                f"History shard is missing arrays: {sorted(missing)}"
            )
        n = len(arrays["sample_ids"])
        max_length = int(self.manifest["sampling"]["maximum_history_length"])
        expected_shapes = {
            "object_ids": (n,),
            "history_anchor_ids": (n, max_length),
            "history_image_paths": (n, max_length),
            "history_lengths": (n,),
            "target_surface_gain": (n, CANONICAL_ANCHOR_COUNT),
            "valid_candidate_mask": (n, CANONICAL_ANCHOR_COUNT),
            "visibility_cache_ids": (n,),
            "draw_indices": (n,),
            "sample_seeds": (n,),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise HistoryDatasetError(
                    f"{key} must have shape {list(shape)}, "
                    f"got {list(arrays[key].shape)}"
                )
        if n != split_info.get("num_samples"):
            raise HistoryDatasetError("History shard sample count disagrees")
        if arrays["valid_candidate_mask"].dtype != np.bool_:
            raise HistoryDatasetError("valid_candidate_mask must have boolean dtype")
        if not np.issubdtype(arrays["history_anchor_ids"].dtype, np.integer):
            raise HistoryDatasetError("history_anchor_ids must have integer dtype")
        if not np.issubdtype(arrays["history_lengths"].dtype, np.integer):
            raise HistoryDatasetError("history_lengths must have integer dtype")
        if arrays["target_surface_gain"].dtype != np.float32:
            raise HistoryDatasetError("target_surface_gain must have float32 dtype")
        if len(set(arrays["sample_ids"].tolist())) != n:
            raise HistoryDatasetError("sample_ids must be unique")
        declared_objects = set(split_info["object_ids"])
        actual_objects = set(arrays["object_ids"].tolist())
        if actual_objects != declared_objects:
            raise HistoryDatasetError("History shard object IDs disagree with manifest")
        for index in range(n):
            length = int(arrays["history_lengths"][index])
            if length not in self.manifest["sampling"]["history_lengths"]:
                raise HistoryDatasetError("Encountered an unconfigured history length")
            anchors = arrays["history_anchor_ids"][index]
            paths = arrays["history_image_paths"][index]
            _validate_history(tuple(int(value) for value in anchors[:length]))
            if np.any(anchors[length:] != -1):
                raise HistoryDatasetError("Padded history anchor IDs must be -1")
            if np.any(paths[length:] != ""):
                raise HistoryDatasetError("Padded history image paths must be empty")
            valid = arrays["valid_candidate_mask"][index]
            expected_valid = np.ones(
                CANONICAL_ANCHOR_COUNT, dtype=np.bool_
            )
            expected_valid[self.manifest["invalid_anchor_ids"]] = False
            expected_valid[anchors[:length]] = False
            if not np.array_equal(valid, expected_valid) or not valid.any():
                raise HistoryDatasetError(
                    "Invalid acquired/candidate mask relationship"
                )
            target = arrays["target_surface_gain"][index]
            if not np.isfinite(target).all() or np.any(target < -1e-7):
                raise HistoryDatasetError("Surface-gain labels must be non-negative")
            if not np.allclose(target[anchors[:length]], 0, atol=1e-7, rtol=0):
                raise HistoryDatasetError("Acquired anchors must have zero gains")
            object_id = str(arrays["object_ids"][index])
            expected_paths = tuple(
                _relative_image_path(object_id, int(anchor_id))
                for anchor_id in anchors[:length]
            )
            if tuple(paths[:length].tolist()) != expected_paths:
                raise HistoryDatasetError(
                    "History image paths disagree with object/anchor IDs"
                )
            expected_cache_id = self.manifest["visibility_cache_ids"].get(
                object_id
            )
            if str(arrays["visibility_cache_ids"][index]) != expected_cache_id:
                raise HistoryDatasetError(
                    "History visibility cache ID disagrees with manifest"
                )


def collate_history_samples(samples: Sequence[HistorySample]) -> HistoryBatch:
    """Pad observation axes while retaining a separate 48-candidate mask."""

    if not samples:
        raise HistoryDatasetError("cannot collate an empty history batch")
    max_length = max(sample.history_length for sample in samples)
    anchors = torch.full((len(samples), max_length), -1, dtype=torch.int64)
    padding = torch.ones((len(samples), max_length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = sample.history_length
        anchors[row, :length] = torch.as_tensor(
            sample.history_anchor_ids, dtype=torch.int64
        )
        padding[row, :length] = False
    target = torch.from_numpy(
        np.stack([sample.target_surface_gain for sample in samples])
    )
    candidates = torch.from_numpy(
        np.stack([sample.valid_candidate_mask for sample in samples])
    )
    have_images = [sample.history_images is not None for sample in samples]
    if any(have_images) and not all(have_images):
        raise HistoryDatasetError(
            "batch cannot mix samples with loaded and unloaded history images"
        )
    batched_images: Tensor | None = None
    if all(have_images):
        converted: list[list[Tensor]] = []
        image_shape: tuple[int, ...] | None = None
        for sample in samples:
            row = []
            assert sample.history_images is not None
            for image in sample.history_images:
                tensor = torch.as_tensor(image)
                if image_shape is None:
                    image_shape = tuple(tensor.shape)
                elif tuple(tensor.shape) != image_shape:
                    raise HistoryDatasetError(
                        "all transformed history images in a batch must share shape"
                    )
                row.append(tensor)
            converted.append(row)
        assert image_shape is not None
        dtype = converted[0][0].dtype
        batched_images = torch.zeros(
            (len(samples), max_length, *image_shape), dtype=dtype
        )
        for row_index, row in enumerate(converted):
            for column, image in enumerate(row):
                batched_images[row_index, column] = image
    return HistoryBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        object_ids=tuple(sample.object_id for sample in samples),
        history_image_paths=tuple(sample.history_image_paths for sample in samples),
        history_anchor_ids=anchors,
        history_lengths=torch.as_tensor(
            [sample.history_length for sample in samples], dtype=torch.int64
        ),
        history_padding_mask=padding,
        target_surface_gain=target,
        valid_candidate_mask=candidates,
        visibility_cache_ids=tuple(
            sample.visibility_cache_id for sample in samples
        ),
        rotation_metadata=tuple(sample.rotation_metadata for sample in samples),
        sampling_metadata=tuple(sample.sampling_metadata for sample in samples),
        history_images=batched_images,
    )


def build_history_dataset(
    output_dir: str | Path,
    *,
    data_root: str | Path,
    visibility_cache_root: str | Path,
    split_manifest: str | Path,
    splits: Sequence[str],
    history_lengths: Sequence[int],
    histories_per_object_per_length: int,
    seed: int,
    coverage_target: str,
    invalid_anchor_ids: Sequence[int] = (),
    expected_visibility_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    object_ids: Sequence[str] | None = None,
    limit_per_split: int | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Generate atomic split shards and a deterministic manifest.

    Every label comes from the shared visibility-cache gain helper. Two
    histories per object are additionally checked against explicit coverage
    differences. When provided, ``progress`` receives human-readable status
    messages as the build advances.
    """

    destination = Path(output_dir).resolve()
    data_root_path = Path(data_root).resolve()
    cache_root = Path(visibility_cache_root).resolve()
    split_path = Path(split_manifest).resolve()
    selected_splits = _validate_splits(splits)
    lengths = _validate_sampling(
        history_lengths,
        histories_per_object_per_length,
        seed,
        invalid_anchor_ids,
    )
    invalid_ids = tuple(int(value) for value in invalid_anchor_ids)
    if coverage_target not in VISIBILITY_TARGETS:
        raise HistoryDatasetError(
            f"coverage_target must be one of {VISIBILITY_TARGETS}"
        )
    if limit_per_split is not None and (
        isinstance(limit_per_split, bool)
        or not isinstance(limit_per_split, int)
        or limit_per_split <= 0
    ):
        raise HistoryDatasetError("limit_per_split must be a positive integer")
    split_objects = load_object_split(split_path)
    requested_filter = None
    if object_ids:
        if len(set(object_ids)) != len(object_ids):
            raise HistoryDatasetError("Explicit object IDs must be unique")
        requested_filter = set(object_ids)
        for object_id in requested_filter:
            _validate_object_id(object_id)
        known = {
            f"{category}/{object_id}"
            for split in selected_splits
            for category, object_id in split_objects[split]
        }
        unknown = requested_filter - known
        if unknown:
            raise HistoryDatasetError(
                f"Explicit objects are outside selected splits: {sorted(unknown)}"
            )
    objects_by_split: dict[str, list[str]] = {}
    for split in selected_splits:
        values = sorted(
            f"{category}/{object_id}"
            for category, object_id in split_objects[split]
        )
        if requested_filter is not None:
            values = [value for value in values if value in requested_filter]
        if limit_per_split is not None:
            values = values[:limit_per_split]
        if values:
            objects_by_split[split] = values
    if not objects_by_split:
        raise HistoryDatasetError("no objects selected for history generation")
    all_objects = [
        value
        for split in selected_splits
        for value in objects_by_split.get(split, [])
    ]
    total_samples = (
        len(all_objects) * len(lengths) * histories_per_object_per_length
    )
    _report_progress(
        progress,
        "Validating inputs for "
        f"{len(all_objects)} object(s); planning {total_samples} sample(s).",
    )
    _preflight_inputs(data_root_path, cache_root, all_objects)
    if destination.exists():
        if not overwrite:
            raise HistoryDatasetError(
                f"History dataset destination already exists: {destination}"
            )
        if not destination.is_dir():
            raise HistoryDatasetError(
                "Refusing to overwrite a non-directory history destination"
            )
        try:
            _load_manifest(destination / "manifest.json")
        except HistoryDatasetError as exc:
            raise HistoryDatasetError(
                "Refusing to overwrite a directory that is not a valid "
                "history dataset"
            ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        sampling = {
            "strategy": HISTORY_SAMPLING_STRATEGY,
            "seed": seed,
            "histories_per_object_per_length": histories_per_object_per_length,
            "history_lengths": list(lengths),
            "maximum_history_length": max(lengths),
            "duplicate_anchors_within_history": False,
            "duplicate_anchor_subsets_per_object_length": False,
            "history_order": "seeded_acquisition_order",
            "sample_seed_protocol": (
                "sha256_global_seed_split_object_length_draw_v1"
            ),
        }
        split_entries: dict[str, Any] = {}
        cache_ids: dict[str, str] = {}
        visibility_definition: str | None = None
        for split in selected_splits:
            selected = objects_by_split.get(split)
            if not selected:
                continue
            _report_progress(
                progress,
                f"Building split {split!r} ({len(selected)} object(s)).",
            )
            records: list[dict[str, Any]] = []
            for object_index, object_id in enumerate(selected, start=1):
                _report_progress(
                    progress,
                    f"[{split}] object {object_index}/{len(selected)}: {object_id}",
                )
                cache = load_visibility_cache(
                    object_id,
                    cache_root=cache_root,
                    expected_metadata=expected_visibility_metadata,
                )
                current_definition = str(cache.metadata["visibility_definition"])
                if visibility_definition is None:
                    visibility_definition = current_definition
                elif current_definition != visibility_definition:
                    raise HistoryDatasetError(
                        "Visibility definition differs between object caches"
                    )
                cache_id = visibility_cache_fingerprint(cache)
                cache_ids[object_id] = cache_id
                object_records = _generate_object_records(
                    object_id=object_id,
                    split=split,
                    cache=cache,
                    cache_id=cache_id,
                    data_root=data_root_path,
                    history_lengths=lengths,
                    histories_per_length=histories_per_object_per_length,
                    seed=seed,
                    coverage_target=coverage_target,
                    invalid_anchor_ids=invalid_ids,
                )
                for record in object_records[:2]:
                    assert_surface_gain_matches_coverage(
                        cache,
                        record["history_anchor_ids"],
                        record["target_surface_gain"],
                        target=coverage_target,
                    )
                records.extend(object_records)
            shard_path = temporary / f"{split}.npz"
            shard_metadata = {
                "schema_version": HISTORY_DATASET_SCHEMA_VERSION,
                "split": split,
                "coverage_target": coverage_target,
                "anchor_ordering": CANONICAL_ORDERING,
                "sampling": sampling,
                "rotation_metadata": NO_ROTATION_METADATA,
            }
            _save_shard(records, shard_path, shard_metadata, max(lengths))
            _report_progress(
                progress,
                f"Finished split {split!r}: wrote {len(records)} sample(s).",
            )
            split_entries[split] = {
                "file": shard_path.name,
                "sha256": sha256_file(shard_path),
                "num_objects": len(selected),
                "num_samples": len(records),
                "object_ids": selected,
                "history_length_counts": {
                    str(length): len(selected)
                    * histories_per_object_per_length
                    for length in lengths
                },
            }
        manifest_core = {
            "schema_version": HISTORY_DATASET_SCHEMA_VERSION,
            "dataset_type": "phase3_direct_surface_gain_histories",
            "coverage_target": coverage_target,
            "visibility_definition": visibility_definition,
            "anchor_count": CANONICAL_ANCHOR_COUNT,
            "anchor_ordering": CANONICAL_ORDERING,
            "split_manifest_sha256": sha256_file(split_path),
            "split_manifest_name": split_path.name,
            "object_disjoint": True,
            "invalid_anchor_ids": list(invalid_ids),
            "sampling": sampling,
            "rotation_metadata": NO_ROTATION_METADATA,
            "image_paths": {
                "format": "relative_to_data_root",
                "data_root_relative_to_dataset": os.path.relpath(
                    data_root_path, destination
                ),
            },
            "visibility_cache_ids": dict(sorted(cache_ids.items())),
            "visibility_cache_expected_metadata": dict(
                expected_visibility_metadata or {}
            ),
            "provenance": dict(provenance or {}),
            "splits": split_entries,
            "validation": {
                "label_check": "explicit_coverage_difference",
                "histories_checked_per_object": min(
                    2, len(lengths) * histories_per_object_per_length
                ),
            },
        }
        manifest = dict(manifest_core, dataset_id=_json_sha256(manifest_core))
        _report_progress(progress, "Writing dataset manifest and finalizing.")
        _write_json(manifest, temporary / "manifest.json")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination / "manifest.json"


def _report_progress(
    progress: Callable[[str], None] | None, message: str
) -> None:
    if progress is not None:
        progress(message)


def assert_surface_gain_matches_coverage(
    cache: VisibilityCache,
    history_anchor_ids: Sequence[int],
    target_surface_gain: NDArray[np.floating],
    *,
    target: str,
    atol: float = 1e-7,
) -> None:
    """Check all 48 targets against brute-force coverage differences."""

    history = _validate_history(history_anchor_ids)
    labels = np.asarray(target_surface_gain, dtype=np.float64)
    if labels.shape != (CANONICAL_ANCHOR_COUNT,):
        raise HistoryDatasetError("surface-gain labels must have shape [48]")
    before = cache.coverage(history, target=target)
    expected = np.asarray(
        [
            cache.coverage((*history, anchor_id), target=target) - before
            for anchor_id in range(CANONICAL_ANCHOR_COUNT)
        ],
        dtype=np.float64,
    )
    if not np.allclose(labels, expected, atol=atol, rtol=1e-6):
        difference = float(np.max(np.abs(labels - expected)))
        raise HistoryDatasetError(
            f"surface-gain label differs from explicit coverage by {difference}"
        )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generate_object_records(
    *,
    object_id: str,
    split: str,
    cache: VisibilityCache,
    cache_id: str,
    data_root: Path,
    history_lengths: Sequence[int],
    histories_per_length: int,
    seed: int,
    coverage_target: str,
    invalid_anchor_ids: Sequence[int],
) -> list[dict[str, Any]]:
    allowed = sorted(
        set(range(CANONICAL_ANCHOR_COUNT)) - set(invalid_anchor_ids)
    )
    records: list[dict[str, Any]] = []
    for length in history_lengths:
        seen_subsets: set[tuple[int, ...]] = set()
        for draw_index in range(histories_per_length):
            sample_seed = _sample_seed(
                seed, split, object_id, length, draw_index
            )
            generator = np.random.Generator(
                np.random.PCG64(int(sample_seed, 16))
            )
            for _attempt in range(10_000):
                history = tuple(
                    int(value)
                    for value in generator.choice(
                        allowed, size=length, replace=False
                    )
                )
                subset = tuple(sorted(history))
                if subset not in seen_subsets:
                    seen_subsets.add(subset)
                    break
            else:
                raise HistoryDatasetError(
                    f"could not sample another unique history for {object_id}, "
                    f"length {length}"
                )
            image_paths = tuple(
                _relative_image_path(object_id, anchor_id)
                for anchor_id in history
            )
            for relative in image_paths:
                if not (data_root / relative).is_file():
                    raise HistoryDatasetError(
                        f"Missing history RGB: {data_root / relative}"
                    )
            gains = cache.candidate_gains(
                history, target=coverage_target
            ).astype(np.float32)
            gains = np.maximum(gains, np.float32(0))
            valid = np.ones(CANONICAL_ANCHOR_COUNT, dtype=np.bool_)
            valid[list(invalid_anchor_ids)] = False
            valid[list(history)] = False
            sample_id = (
                f"{split}/{object_id}/h{length:02d}/d{draw_index:04d}/"
                f"{hashlib.sha256(bytes(history)).hexdigest()[:12]}"
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "history_image_paths": image_paths,
                    "history_anchor_ids": history,
                    "history_length": length,
                    "target_surface_gain": gains,
                    "valid_candidate_mask": valid,
                    "visibility_cache_id": cache_id,
                    "draw_index": draw_index,
                    "sample_seed": sample_seed,
                }
            )
    return records


def _save_shard(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    metadata: Mapping[str, Any],
    max_length: int,
) -> None:
    n = len(records)
    anchors = np.full((n, max_length), -1, dtype=np.int16)
    paths = np.full(
        (n, max_length), "", dtype=f"<U{_maximum_path_length(records)}"
    )
    for index, record in enumerate(records):
        length = int(record["history_length"])
        anchors[index, :length] = record["history_anchor_ids"]
        paths[index, :length] = record["history_image_paths"]
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "sample_ids": _unicode_array(
            record["sample_id"] for record in records
        ),
        "object_ids": _unicode_array(
            record["object_id"] for record in records
        ),
        "history_anchor_ids": anchors,
        "history_image_paths": paths,
        "history_lengths": np.asarray(
            [record["history_length"] for record in records], dtype=np.int16
        ),
        "target_surface_gain": np.stack(
            [record["target_surface_gain"] for record in records]
        ).astype(np.float32),
        "valid_candidate_mask": np.stack(
            [record["valid_candidate_mask"] for record in records]
        ).astype(np.bool_),
        "visibility_cache_ids": _unicode_array(
            record["visibility_cache_id"] for record in records
        ),
        "draw_indices": np.asarray(
            [record["draw_index"] for record in records], dtype=np.int32
        ),
        "sample_seeds": _unicode_array(
            record["sample_seed"] for record in records
        ),
    }
    _atomic_deterministic_npz(path, arrays)


def _atomic_deterministic_npz(
    path: Path, arrays: Mapping[str, NDArray[Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for key in sorted(arrays):
                    buffer = io.BytesIO()
                    array = np.asarray(arrays[key])
                    if array.ndim:
                        array = np.ascontiguousarray(array)
                    np.lib.format.write_array(
                        buffer,
                        array,
                        allow_pickle=False,
                    )
                    info = zipfile.ZipInfo(
                        f"{key}.npy", date_time=_FIXED_ZIP_TIME
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(
                        info, buffer.getvalue(), compresslevel=6
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_shard(
    path: Path,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise HistoryDatasetError(
            f"Could not load history shard {path}: {exc}"
        ) from exc
    raw_metadata = arrays.pop("metadata_json", None)
    if raw_metadata is None or raw_metadata.shape != ():
        raise HistoryDatasetError(
            "History shard metadata_json must be a scalar"
        )
    try:
        metadata = json.loads(str(raw_metadata.item()))
    except json.JSONDecodeError as exc:
        raise HistoryDatasetError(
            "History shard metadata_json is invalid"
        ) from exc
    if metadata.get("schema_version") != HISTORY_DATASET_SCHEMA_VERSION:
        raise HistoryDatasetError("Unsupported history shard schema_version")
    return arrays, metadata


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HistoryDatasetError(
            f"History manifest does not exist: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryDatasetError(
            f"Could not read history manifest {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise HistoryDatasetError("History manifest root must be an object")
    if raw.get("schema_version") != HISTORY_DATASET_SCHEMA_VERSION:
        raise HistoryDatasetError(
            "Unsupported history manifest schema_version"
        )
    required = {
        "dataset_id",
        "coverage_target",
        "anchor_count",
        "anchor_ordering",
        "object_disjoint",
        "sampling",
        "rotation_metadata",
        "image_paths",
        "splits",
    }
    missing = required - set(raw)
    if missing:
        raise HistoryDatasetError(
            f"History manifest is missing fields: {sorted(missing)}"
        )
    if raw["anchor_count"] != CANONICAL_ANCHOR_COUNT:
        raise HistoryDatasetError("History dataset anchor count must be 48")
    if raw["anchor_ordering"] != CANONICAL_ORDERING:
        raise HistoryDatasetError(
            "History dataset anchor ordering is not canonical"
        )
    if raw["coverage_target"] not in VISIBILITY_TARGETS:
        raise HistoryDatasetError(
            "History dataset has invalid coverage_target"
        )
    if raw["object_disjoint"] is not True:
        raise HistoryDatasetError("History dataset must be object-disjoint")
    if not isinstance(raw["image_paths"], Mapping):
        raise HistoryDatasetError("History image_paths must be a mapping")
    data_root_relative = raw["image_paths"].get(
        "data_root_relative_to_dataset"
    )
    if (
        raw["image_paths"].get("format") != "relative_to_data_root"
        or not isinstance(data_root_relative, str)
        or not data_root_relative
        or Path(data_root_relative).is_absolute()
    ):
        raise HistoryDatasetError(
            "History image path convention is invalid"
        )
    if not isinstance(raw.get("sampling"), Mapping):
        raise HistoryDatasetError("History sampling must be a mapping")
    if not isinstance(raw.get("invalid_anchor_ids"), list):
        raise HistoryDatasetError("History invalid_anchor_ids must be a list")
    _validate_sampling(
        raw["sampling"].get("history_lengths", []),
        raw["sampling"].get("histories_per_object_per_length"),
        raw["sampling"].get("seed"),
        raw["invalid_anchor_ids"],
    )
    if raw["sampling"].get("strategy") != HISTORY_SAMPLING_STRATEGY:
        raise HistoryDatasetError("History sampling strategy is unsupported")
    if not isinstance(raw.get("visibility_cache_ids"), Mapping):
        raise HistoryDatasetError(
            "History visibility_cache_ids must be a mapping"
        )
    owners: dict[str, str] = {}
    if not isinstance(raw["splits"], Mapping):
        raise HistoryDatasetError("History manifest splits must be a mapping")
    for split, info in raw["splits"].items():
        if split not in _SPLITS or not isinstance(info, Mapping):
            raise HistoryDatasetError(
                "History manifest contains an invalid split"
            )
        object_ids = info.get("object_ids")
        if not isinstance(object_ids, list):
            raise HistoryDatasetError(
                f"History split {split} object_ids must be a list"
            )
        for object_id in object_ids:
            _validate_object_id(object_id)
            if object_id in owners:
                raise HistoryDatasetError(
                    f"Object {object_id} appears in both "
                    f"{owners[object_id]} and {split}"
                )
            owners[object_id] = split
    core = dict(raw)
    dataset_id = core.pop("dataset_id")
    if dataset_id != _json_sha256(core):
        raise HistoryDatasetError(
            "History manifest dataset_id does not match contents"
        )
    return raw


def _validate_sampling(
    history_lengths: Sequence[int],
    histories_per_length: int,
    seed: int,
    invalid_anchor_ids: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(history_lengths, (str, bytes)) or not history_lengths:
        raise HistoryDatasetError(
            "history_lengths must be a non-empty sequence"
        )
    lengths = tuple(history_lengths)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in lengths
    ):
        raise HistoryDatasetError("history_lengths must contain integers")
    if len(set(lengths)) != len(lengths) or tuple(sorted(lengths)) != lengths:
        raise HistoryDatasetError(
            "history_lengths must be unique and increasing"
        )
    invalid = tuple(invalid_anchor_ids)
    for anchor_id in invalid:
        if isinstance(anchor_id, bool) or not isinstance(
            anchor_id, (int, np.integer)
        ):
            raise HistoryDatasetError(
                "invalid_anchor_ids must contain integers"
            )
        try:
            canonical_anchors().by_id(int(anchor_id))
        except (IndexError, TypeError) as exc:
            raise HistoryDatasetError(
                f"invalid candidate anchor ID: {anchor_id!r}"
            ) from exc
    if len(set(invalid)) != len(invalid):
        raise HistoryDatasetError("invalid_anchor_ids must be unique")
    available = CANONICAL_ANCHOR_COUNT - len(invalid)
    if any(length <= 0 or length >= available for length in lengths):
        raise HistoryDatasetError(
            "history lengths must leave at least one valid candidate"
        )
    if (
        isinstance(histories_per_length, bool)
        or not isinstance(histories_per_length, int)
        or histories_per_length <= 0
    ):
        raise HistoryDatasetError(
            "histories_per_object_per_length must be a positive integer"
        )
    for length in lengths:
        if histories_per_length > math.comb(available, length):
            raise HistoryDatasetError(
                f"Requested more unique length-{length} histories than exist"
            )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HistoryDatasetError("seed must be a non-negative integer")
    return lengths


def _validate_splits(splits: Sequence[str]) -> tuple[str, ...]:
    if isinstance(splits, (str, bytes)) or not splits:
        raise HistoryDatasetError("splits must be a non-empty sequence")
    selected = tuple(splits)
    if len(set(selected)) != len(selected) or any(
        split not in _SPLITS for split in selected
    ):
        raise HistoryDatasetError(
            "splits must be unique values drawn from train, val, test"
        )
    return tuple(split for split in _SPLITS if split in selected)


def _validate_history(values: Sequence[int]) -> tuple[int, ...]:
    anchors = tuple(values)
    if not anchors:
        raise HistoryDatasetError("history must contain at least one anchor")
    if len(set(anchors)) != len(anchors):
        raise HistoryDatasetError("history anchors must be unique")
    for anchor_id in anchors:
        if isinstance(anchor_id, bool) or not isinstance(
            anchor_id, (int, np.integer)
        ):
            raise HistoryDatasetError("history anchor IDs must be integers")
        try:
            canonical_anchors().by_id(int(anchor_id))
        except (IndexError, TypeError) as exc:
            raise HistoryDatasetError(
                f"history anchor ID is invalid: {anchor_id!r}"
            ) from exc
    return tuple(int(value) for value in anchors)


def _validate_object_id(object_id: str) -> None:
    try:
        visibility_cache_path("unused", object_id)
    except VisibilityCacheError as exc:
        raise HistoryDatasetError(str(exc)) from exc


def _preflight_inputs(
    data_root: Path, cache_root: Path, object_ids: Sequence[str]
) -> None:
    if not data_root.is_dir():
        raise HistoryDatasetError(
            f"NUM data root is not a directory: {data_root}"
        )
    missing_caches = [
        object_id
        for object_id in object_ids
        if not visibility_cache_path(cache_root, object_id).is_file()
    ]
    missing_images: list[str] = []
    for object_id in object_ids:
        for anchor_id in range(CANONICAL_ANCHOR_COUNT):
            relative = _relative_image_path(object_id, anchor_id)
            if not (data_root / relative).is_file():
                missing_images.append(f"{object_id}:{anchor_id}")
                break
    if missing_caches or missing_images:
        details = []
        if missing_caches:
            details.append(
                f"missing visibility caches ({len(missing_caches)}): "
                + ", ".join(missing_caches[:10])
            )
        if missing_images:
            details.append(
                f"objects with incomplete RGB ({len(missing_images)}): "
                + ", ".join(missing_images[:10])
            )
        raise HistoryDatasetError("; ".join(details))


def _sample_seed(
    seed: int,
    split: str,
    object_id: str,
    length: int,
    draw_index: int,
) -> str:
    payload = json.dumps(
        [seed, split, object_id, length, draw_index],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _relative_image_path(object_id: str, anchor_id: int) -> str:
    return (
        f"{object_id}/images/"
        + _IMAGE_NAME.format(anchor_id=anchor_id)
    )


def _safe_child(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HistoryDatasetError(
            f"{field} must be a non-empty relative path"
        )
    candidate = Path(relative)
    if candidate.is_absolute():
        raise HistoryDatasetError(f"{field} must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise HistoryDatasetError(
            f"{field} escapes its configured root"
        )
    return resolved


def _maximum_path_length(records: Sequence[Mapping[str, Any]]) -> int:
    return max(
        1,
        max(
            len(path)
            for record in records
            for path in record["history_image_paths"]
        ),
    )


def _unicode_array(values: Any) -> NDArray[np.str_]:
    materialized = tuple(str(value) for value in values)
    width = max(1, *(len(value) for value in materialized))
    return np.asarray(materialized, dtype=f"<U{width}")


def _json_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


SurfaceGainHistoryDataset = HistoryDataset


__all__ = [
    "HISTORY_DATASET_SCHEMA_VERSION",
    "HISTORY_SAMPLING_STRATEGY",
    "HistoryBatch",
    "HistoryDataset",
    "HistoryDatasetError",
    "HistorySample",
    "SurfaceGainHistoryDataset",
    "assert_surface_gain_matches_coverage",
    "build_history_dataset",
    "collate_history_samples",
]
