"""Deterministic loader for the original PUN Neural Uncertainty Map data.

The released NUM layout is::

    root/category_id/object_id/images/viewpoint_<id>_offset_phi_0.png
    root/category_id/object_id/uncertainties/viewpoint_<id>_offset_phi_0.json

Only the canonical, non-``example`` images at rotation offset zero are Phase 1
samples.  Target arrays are kept in their official 48-entry ordering and are
not converted into Phase 2 surface-gain utilities.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from nbv.geometry import CANONICAL_ANCHOR_COUNT, canonical_anchors


NUMSplit = Literal["train", "val", "test", "all"]
ObjectKey = tuple[str, str]
ImageTransform = Callable[[Image.Image], Any]

_SPLIT_NAMES = ("train", "val", "test")
_IMAGE_NAME = re.compile(r"^viewpoint_(\d+)_offset_phi_0\.png$")


class NUMDatasetError(ValueError):
    """Raised when NUM files or split metadata violate the expected schema."""


@dataclass(frozen=True, slots=True)
class NUMSampleRecord:
    """Paths and immutable identity fields for one NUM sample."""

    category_id: str
    object_id: str
    image_path: Path
    target_path: Path
    source_anchor_id: int
    split: NUMSplit

    @property
    def sample_id(self) -> str:
        """Return a stable identifier independent of the dataset root path."""

        return f"{self.category_id}/{self.object_id}/{self.source_anchor_id}"


@dataclass(frozen=True, slots=True)
class NUMSample:
    """One preprocessed image, official target map, and its provenance."""

    image: Any
    target_map: NDArray[np.float32]
    record: NUMSampleRecord
    target_name: str

    @property
    def category_id(self) -> str:
        return self.record.category_id

    @property
    def object_id(self) -> str:
        return self.record.object_id

    @property
    def source_anchor_id(self) -> int:
        return self.record.source_anchor_id

    @property
    def split(self) -> NUMSplit:
        return self.record.split


class NUMDataset(Sequence[NUMSample]):
    """Index official single-image NUM samples in stable canonical order.

    ``train``, ``val``, and ``test`` require an explicit object-level split
    manifest.  This makes leakage impossible to introduce through a random
    per-image split.  Use ``split="all"`` only for inspection or for preparing
    a manifest.

    Images are converted to RGB before ``transform`` is called.  The default
    transform returns float32 ``[3, H, W]`` arrays in ``[0, 1]`` without
    resizing or backbone-specific normalization.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: NUMSplit,
        split_manifest: str | Path | None = None,
        target_name: str = "PSNR",
        transform: ImageTransform | None = None,
        require_complete_objects: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NUMDatasetError(f"NUM root is not a directory: {self.root}")
        if split not in (*_SPLIT_NAMES, "all"):
            raise NUMDatasetError(
                f"split must be one of {(*_SPLIT_NAMES, 'all')}, got {split!r}"
            )
        if not isinstance(target_name, str) or not target_name.strip():
            raise NUMDatasetError("target_name must be a non-empty string")
        if not isinstance(require_complete_objects, bool):
            raise TypeError("require_complete_objects must be a boolean")

        selected_objects: frozenset[ObjectKey] | None = None
        if split == "all":
            if split_manifest is not None:
                raise NUMDatasetError(
                    "split_manifest must be omitted when split='all'"
                )
        else:
            if split_manifest is None:
                raise NUMDatasetError(
                    f"split_manifest is required when split={split!r}"
                )
            selected_objects = load_object_split(split_manifest)[split]

        self.split = split
        self.target_name = target_name.strip()
        self.transform = transform if transform is not None else rgb_to_float_chw
        self.require_complete_objects = require_complete_objects
        self._records = _discover_records(
            self.root,
            split,
            selected_objects,
            require_complete_objects=require_complete_objects,
        )
        if not self._records:
            raise NUMDatasetError(
                f"No NUM samples found for split {split!r} under {self.root}"
            )

    @property
    def records(self) -> tuple[NUMSampleRecord, ...]:
        """Return the deterministic, immutable sample index."""

        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> NUMSample | list[NUMSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._records[index]
        image = _load_rgb_image(record.image_path)
        target = _load_target(record.target_path, self.target_name)
        return NUMSample(
            image=self.transform(image),
            target_map=target,
            record=record,
            target_name=self.target_name,
        )

    def __iter__(self) -> Iterator[NUMSample]:
        for index in range(len(self)):
            yield self[index]


def rgb_to_float_chw(image: Image.Image) -> NDArray[np.float32]:
    """Convert a PIL RGB image to contiguous float32 ``[3, H, W]`` data."""

    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(np.transpose(pixels, (2, 0, 1)))


def load_object_split(path: str | Path) -> dict[str, frozenset[ObjectKey]]:
    """Load and validate a versioned object-disjoint NUM split manifest.

    The JSON schema is::

        {"schema_version": 1,
         "splits": {"train": ["category/object"], "val": [...],
                    "test": [...]}}
    """

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NUMDatasetError(f"Split manifest does not exist: {source}") from exc
    except OSError as exc:
        raise NUMDatasetError(f"Could not read split manifest {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise NUMDatasetError(f"Invalid split JSON in {source}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise NUMDatasetError("Split manifest root must be an object")
    if raw.get("schema_version") != 1:
        raise NUMDatasetError("Split manifest schema_version must be 1")
    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise NUMDatasetError("Split manifest 'splits' must be an object")
    if set(raw_splits) != set(_SPLIT_NAMES):
        raise NUMDatasetError(
            f"Split manifest must contain exactly {_SPLIT_NAMES}"
        )

    splits: dict[str, frozenset[ObjectKey]] = {}
    owner: dict[ObjectKey, str] = {}
    for split_name in _SPLIT_NAMES:
        entries = raw_splits[split_name]
        if not isinstance(entries, list):
            raise NUMDatasetError(f"Split {split_name!r} must be a list")
        parsed: set[ObjectKey] = set()
        for entry in entries:
            key = _parse_object_key(entry, split_name)
            if key in parsed:
                raise NUMDatasetError(
                    f"Duplicate object {_format_object_key(key)!r} in {split_name}"
                )
            if key in owner:
                raise NUMDatasetError(
                    f"Object {_format_object_key(key)!r} appears in both "
                    f"{owner[key]} and {split_name}"
                )
            parsed.add(key)
            owner[key] = split_name
        splits[split_name] = frozenset(parsed)
    return splits


def _discover_records(
    root: Path,
    split: NUMSplit,
    selected_objects: frozenset[ObjectKey] | None,
    *,
    require_complete_objects: bool,
) -> tuple[NUMSampleRecord, ...]:
    records: list[NUMSampleRecord] = []
    discovered_objects: set[ObjectKey] = set()
    expected_anchor_ids = set(range(CANONICAL_ANCHOR_COUNT))

    for category_path in _sorted_directories(root):
        for object_path in _sorted_directories(category_path):
            object_key = (category_path.name, object_path.name)
            if selected_objects is not None and object_key not in selected_objects:
                continue
            discovered_objects.add(object_key)
            images_dir = object_path / "images"
            targets_dir = object_path / "uncertainties"
            if not images_dir.is_dir() or not targets_dir.is_dir():
                if not require_complete_objects:
                    continue
                raise NUMDatasetError(
                    f"NUM object {_format_object_key(object_key)!r} must contain "
                    "'images' and 'uncertainties' directories"
                )

            object_records: list[NUMSampleRecord] = []
            found_anchor_ids: set[int] = set()
            for image_path in sorted(images_dir.iterdir(), key=lambda path: path.name):
                if not image_path.is_file():
                    continue
                match = _IMAGE_NAME.fullmatch(image_path.name)
                if match is None:
                    continue
                source_anchor_id = int(match.group(1))
                try:
                    canonical_anchors().by_id(source_anchor_id)
                except (IndexError, TypeError) as exc:
                    raise NUMDatasetError(
                        f"Invalid source anchor in image name: {image_path}"
                    ) from exc
                if source_anchor_id in found_anchor_ids:
                    raise NUMDatasetError(
                        f"Duplicate anchor {source_anchor_id} for object "
                        f"{_format_object_key(object_key)!r}"
                    )
                target_path = targets_dir / f"{image_path.stem}.json"
                if not target_path.is_file():
                    if not require_complete_objects:
                        continue
                    raise NUMDatasetError(
                        f"Missing target for NUM image {image_path}: {target_path}"
                    )
                found_anchor_ids.add(source_anchor_id)
                object_records.append(
                    NUMSampleRecord(
                        category_id=category_path.name,
                        object_id=object_path.name,
                        image_path=image_path,
                        target_path=target_path,
                        source_anchor_id=source_anchor_id,
                        split=split,
                    )
                )

            if require_complete_objects and found_anchor_ids != expected_anchor_ids:
                missing = sorted(expected_anchor_ids - found_anchor_ids)
                extra = sorted(found_anchor_ids - expected_anchor_ids)
                raise NUMDatasetError(
                    f"NUM object {_format_object_key(object_key)!r} does not have "
                    f"exactly anchors 0..{CANONICAL_ANCHOR_COUNT - 1}; "
                    f"missing={missing}, extra={extra}"
                )
            records.extend(
                sorted(object_records, key=lambda record: record.source_anchor_id)
            )

    if selected_objects is not None:
        missing_objects = selected_objects - discovered_objects
        if missing_objects:
            formatted = sorted(_format_object_key(key) for key in missing_objects)
            raise NUMDatasetError(
                f"Objects from split {split!r} are missing under {root}: {formatted}"
            )
    return tuple(records)


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise NUMDatasetError(f"Could not read NUM image {path}: {exc}") from exc


def _load_target(path: Path, target_name: str) -> NDArray[np.float32]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NUMDatasetError(f"Could not read NUM target {path}: {exc}") from exc
    if not isinstance(raw, Mapping) or target_name not in raw:
        raise NUMDatasetError(
            f"NUM target {path} does not contain {target_name!r}"
        )
    values = raw[target_name]
    if not isinstance(values, list) or len(values) != CANONICAL_ANCHOR_COUNT:
        raise NUMDatasetError(
            f"NUM target {path} field {target_name!r} must have shape "
            f"[{CANONICAL_ANCHOR_COUNT}]"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise NUMDatasetError(
            f"NUM target {path} field {target_name!r} must be numeric"
        )
    if any(not math.isfinite(float(value)) for value in values):
        raise NUMDatasetError(
            f"NUM target {path} field {target_name!r} must be finite"
        )
    target = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(target)):
        raise NUMDatasetError(
            f"NUM target {path} field {target_name!r} exceeds float32 range"
        )
    return target


def _sorted_directories(path: Path) -> list[Path]:
    return sorted(
        (child for child in path.iterdir() if child.is_dir()),
        key=lambda child: child.name,
    )


def _parse_object_key(value: object, split_name: str) -> ObjectKey:
    if not isinstance(value, str):
        raise NUMDatasetError(
            f"Every entry in split {split_name!r} must be 'category/object'"
        )
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise NUMDatasetError(
            f"Invalid object key {value!r} in split {split_name!r}; expected "
            "'category/object'"
        )
    return parts[0], parts[1]


def _format_object_key(key: ObjectKey) -> str:
    return f"{key[0]}/{key[1]}"
