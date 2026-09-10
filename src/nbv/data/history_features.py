"""Lazy history views over cached independent single-image features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nbv.data.history_dataset import HistoryDataset, HistoryDatasetError
from nbv.features.cache import CachedFeatureDataset, load_feature_cache


class HistoryFeatureError(ValueError):
    """Raised when cached features cannot represent a history dataset."""


class FrozenFeatureLookup:
    """Index one split's immutable Phase 1 feature vectors by object/anchor."""

    def __init__(
        self,
        cache: CachedFeatureDataset,
        *,
        split: str,
        expected_metadata: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        if cache.metadata.get("split") != split:
            raise HistoryFeatureError("feature-cache split does not match history split")
        for key, expected in dict(expected_metadata or {}).items():
            if cache.metadata.get(key) != expected:
                raise HistoryFeatureError(
                    f"feature cache has incompatible {key}: "
                    f"expected {expected!r}, got {cache.metadata.get(key)!r}"
                )
        self.cache = cache
        self.split = split
        self.source_path = None if source_path is None else str(Path(source_path).resolve())
        self._indices = {sample_id: index for index, sample_id in enumerate(cache.sample_ids)}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        split: str,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> "FrozenFeatureLookup":
        return cls(
            load_feature_cache(path),
            split=split,
            expected_metadata=expected_metadata,
            source_path=path,
        )

    @property
    def feature_dim(self) -> int:
        return int(self.cache.features.shape[1])

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.cache.metadata

    def key(self, object_id: str, anchor_id: int) -> str:
        return f"{object_id}/{anchor_id}"

    def contains(self, object_id: str, anchor_id: int) -> bool:
        return self.key(object_id, anchor_id) in self._indices

    def feature(self, object_id: str, anchor_id: int) -> Tensor:
        key = self.key(object_id, anchor_id)
        try:
            index = self._indices[key]
        except KeyError as exc:
            raise HistoryFeatureError(f"missing frozen feature {key!r}") from exc
        return self.cache.features[index]


@dataclass(frozen=True, slots=True)
class HistoryFeatureSample:
    sample_id: str
    object_id: str
    history_features: Tensor
    history_anchor_ids: Tensor
    target_surface_gain: Tensor
    valid_candidate_mask: Tensor
    visibility_cache_id: str


@dataclass(frozen=True, slots=True)
class HistoryFeatureBatch:
    sample_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    history_features: Tensor
    history_anchor_ids: Tensor
    history_lengths: Tensor
    history_padding_mask: Tensor
    target_surface_gain: Tensor
    valid_candidate_mask: Tensor
    visibility_cache_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "HistoryFeatureBatch":
        return HistoryFeatureBatch(
            sample_ids=self.sample_ids,
            object_ids=self.object_ids,
            history_features=self.history_features.to(device, non_blocking=True),
            history_anchor_ids=self.history_anchor_ids.to(
                device, non_blocking=True
            ),
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
        )


class HistoryFeatureDataset(Sequence[HistoryFeatureSample]):
    """Resolve Step 15 records to cached one-image VGGT features lazily."""

    def __init__(self, histories: HistoryDataset, lookup: FrozenFeatureLookup) -> None:
        if histories.load_images:
            raise HistoryFeatureError("cached-feature training must use path-only histories")
        if histories.split != lookup.split:
            raise HistoryFeatureError("history and feature-cache splits differ")
        self.histories = histories
        self.lookup = lookup
        missing: set[str] = set()
        arrays = histories._arrays
        for row, object_id in enumerate(arrays["object_ids"].tolist()):
            length = int(arrays["history_lengths"][row])
            for anchor_id in arrays["history_anchor_ids"][row, :length].tolist():
                if not lookup.contains(str(object_id), int(anchor_id)):
                    missing.add(lookup.key(str(object_id), int(anchor_id)))
        if missing:
            examples = ", ".join(sorted(missing)[:10])
            raise HistoryFeatureError(
                f"feature cache is missing {len(missing)} acquired views: {examples}"
            )

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int | slice) -> HistoryFeatureSample | list[HistoryFeatureSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        sample = self.histories[index]
        features = torch.stack([
            self.lookup.feature(sample.object_id, anchor_id)
            for anchor_id in sample.history_anchor_ids
        ]).float()
        return HistoryFeatureSample(
            sample_id=sample.sample_id,
            object_id=sample.object_id,
            history_features=features,
            history_anchor_ids=torch.as_tensor(sample.history_anchor_ids, dtype=torch.int64),
            target_surface_gain=torch.from_numpy(sample.target_surface_gain.copy()),
            valid_candidate_mask=torch.from_numpy(sample.valid_candidate_mask.copy()),
            visibility_cache_id=sample.visibility_cache_id,
        )


class MaterializedHistoryFeatureDataset(HistoryFeatureDataset):
    """In-memory features produced once from complete joint histories.

    Unlike :class:`HistoryFeatureDataset`, these vectors are keyed by history
    sample rather than by object/view. That distinction matters for a joint
    backbone because a view's representation depends on every other view in
    the same acquired history.
    """

    def __init__(
        self,
        histories: HistoryDataset,
        samples: Sequence[HistoryFeatureSample],
    ) -> None:
        if not histories.load_images:
            raise HistoryFeatureError(
                "materialized joint features require their source RGB histories"
            )
        if len(samples) != len(histories):
            raise HistoryFeatureError(
                "materialized features must contain one sample per history"
            )
        materialized = tuple(samples)
        expected_ids = tuple(str(value) for value in histories._arrays["sample_ids"])
        actual_ids = tuple(sample.sample_id for sample in materialized)
        if actual_ids != expected_ids:
            raise HistoryFeatureError(
                "materialized feature samples must retain history dataset order"
            )
        feature_dims = {
            int(sample.history_features.shape[1])
            for sample in materialized
            if sample.history_features.ndim == 2
        }
        if len(feature_dims) != 1 or any(
            sample.history_features.ndim != 2 for sample in materialized
        ):
            raise HistoryFeatureError(
                "materialized history features must share one [H, D] shape contract"
            )
        for sample in materialized:
            if sample.history_features.requires_grad:
                raise HistoryFeatureError("materialized features must be detached")
            if sample.history_features.device.type != "cpu":
                raise HistoryFeatureError("materialized features must reside on CPU")
            if not torch.isfinite(sample.history_features).all():
                raise HistoryFeatureError("materialized features must be finite")
            if sample.history_features.shape[0] != sample.history_anchor_ids.shape[0]:
                raise HistoryFeatureError(
                    "materialized feature and anchor history lengths differ"
                )
        self.histories = histories
        self.samples = materialized
        self.feature_dim = feature_dims.pop()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def storage_bytes(self) -> int:
        return sum(
            sample.history_features.numel() * sample.history_features.element_size()
            for sample in self.samples
        )

    def __getitem__(
        self, index: int | slice
    ) -> HistoryFeatureSample | list[HistoryFeatureSample]:
        if isinstance(index, slice):
            return list(self.samples[index])
        return self.samples[index]


def collate_history_features(
    samples: Sequence[HistoryFeatureSample],
) -> HistoryFeatureBatch:
    if not samples:
        raise HistoryDatasetError("cannot collate an empty history-feature batch")
    feature_dim = int(samples[0].history_features.shape[1])
    max_length = max(int(sample.history_features.shape[0]) for sample in samples)
    feature_dtype = samples[0].history_features.dtype
    if not feature_dtype.is_floating_point:
        raise HistoryFeatureError("history features must be floating point")
    if any(sample.history_features.dtype != feature_dtype for sample in samples):
        raise HistoryFeatureError("all history features in a batch must share one dtype")
    features = torch.zeros(
        (len(samples), max_length, feature_dim), dtype=feature_dtype
    )
    anchors = torch.full((len(samples), max_length), -1, dtype=torch.int64)
    padding = torch.ones((len(samples), max_length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        if sample.history_features.ndim != 2 or sample.history_features.shape[1] != feature_dim:
            raise HistoryFeatureError("all history feature tensors must have shape [H, D]")
        length = int(sample.history_features.shape[0])
        if sample.history_anchor_ids.shape != (length,):
            raise HistoryFeatureError("history anchors must match the feature sequence")
        features[row, :length] = sample.history_features
        anchors[row, :length] = sample.history_anchor_ids
        padding[row, :length] = False
    return HistoryFeatureBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        object_ids=tuple(sample.object_id for sample in samples),
        history_features=features,
        history_anchor_ids=anchors,
        history_lengths=(~padding).sum(dim=1),
        history_padding_mask=padding,
        target_surface_gain=torch.stack([sample.target_surface_gain for sample in samples]).float(),
        valid_candidate_mask=torch.stack([sample.valid_candidate_mask for sample in samples]).bool(),
        visibility_cache_ids=tuple(sample.visibility_cache_id for sample in samples),
    )


__all__ = [
    "FrozenFeatureLookup",
    "HistoryFeatureBatch",
    "HistoryFeatureDataset",
    "HistoryFeatureError",
    "HistoryFeatureSample",
    "MaterializedHistoryFeatureDataset",
    "collate_history_features",
]
