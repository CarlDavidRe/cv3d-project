"""Independent single-image VGGT NUM policy for Phase 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from nbv.features import select_feature_components
from nbv.policies.aggregation import (
    PUN_AGGREGATION,
    PUN_ALIGNMENT,
    aggregate_pun_psnr,
    align_pun_history,
)
from nbv.policies.base import ObservationState


class _Extractor(Protocol):
    def extract(self, images: Tensor) -> Any: ...


class VGGTPolicy:
    """Predict one source-relative PSNR map per acquired image, independently.

    A Phase 1 fixed-vector feature cache may satisfy any observation. Cache
    misses are extracted as one-image VGGT sequences; images in the history
    never interact inside the backbone.
    """

    name = "vggt"
    is_oracle = False
    score_semantics = "negative_product_of_aligned_raw_psnr_after_small_filter"
    training_target_semantics = "phase1_single_image_num_psnr"

    def __init__(
        self,
        head: nn.Module,
        feature_components: tuple[str, ...] | list[str],
        *,
        cached_features: Mapping[str, Tensor | np.ndarray] | None = None,
        extractor: _Extractor | None = None,
        extractor_factory: Callable[[], _Extractor] | None = None,
        device: str | torch.device = "cpu",
        inference_batch_size: int = 16,
        interpolation_degrees: float = 30.0,
        suppression_threshold: float = 0.1,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not feature_components or any(
            not isinstance(item, str) for item in feature_components
        ):
            raise ValueError("feature_components must contain feature names")
        if extractor is not None and extractor_factory is not None:
            raise ValueError("Supply extractor or extractor_factory, not both")
        if (
            isinstance(inference_batch_size, bool)
            or not isinstance(inference_batch_size, int)
            or inference_batch_size < 1
        ):
            raise ValueError("inference_batch_size must be a positive integer")
        self.device = _resolve_device(device)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.feature_components = tuple(feature_components)
        self.cached_features = dict(cached_features or {})
        self._extractor = extractor
        self._extractor_factory = extractor_factory
        self.inference_batch_size = inference_batch_size
        self.interpolation_degrees = float(interpolation_degrees)
        self.suppression_threshold = float(suppression_threshold)
        self._base_provenance = dict(provenance or {})
        self._predictions: dict[tuple[str, int], tuple[str, np.ndarray]] = {}
        self._prediction_sources: dict[tuple[str, int], str] = {}
        self._filter_fallback_steps: list[int] = []
        self._filtered_candidate_counts: list[int] = []

    @property
    def provenance(self) -> dict[str, Any]:
        counts = {"feature_cache": 0, "live_vggt": 0}
        for source in self._prediction_sources.values():
            counts[source] += 1
        return {
            **self._base_provenance,
            "alignment": PUN_ALIGNMENT,
            "aggregation": PUN_AGGREGATION,
            "interpolation_degrees": self.interpolation_degrees,
            "suppression_threshold": self.suppression_threshold,
            "target_name": "PSNR",
            "target_direction": "lower_is_more_uncertain",
            "common_score_orientation": "negative_raw_product_higher_is_better",
            "history_mode": "independent_single_image",
            "feature_components": list(self.feature_components),
            "prediction_source_counts": counts,
            "filter_fallback_steps": self._filter_fallback_steps.copy(),
            "filtered_candidate_counts": self._filtered_candidate_counts.copy(),
        }

    @property
    def profiling_mode(self) -> str:
        sources = set(self._prediction_sources.values())
        if not sources or sources == {"feature_cache"}:
            return "cached_features_plus_live_head"
        if sources == {"live_vggt"}:
            return "live_vggt_plus_head"
        return "mixed_cached_and_live_vggt_plus_head"

    def score(self, observation_state: ObservationState) -> np.ndarray:
        if not observation_state.acquired_observations:
            raise ValueError("VGGTPolicy requires at least one acquired observation")
        self._predict_missing(observation_state)
        source_ids = observation_state.acquired_anchor_ids
        relative_maps = np.stack([
            self._predictions[(observation_state.object_id, anchor_id)][1]
            for anchor_id in source_ids
        ])
        aligned = align_pun_history(
            relative_maps,
            source_ids,
            observation_state.anchor_directions,
            interpolation_degrees=self.interpolation_degrees,
        )
        aggregate = aggregate_pun_psnr(
            aligned,
            observation_state.valid_candidate_mask,
            suppression_threshold=self.suppression_threshold,
        )
        if aggregate.filter_fallback_used:
            self._filter_fallback_steps.append(observation_state.step_index)
        self._filtered_candidate_counts.append(int(
            observation_state.valid_candidate_mask.sum()
            - aggregate.kept_candidate_mask.sum()
        ))
        return aggregate.policy_scores

    def save_prediction_cache(self, path: str | Path) -> Path:
        """Atomically save acquired-only raw NUM maps and strict provenance."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(self._predictions.items())
        keys = [key for key, _ in entries]
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata_json=np.asarray(json.dumps(
                        self.provenance, sort_keys=True, allow_nan=False
                    )),
                    object_ids=np.asarray([key[0] for key in keys], dtype=str),
                    anchor_ids=np.asarray([key[1] for key in keys], dtype=np.int64),
                    image_paths=np.asarray([value[0] for _, value in entries], dtype=str),
                    prediction_sources=np.asarray([
                        self._prediction_sources[key] for key in keys
                    ], dtype=str),
                    raw_prediction_maps=(
                        np.stack([value[1] for _, value in entries])
                        if entries else np.empty((0, 48), dtype=np.float64)
                    ),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def _predict_missing(self, state: ObservationState) -> None:
        missing = [
            item for item in state.acquired_observations
            if (state.object_id, item.anchor_id) not in self._predictions
        ]
        for start in range(0, len(missing), self.inference_batch_size):
            observations = missing[start:start + self.inference_batch_size]
            cached_indices: list[int] = []
            cached_vectors: list[Tensor] = []
            live_indices: list[int] = []
            for index, observation in enumerate(observations):
                sample_id = f"{state.object_id}/{observation.anchor_id}"
                vector = self.cached_features.get(sample_id)
                if vector is None:
                    live_indices.append(index)
                else:
                    cached_indices.append(index)
                    cached_vectors.append(torch.as_tensor(vector))

            predictions: dict[int, tuple[np.ndarray, str]] = {}
            if cached_vectors:
                features = torch.stack(cached_vectors).float().to(self.device)
                predictions.update(self._run_head(features, cached_indices, "feature_cache"))
            if live_indices:
                extractor = self._get_extractor()
                images = np.stack([
                    np.transpose(observations[index].rgb, (2, 0, 1))
                    for index in live_indices
                ]).astype(np.float32) / np.float32(255.0)
                frozen = extractor.extract(images)
                features = select_feature_components(
                    frozen, self.feature_components
                ).float().to(self.device)
                predictions.update(self._run_head(features, live_indices, "live_vggt"))

            for index, observation in enumerate(observations):
                prediction, source = predictions[index]
                key = (state.object_id, observation.anchor_id)
                self._predictions[key] = (observation.image_path, prediction)
                self._prediction_sources[key] = source

    def _run_head(
        self, features: Tensor, indices: list[int], source: str
    ) -> dict[int, tuple[np.ndarray, str]]:
        with torch.inference_mode():
            values = self.head(features)
        if values.shape != (len(indices), 48):
            raise ValueError("VGGT Phase 1 head must return [batch, 48]")
        array = values.detach().cpu().double().numpy()
        if not np.isfinite(array).all():
            raise ValueError("VGGT Phase 1 head returned non-finite predictions")
        return {
            index: (prediction.copy(), source)
            for index, prediction in zip(indices, array, strict=True)
        }

    def _get_extractor(self) -> _Extractor:
        if self._extractor is None:
            if self._extractor_factory is None:
                raise ValueError(
                    "An acquired image is absent from the pinned feature cache and "
                    "live VGGT extraction is disabled"
                )
            self._extractor = self._extractor_factory()
        return self._extractor


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved
