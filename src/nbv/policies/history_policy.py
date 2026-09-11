"""Closed-loop adapter for Phase 3 direct surface-gain history models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from nbv.models.independent_multiview import independent_feature_batch
from nbv.policies.base import ObservationState
from nbv.geometry.anchors import CANONICAL_ORDERING


class _Extractor(Protocol):
    def extract(self, images: Tensor) -> Any: ...


class IndependentHistoryPolicy:
    """Predict direct gain from acquired-only independent VGGT vectors."""

    name = "vggt_independent_history"
    is_oracle = False
    score_semantics = "predicted_direct_surface_gain_higher_is_better"
    training_target_semantics = "phase3_history_dependent_surface_gain"

    def __init__(
        self,
        model: nn.Module,
        feature_components: Sequence[str],
        *,
        cached_features: Mapping[str, Tensor | np.ndarray] | None = None,
        feature_lookup: object | None = None,
        extractor: _Extractor | None = None,
        extractor_factory: Callable[[], _Extractor] | None = None,
        device: str | torch.device = "cpu",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not feature_components or any(
            not isinstance(value, str) for value in feature_components
        ):
            raise ValueError("feature_components must contain feature names")
        if extractor is not None and extractor_factory is not None:
            raise ValueError("Supply extractor or extractor_factory, not both")
        if cached_features is not None and feature_lookup is not None:
            raise ValueError("Supply cached_features or feature_lookup, not both")
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.feature_components = tuple(feature_components)
        self.cached_features = dict(cached_features or {})
        self.feature_lookup = feature_lookup
        self._extractor = extractor
        self._extractor_factory = extractor_factory
        self._features: dict[tuple[str, int], tuple[Tensor, str]] = {}
        self._base_provenance = dict(provenance or {})

    @property
    def provenance(self) -> dict[str, Any]:
        counts = {"feature_cache": 0, "live_vggt": 0}
        for _, source in self._features.values():
            counts[source] += 1
        return {
            **self._base_provenance,
            "history_mode": "independent_single_image_features_then_masked_mean",
            "backbone_cross_view_interaction": False,
            "feature_components": list(self.feature_components),
            "prediction_source_counts": counts,
            "target_direction": "higher_is_more_surface_gain",
        }

    @property
    def profiling_mode(self) -> str:
        sources = {source for _, source in self._features.values()}
        if not sources or sources == {"feature_cache"}:
            return "cached_independent_features_plus_live_history_head"
        if sources == {"live_vggt"}:
            return "live_independent_vggt_plus_history_head"
        return "mixed_independent_features_plus_history_head"

    def score(self, observation_state: ObservationState) -> np.ndarray:
        if not observation_state.acquired_observations:
            raise ValueError("IndependentHistoryPolicy requires an acquired history")
        if observation_state.anchor_ordering != CANONICAL_ORDERING:
            raise ValueError("IndependentHistoryPolicy requires canonical anchor ordering")
        self._load_missing(observation_state)
        vectors = torch.stack([
            self._features[(observation_state.object_id, anchor_id)][0]
            for anchor_id in observation_state.acquired_anchor_ids
        ]).unsqueeze(0).float().to(self.device)
        anchor_ids = torch.as_tensor(
            [observation_state.acquired_anchor_ids], dtype=torch.int64, device=self.device
        )
        padding = torch.zeros(anchor_ids.shape, dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            predictions = self.model(vectors, anchor_ids, padding)
        if predictions.shape != (1, 48):
            raise ValueError("Phase 3 history model must return [1, 48]")
        result = predictions[0].detach().cpu().double().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Phase 3 history model returned non-finite scores")
        return result

    def _load_missing(self, state: ObservationState) -> None:
        missing = [
            observation for observation in state.acquired_observations
            if (state.object_id, observation.anchor_id) not in self._features
        ]
        live = []
        for observation in missing:
            vector = self._cached(state.object_id, observation.anchor_id)
            if vector is None:
                live.append(observation)
            else:
                self._store(state.object_id, observation.anchor_id, vector, "feature_cache")
        if live:
            extractor = self._get_extractor()
            images = torch.from_numpy(np.stack([
                np.transpose(observation.rgb, (2, 0, 1))
                for observation in live
            ]).copy()).float().div_(255.0).unsqueeze(0)
            padding = torch.zeros((1, len(live)), dtype=torch.bool)
            vectors = independent_feature_batch(
                images, padding, extractor, self.feature_components
            )[0]
            for observation, vector in zip(live, vectors, strict=True):
                self._store(state.object_id, observation.anchor_id, vector, "live_vggt")

    def _cached(self, object_id: str, anchor_id: int) -> Tensor | np.ndarray | None:
        key = f"{object_id}/{anchor_id}"
        if self.feature_lookup is not None:
            contains = getattr(self.feature_lookup, "contains", None)
            feature = getattr(self.feature_lookup, "feature", None)
            if contains is None or feature is None:
                raise TypeError("feature_lookup must expose contains() and feature()")
            return feature(object_id, anchor_id) if contains(object_id, anchor_id) else None
        return self.cached_features.get(key)

    def _store(
        self,
        object_id: str,
        anchor_id: int,
        vector: Tensor | np.ndarray,
        source: str,
    ) -> None:
        tensor = torch.as_tensor(vector).detach().cpu().float()
        expected_dim = getattr(self.model, "feature_dim", None)
        if tensor.ndim != 1 or (expected_dim is not None and tensor.shape[0] != expected_dim):
            raise ValueError("cached/live feature dimension does not match the history model")
        if not torch.isfinite(tensor).all():
            raise ValueError("cached/live history feature must be finite")
        self._features[(object_id, anchor_id)] = (tensor, source)

    def _get_extractor(self) -> _Extractor:
        if self._extractor is None:
            if self._extractor_factory is None:
                raise ValueError(
                    "An acquired image is absent from the frozen feature cache and "
                    "live independent VGGT extraction is disabled"
                )
            self._extractor = self._extractor_factory()
        return self._extractor


class JointHistoryPolicy:
    """Predict direct gain from one live joint VGGT forward per history state."""

    name = "vggt_joint_history"
    is_oracle = False
    score_semantics = "predicted_direct_surface_gain_higher_is_better"
    training_target_semantics = "phase3_history_dependent_surface_gain"
    profiling_mode = "live_joint_vggt_plus_history_head"

    def __init__(
        self,
        model: nn.Module,
        *,
        policy_name: str | None = None,
        device: str | torch.device = "cpu",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if policy_name is not None and (not isinstance(policy_name, str) or not policy_name):
            raise ValueError("policy_name must be a non-empty string or null")
        self.name = policy_name or type(self).name
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self._base_provenance = dict(provenance or {})

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **self._base_provenance,
            "history_mode": "joint_multiview_then_masked_mean",
            "backbone_cross_view_interaction": True,
            "target_direction": "higher_is_more_surface_gain",
        }

    def score(self, observation_state: ObservationState) -> np.ndarray:
        observations = observation_state.acquired_observations
        if not observations:
            raise ValueError("JointHistoryPolicy requires an acquired history")
        if observation_state.anchor_ordering != CANONICAL_ORDERING:
            raise ValueError("JointHistoryPolicy requires canonical anchor ordering")
        images = torch.from_numpy(
            np.stack(
                [np.transpose(observation.rgb, (2, 0, 1)) for observation in observations]
            ).copy()
        ).float().div_(255.0).unsqueeze(0).to(self.device)
        anchor_ids = torch.as_tensor(
            [observation_state.acquired_anchor_ids],
            dtype=torch.int64,
            device=self.device,
        )
        padding = torch.zeros(anchor_ids.shape, dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            predictions = self.model(images, anchor_ids, padding)
        if predictions.shape != (1, 48):
            raise ValueError("Phase 3 joint history model must return [1, 48]")
        result = predictions[0].detach().cpu().double().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Phase 3 joint history model returned non-finite scores")
        return result


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


__all__ = ["IndependentHistoryPolicy", "JointHistoryPolicy"]
