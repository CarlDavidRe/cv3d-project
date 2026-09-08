"""Official-checkpoint PUN adapter for the common closed-loop evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn

from nbv.policies.aggregation import (
    PUN_AGGREGATION,
    PUN_ALIGNMENT,
    aggregate_pun_psnr,
    align_pun_history,
)
from nbv.policies.base import ObservationState


class PUNPolicy:
    """Infer source-relative PSNR maps independently, then aggregate history."""

    name = "pun"
    is_oracle = False
    score_semantics = "negative_product_of_aligned_raw_psnr_after_small_filter"
    training_target_semantics = "official_single_image_num_psnr"
    profiling_mode = "live_model_inference_incremental"

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], Tensor],
        *,
        device: str | torch.device = "cpu",
        inference_batch_size: int = 16,
        interpolation_degrees: float = 30.0,
        suppression_threshold: float = 0.1,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(transform):
            raise TypeError("transform must be callable")
        if (
            isinstance(inference_batch_size, bool)
            or not isinstance(inference_batch_size, int)
            or inference_batch_size < 1
        ):
            raise ValueError("inference_batch_size must be a positive integer")
        resolved = _resolve_device(device)
        self.model = model.to(resolved).eval().requires_grad_(False)
        self.transform = transform
        self.device = resolved
        self.inference_batch_size = inference_batch_size
        self.interpolation_degrees = float(interpolation_degrees)
        self.suppression_threshold = float(suppression_threshold)
        self._base_provenance = dict(provenance or {})
        self._predictions: dict[tuple[str, int], tuple[str, np.ndarray]] = {}
        self._filter_fallback_steps: list[int] = []
        self._filtered_candidate_counts: list[int] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **self._base_provenance,
            "alignment": PUN_ALIGNMENT,
            "aggregation": PUN_AGGREGATION,
            "interpolation_degrees": self.interpolation_degrees,
            "suppression_threshold": self.suppression_threshold,
            "target_name": "PSNR",
            "target_direction": "lower_is_more_uncertain",
            "common_score_orientation": "negative_raw_product_higher_is_better",
            "filter_fallback_steps": self._filter_fallback_steps.copy(),
            "filtered_candidate_counts": self._filtered_candidate_counts.copy(),
        }

    def score(self, observation_state: ObservationState) -> np.ndarray:
        if not observation_state.acquired_observations:
            raise ValueError("PUNPolicy requires at least one acquired observation")
        self._predict_missing(observation_state)
        source_ids = observation_state.acquired_anchor_ids
        relative_maps = np.stack(
            [self._predictions[(observation_state.object_id, anchor_id)][1]
             for anchor_id in source_ids]
        )
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
        self._filtered_candidate_counts.append(
            int(observation_state.valid_candidate_mask.sum()
                - aggregate.kept_candidate_mask.sum())
        )
        return aggregate.policy_scores

    def save_prediction_cache(self, path: str | Path) -> Path:
        """Save acquired-only raw UPNet maps separately from policy scores."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(self._predictions.items())
        anchor_ids = np.asarray([key[1] for key, _ in entries], dtype=np.int64)
        object_ids = np.asarray([key[0] for key, _ in entries], dtype=str)
        image_paths = np.asarray([value[0] for _, value in entries], dtype=str)
        maps = (
            np.stack([value[1] for _, value in entries])
            if entries else np.empty((0, 48), dtype=np.float64)
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata_json=np.asarray(
                        json.dumps(self.provenance, sort_keys=True, allow_nan=False)
                    ),
                    object_ids=object_ids,
                    anchor_ids=anchor_ids,
                    image_paths=image_paths,
                    raw_prediction_maps=maps,
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
            images = []
            for observation in observations:
                image = Image.fromarray(np.asarray(observation.rgb))
                transformed = torch.as_tensor(self.transform(image), dtype=torch.float32)
                if transformed.ndim != 3 or transformed.shape[0] != 3:
                    raise ValueError("PUN transform must return [3, H, W]")
                images.append(transformed)
            with torch.inference_mode():
                predictions = self.model(
                    torch.stack(images).to(self.device, non_blocking=True)
                )
            if predictions.shape != (len(observations), 48):
                raise ValueError("PUN model must return [batch, 48]")
            values = predictions.detach().cpu().double().numpy()
            if not np.isfinite(values).all():
                raise ValueError("PUN model returned non-finite predictions")
            for observation, prediction in zip(observations, values, strict=True):
                self._predictions[(state.object_id, observation.anchor_id)] = (
                    observation.image_path,
                    prediction.copy(),
                )


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved
