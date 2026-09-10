"""Greedy camera-space diversity baseline using known anchor directions."""

from __future__ import annotations

import numpy as np

from nbv.policies.base import ObservationState


class FarthestPolicy:
    """Score each anchor by its nearest acquired-view angular distance."""

    name = "farthest"
    is_oracle = False
    score_semantics = "max_min_angular_distance_radians"
    provenance = {"trainable_parameter_count": 0, "frozen_parameter_count": 0}

    def score(self, observation_state: ObservationState) -> np.ndarray:
        directions = np.asarray(observation_state.anchor_directions, dtype=np.float64)
        if directions.shape != (48, 3):
            raise ValueError("anchor_directions must have shape [48, 3]")
        acquired = observation_state.acquired_anchor_ids
        if not acquired:
            raise ValueError("FarthestPolicy requires at least one acquired anchor")

        # Use the same fixed-order, extended-precision reduction as
        # AnchorSet.angular_distance_matrix.  BLAS-backed matrix multiplication
        # can round near-tied anchor scores differently across platforms.
        precise = directions.astype(np.longdouble)
        cosine = (
            precise[:, None, 0] * precise[None, :, 0]
            + precise[:, None, 1] * precise[None, :, 1]
            + precise[:, None, 2] * precise[None, :, 2]
        )
        acquired_ids = np.asarray(acquired, dtype=np.int64)
        nearest_cosine = np.max(
            np.clip(cosine[:, acquired_ids], -1.0, 1.0), axis=1
        )
        scores = np.asarray(np.arccos(nearest_cosine), dtype=np.float64)
        scores[acquired_ids] = 0.0
        return scores
