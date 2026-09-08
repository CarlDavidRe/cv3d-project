"""Greedy camera-space diversity baseline using known anchor directions."""

from __future__ import annotations

import numpy as np

from nbv.policies.base import ObservationState


class FarthestPolicy:
    """Score each anchor by its nearest acquired-view angular distance."""

    name = "farthest"
    is_oracle = False
    score_semantics = "max_min_angular_distance_radians"

    def score(self, observation_state: ObservationState) -> np.ndarray:
        directions = np.asarray(observation_state.anchor_directions, dtype=np.float64)
        if directions.shape != (48, 3):
            raise ValueError("anchor_directions must have shape [48, 3]")
        acquired = observation_state.acquired_anchor_ids
        if not acquired:
            raise ValueError("FarthestPolicy requires at least one acquired anchor")

        # For unit vectors, the nearest angular distance is arccos of the
        # largest cosine similarity to an acquired camera direction.
        cosine = directions @ directions[np.asarray(acquired, dtype=np.int64)].T
        nearest_cosine = np.max(np.clip(cosine, -1.0, 1.0), axis=1)
        scores = np.arccos(nearest_cosine)
        scores[np.asarray(acquired, dtype=np.int64)] = 0.0
        return scores
