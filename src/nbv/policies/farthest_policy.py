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

        # For unit vectors, the nearest angular distance is arccos of the
        # largest cosine similarity to an acquired camera direction.
        # Build the full pairwise matrix before selecting history columns.  A
        # skinny GEMM (directions @ directions[acquired].T) can round symmetric
        # anchor scores differently on different BLAS implementations and flip
        # an otherwise canonical near-tie by one ULP.  This also matches
        # AnchorSet.angular_distance_matrix, the canonical geometry reference.
        cosine = directions @ directions.T
        acquired_ids = np.asarray(acquired, dtype=np.int64)
        nearest_cosine = np.max(
            np.clip(cosine[:, acquired_ids], -1.0, 1.0), axis=1
        )
        scores = np.arccos(nearest_cosine)
        scores[acquired_ids] = 0.0
        return scores
