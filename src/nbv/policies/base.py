"""The policy input contract contains acquired observations and known poses only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nbv.data.observation_store import AcquiredObservation


@dataclass(frozen=True, slots=True)
class ObservationState:
    object_id: str
    acquired_observations: tuple[AcquiredObservation, ...]
    anchor_directions: np.ndarray
    camera_to_world: np.ndarray
    valid_candidate_mask: np.ndarray
    step_index: int
    seed: int
    anchor_ordering: str

    @property
    def acquired_anchor_ids(self) -> tuple[int, ...]:
        return tuple(item.anchor_id for item in self.acquired_observations)


class NBVPolicy(Protocol):
    name: str
    is_oracle: bool
    score_semantics: str

    def score(self, observation_state: ObservationState) -> np.ndarray:
        """Return 48 higher-is-better scores in canonical rollout coordinates."""
        ...
