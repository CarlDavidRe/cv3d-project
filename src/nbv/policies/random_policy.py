"""Seeded random ranking independent of object order and other policy runs."""

import hashlib
import json

import numpy as np

from nbv.policies.base import ObservationState


class RandomPolicy:
    name = "random"
    is_oracle = False
    score_semantics = "seeded_random_ranking"

    def score(self, observation_state: ObservationState) -> np.ndarray:
        key = json.dumps([
            observation_state.seed, observation_state.object_id,
            observation_state.step_index,
        ], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "big")
        return np.random.Generator(np.random.PCG64(seed)).random(48)
