"""Explicit privileged marker; the evaluator supplies one-step true gains."""

from nbv.policies.base import ObservationState


class OraclePolicy:
    name = "oracle"
    is_oracle = True
    score_semantics = "true_incremental_surface_gain"

    def score(self, observation_state: ObservationState):
        raise RuntimeError("Oracle scores must be supplied by the evaluator")
