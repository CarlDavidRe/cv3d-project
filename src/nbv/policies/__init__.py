"""Policies for the shared closed-loop evaluator."""

from nbv.policies.base import NBVPolicy, ObservationState
from nbv.policies.farthest_policy import FarthestPolicy
from nbv.policies.oracle_policy import OraclePolicy
from nbv.policies.pun_policy import PUNPolicy
from nbv.policies.random_policy import RandomPolicy

__all__ = [
    "NBVPolicy",
    "ObservationState",
    "FarthestPolicy",
    "OraclePolicy",
    "PUNPolicy",
    "RandomPolicy",
]
