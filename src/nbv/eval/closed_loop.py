"""One deterministic acquisition loop; ground-truth geometry stays evaluator-owned."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import VisibilityCache
from nbv.eval.metrics import (
    ndcg_at_k,
    normalized_regret,
    reachable_normalized_coverage,
    spearman_rank,
)
from nbv.eval.result_schema import ROLLOUT_SCHEMA_VERSION, RolloutResult, visibility_fingerprint
from nbv.geometry.anchors import CANONICAL_ORDERING, canonical_anchors
from nbv.geometry.coverage import VISIBILITY_TARGETS
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world
from nbv.policies.base import NBVPolicy, ObservationState


@dataclass(frozen=True)
class RolloutConfig:
    max_acquired_views: int = 10
    initial_anchor_ids: tuple[int, ...] = (0,)
    invalid_anchor_ids: tuple[int, ...] = ()
    coverage_target: str = "vis_a"
    seed: int = 0

    def __post_init__(self):
        if type(self.max_acquired_views) is not int or self.max_acquired_views <= 0:
            raise ValueError("max_acquired_views must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for field in ("initial_anchor_ids", "invalid_anchor_ids"):
            ids = tuple(getattr(self, field))
            for anchor_id in ids:
                canonical_anchors().by_id(anchor_id)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{field} must contain unique anchors")
            object.__setattr__(self, field, ids)
        if not self.initial_anchor_ids or len(self.initial_anchor_ids) > self.max_acquired_views:
            raise ValueError("The total view budget must include all non-empty initial views")
        if set(self.initial_anchor_ids) & set(self.invalid_anchor_ids):
            raise ValueError("Initial anchors cannot be invalid")
        if self.coverage_target not in VISIBILITY_TARGETS:
            raise ValueError(f"coverage_target must be one of {VISIBILITY_TARGETS}")


def canonical_masked_argmax(scores: np.ndarray, valid: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid)
    if scores.shape != (48,):
        raise ValueError("Policy scores must have shape [48]")
    if valid.shape != (48,) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("valid must be a boolean [48] mask with a candidate")
    if not np.isfinite(scores[valid]).all():
        raise ValueError("Policy scores must be finite on valid candidates")
    candidates = np.flatnonzero(valid)
    return int(candidates[np.argmax(scores[valid])])


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.setflags(write=False)
    return result


def _resident_memory_bytes() -> int | None:
    """Return current Linux RSS, with a portable peak-RSS fallback."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * (1024 if os.name != "darwin" else 1)) if peak else None


def _policy_cuda_device(policy: NBVPolicy) -> torch.device | None:
    device = getattr(policy, "device", None)
    if device is None:
        return None
    resolved = torch.device(device)
    return resolved if resolved.type == "cuda" else None


def run_rollout(
    cache: VisibilityCache,
    observations: ObservationStore,
    policy: NBVPolicy,
    config: RolloutConfig = RolloutConfig(),
) -> RolloutResult:
    """Count initial views in the budget; continue zero-gain steps until budget/exhaustion.

    Only the explicit oracle branch sees true gains. Policies get fresh snapshots
    of acquired RGB, camera geometry, and masks, never this cache or the store.
    Timing measures CPU score generation (Oracle: copying precomputed gains),
    excluding geometry evaluation and RGB loading. It is not live model latency.
    """
    if cache.metadata["object_id"] != observations.object_id:
        raise ValueError("Visibility cache and observation object IDs disagree")
    anchors = canonical_anchors()
    available = observations.available_mask
    available[list(config.invalid_anchor_ids)] = False
    history = list(config.initial_anchor_ids)
    if not available[history].all():
        raise ValueError("Every initial anchor must have available RGB")
    for anchor_id in history:
        observations.acquire(anchor_id)
    reachable_anchor_ids = np.flatnonzero(available)
    reachable_ceiling = float(
        np.clip(
            cache.coverage(reachable_anchor_ids, target=config.coverage_target),
            0,
            1,
        )
    )
    radius = float(cache.metadata.get("camera_radius", 1.0))
    camera_poses = np.stack([
        anchor_camera_to_world(
            anchor, radius,
            convention=cache.metadata.get("camera_convention", CAMERA_CONVENTION),
        )
        for anchor in anchors
    ])
    coverage = [float(np.clip(cache.coverage(history, target=config.coverage_target), 0, 1))]
    counts = [len(history)]
    steps, all_scores, all_gains, all_masks = [], [], [], []
    stop_reason = "view_budget_exhausted"
    while len(history) < config.max_acquired_views:
        valid = available & anchors.valid_candidate_mask(history)
        if not valid.any():
            stop_reason = "no_valid_candidates"
            break
        gains = cache.candidate_gains(history, target=config.coverage_target)
        cuda_device = _policy_cuda_device(policy)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        rss_before = _resident_memory_bytes()
        if policy.is_oracle:
            started = perf_counter()
            scores = gains.copy()
        else:
            state = ObservationState(
                object_id=observations.object_id,
                acquired_observations=tuple(observations.acquire(a) for a in history),
                anchor_directions=_readonly_copy(anchors.directions),
                camera_to_world=_readonly_copy(camera_poses),
                valid_candidate_mask=_readonly_copy(valid),
                step_index=len(steps), seed=config.seed, anchor_ordering=CANONICAL_ORDERING,
            )
            started = perf_counter()
            scores = policy.score(state)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        elapsed_ms = (perf_counter() - started) * 1000
        rss_after = _resident_memory_bytes()
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(cuda_device))
            if cuda_device is not None else None
        )
        scores = np.asarray(scores, dtype=np.float64).copy()
        selected = canonical_masked_argmax(scores, valid)
        correlation = spearman_rank(scores, gains, valid)
        before = coverage[-1]
        old_history = history.copy()
        acquisition_started = perf_counter()
        observations.acquire(selected)
        acquisition_ms = (perf_counter() - acquisition_started) * 1000
        history.append(selected)
        after = float(np.clip(cache.coverage(history, target=config.coverage_target), 0, 1))
        if after < before - 1e-12 or not np.isclose(after - before, gains[selected], atol=1e-12, rtol=1e-10):
            raise ValueError("Selected gain does not match the explicit coverage difference")
        steps.append({
            "object_id": observations.object_id, "policy": policy.name, "seed": config.seed,
            "step_index": len(steps), "history_anchor_ids": old_history,
            "acquired_view_count_before": len(old_history), "acquired_view_count_after": len(history),
            "valid_candidate_mask": valid.tolist(), "selected_anchor": selected,
            "selected_true_gain": float(gains[selected]), "oracle_true_gain": float(gains[valid].max()),
            "coverage_before": before, "coverage_after": after,
            "reachable_normalized_coverage_before": reachable_normalized_coverage(
                before, reachable_ceiling
            ),
            "reachable_normalized_coverage_after": reachable_normalized_coverage(
                after, reachable_ceiling
            ),
            "normalized_regret": normalized_regret(scores, gains, valid),
            "spearman": float(correlation) if np.isfinite(correlation) else None,
            "ndcg_at_5": ndcg_at_k(scores, gains, valid),
            "policy_ms": elapsed_ms, "acquisition_ms": acquisition_ms,
            "process_rss_bytes": rss_after,
            "process_rss_delta_bytes": (
                max(0, rss_after - rss_before)
                if rss_before is not None and rss_after is not None else None
            ),
            "peak_cuda_allocated_bytes": peak_cuda_bytes,
            "profiling_mode": getattr(policy, "profiling_mode", "analytic_cpu"),
        })
        all_scores.append(scores)
        all_gains.append(gains)
        all_masks.append(valid)
        coverage.append(after)
        counts.append(len(history))
    metadata = {
        "schema_version": ROLLOUT_SCHEMA_VERSION, "phase": "phase2",
        "object_id": observations.object_id, "policy": policy.name, "seed": config.seed,
        "policy_is_oracle": bool(policy.is_oracle), "policy_score_semantics": policy.score_semantics,
        "policy_provenance": getattr(policy, "provenance", {}),
        "training_target_semantics": getattr(policy, "training_target_semantics", "none"),
        "coverage_target": config.coverage_target,
        "reachable_coverage_ceiling": reachable_ceiling,
        "reachable_coverage_anchor_ids": reachable_anchor_ids.tolist(),
        "reachable_normalized_coverage_definition": (
            "absolute_coverage_divided_by_union_coverage_of_all_available_"
            "non_invalid_anchors; zero_when_ceiling_is_zero"
        ),
        "visibility_definition": cache.metadata["visibility_definition"],
        "visibility_cache_fingerprint": visibility_fingerprint(cache),
        "visibility_cache_metadata": cache.metadata.copy(),
        "rollout_config": dict(asdict(config), initial_anchor_ids=list(config.initial_anchor_ids),
                               invalid_anchor_ids=list(config.invalid_anchor_ids)),
        "available_anchor_mask": available.tolist(), "stop_reason": stop_reason,
        "tie_breaking": "lowest_valid_canonical_anchor_id", "zero_gain_behavior": "continue",
        "random_protocol": "sha256_seed_object_step_128bit_pcg64_v1",
        "timing_protocol": "wall_clock_policy_scoring_with_cuda_synchronization_excludes_geometry_and_evaluator_rgb_loading",
        "memory_protocol": "per_step_process_rss_and_cuda_peak_allocated; cuda_stats_reset_before_policy_score",
        "ranking_aggregation": "per_step_mean_excluding_undefined_with_valid_counts",
    }
    return RolloutResult(
        metadata=metadata, steps=steps, acquired_anchor_ids=np.asarray(history, dtype=np.int64),
        acquired_view_counts=np.asarray(counts, dtype=np.int64), coverage=np.asarray(coverage),
        scores=np.asarray(all_scores, dtype=np.float64).reshape(-1, 48),
        candidate_gains=np.asarray(all_gains, dtype=np.float64).reshape(-1, 48),
        valid_masks=np.asarray(all_masks, dtype=bool).reshape(-1, 48),
        image_paths=tuple(observations.acquire(a).image_path for a in history),
    )


def replay_rollout(
    saved: RolloutResult, cache: VisibilityCache, observations: ObservationStore
) -> RolloutResult:
    """Recompute selection, geometry, masks, and metrics from saved scores.

    This validates a diagnostic replay, not a fresh model inference. Recorded
    wall-clock timing is deliberately excluded from deterministic comparisons.
    """
    if visibility_fingerprint(cache) != saved.metadata["visibility_cache_fingerprint"]:
        raise ValueError("Replay visibility cache fingerprint differs")

    class RecordedPolicy:
        name = saved.metadata["policy"]
        is_oracle = saved.metadata["policy_is_oracle"]
        score_semantics = saved.metadata["policy_score_semantics"]

        def score(self, state):
            return saved.scores[state.step_index].copy()

    replayed = run_rollout(cache, observations, RecordedPolicy(), RolloutConfig(**saved.metadata["rollout_config"]))
    for key in ("available_anchor_mask", "stop_reason"):
        if replayed.metadata[key] != saved.metadata[key]:
            raise ValueError(f"Replay mismatch: {key}")
    for key in ("acquired_anchor_ids", "acquired_view_counts", "coverage", "scores", "candidate_gains", "valid_masks"):
        if not np.array_equal(getattr(replayed, key), getattr(saved, key), equal_nan=True):
            raise ValueError(f"Replay mismatch: {key}")
    if len(replayed.steps) != len(saved.steps):
        raise ValueError("Replay step count differs")
    for actual, expected in zip(replayed.steps, saved.steps):
        for key in expected.keys() - {
            "policy_ms", "acquisition_ms", "process_rss_bytes",
            "process_rss_delta_bytes", "peak_cuda_allocated_bytes", "profiling_mode",
        }:
            if actual[key] != expected[key]:
                raise ValueError(f"Replay step mismatch: {key}")
    return replayed
