"""Replayable Phase 2 arrays and strict JSON/CSV summaries."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nbv.data.visibility_cache import VisibilityCache, visibility_cache_fingerprint
from nbv.eval.metrics import coverage_auc, reachable_normalized_coverage


ROLLOUT_SCHEMA_VERSION = 2
SUPPORTED_ROLLOUT_SCHEMA_VERSIONS = (1, ROLLOUT_SCHEMA_VERSION)


def visibility_fingerprint(cache: VisibilityCache) -> str:
    """Identify both the actual face arrays and their generation provenance."""
    return visibility_cache_fingerprint(cache)


@dataclass(frozen=True)
class RolloutResult:
    metadata: dict[str, Any]
    steps: list[dict[str, Any]]
    acquired_anchor_ids: np.ndarray
    acquired_view_counts: np.ndarray
    coverage: np.ndarray
    scores: np.ndarray
    candidate_gains: np.ndarray
    valid_masks: np.ndarray
    image_paths: tuple[str, ...]

    @property
    def reachable_normalized_coverage(self) -> np.ndarray | None:
        """Coverage normalized by the eligible-view union, when recorded."""

        ceiling = self.metadata.get("reachable_coverage_ceiling")
        if ceiling is None:
            return None
        return np.asarray(
            [reachable_normalized_coverage(value, ceiling) for value in self.coverage],
            dtype=np.float64,
        )

    def summary(self) -> dict[str, Any]:
        reachable = self.reachable_normalized_coverage
        result = {
            "object_id": self.metadata["object_id"],
            "policy": self.metadata["policy"],
            "seed": self.metadata["seed"],
            "coverage_target": self.metadata["coverage_target"],
            "stop_reason": self.metadata["stop_reason"],
            "acquired_views": int(self.acquired_view_counts[-1]),
            "initial_coverage": float(self.coverage[0]),
            "final_coverage": float(self.coverage[-1]),
            "coverage_auc": coverage_auc(self.coverage, self.acquired_view_counts),
            "reachable_coverage_ceiling": self.metadata.get(
                "reachable_coverage_ceiling"
            ),
            "initial_reachable_normalized_coverage": (
                float(reachable[0]) if reachable is not None else None
            ),
            "final_reachable_normalized_coverage": (
                float(reachable[-1]) if reachable is not None else None
            ),
            "reachable_normalized_coverage_auc": (
                coverage_auc(reachable, self.acquired_view_counts)
                if reachable is not None else None
            ),
            "coverage_interval_start": int(self.acquired_view_counts[0]),
            "coverage_interval_end": int(self.acquired_view_counts[-1]),
            "num_steps": len(self.steps),
        }
        for key in ("normalized_regret", "spearman", "ndcg_at_5"):
            values = [s[key] for s in self.steps if s[key] is not None]
            result[f"{key}_mean"] = float(np.mean(values)) if values else None
            result[f"{key}_valid_count"] = len(values)
        times = [s["policy_ms"] for s in self.steps]
        result["median_policy_ms"] = float(np.median(times)) if times else None
        for source, destination in (
            ("process_rss_bytes", "peak_process_rss_bytes"),
            ("process_rss_delta_bytes", "peak_process_rss_delta_bytes"),
            ("peak_cuda_allocated_bytes", "peak_cuda_allocated_bytes"),
        ):
            values = [s.get(source) for s in self.steps if s.get(source) is not None]
            result[destination] = max(values) if values else None
        return result


def save_rollout(result: RolloutResult, path: str | Path) -> Path:
    """Atomic NPZ; no pickle, geometry objects, or executable policy payloads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True, allow_nan=False)),
                steps_json=np.asarray(json.dumps(result.steps, sort_keys=True, allow_nan=False)),
                acquired_anchor_ids=result.acquired_anchor_ids,
                acquired_view_counts=result.acquired_view_counts,
                coverage=result.coverage,
                scores=result.scores,
                candidate_gains=result.candidate_gains,
                valid_masks=result.valid_masks,
                image_paths=np.asarray(result.image_paths, dtype=str),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_rollout(path: str | Path) -> RolloutResult:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("schema_version") not in SUPPORTED_ROLLOUT_SCHEMA_VERSIONS:
            raise ValueError("Unsupported rollout schema_version")
        if (
            metadata["schema_version"] >= 2
            and "reachable_coverage_ceiling" not in metadata
        ):
            raise ValueError(
                "Rollout schema 2 requires metadata.reachable_coverage_ceiling"
            )
        return RolloutResult(
            metadata=metadata,
            steps=json.loads(str(payload["steps_json"].item())),
            image_paths=tuple(payload["image_paths"].tolist()),
            **{key: payload[key].copy() for key in (
                "acquired_anchor_ids", "acquired_view_counts", "coverage",
                "scores", "candidate_gains", "valid_masks",
            )},
        )


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                for key, value in row.items()
            })
