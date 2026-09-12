"""Read-only discovery and loading helpers for the local dashboard."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExperimentRun:
    """A completed experiment run with a summary artifact."""

    phase: str
    experiment: str
    seed: str
    root: Path

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def label(self) -> str:
        return f"{self.phase} / {self.experiment} / {self.seed}"


@dataclass(frozen=True)
class ReconstructionVisualization:
    """A generated reconstruction viewer and its associated artifacts."""

    root: Path
    metadata: dict[str, Any]

    @property
    def object_id(self) -> str:
        return str(self.metadata.get("object_id", self.root.name))

    @property
    def policy(self) -> str:
        return str(self.metadata.get("policy", "unknown"))

    @property
    def acquired_view_count(self) -> int:
        return int(self.metadata.get("acquired_view_count", 0))

    @property
    def label(self) -> str:
        return f"{self.object_id} · {self.policy} · {self.acquired_view_count} views"

    def artifact(self, name: str) -> Path | None:
        candidate = self.root / name
        return candidate if candidate.is_file() else None


def discover_runs(outputs_root: Path) -> list[ExperimentRun]:
    """Discover the experiment set represented by the project figures.

    Phase 1 keeps all experiment runs. Phase 2 mirrors the five explicit runs
    in ``plot_all_policy_coverage.py``. Phase 3 mirrors that script's recursive
    completed-run selection.
    """

    outputs_root = outputs_root.resolve()
    runs: list[ExperimentRun] = []
    if not outputs_root.is_dir():
        return runs

    phase1_root = outputs_root / "phase1"
    if phase1_root.is_dir():
        for run_root in phase1_root.rglob("seed_*"):
            if run_root.is_dir() and (run_root / "metrics").is_dir():
                relative = run_root.relative_to(outputs_root)
                runs.append(
                    ExperimentRun(
                        phase="phase1",
                        experiment="/".join(relative.parts[1:-1]),
                        seed=relative.parts[-1],
                        root=run_root,
                    )
                )

    for experiment in (
        "phase2_random",
        "phase2_farthest",
        "phase2_pun",
        "phase2_vggt",
        "phase2_oracle",
    ):
        phase2_root = outputs_root / "phase2" / experiment
        if not phase2_root.is_dir():
            continue
        for run_root in phase2_root.glob("seed_*"):
            if not (run_root / "metrics" / "summary.json").is_file():
                continue
            runs.append(
                ExperimentRun(
                    phase="phase2",
                    experiment=experiment,
                    seed=run_root.name,
                    root=run_root,
                )
            )

    phase3_root = outputs_root / "phase3"
    for completion_path in phase3_root.rglob("metrics/phase3_completion.json") if phase3_root.is_dir() else ():
        run_root = completion_path.parent.parent
        if not (run_root / "metrics" / "coverage.csv").is_file():
            continue
        try:
            completion = load_json(completion_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if completion.get("status") != "complete":
            continue
        relative = run_root.relative_to(outputs_root)
        parts = relative.parts
        runs.append(
            ExperimentRun(
                phase="phase3",
                experiment="/".join(parts[1:-1]),
                seed=parts[-1],
                root=run_root,
            )
        )
    return sorted(runs, key=lambda run: (run.phase, run.experiment, run.seed))


def discover_all_visualizations(outputs_root: Path) -> list[ReconstructionVisualization]:
    """Discover generated viewers globally so shared evaluation runs can reuse them."""

    visualizations: list[ReconstructionVisualization] = []
    if not outputs_root.is_dir():
        return visualizations
    for metadata_path in outputs_root.rglob("reconstruction_visualizations/*/metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        visualizations.append(
            ReconstructionVisualization(root=metadata_path.parent, metadata=metadata)
        )
    return sorted(
        visualizations,
        key=lambda item: (item.object_id, item.policy, item.acquired_view_count),
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary_policies(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the comparable closed-loop policy summaries across phases."""

    for key in ("policies", "closed_loop"):
        value = summary.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def discover_visualizations(run: ExperimentRun) -> list[ReconstructionVisualization]:
    root = run.metrics_dir / "reconstruction_visualizations"
    visualizations: list[ReconstructionVisualization] = []
    if not root.is_dir():
        return visualizations
    for metadata_path in root.glob("*/metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        visualizations.append(
            ReconstructionVisualization(root=metadata_path.parent, metadata=metadata)
        )
    return sorted(
        visualizations,
        key=lambda item: (item.object_id, item.policy, item.acquired_view_count),
    )


def filter_rows(
    rows: Iterable[dict[str, str]],
    *,
    object_id: str | None = None,
    policy: str | None = None,
    acquired_view_count: int | None = None,
) -> list[dict[str, str]]:
    """Filter reconstruction rows using their stable identity fields."""

    selected = []
    for row in rows:
        if object_id is not None and row.get("object_id") != object_id:
            continue
        if policy is not None and row.get("policy") != policy:
            continue
        if acquired_view_count is not None:
            try:
                if int(row.get("acquired_view_count", "")) != acquired_view_count:
                    continue
            except ValueError:
                continue
        selected.append(row)
    return selected


def matching_visualization(
    visualizations: Iterable[ReconstructionVisualization],
    *,
    object_id: str,
    policy: str,
    acquired_view_count: int,
) -> ReconstructionVisualization | None:
    for visualization in visualizations:
        if (
            visualization.object_id == object_id
            and visualization.policy == policy
            and visualization.acquired_view_count == acquired_view_count
        ):
            return visualization
    return None
