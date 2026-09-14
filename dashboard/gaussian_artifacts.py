"""Protocol checks for Gaussian artifacts displayed by the dashboard."""

from __future__ import annotations

import json
from pathlib import Path


GAUSSIAN_ITERATIONS_PER_VIEW = 1_500
GAUSSIAN_ALIGNMENT_REPAIR_VERSION = "acquired_rendered_silhouette_sim3_v2"


def is_current_gaussian_training(summary: dict, backend: str | None = None) -> bool:
    """Accept only results trained with the dashboard's per-view budget."""
    try:
        view_count = int(summary["acquired_view_count"])
        budget = summary["training_budget"]
        settings = summary["settings"]
    except (KeyError, TypeError, ValueError):
        return False
    total = GAUSSIAN_ITERATIONS_PER_VIEW * view_count
    return (
        view_count > 0
        and budget == {
            "iterations_per_view": GAUSSIAN_ITERATIONS_PER_VIEW,
            "view_count": view_count,
            "total_iterations": total,
        }
        and settings.get("iterations") == total
        and (backend is None or summary.get("backend") == backend)
    )


def is_current_gaussian_repair(summary: dict, backend: str | None = None) -> bool:
    """Require both current training and rendered-IoU-gated repair metadata."""
    try:
        version = summary["alignment_repair"]["identity"]["version"]
    except (KeyError, TypeError):
        return False
    return (
        version == GAUSSIAN_ALIGNMENT_REPAIR_VERSION
        and is_current_gaussian_training(summary, backend)
    )


def summary_file_matches(path: Path, *, backend: str, repaired: bool) -> bool:
    """Validate one artifact summary without leaking parse errors into the UI."""
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    validator = is_current_gaussian_repair if repaired else is_current_gaussian_training
    return validator(summary, backend)
