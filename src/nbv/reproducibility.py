"""Seed control and run provenance for reproducible experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nbv.config import save_config, validate_config


@dataclass(frozen=True)
class RunContext:
    """Filesystem locations created for a single experiment run."""

    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    figure_dir: Path
    log_path: Path


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def initialize_run(
    config: Mapping[str, Any], repository_root: str | Path
) -> RunContext:
    """Create run directories and persist config plus environment provenance."""

    validate_config(config)
    root = Path(repository_root).resolve()
    experiment = config["experiment"]
    run_id = (
        f"{experiment['phase']}/{experiment['name']}/seed_{experiment['seed']}"
    )
    run_dir = resolve_run_directory(config, root)
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    figure_dir = run_dir / "figures"
    for directory in (checkpoint_dir, metrics_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")
    metadata = collect_metadata(root, int(experiment["seed"]))
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        figure_dir=figure_dir,
        log_path=run_dir / "run.log",
    )


def resolve_run_directory(
    config: Mapping[str, Any], repository_root: str | Path
) -> Path:
    """Resolve a validated run directory below the configured output root."""

    validate_config(config)
    root = Path(repository_root).resolve()
    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    experiment = config["experiment"]
    run_dir = (
        output_root
        / experiment["phase"]
        / experiment["name"]
        / f"seed_{experiment['seed']}"
    ).resolve()
    if not run_dir.is_relative_to(output_root):
        raise ValueError("Resolved run directory escapes paths.output_root")
    return run_dir


def collect_metadata(repository_root: str | Path, seed: int) -> dict[str, Any]:
    """Collect compact machine-readable provenance for a run."""

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(repository_root)),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "seed": seed,
    }


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
