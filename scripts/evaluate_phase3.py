#!/usr/bin/env python3
"""Run the Step 18 controlled Phase 3 comparison and reporting pipeline."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.data import HistoryDatasetError, HistoryFeatureError  # noqa: E402
from nbv.experiments.phase3_controlled import (  # noqa: E402
    parse_phase3_controlled_settings,
    run_phase3_controlled,
)
from nbv.features import FeatureCacheError, FeatureExtractorError  # noqa: E402
from nbv.logging_utils import configure_logging  # noqa: E402
from nbv.reproducibility import resolve_run_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase3_controlled.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified joint-test shards and replayable object rollouts.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        help=(
            "Evaluate a trained expressive joint-variant checkpoint in place of "
            "the capacity-matched joint checkpoint."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        help="Required with --joint-checkpoint to keep variant outputs separate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        if args.joint_checkpoint is not None:
            if not args.experiment_name:
                raise ValueError("--experiment-name is required with --joint-checkpoint")
            checkpoint = args.joint_checkpoint.resolve()
            if not checkpoint.is_file():
                raise ValueError(f"joint checkpoint does not exist: {checkpoint}")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config["experiment"]["name"] = args.experiment_name
            controlled = config["phase3"]["controlled"]
            controlled["comparison_mode"] = "expressive_joint_variant"
            controlled["joint"]["checkpoint"] = str(checkpoint)
            controlled["joint"]["checkpoint_sha256"] = digest
        elif args.experiment_name:
            raise ValueError("--experiment-name requires --joint-checkpoint")
        parse_phase3_controlled_settings(config, REPOSITORY_ROOT)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    run_dir = resolve_run_directory(config, REPOSITORY_ROOT)
    logger = configure_logging(run_dir / "run.log", args.verbose)
    try:
        completed = run_phase3_controlled(
            config,
            REPOSITORY_ROOT,
            logger=logger,
            resume=args.resume,
        )
    except (
        FeatureCacheError,
        FeatureExtractorError,
        HistoryDatasetError,
        HistoryFeatureError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("Phase 3 controlled evaluation failed: %s", exc)
        return 1
    logger.info("Summary: %s", completed / "metrics" / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
