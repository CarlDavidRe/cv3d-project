#!/usr/bin/env python3
"""Train the Step 17 joint frozen-VGGT direct surface-gain control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.data import HistoryDatasetError  # noqa: E402
from nbv.experiments.phase3_joint import (  # noqa: E402
    parse_phase3_joint_settings,
    run_phase3_joint,
)
from nbv.features import FeatureCacheError, FeatureExtractorError  # noqa: E402
from nbv.logging_utils import configure_logging  # noqa: E402
from nbv.reproducibility import resolve_run_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase3_joint.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        settings = parse_phase3_joint_settings(config, REPOSITORY_ROOT)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    run_dir = resolve_run_directory(config, REPOSITORY_ROOT)
    logger = configure_logging(run_dir / "run.log", args.verbose)
    logger.info(
        "Phase 3 joint control: complete histories through %s on %s",
        settings.model_id,
        settings.device,
    )
    try:
        completed = run_phase3_joint(config, REPOSITORY_ROOT, logger=logger)
    except (
        FeatureCacheError,
        FeatureExtractorError,
        HistoryDatasetError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("Phase 3 joint training failed: %s", exc)
        return 1
    logger.info("Summary: %s", completed / "metrics" / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
