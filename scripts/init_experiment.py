#!/usr/bin/env python3
"""Initialize a reproducible run directory; no training is performed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.logging_utils import configure_logging  # noqa: E402
from nbv.reproducibility import initialize_run, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config, args.set)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    experiment = config["experiment"]
    seed_everything(experiment["seed"], experiment["deterministic"])
    context = initialize_run(config, REPOSITORY_ROOT)
    logger = configure_logging(context.log_path, args.verbose)
    logger.info("Initialized run %s", context.run_id)
    logger.info("Artifacts: %s", context.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
