#!/usr/bin/env python3
"""Evaluate Phase 2 closed-loop policies, or verify a saved rollout replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nbv.config import load_config
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.closed_loop import replay_rollout
from nbv.eval.result_schema import load_rollout
from nbv.experiments.closed_loop import run_closed_loop_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/experiments/phase2_closed_loop.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--object", action="append", default=[], metavar="CATEGORY/OBJECT")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--skip-missing-caches", action="store_true")
    parser.add_argument("--replay", type=Path, help="Verify saved scores/actions/geometry; does not write results.")
    parser.add_argument("--cache", type=Path, help="Optional relocated visibility cache for --replay.")
    parser.add_argument("--data-root", type=Path, help="Optional relocated NUM root for --replay.")
    args = parser.parse_args()
    try:
        if args.replay:
            saved = load_rollout(args.replay)
            cache = load_visibility_cache(args.cache or saved.metadata["visibility_cache_path"])
            if args.data_root:
                store = ObservationStore.from_num_object(args.data_root, saved.metadata["object_id"])
            else:
                # Reconstruct all RGB paths from the original NUM object's image directory.
                data_root = Path(saved.image_paths[0]).parents[3]
                store = ObservationStore.from_num_object(data_root, saved.metadata["object_id"])
            replay_rollout(saved, cache, store)
            print(f"Replay verified: {args.replay}")
            return 0
        if args.cache or args.data_root:
            raise ValueError("--cache and --data-root require --replay; use --set paths.* for evaluation")
        config = load_config(args.config, args.set)
        settings = config["phase2"]["evaluation"]
        for arg, key in ((args.object, "object_ids"), (args.limit, "limit"), (args.split, "split")):
            if arg is not None and arg != []:
                settings[key] = arg
        if args.skip_missing_caches:
            settings["skip_missing_caches"] = True
        result = run_closed_loop_experiment(config, REPOSITORY_ROOT)
        print(f"Closed-loop results: {result}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Closed-loop error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
