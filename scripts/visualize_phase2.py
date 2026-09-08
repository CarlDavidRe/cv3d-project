#!/usr/bin/env python3
"""Replay and render a saved Phase 2 rollout as a self-contained SVG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.closed_loop import replay_rollout
from nbv.eval.result_schema import load_rollout
from nbv.visualization import write_phase2_rollout_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decision", type=int, default=-1, help="Zero-based decision index; default is final.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        saved = load_rollout(args.rollout)
        cache = load_visibility_cache(args.cache or saved.metadata["visibility_cache_path"])
        data_root = args.data_root or Path(saved.image_paths[0]).parents[3]
        store = ObservationStore.from_num_object(data_root, saved.metadata["object_id"])
        replay_rollout(saved, cache, store)
        output = args.output or args.rollout.with_suffix(".svg")
        write_phase2_rollout_demo(saved, cache, output, decision_index=args.decision)
        print(f"Phase 2 rollout demo: {output}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Phase 2 visualization error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
