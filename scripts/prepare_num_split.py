#!/usr/bin/env python3
"""Create the frozen Phase 1 object-disjoint NUM split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.data import (  # noqa: E402
    NUMDataset,
    build_num_v1_manifest,
    complete_objects_by_category,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the PUN-compatible object-level NUM split"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/NUM"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/splits/num_v1.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = NUMDataset(
        args.data_root,
        split="all",
        require_complete_objects=False,
        transform=lambda image: image,
    )
    complete = complete_objects_by_category(dataset.records)
    complete_keys = {
        f"{category_id}/{object_id}"
        for category_id, object_ids in complete.items()
        for object_id in object_ids
    }
    all_keys = {
        f"{category_path.name}/{object_path.name}"
        for category_path in args.data_root.iterdir()
        if category_path.is_dir()
        for object_path in category_path.iterdir()
        if object_path.is_dir()
    }
    manifest = build_num_v1_manifest(
        complete,
        excluded_objects=all_keys - complete_keys,
    )
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"Refusing to change frozen split manifest: {args.output}"
            )
        print(f"split unchanged: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote split: {args.output}")

    counts = manifest["object_counts"]
    print(
        f"objects: train={counts['train']} val={counts['val']} "
        f"test={counts['test']} excluded={len(all_keys - complete_keys)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
