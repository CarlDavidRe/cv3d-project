#!/usr/bin/env python3
"""Migrate flattened Gaussian-splatting object directories to category/object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/gaussian_splatting_variant_comparison"
DEFAULT_SPLIT_MANIFEST = ROOT / "data/splits/num_v1.json"


def split_object_id(object_id: str) -> tuple[str, str]:
    if not isinstance(object_id, str):
        raise ValueError("object IDs must be strings")
    parts = object_id.split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe object ID {object_id!r}; expected category/object")
    return parts[0], parts[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print moves only"
    )
    return parser


def load_test_objects(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    object_ids = payload["splits"]["test"]
    if not isinstance(object_ids, list):
        raise ValueError("split manifest must contain splits.test category/object IDs")
    for object_id in object_ids:
        split_object_id(object_id)
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("splits.test contains duplicate object IDs")
    return object_ids


def _rewrite_strings(value: Any, old_fragment: str, new_fragment: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_fragment, new_fragment)
    if isinstance(value, list):
        return [_rewrite_strings(item, old_fragment, new_fragment) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_strings(item, old_fragment, new_fragment)
            for key, item in value.items()
        }
    return value


def rewrite_json_paths(destination: Path, flat_name: str, object_id: str) -> int:
    output_directory_name = destination.parent.parent.name
    old_fragment = f"/{output_directory_name}/{flat_name}/"
    new_fragment = f"/{output_directory_name}/{object_id}/"
    rewritten = 0
    for path in destination.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = _rewrite_strings(payload, old_fragment, new_fragment)
        if updated != payload:
            path.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rewritten += 1
    return rewritten


def migrate_layout(
    output_root: Path, object_ids: Sequence[str], *, dry_run: bool = False
) -> tuple[int, int]:
    moves: list[tuple[Path, Path, str, str]] = []
    seen: set[str] = set()
    for object_id in object_ids:
        category_id, object_key = split_object_id(object_id)
        if object_id in seen:
            raise ValueError(f"duplicate object ID: {object_id}")
        seen.add(object_id)
        flat_name = f"{category_id}_{object_key}"
        source = output_root / flat_name
        destination = output_root / category_id / object_key
        if not source.exists():
            continue
        if not source.is_dir():
            raise ValueError(f"legacy output is not a directory: {source}")
        if destination.exists():
            raise FileExistsError(
                f"refusing to merge legacy and nested outputs: {source} -> "
                f"{destination}"
            )
        moves.append((source, destination, flat_name, object_id))

    for source, destination, _, _ in moves:
        print(f"{'WOULD MOVE' if dry_run else 'MOVE'} {source} -> {destination}")

    if dry_run:
        return len(moves), 0

    rewritten = 0
    for source, destination, flat_name, object_id in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        rewritten += rewrite_json_paths(destination, flat_name, object_id)
    return len(moves), rewritten


def main() -> int:
    args = build_parser().parse_args()
    try:
        object_ids = load_test_objects(args.split_manifest)
        moved, rewritten = migrate_layout(
            args.output_root, object_ids, dry_run=args.dry_run
        )
        if args.dry_run:
            print(f"Migration dry run: {moved} object directories would move.")
        else:
            print(
                f"Migration complete: moved {moved} object directories and "
                f"rewrote {rewritten} JSON files."
            )
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Gaussian-splatting migration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
