#!/usr/bin/env python3
"""Precompute deterministic mesh-surface visibility for canonical anchors."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.data import (  # noqa: E402
    IncompatibleVisibilityCacheError,
    VisibilityCache,
    VisibilityCacheError,
    load_object_split,
    load_visibility_cache,
    save_visibility_cache,
    visibility_cache_path,
)
from nbv.geometry import (  # noqa: E402
    CAMERA_CONVENTION,
    CANONICAL_ORDERING,
    DEPTH_RENDERER,
    PerspectiveCamera,
    SURFACE_SAMPLING_ALGORITHM,
    canonical_anchors,
    compute_anchor_visibility,
    sample_obj_surface,
)
from nbv.geometry.mesh_sampling import sha256_file  # noqa: E402
from nbv.reproducibility import seed_everything  # noqa: E402
from nbv.visualization import write_visibility_debug_svg  # noqa: E402


PUN_SOURCE_REVISION = "aa6f8f4f12154854a4c1867209725c80475af102"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase2_visibility.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        metavar="CATEGORY/OBJECT",
        help="Precompute one or more explicit objects instead of a split.",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default=None
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing compatible or incompatible caches.",
    )
    parser.add_argument(
        "--debug-anchor",
        type=int,
        default=None,
        help="Write an SVG showing visible points for this anchor.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config, args.set)
        settings = _visibility_settings(config)
        object_ids = _select_objects(config, args.object, args.split, args.limit)
    except (ConfigError, ValueError, VisibilityCacheError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    experiment = config["experiment"]
    seed_everything(experiment["seed"], experiment["deterministic"])
    mesh_root = _repository_path(config["paths"]["mesh_root"])
    cache_root = _repository_path(config["paths"]["visibility_cache_root"])
    debug_root = (
        _repository_path(config["paths"]["output_root"])
        / "phase2"
        / "visibility_debug"
    )
    mesh_relative_path = Path(settings["mesh_relative_path"])
    if not mesh_root.is_dir():
        print(
            f"Configuration error: ShapeNet mesh root is not a directory: "
            f"{mesh_root}",
            file=sys.stderr,
        )
        return 2
    if mesh_relative_path.is_absolute() or ".." in mesh_relative_path.parts:
        print(
            "Configuration error: mesh_relative_path must be safe and relative",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for object_id in object_ids:
        mesh_path = mesh_root / object_id / mesh_relative_path
        destination = visibility_cache_path(cache_root, object_id)
        if not mesh_path.is_file():
            print(f"ERROR {object_id}: missing mesh {mesh_path}", file=sys.stderr)
            failures += 1
            continue
        mesh_sha256 = sha256_file(mesh_path)
        expected = _expected_metadata(object_id, mesh_sha256, settings)
        if destination.exists() and not args.overwrite:
            try:
                cache = load_visibility_cache(
                    destination, expected_metadata=expected
                )
            except IncompatibleVisibilityCacheError as exc:
                print(
                    f"ERROR {object_id}: {exc}; pass --overwrite to rebuild",
                    file=sys.stderr,
                )
                failures += 1
                continue
            except VisibilityCacheError as exc:
                print(
                    f"ERROR {object_id}: {exc}; pass --overwrite to rebuild",
                    file=sys.stderr,
                )
                failures += 1
                continue
            print(f"SKIP {object_id}: compatible cache {destination}")
            _maybe_write_debug(cache, object_id, args.debug_anchor, debug_root)
            continue

        try:
            started = time.perf_counter()
            mesh, sample = sample_obj_surface(
                mesh_path,
                n_surface=settings["n_surface"],
                seed=settings["sampling_seed"],
                mesh_scale=settings["mesh_scale"],
            )
            camera = PerspectiveCamera(
                height=settings["render_resolution"][0],
                width=settings["render_resolution"][1],
                horizontal_fov_degrees=settings["horizontal_fov_degrees"],
                near=settings["near"],
                far=settings["far"],
            )

            def progress(anchor_id: int, visible_count: int, total: int) -> None:
                if anchor_id in {0, 11, 23, 35, 47}:
                    print(
                        f"  {object_id}: anchor {anchor_id:02d}/47, "
                        f"visible={visible_count}/{total}"
                    )

            visibility = compute_anchor_visibility(
                mesh,
                sample.points,
                camera=camera,
                camera_radius=settings["camera_radius"],
                depth_tolerance=settings["depth_tolerance"],
                depth_neighborhood_radius=settings["depth_neighborhood_radius"],
                cull_backfaces=settings["cull_backfaces"],
                progress=progress,
            )
            metadata = dict(expected)
            metadata.update(sample.metadata)
            metadata["mesh_relative_path"] = mesh_relative_path.as_posix()
            cache = VisibilityCache(
                surface_points=sample.points,
                visibility=visibility,
                anchor_ids=np.arange(48, dtype=np.int16),
                sample_face_indices=sample.face_indices,
                sample_barycentric=sample.barycentric,
                metadata=metadata,
            )
            save_visibility_cache(cache, destination)
            elapsed = time.perf_counter() - started
            size_mib = destination.stat().st_size / (1024 * 1024)
            print(
                f"WROTE {object_id}: {destination} ({size_mib:.2f} MiB, "
                f"{elapsed:.1f}s)"
            )
            _print_sanity_statistics(cache)
            _maybe_write_debug(cache, object_id, args.debug_anchor, debug_root)
        except (OSError, ValueError) as exc:
            print(f"ERROR {object_id}: {exc}", file=sys.stderr)
            failures += 1

    print(
        f"visibility precompute complete: objects={len(object_ids)}, "
        f"failures={failures}"
    )
    return 1 if failures else 0


def _visibility_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    phase2 = config.get("phase2")
    if not isinstance(phase2, Mapping) or not isinstance(
        phase2.get("visibility"), Mapping
    ):
        raise ValueError("phase2.visibility must be a mapping")
    raw = dict(phase2["visibility"])
    required = {
        "split",
        "split_manifest",
        "mesh_relative_path",
        "n_surface",
        "sampling_seed",
        "mesh_scale",
        "camera_radius",
        "horizontal_fov_degrees",
        "render_resolution",
        "near",
        "far",
        "depth_tolerance",
        "depth_neighborhood_radius",
        "cull_backfaces",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"phase2.visibility is missing {sorted(missing)}")
    resolution = raw["render_resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in resolution
        )
    ):
        raise ValueError("render_resolution must be [positive_height, positive_width]")
    for key in ("n_surface", "sampling_seed", "depth_neighborhood_radius"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int):
            raise ValueError(f"{key} must be an integer")
    if raw["n_surface"] <= 0 or raw["depth_neighborhood_radius"] < 0:
        raise ValueError(
            "n_surface must be positive and neighborhood radius non-negative"
        )
    if raw["sampling_seed"] < 0:
        raise ValueError("sampling_seed must be non-negative")
    numeric_fields = (
        "mesh_scale",
        "camera_radius",
        "horizontal_fov_degrees",
        "near",
        "far",
        "depth_tolerance",
    )
    for key in numeric_fields:
        value = raw[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
    if raw["mesh_scale"] <= 0 or raw["camera_radius"] <= 0:
        raise ValueError("mesh_scale and camera_radius must be positive")
    if raw["depth_tolerance"] < 0:
        raise ValueError("depth_tolerance must be non-negative")
    PerspectiveCamera(
        height=resolution[0],
        width=resolution[1],
        horizontal_fov_degrees=raw["horizontal_fov_degrees"],
        near=raw["near"],
        far=raw["far"],
    )
    if not isinstance(raw["cull_backfaces"], bool):
        raise ValueError("cull_backfaces must be a boolean")
    return raw


def _select_objects(
    config: Mapping[str, Any],
    explicit: list[str],
    split_override: str | None,
    limit_override: int | None,
) -> list[str]:
    settings = config["phase2"]["visibility"]
    if explicit:
        object_ids = sorted(set(explicit))
    else:
        split = split_override or settings["split"]
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be train, val, test, or all")
        manifest = load_object_split(_repository_path(settings["split_manifest"]))
        keys = (
            set().union(*manifest.values()) if split == "all" else manifest[split]
        )
        object_ids = sorted(f"{category}/{object_id}" for category, object_id in keys)
    for object_id in object_ids:
        visibility_cache_path(Path("unused"), object_id)
    limit = limit_override if limit_override is not None else settings.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer or null")
        object_ids = object_ids[:limit]
    if not object_ids:
        raise ValueError("no objects selected")
    return object_ids


def _expected_metadata(
    object_id: str, mesh_sha256: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "object_id": object_id,
        "mesh_sha256": mesh_sha256,
        "n_surface": settings["n_surface"],
        "sampling_seed": settings["sampling_seed"],
        "sampling_algorithm": SURFACE_SAMPLING_ALGORITHM,
        "numpy_version": np.__version__,
        "mesh_scale": float(settings["mesh_scale"]),
        "anchor_ordering": CANONICAL_ORDERING,
        "anchor_count": 48,
        "render_resolution": list(settings["render_resolution"]),
        "horizontal_fov_degrees": float(settings["horizontal_fov_degrees"]),
        "camera_radius": float(settings["camera_radius"]),
        "near": float(settings["near"]),
        "far": float(settings["far"]),
        "depth_tolerance": float(settings["depth_tolerance"]),
        "depth_neighborhood_radius": settings["depth_neighborhood_radius"],
        "cull_backfaces": settings["cull_backfaces"],
        "renderer": DEPTH_RENDERER,
        "camera_convention": CAMERA_CONVENTION,
        "pun_source_revision": PUN_SOURCE_REVISION,
    }


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _print_sanity_statistics(cache: VisibilityCache) -> None:
    total = len(cache.surface_points)
    for anchor_id in (0, 12, 24, 36, 46):
        count = int(cache.visibility[anchor_id].sum())
        print(f"  anchor {anchor_id:02d}: {count}/{total} ({100 * count / total:.2f}%)")


def _maybe_write_debug(
    cache: VisibilityCache,
    object_id: str,
    anchor_id: int | None,
    debug_root: Path,
) -> None:
    if anchor_id is None:
        return
    canonical_anchors().by_id(anchor_id)
    category, instance = object_id.split("/")
    path = debug_root / category / instance / f"anchor_{anchor_id}.svg"
    write_visibility_debug_svg(cache, anchor_id, path)
    print(f"  debug visualization: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
