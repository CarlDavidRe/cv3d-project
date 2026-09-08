#!/usr/bin/env python3
"""Build deterministic Phase 3 direct surface-gain history shards."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import (  # noqa: E402
    ConfigError,
    load_config,
    validate_artifact_path,
)
from nbv.data import (  # noqa: E402
    HISTORY_SAMPLING_STRATEGY,
    HistoryDatasetError,
    build_history_dataset,
)
from nbv.geometry import (  # noqa: E402
    CAMERA_CONVENTION,
    CANONICAL_ORDERING,
    FACE_VISIBILITY_RENDERER,
)
from nbv.geometry.mesh import MESH_CENTERING, sha256_file  # noqa: E402
from nbv.reproducibility import seed_everything  # noqa: E402


_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase3_histories.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        default=[],
        help="Build only this split; may be repeated.",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        metavar="CATEGORY/OBJECT",
        help="Build only explicit objects from the selected split or splits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit each selected split after canonical sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing generated dataset directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = _repository_path(args.config)
        config = load_config(config_path, args.set)
        settings = _history_settings(config)
        if config["experiment"]["phase"] != "phase3":
            raise HistoryDatasetError(
                "history dataset generation requires experiment.phase=phase3"
            )
        visibility_config_path = _repository_path(
            settings["visibility_config"]
        )
        evaluation_config_path = _repository_path(
            settings["evaluation_config"]
        )
        visibility_config = load_config(visibility_config_path)
        evaluation_config = load_config(evaluation_config_path)
        geometry = _geometry_metadata(visibility_config)
        _validate_shared_protocol(
            config,
            settings,
            visibility_config,
            evaluation_config,
        )
        paths = config["paths"]
        data_root = _repository_path(paths["data_root"])
        cache_root = _repository_path(paths["visibility_cache_root"])
        split_manifest = _repository_path(settings["split_manifest"])
        output_root = _repository_path(paths["history_dataset_root"])
        validate_artifact_path(
            output_root, "paths.history_dataset_root"
        )
        output_dir = output_root / settings["dataset_name"]
        splits = args.split or settings["splits"]
        seed_everything(
            config["experiment"]["seed"],
            config["experiment"]["deterministic"],
        )
        manifest = build_history_dataset(
            output_dir,
            data_root=data_root,
            visibility_cache_root=cache_root,
            split_manifest=split_manifest,
            splits=splits,
            history_lengths=settings["history_lengths"],
            histories_per_object_per_length=settings[
                "histories_per_object_per_length"
            ],
            seed=config["experiment"]["seed"],
            coverage_target=settings["coverage_target"],
            invalid_anchor_ids=settings["invalid_anchor_ids"],
            expected_visibility_metadata=geometry,
            provenance={
                "history_config": str(config_path),
                "history_config_sha256": sha256_file(config_path),
                "visibility_config": str(visibility_config_path),
                "visibility_config_sha256": sha256_file(
                    visibility_config_path
                ),
                "evaluation_config": str(evaluation_config_path),
                "evaluation_config_sha256": sha256_file(
                    evaluation_config_path
                ),
                "split_manifest": str(split_manifest),
                "resolved_config": config,
            },
            object_ids=args.object or None,
            limit_per_split=(
                args.limit
                if args.limit is not None
                else settings["limit_per_split"]
            ),
            overwrite=args.overwrite,
        )
    except (
        ConfigError,
        HistoryDatasetError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"History dataset error: {exc}", file=sys.stderr)
        return 2
    print(f"WROTE Phase 3 history dataset: {manifest}")
    return 0


def _history_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    phase3 = config.get("phase3")
    if not isinstance(phase3, Mapping):
        raise HistoryDatasetError("phase3 must be a mapping")
    histories = phase3.get("histories")
    if not isinstance(histories, Mapping):
        raise HistoryDatasetError("phase3.histories must be a mapping")
    raw = dict(histories)
    required = {
        "dataset_name",
        "split_manifest",
        "visibility_config",
        "evaluation_config",
        "splits",
        "history_lengths",
        "histories_per_object_per_length",
        "coverage_target",
        "invalid_anchor_ids",
        "sampling_strategy",
        "rotation_augmentation",
        "limit_per_split",
    }
    missing = required - set(raw)
    if missing:
        raise HistoryDatasetError(
            f"phase3.histories is missing {sorted(missing)}"
        )
    if (
        not isinstance(raw["dataset_name"], str)
        or not _SAFE_NAME.fullmatch(raw["dataset_name"])
    ):
        raise HistoryDatasetError(
            "dataset_name must be a safe lowercase artifact name"
        )
    for key in ("split_manifest", "visibility_config", "evaluation_config"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise HistoryDatasetError(f"{key} must be a non-empty path")
    if raw["sampling_strategy"] != HISTORY_SAMPLING_STRATEGY:
        raise HistoryDatasetError(
            f"sampling_strategy must be {HISTORY_SAMPLING_STRATEGY!r}"
        )
    if raw["rotation_augmentation"] is not False:
        raise HistoryDatasetError(
            "Step 15 requires non-rotated data; rotations are a later ablation"
        )
    if not isinstance(raw["invalid_anchor_ids"], list):
        raise HistoryDatasetError("invalid_anchor_ids must be a list")
    if raw["limit_per_split"] is not None and (
        isinstance(raw["limit_per_split"], bool)
        or not isinstance(raw["limit_per_split"], int)
        or raw["limit_per_split"] <= 0
    ):
        raise HistoryDatasetError(
            "limit_per_split must be a positive integer or null"
        )
    return raw


def _geometry_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    visibility = config.get("phase2", {}).get("visibility")
    if not isinstance(visibility, Mapping):
        raise HistoryDatasetError(
            "visibility config must contain phase2.visibility"
        )
    required = {
        "mesh_relative_path",
        "mesh_scale",
        "mesh_centering",
        "camera_radius",
        "horizontal_fov_degrees",
        "render_resolution",
        "near",
        "far",
        "cull_backfaces",
    }
    missing = required - set(visibility)
    if missing:
        raise HistoryDatasetError(
            f"visibility config is missing {sorted(missing)}"
        )
    if visibility["mesh_centering"] != MESH_CENTERING:
        raise HistoryDatasetError(
            f"mesh_centering must be {MESH_CENTERING!r}"
        )
    return {
        "schema_version": 2,
        "mesh_relative_path": visibility["mesh_relative_path"],
        "mesh_scale": float(visibility["mesh_scale"]),
        "mesh_centering": visibility["mesh_centering"],
        "camera_radius": float(visibility["camera_radius"]),
        "horizontal_fov_degrees": float(
            visibility["horizontal_fov_degrees"]
        ),
        "render_resolution": list(visibility["render_resolution"]),
        "near": float(visibility["near"]),
        "far": float(visibility["far"]),
        "cull_backfaces": visibility["cull_backfaces"],
        "anchor_count": 48,
        "anchor_ordering": CANONICAL_ORDERING,
        "renderer": FACE_VISIBILITY_RENDERER,
        "camera_convention": CAMERA_CONVENTION,
        "visibility_definition": (
            "pun_unoccluded_rasterized_mesh_faces_v1"
        ),
        "available_visibility_targets": ["vis", "vis_a"],
    }


def _validate_shared_protocol(
    history_config: Mapping[str, Any],
    settings: Mapping[str, Any],
    visibility_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> None:
    evaluation = evaluation_config.get("phase2", {}).get("evaluation")
    visibility = visibility_config.get("phase2", {}).get("visibility")
    if not isinstance(evaluation, Mapping):
        raise HistoryDatasetError(
            "evaluation config must contain phase2.evaluation"
        )
    if not isinstance(visibility, Mapping):
        raise HistoryDatasetError(
            "visibility config must contain phase2.visibility"
        )
    if settings["coverage_target"] != evaluation.get("coverage_target"):
        raise HistoryDatasetError(
            "Phase 3 coverage_target must equal the Phase 2 evaluator target"
        )
    if settings["coverage_target"] != visibility.get("target"):
        raise HistoryDatasetError(
            "Phase 3 coverage_target must equal the visibility config target"
        )
    history_split = _repository_path(settings["split_manifest"]).resolve()
    for owner, configured in (
        ("visibility", visibility.get("split_manifest")),
        ("evaluation", evaluation.get("split_manifest")),
    ):
        if _repository_path(configured).resolve() != history_split:
            raise HistoryDatasetError(
                f"Phase 3 and Phase 2 {owner} split manifests differ"
            )
    history_data = _repository_path(history_config["paths"]["data_root"])
    for owner, other_config in (
        ("visibility", visibility_config),
        ("evaluation", evaluation_config),
    ):
        other_data = _repository_path(other_config["paths"]["data_root"])
        if other_data.resolve() != history_data.resolve():
            raise HistoryDatasetError(
                f"Phase 3 and Phase 2 {owner} data roots differ"
            )
    history_cache = _repository_path(
        history_config["paths"]["visibility_cache_root"]
    )
    for owner, other_config in (
        ("visibility", visibility_config),
        ("evaluation", evaluation_config),
    ):
        other_cache = _repository_path(
            other_config["paths"]["visibility_cache_root"]
        )
        if other_cache.resolve() != history_cache.resolve():
            raise HistoryDatasetError(
                f"Phase 3 and Phase 2 {owner} visibility cache roots differ"
            )
    configured_visibility = _repository_path(
        evaluation_config["phase2"]["visibility_config"]
    )
    requested_visibility = _repository_path(settings["visibility_config"])
    if configured_visibility.resolve() != requested_visibility.resolve():
        raise HistoryDatasetError(
            "Phase 3 and Phase 2 evaluation visibility configs differ"
        )


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
