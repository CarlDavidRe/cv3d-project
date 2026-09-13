#!/usr/bin/env python3
"""Train/evaluate 2DGS geometry and render qualitative 3DGS for one cached history."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.gaussian_splatting import (
    GaussianSplatSettings,
    evaluate_2dgs_geometry,
    known_num_splat_cameras,
    load_splat_images,
    orbit_camera_poses,
    render_splats,
    require_gsplat,
    save_gaussian_checkpoint,
    train_gaussian_splats,
    write_point_ply,
    write_image_comparison_gallery,
    write_render_gallery,
    write_summary,
)
from nbv.eval.reconstruction import align_vggt_to_num
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world
from nbv.geometry.visibility import PerspectiveCamera

from scripts.visualize_reconstruction import (
    GT_COLOR,
    PREDICTION_COLOR,
    cached_path,
    find_ground_truth,
    load_prediction,
    mapping_sha256,
    parse_history,
    select_row,
    validate_object_id,
    write_interactive_html,
    write_ply,
    write_preview,
)


def resolve_metric_cache(
    root: Path,
    object_id: str,
    recorded: str,
    prediction_identity: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    """Resolve a metric cache, tolerating a stale hash recorded in an old CSV."""

    try:
        path = cached_path(root, "metrics", object_id, recorded)
    except ValueError as recorded_error:
        prediction_id = mapping_sha256(prediction_identity)
        matches: list[tuple[Path, dict[str, object]]] = []
        directory = root / "metrics" / object_id
        for candidate in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            identity = payload.get("identity")
            if (
                isinstance(identity, dict)
                and identity.get("prediction_id") == prediction_id
                and isinstance(identity.get("target_id"), str)
            ):
                matches.append((candidate, payload))
        if not matches:
            raise recorded_error
        target_ids = {match[1]["identity"]["target_id"] for match in matches}
        if len(target_ids) != 1:
            raise ValueError(
                "compatible metric caches disagree on the ground-truth target for "
                f"prediction {prediction_id}"
            )
        return matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="reconstruction_per_object.csv")
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--views", type=int)
    parser.add_argument(
        "--backend", choices=("2dgs", "3dgs", "both"), default="both",
        help="2DGS writes geometry metrics; 3DGS writes visualization only",
    )
    parser.add_argument("--iterations", type=int, default=1_500)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--position-learning-rate", type=float, default=2e-4)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--surface-point-count", type=int, default=10_000)
    parser.add_argument("--preview-points", type=int, default=5_000)
    parser.add_argument("--interactive-points", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/NUM")
    parser.add_argument(
        "--reconstruction-cache-root", type=Path,
        default=ROOT / "data/cache/reconstruction",
    )
    parser.add_argument(
        "--visibility-cache-root", type=Path,
        default=ROOT / "data/cache/visibility",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        backends = ("2dgs", "3dgs") if args.backend == "both" else (args.backend,)
        for backend in backends:
            require_gsplat(backend)
        object_id = validate_object_id(args.object_id)
        row = select_row(args.metrics, object_id, args.policy, args.views)
        history = parse_history(row["history_anchor_ids"])
        prediction_path = cached_path(
            args.reconstruction_cache_root, "predictions", object_id,
            row["prediction_cache_path"],
        )
        raw_points, predicted_cameras, identity = load_prediction(prediction_path)
        metric_path, metric_payload = resolve_metric_cache(
            args.reconstruction_cache_root,
            object_id,
            row["metric_cache_path"],
            identity,
        )
        if identity.get("history_anchor_ids") != history:
            raise ValueError("prediction cache history does not match the metric row")
        target_path, target_points = find_ground_truth(
            args.reconstruction_cache_root,
            object_id,
            str(metric_payload["identity"]["target_id"]),
        )
        visibility_path = (
            args.visibility_cache_root / object_id.split("/")[0]
            / f"{object_id.split('/')[1]}.npz"
        )
        visibility = load_visibility_cache(visibility_path)
        metadata = visibility.metadata
        radius = float(metadata["camera_radius"])
        convention = str(metadata.get("camera_convention", CAMERA_CONVENTION))
        anchors = canonical_anchors()
        training_poses = np.stack([
            anchor_camera_to_world(anchors.by_id(anchor), radius, convention=convention)
            for anchor in history
        ])
        all_poses = np.stack([
            anchor_camera_to_world(anchor, radius, convention=convention)
            for anchor in anchors
        ])
        transform, alignment = align_vggt_to_num(
            raw_points, predicted_cameras, training_poses, target_points
        )
        initial_points = raw_points @ transform[:3, :3].T + transform[:3, 3]
        camera = PerspectiveCamera(
            height=args.resolution,
            width=args.resolution,
            horizontal_fov_degrees=float(metadata["horizontal_fov_degrees"]),
            near=float(metadata["near"]),
            far=float(metadata["far"]),
        )
        training_cameras = known_num_splat_cameras(training_poses, camera)
        evaluation_cameras = known_num_splat_cameras(all_poses, camera)
        gallery_cameras = known_num_splat_cameras(
            orbit_camera_poses(args.render_frames, radius), camera
        )
        store = ObservationStore.from_num_object(args.data_root, object_id)
        image_paths = [store.acquire(anchor).image_path for anchor in history]
        all_image_paths = [store.acquire(anchor.anchor_id).image_path for anchor in anchors]
        view_count = int(row["acquired_view_count"])
        category_id, object_key = object_id.split("/", 1)
        output = args.output_dir or (
            args.metrics.parent / "gaussian_splatting" / category_id / object_key
            / f"{args.policy}_{view_count}views"
        )
        output.mkdir(parents=True, exist_ok=True)
        common = {
            "schema_version": 1,
            "object_id": object_id,
            "policy": args.policy,
            "acquired_view_count": view_count,
            "history_anchor_ids": history,
            "initialization": "camera_aligned_vggt_filtered_point_map",
            "alignment": alignment,
            "prediction_cache": str(prediction_path.resolve()),
            "ground_truth_cache": str(target_path.resolve()),
            "camera_source": "known_num_cameras",
        }
        for backend in backends:
            settings = GaussianSplatSettings(
                backend=backend,
                iterations=args.iterations,
                resolution=args.resolution,
                learning_rate=args.learning_rate,
                position_learning_rate=args.position_learning_rate,
                alpha_threshold=args.alpha_threshold,
                surface_point_count=args.surface_point_count,
                seed=args.seed,
            )
            destination = output / backend
            destination.mkdir(parents=True, exist_ok=True)
            model, training_history = train_gaussian_splats(
                initial_points,
                image_paths,
                training_cameras,
                settings,
            )
            save_gaussian_checkpoint(
                destination / "checkpoint.pt", model, settings, training_history
            )
            renders, _, _ = render_splats(
                model, gallery_cameras, settings, include_depth=False
            )
            write_render_gallery(
                destination / "turntable.html",
                renders,
                f"{backend.upper()} · {object_id} · {args.policy} · {view_count} views",
            )
            summary = {
                **common,
                "backend": backend,
                "settings": asdict(settings),
                "training_history": training_history,
                "turntable": str((destination / "turntable.html").resolve()),
                "geometry_evaluation_enabled": backend == "2dgs",
            }
            if backend == "2dgs":
                surface, metrics = evaluate_2dgs_geometry(
                    model, evaluation_cameras, target_points, settings
                )
                write_point_ply(destination / "surface.ply", surface, (234, 88, 12))
                write_ply(destination / "ground_truth.ply", target_points, GT_COLOR)
                write_ply(
                    destination / "comparison.ply",
                    np.concatenate((target_points, surface)),
                    (GT_COLOR,) * len(target_points)
                    + (PREDICTION_COLOR,) * len(surface),
                )
                write_preview(
                    destination / "comparison.png",
                    target_points,
                    surface,
                    max_points=args.preview_points,
                    title=(
                        "Ground truth (blue) vs 2DGS depth-fused surface (orange)"
                    ),
                )
                write_interactive_html(
                    destination / "comparison_interactive.html",
                    target_points,
                    surface,
                    max_points=args.interactive_points,
                    title=(
                        f"{object_id} · {args.policy} · {view_count} views · "
                        "2DGS surface vs ground truth"
                    ),
                )
                summary.update({
                    "surface_extraction": (
                        "median 2DGS depth from all 48 canonical NUM cameras; "
                        "alpha filtered, voxel fused, deterministically sampled"
                    ),
                    "surface_point_count": len(surface),
                    "metrics": metrics,
                    "metric_status": "evaluation_candidate_without_ground_truth_alignment",
                    "ground_truth_comparison": {
                        "static": str((destination / "comparison.png").resolve()),
                        "interactive": str(
                            (destination / "comparison_interactive.html").resolve()
                        ),
                        "combined_point_cloud": str(
                            (destination / "comparison.ply").resolve()
                        ),
                        "colors": {
                            "ground_truth": list(GT_COLOR),
                            "surface": list(PREDICTION_COLOR),
                        },
                    },
                })
                metric_row = {
                    "object_id": object_id,
                    "policy": args.policy,
                    "acquired_view_count": view_count,
                    "backend": "2dgs",
                    **metrics,
                }
                with (destination / "metrics.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(metric_row))
                    writer.writeheader()
                    writer.writerow(metric_row)
            else:
                canonical_renders, _, _ = render_splats(
                    model, evaluation_cameras, settings, include_depth=False
                )
                reference_images, _ = load_splat_images(
                    all_image_paths, args.resolution, 245 / 255
                )
                write_image_comparison_gallery(
                    destination / "ground_truth_comparison.html",
                    reference_images,
                    canonical_renders,
                    f"3DGS vs NUM RGB · {object_id} · {args.policy} · {view_count} views",
                    training_view_ids=history,
                )
                summary["metric_status"] = (
                    "disabled_by_contract_3dgs_is_qualitative_visualization_only"
                )
                summary["ground_truth_comparison"] = {
                    "type": "canonical_render_vs_corresponding_num_rgb",
                    "interactive": str(
                        (destination / "ground_truth_comparison.html").resolve()
                    ),
                    "training_view_ids": history,
                    "held_out_view_ids": [
                        anchor.anchor_id for anchor in anchors
                        if anchor.anchor_id not in history
                    ],
                }
            write_summary(destination / "summary.json", summary)
            print(f"{backend.upper()} output: {destination}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Gaussian splatting error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
