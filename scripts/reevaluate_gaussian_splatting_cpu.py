#!/usr/bin/env python3
"""Recover 2DGS geometry from saved checkpoints on CPU, without retraining."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.gaussian_splatting import (
    GaussianParameters, GaussianSplatSettings, fuse_depth_surfaces,
    known_num_splat_cameras, write_point_ply, write_summary,
)
from nbv.eval.gaussian_splatting_cpu import RENDERER_VERSION, render_depth_cpu
from nbv.eval.reconstruction import point_cloud_metrics
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world
from nbv.geometry.visibility import PerspectiveCamera
from scripts.visualize_reconstruction import (
    GT_COLOR, PREDICTION_COLOR, validate_object_id, write_interactive_html,
    write_ply, write_preview,
)

OUTPUT_FILES = ("surface.ply", "ground_truth.ply", "comparison.ply", "comparison.png",
                "comparison_interactive.html", "metrics.csv", "summary.json")


def read_point_ply(path: Path) -> np.ndarray:
    """Read the ASCII point-cloud artifacts written by this project."""
    with path.open(encoding="ascii") as handle:
        if handle.readline().strip() != "ply" or handle.readline().strip() != "format ascii 1.0":
            raise ValueError(f"expected ASCII PLY: {path}")
        count = None
        for line in handle:
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        points = np.loadtxt(handle, usecols=(0, 1, 2), ndmin=2)
    if count != len(points) or not len(points) or not np.isfinite(points).all():
        raise ValueError(f"invalid point cloud: {path}")
    return points


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def recover(source: Path, destination: Path, visibility_root: Path, *, force: bool = False,
            dry_run: bool = False) -> str:
    summary = json.loads((source / "summary.json").read_text())
    if summary["backend"] != "2dgs":
        raise ValueError(f"not a 2DGS artifact: {source}")
    category, key = validate_object_id(summary["object_id"]).split("/")
    visibility_path = visibility_root / category / f"{key}.npz"
    identity = {"renderer": RENDERER_VERSION, "inputs": {
        name: digest(path) for name, path in (
            ("checkpoint", source / "checkpoint.pt"),
            ("ground_truth", source / "ground_truth.ply"),
            ("summary", source / "summary.json"),
            ("visibility", visibility_path),
        )
    }}
    if not force and all((destination / name).is_file() for name in OUTPUT_FILES):
        previous = json.loads((destination / "summary.json").read_text())
        if previous.get("cpu_recovery", {}).get("identity") == identity:
            return "skipped (matching completed recovery)"
    target = read_point_ply(source / "ground_truth.ply")
    metadata = load_visibility_cache(visibility_path).metadata
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    settings = GaussianSplatSettings(**checkpoint["settings"])
    if settings.backend != "2dgs":
        raise ValueError("checkpoint is not 2DGS")
    model = GaussianParameters(checkpoint["state_dict"]["means"].numpy(),
                               np.full(checkpoint["state_dict"]["means"].shape, 0.5), "2dgs")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    poses = np.stack([
        anchor_camera_to_world(anchor, float(metadata["camera_radius"]),
                               convention=str(metadata.get("camera_convention", CAMERA_CONVENTION)))
        for anchor in canonical_anchors()
    ])
    cameras = known_num_splat_cameras(poses, PerspectiveCamera(
        height=settings.resolution, width=settings.resolution,
        horizontal_fov_degrees=float(metadata["horizontal_fov_degrees"]),
        near=float(metadata["near"]), far=float(metadata["far"]),
    ))
    if dry_run:
        return f"ready ({len(model.means)} splats, {len(poses)} cameras, {settings.resolution}px)"
    start = time.monotonic()

    def progress(done: int, total: int) -> None:
        if done == 1 or done % 8 == 0 or done == total:
            print(f"  cameras {done}/{total}, {time.monotonic()-start:.1f}s", flush=True)

    with torch.inference_mode():
        alpha, depth = render_depth_cpu(model, cameras, settings.resolution, progress=progress)
    surface = fuse_depth_surfaces(depth, alpha, cameras,
        alpha_threshold=settings.alpha_threshold, point_count=settings.surface_point_count,
        seed=settings.seed)
    metrics = point_cloud_metrics(surface, target, chunk_size=settings.metric_chunk_size,
                                  fscore_thresholds=settings.fscore_thresholds, device="cpu")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the summary last: a failed run cannot be mistaken for completion.
    with tempfile.TemporaryDirectory(prefix=".cpu-recovery-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        write_point_ply(stage / "surface.ply", surface, (234, 88, 12))
        write_ply(stage / "ground_truth.ply", target, GT_COLOR)
        write_ply(stage / "comparison.ply", np.concatenate((target, surface)),
                  (GT_COLOR,)*len(target) + (PREDICTION_COLOR,)*len(surface))
        title = "Ground truth (blue) vs CPU 2DGS median-depth surface (orange)"
        write_preview(stage / "comparison.png", target, surface, max_points=5000, title=title)
        write_interactive_html(stage / "comparison_interactive.html", target, surface,
                               max_points=8000, title=title)
        row = {key: summary[key] for key in ("object_id", "policy", "acquired_view_count", "backend")}
        row.update(metrics)
        with (stage / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        summary.update({
            "settings": checkpoint["settings"], "metrics": metrics,
            "surface_point_count": len(surface),
            "surface_extraction": "CPU 2DGS median center-z depth from all canonical NUM cameras; alpha filtered and voxel fused",
            "cpu_recovery": {"identity": identity, "source": str(source.resolve()),
                             "elapsed_seconds": time.monotonic()-start,
                             "cuda_parity_verified": False},
            "ground_truth_comparison": {
                "static": str((destination / "comparison.png").resolve()),
                "interactive": str((destination / "comparison_interactive.html").resolve()),
                "combined_point_cloud": str((destination / "comparison.ply").resolve()),
                "colors": {"ground_truth": list(GT_COLOR), "surface": list(PREDICTION_COLOR)},
            },
            "turntable": str((source / "turntable.html").resolve()),
        })
        write_summary(stage / "summary.json", summary)
        destination.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            os.replace(stage / name, destination / name)
    return f"complete: F1@1%={metrics['fscore_1pct']:.4f}, F1@10%={metrics['fscore_10pct']:.4f}, {time.monotonic()-start:.1f}s"


def write_aggregate(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("**/2dgs/metrics.csv")):
        summary_path = path.parent / "summary.json"
        if not summary_path.is_file() or "cpu_recovery" not in json.loads(summary_path.read_text()):
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"variant": path.parent.parent.name, **row})
    if rows:
        path = root / "recovered_metrics.csv"
        with path.with_suffix(".csv.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(path.with_suffix(".csv.tmp"), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "outputs/gaussian_splatting_per_view_budget")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/gaussian_splatting_per_view_budget_cpu_recovery")
    parser.add_argument("--visibility-cache-root", type=Path, default=ROOT / "data/cache/visibility")
    parser.add_argument("--object-id", help="Only this category/object")
    parser.add_argument("--variant", help="Only this folder, e.g. phase2_vggt")
    parser.add_argument("--views", nargs="+", type=int)
    parser.add_argument("--limit", type=int, help="Process at most this many matching artifacts")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--force", action="store_true", help="Recompute matching completed recoveries")
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs without rendering or writing")
    args = parser.parse_args(argv)
    if args.threads <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("threads and limit must be positive")
    source_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        parser.error("input and output roots must be separate, non-nested directories")
    torch.set_num_threads(args.threads)
    sources = []
    for path in sorted(source_root.glob("**/2dgs/summary.json")):
        summary = json.loads(path.read_text())
        if args.object_id and summary.get("object_id") != args.object_id:
            continue
        if args.variant and path.parent.parent.name != args.variant:
            continue
        if args.views and summary.get("acquired_view_count") not in args.views:
            continue
        sources.append(path.parent)
    if args.limit is not None:
        sources = sources[:args.limit]
    if not sources:
        print("No matching 2DGS summaries found", file=sys.stderr)
        return 1
    failures = 0
    for index, source in enumerate(sources, 1):
        relative = source.relative_to(source_root)
        print(f"[{index}/{len(sources)}] {relative}", flush=True)
        try:
            print("  " + recover(source, output_root / relative, args.visibility_cache_root,
                                 force=args.force, dry_run=args.dry_run), flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
    if not args.dry_run:
        write_aggregate(output_root)
    print(f"Finished: {len(sources)-failures} successful/ready/skipped, {failures} failed", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
