from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        print(f"skip: {source.relative_to(REPO_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"copy: {source.relative_to(REPO_ROOT)}")


def copy_matching_files(source_root: Path, destination_root: Path, pattern: str) -> None:
    if not source_root.exists():
        return

    for source in source_root.glob(pattern):
        relative = source.relative_to(source_root)
        copy_file(source, destination_root / relative)


def compact_rollouts(source_root: Path, destination_root: Path) -> None:
    rollout_root = source_root / "rollouts"
    if not rollout_root.exists():
        return

    for source in rollout_root.glob("*/*/*.npz"):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        with np.load(source, allow_pickle=False) as payload:
            np.savez_compressed(
                destination,
                metadata_json=payload["metadata_json"],
                steps_json=payload["steps_json"],
                acquired_anchor_ids=payload["acquired_anchor_ids"],
                acquired_view_counts=payload["acquired_view_counts"],
                coverage=payload["coverage"],
            )

        print(f"compact: {source.relative_to(REPO_ROOT)}")


def export_closed_loop_root(source_root: Path, destination_root: Path) -> None:
    for filename in (
        "comparison.csv",
        "closed_loop_comparison.csv",
        "coverage.csv",
        "per_step.csv",
        "reconstruction_curves.csv",
    ):
        copy_file(
            source_root / "metrics" / filename,
            destination_root / "metrics" / filename,
        )

    compact_rollouts(source_root, destination_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    destination = args.destination.resolve()

    if destination.exists():
        raise SystemExit(
            f"Destination already exists: {destination}\n"
            "Delete it or choose another folder."
        )

    print(f"Creating lightweight dashboard bundle at:\n{destination}\n")

    # Dashboard application and theme.
    copy_file(
        REPO_ROOT / "dashboard" / "app.py",
        destination / "dashboard" / "app.py",
    )
    copy_file(
        REPO_ROOT / "dashboard" / "requirements.txt",
        destination / "dashboard" / "requirements.txt",
    )
    copy_file(
        REPO_ROOT / ".streamlit" / "config.toml",
        destination / ".streamlit" / "config.toml",
    )

    # Small static dashboard inputs.
    copy_file(
        REPO_ROOT / "dashboard" / "data" / "phase1_test_predictions.npz",
        destination / "dashboard" / "data" / "phase1_test_predictions.npz",
    )
    copy_file(
        REPO_ROOT / "dashboard_rendering_objects" / "README.md",
        destination / "dashboard_rendering_objects" / "README.md",
    )
    copy_file(
        REPO_ROOT / "data" / "splits" / "num_v1.json",
        destination / "data" / "splits" / "num_v1.json",
    )
    copy_file(
        REPO_ROOT / "src" / "nbv" / "geometry" / "anchors_v1.csv",
        destination / "src" / "nbv" / "geometry" / "anchors_v1.csv",
    )

    # Phase 1 only needs experiment summaries.
    phase1_source = REPO_ROOT / "outputs" / "phase1" / "backbone_sweep"
    phase1_destination = destination / "outputs" / "phase1" / "backbone_sweep"
    copy_matching_files(
        phase1_source,
        phase1_destination,
        "seed_*/variants/*/summary.json",
    )

    # Phase 2.
    phase2_relative = (
        Path("outputs")
        / "phase2"
        / "phase2_closed_loop_reconstruction"
        / "seed_0"
    )
    export_closed_loop_root(
        REPO_ROOT / phase2_relative,
        destination / phase2_relative,
    )

    # Phase 3 variants.
    for experiment in (
        "controlled_history_comparison_reconstruction",
        "controlled_pose_deepsets",
        "controlled_token_attention",
    ):
        relative = Path("outputs") / "phase3" / experiment / "seed_0"
        export_closed_loop_root(
            REPO_ROOT / relative,
            destination / relative,
        )

    # Small 2DGS metric files used by the closed-loop comparison charts.
    gs_relative = Path("outputs") / "gaussian_splatting_cpu_recovery"
    gs_source = REPO_ROOT / gs_relative
    gs_destination = destination / gs_relative

    copy_matching_files(
        gs_source,
        gs_destination,
        "*/*/*views/phase*/2dgs/metrics.csv",
    )
    copy_matching_files(
        gs_source,
        gs_destination,
        "*/*/*views/phase*/2dgs/summary.json",
    )

    readme = destination / "README.txt"
    readme.write_text(
        "CV3D lightweight dashboard bundle\n"
        "================================\n\n"
        "Run from this directory with:\n\n"
        "    streamlit run dashboard/app.py\n\n"
        "This lightweight bundle contains metrics, compact rollout replays, "
        "and dashboard data.\n"
        "The full NUM image dataset, ShapeNet meshes, checkpoints, and large "
        "reconstruction viewers are intentionally omitted.\n",
        encoding="utf-8",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()