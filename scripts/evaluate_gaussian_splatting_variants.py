#!/usr/bin/env python3
"""Render one object for every Phase 2/3 variant at incremental view counts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWS = (1, 2, 3, 5, 10)
PHASE2_POLICIES = ("random", "farthest", "pun", "vggt", "oracle")
PHASE3_POLICIES = (
    "vggt_independent_history",
    "vggt_joint_history",
    "vggt_joint_pose_deepsets",
    "vggt_joint_token_attention",
)


@dataclass(frozen=True, slots=True)
class Variant:
    phase: str
    policy: str
    metrics: Path

    @property
    def key(self) -> str:
        return f"{self.phase}_{self.policy}"


@dataclass(frozen=True, slots=True)
class RunResult:
    variant: str
    policy: str
    views: int
    output: str
    status: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True, help="ShapeNet category/object ID")
    parser.add_argument(
        "--views", type=int, nargs="+", default=list(DEFAULT_VIEWS),
        help="Incremental acquired-view counts (default: 1 2 3 5 10)",
    )
    parser.add_argument(
        "--backend", choices=("2dgs", "3dgs", "both"), default="both",
    )
    parser.add_argument(
        "--iterations", type=int, default=1_500, metavar="PER_VIEW",
        help="Optimization iterations per acquired view (default: 1500)",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--phase2-root", type=Path, default=ROOT / "outputs/phase2")
    parser.add_argument("--phase3-root", type=Path, default=ROOT / "outputs/phase3")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/gaussian_splatting_variant_comparison",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/NUM")
    parser.add_argument(
        "--reconstruction-cache-root", type=Path,
        default=ROOT / "data/cache/reconstruction",
    )
    parser.add_argument(
        "--visibility-cache-root", type=Path,
        default=ROOT / "data/cache/visibility",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even when all requested backend summaries already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print commands without starting CUDA training",
    )
    return parser


def normalize_views(values: Sequence[int]) -> tuple[int, ...]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("--views must contain one or more positive integers")
    return tuple(sorted(set(values)))


def _metrics_candidates(root: Path) -> list[Path]:
    return sorted(root.glob("**/metrics/reconstruction_per_object.csv"))


def _policies_in(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row.get("policy", "") for row in csv.DictReader(handle)}


def discover_variants(phase2_root: Path, phase3_root: Path) -> list[Variant]:
    """Resolve each logical policy once, preferring the shallowest metrics file."""

    discovered: list[Variant] = []
    for phase, root, policies in (
        ("phase2", phase2_root, PHASE2_POLICIES),
        ("phase3", phase3_root, PHASE3_POLICIES),
    ):
        candidates = _metrics_candidates(root)
        policy_sets = {path: _policies_in(path) for path in candidates}
        for policy in policies:
            matches = [path for path in candidates if policy in policy_sets[path]]
            if not matches:
                raise ValueError(
                    f"could not find policy {policy!r} below {root}; "
                    "run the corresponding reconstruction evaluation first"
                )
            selected = min(
                matches,
                key=lambda path: (len(path.relative_to(root).parts), str(path)),
            )
            discovered.append(Variant(phase, policy, selected))
    return discovered


def available_object_views(path: Path, object_id: str, policy: str) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["acquired_view_count"])
            for row in csv.DictReader(handle)
            if row.get("object_id") == object_id and row.get("policy") == policy
        }


def validate_inputs(
    variants: Sequence[Variant], object_id: str, views: Sequence[int]
) -> None:
    if len(object_id.split("/")) != 2 or any(
        part in {"", ".", ".."} for part in object_id.split("/")
    ):
        raise ValueError("object-id must have safe 'category/object' form")
    requested = set(views)
    missing = []
    for variant in variants:
        absent = sorted(requested - available_object_views(
            variant.metrics, object_id, variant.policy
        ))
        if absent:
            missing.append(f"{variant.key}: {absent}")
    if missing:
        raise ValueError(
            f"missing reconstruction histories for object {object_id!r}: "
            + "; ".join(missing)
        )


def object_output_root(output_root: Path, object_id: str) -> Path:
    """Mirror NUM's category/object hierarchy below an output root."""

    category_id, object_key = object_id.split("/", 1)
    return output_root / category_id / object_key


def backend_summaries(output: Path, backend: str) -> tuple[Path, ...]:
    names = ("2dgs", "3dgs") if backend == "both" else (backend,)
    return tuple(output / name / "summary.json" for name in names)


def summary_has_training_budget(path: Path, iterations_per_view: int,
                                view_count: int) -> bool:
    """Only resume checkpoints trained with the requested per-view budget."""
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("training_budget") == {
        "iterations_per_view": iterations_per_view,
        "view_count": view_count,
        "total_iterations": iterations_per_view * view_count,
    }


def build_command(
    variant: Variant,
    object_id: str,
    views: int,
    output: Path,
    args: argparse.Namespace,
    backend: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/evaluate_gaussian_splatting.py"),
        str(variant.metrics),
        "--object-id", object_id,
        "--policy", variant.policy,
        "--views", str(views),
        "--backend", backend,
        "--iterations", str(args.iterations),
        "--resolution", str(args.resolution),
        "--render-frames", str(args.render_frames),
        "--data-root", str(args.data_root),
        "--reconstruction-cache-root", str(args.reconstruction_cache_root),
        "--visibility-cache-root", str(args.visibility_cache_root),
        "--output-dir", str(output),
    ]


def write_index(
    path: Path,
    object_id: str,
    variants: Sequence[Variant],
    views: Sequence[int],
    backend: str,
) -> None:
    """Write a compact matrix linking every generated interactive viewer."""

    root = path.parent
    backend_names = ("2dgs", "3dgs") if backend == "both" else (backend,)
    cells = []
    for view_count in views:
        row = [f"<th>{view_count}</th>"]
        for variant in variants:
            destination = root / f"{view_count}views" / variant.key
            links = []
            if "2dgs" in backend_names:
                links.extend((
                    (destination / "2dgs/comparison_interactive.html", "2DGS vs GT"),
                    (destination / "2dgs/turntable.html", "2DGS turntable"),
                ))
            if "3dgs" in backend_names:
                links.extend((
                    (destination / "3dgs/ground_truth_comparison.html", "3DGS vs GT"),
                    (destination / "3dgs/turntable.html", "3DGS turntable"),
                ))
            rendered = []
            for target, label in links:
                relative = target.relative_to(root).as_posix()
                if target.is_file():
                    rendered.append(f'<a href="{html.escape(relative)}">{label}</a>')
                else:
                    rendered.append(f'<span class="missing">{label}</span>')
            row.append("<td>" + "<br>".join(rendered) + "</td>")
        cells.append("<tr>" + "".join(row) + "</tr>")
    headings = "".join(
        f"<th>{html.escape(variant.key)}</th>" for variant in variants
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Gaussian-splatting variant comparison</title><style>
body{{font:14px system-ui;margin:24px;background:#f8fafc;color:#0f172a}}table{{border-collapse:collapse}}
th,td{{border:1px solid #cbd5e1;padding:9px;text-align:left;vertical-align:top}}th{{background:#e2e8f0}}
a{{display:inline-block;margin:2px 0}}.missing{{color:#94a3b8;display:inline-block;margin:2px 0}}
</style></head><body><h1>Gaussian-splatting variant comparison</h1>
<p>Object: <code>{html.escape(object_id)}</code>. Each view count is trained independently.</p>
<table><thead><tr><th>Views</th>{headings}</tr></thead><tbody>{''.join(cells)}</tbody></table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        views = normalize_views(args.views)
        variants = discover_variants(args.phase2_root, args.phase3_root)
        validate_inputs(variants, args.object_id, views)
        object_root = object_output_root(args.output_root, args.object_id)
        results: list[RunResult] = []
        for view_count in views:
            for variant in variants:
                output = object_root / f"{view_count}views" / variant.key
                summaries = backend_summaries(output, args.backend)
                missing_backends = tuple(
                    path.parent.name for path in summaries
                    if args.force or not summary_has_training_budget(
                        path, args.iterations, view_count,
                    )
                )
                if not missing_backends:
                    print(f"SKIP {variant.key} at {view_count} views (already complete)")
                    results.append(RunResult(
                        variant.key, variant.policy, view_count, str(output), "skipped"
                    ))
                    continue
                backend = (
                    args.backend if args.force or len(missing_backends) > 1
                    else missing_backends[0]
                )
                command = build_command(
                    variant, args.object_id, view_count, output, args, backend
                )
                print("DRY RUN" if args.dry_run else "RUN", " ".join(command))
                if args.dry_run:
                    status = "dry_run"
                else:
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    if completed.returncode:
                        raise RuntimeError(
                            f"{variant.key} at {view_count} views failed with "
                            f"exit code {completed.returncode}"
                        )
                    status = "complete"
                results.append(RunResult(
                    variant.key, variant.policy, view_count, str(output), status
                ))
        object_root.mkdir(parents=True, exist_ok=True)
        (object_root / "manifest.json").write_text(
            json.dumps({
                "schema_version": 1,
                "object_id": args.object_id,
                "backend": args.backend,
                "views": list(views),
                "variants": [
                    {**asdict(variant), "metrics": str(variant.metrics.resolve())}
                    for variant in variants
                ],
                "runs": [asdict(result) for result in results],
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_index(
            object_root / "index.html", args.object_id, variants, views, args.backend
        )
        print(f"Comparison index: {object_root / 'index.html'}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Gaussian-splatting batch error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
