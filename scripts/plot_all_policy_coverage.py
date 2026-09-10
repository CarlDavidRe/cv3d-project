#!/usr/bin/env python3
"""Combine Phase 2 and Phase 3 mean-coverage curves in one SVG."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE2 = ROOT / "outputs/phase2/phase2_closed_loop/seed_0"
DEFAULT_PHASE3 = ROOT / "outputs/phase3/controlled_history_comparison/seed_0"
DEFAULT_OUTPUT = ROOT / "outputs/all_policy_comparison/coverage_curves.svg"

POLICIES = (
    ("random", "Random", "#64748b", "phase2"),
    ("farthest", "Farthest", "#2563eb", "phase2"),
    ("pun", "PUN", "#dc2626", "phase2"),
    ("vggt", "Phase 2 VGGT", "#7c3aed", "phase2"),
    ("oracle", "Oracle", "#059669", "phase2"),
    (
        "vggt_independent_history",
        "Phase 3 independent",
        "#ea580c",
        "phase3",
    ),
    ("vggt_joint_history", "Phase 3 joint", "#0891b2", "phase3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--phase3-run", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _coverage_path(run: Path) -> Path:
    return run / "metrics/coverage.csv"


def _read_coverage(path: Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, int]]:
    curves: dict[str, list[tuple[int, float]]] = {}
    cohort_sizes: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"policy", "acquired_view_count", "coverage_mean", "object_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} does not have the expected coverage columns")
        for row in reader:
            policy = row["policy"]
            curves.setdefault(policy, []).append(
                (int(row["acquired_view_count"]), float(row["coverage_mean"]))
            )
            count = int(row["object_count"])
            if policy in cohort_sizes and cohort_sizes[policy] != count:
                raise ValueError(f"{path} has inconsistent object counts for {policy}")
            cohort_sizes[policy] = count
    for points in curves.values():
        points.sort()
    return curves, cohort_sizes


def _coverage_target(run: Path) -> str | None:
    summary = run / "metrics/summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    value = payload.get("coverage_target")
    return str(value) if value is not None else None


def _phase3_is_complete(run: Path) -> bool:
    completion = run / "metrics/phase3_completion.json"
    if not completion.is_file():
        return False
    payload = json.loads(completion.read_text(encoding="utf-8"))
    return payload.get("status") == "complete"


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def write_plot(
    output: Path,
    curves: dict[str, list[tuple[int, float]]],
    cohort_sizes: dict[str, int],
    missing: list[str],
    coverage_target: str,
) -> None:
    width, height = 1280, 720
    left, right, top, bottom = 105.0, 55.0, 185.0, 95.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in curves.values() for point in points]
    x_min = min((point[0] for point in all_points), default=1)
    x_max = max((point[0] for point in all_points), default=10)
    y_min, y_max = 0.0, 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#536076}",
        ".axis{font-size:12px;fill:#536076}.legend{font-size:13px;font-weight:600}",
        ".note{font-size:13px;fill:#92400e}.grid{stroke:#d9dee8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="55" y="42" class="title">All-policy closed-loop surface coverage</text>',
        f'<text x="55" y="68" class="subtitle">Common evaluator target: {escape(coverage_target)}. Phase 2 proxy-trained and Phase 3 direct-gain policies are distinct experiment groups.</text>',
    ]

    for index, (policy, label, color, group) in enumerate(POLICIES):
        row, column = divmod(index, 4)
        x = 60 + column * 295
        y = 103 + row * 30
        available = policy in curves
        shown_color = color if available else "#a8afbd"
        dash = "" if available else ' stroke-dasharray="5 4"'
        suffix = f" (n={cohort_sizes[policy]})" if available else " (pending)"
        lines.extend([
            f'<line x1="{x}" y1="{y}" x2="{x + 30}" y2="{y}" stroke="{shown_color}" stroke-width="4"{dash}/>',
            f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(label + suffix)}</text>',
        ])

    if missing:
        labels = [label for policy, label, _, _ in POLICIES if policy in missing]
        lines.append(
            f'<text x="55" y="169" class="note">Pending: {escape(", ".join(labels))}; no controlled Phase 3 coverage data is present yet.</text>'
        )

    for tick in range(0, 11, 2):
        y = _scale(tick / 10, y_min, y_max, top + plot_height, top)
        lines.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>',
            f'<text x="{left - 14}" y="{y + 4:.2f}" class="axis" text-anchor="end">{tick / 10:.1f}</text>',
        ])
    for tick in range(x_min, x_max + 1):
        x = _scale(tick, x_min, x_max, left, left + plot_width)
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 25}" class="axis" text-anchor="middle">{tick}</text>'
        )
    lines.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#64748b" stroke-width="1.5"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 32}" class="axis" text-anchor="middle">Total acquired views (initial view included)</text>',
        f'<text x="27" y="{top + plot_height / 2}" class="axis" text-anchor="middle" transform="rotate(-90 27 {top + plot_height / 2})">Mean absolute surface coverage</text>',
    ])

    for policy, _label, color, group in POLICIES:
        points = curves.get(policy)
        if not points:
            continue
        coordinates = " ".join(
            f"{_scale(x, x_min, x_max, left, left + plot_width):.2f},"
            f"{_scale(y, y_min, y_max, top + plot_height, top):.2f}"
            for x, y in points
        )
        dash = ' stroke-dasharray="9 5"' if group == "phase3" else ""
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"{dash}/>'
        )
        for x_value, y_value in points:
            x = _scale(x_value, x_min, x_max, left, left + plot_width)
            y = _scale(y_value, y_min, y_max, top + plot_height, top)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    phase2_csv = _coverage_path(args.phase2_run)
    if not phase2_csv.is_file():
        raise FileNotFoundError(f"Missing required Phase 2 coverage data: {phase2_csv}")

    curves, cohort_sizes = _read_coverage(phase2_csv)
    phase2_target = _coverage_target(args.phase2_run)
    phase3_csv = _coverage_path(args.phase3_run)
    if phase3_csv.is_file() and _phase3_is_complete(args.phase3_run):
        phase3_curves, phase3_sizes = _read_coverage(phase3_csv)
        phase3_target = _coverage_target(args.phase3_run)
        if phase2_target and phase3_target and phase2_target != phase3_target:
            raise ValueError(
                f"Coverage targets differ: Phase 2={phase2_target}, Phase 3={phase3_target}"
            )
        curves.update(phase3_curves)
        cohort_sizes.update(phase3_sizes)

    available_sizes = {cohort_sizes[policy] for policy in curves if policy in cohort_sizes}
    if len(available_sizes) != 1:
        raise ValueError(
            "Cannot combine coverage curves from different object-cohort sizes: "
            + ", ".join(str(value) for value in sorted(available_sizes))
        )

    expected = [policy for policy, _, _, _ in POLICIES]
    missing = [policy for policy in expected if policy not in curves]
    write_plot(
        args.output,
        curves,
        cohort_sizes,
        missing,
        phase2_target or "vis_a",
    )
    print(args.output)
    if missing:
        print("pending policies: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
