#!/usr/bin/env python3
"""Plot reconstruction quality for Phase 2 and all completed Phase 3 variants."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import sys

if __package__:
    from .plot_all_policy_coverage import (
        DEFAULT_PHASE2_RUNS,
        DEFAULT_PHASE3_ROOT,
        PHASE2_POLICIES,
        _completed_phase3_runs,
        _phase3_is_complete,
        _phase3_policy_specs,
        _scale,
    )
else:
    from plot_all_policy_coverage import (
        DEFAULT_PHASE2_RUNS,
        DEFAULT_PHASE3_ROOT,
        PHASE2_POLICIES,
        _completed_phase3_runs,
        _phase3_is_complete,
        _phase3_policy_specs,
        _scale,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/all_policy_comparison/reconstruction_curves.svg"
METRICS = (
    ("chamfer_l1_normalized_mean", "Normalized Chamfer-L1 (log-scaled) ↓", "log"),
    ("fscore_1pct_mean", "F-score @ 1% of object diameter ↑", "linear"),
    ("fscore_2pct_mean", "F-score @ 2% of object diameter ↑", "linear"),
    ("fscore_10pct_mean", "F-score @ 10% of object diameter ↑", "linear"),
)
REQUIRED_METRICS = METRICS[:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-run",
        type=Path,
        action="append",
        help=(
            "Phase 2 run to include; repeat to merge multiple runs. Defaults to "
            "the phase2_random/farthest/pun/vggt/oracle runs."
        ),
    )
    parser.add_argument(
        "--phase3-root",
        type=Path,
        default=DEFAULT_PHASE3_ROOT,
        help="root searched recursively for completed Phase 3 evaluation runs",
    )
    parser.add_argument(
        "--phase3-run",
        type=Path,
        action="append",
        help="use only this Phase 3 run (repeat to combine multiple explicit runs)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _reconstruction_path(run: Path) -> Path:
    return run / "metrics/reconstruction_curves.csv"


def _read_reconstruction(
    path: Path,
) -> tuple[dict[str, dict[str, list[tuple[int, float]]]], dict[str, int]]:
    curves: dict[str, dict[str, list[tuple[int, float]]]] = {}
    cohort_sizes: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "policy",
            "acquired_view_count",
            "object_count",
            *(metric for metric, _, _ in REQUIRED_METRICS),
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} does not have the expected reconstruction columns"
            )
        for row in reader:
            policy = row["policy"]
            count = int(row["object_count"])
            if policy in cohort_sizes and cohort_sizes[policy] != count:
                raise ValueError(f"{path} has inconsistent object counts for {policy}")
            cohort_sizes[policy] = count
            policy_curves = curves.setdefault(policy, {})
            for metric, _, _ in METRICS:
                if metric not in row or row[metric] in (None, ""):
                    continue
                policy_curves.setdefault(metric, []).append(
                    (int(row["acquired_view_count"]), float(row[metric]))
                )
    for policy_curves in curves.values():
        for points in policy_curves.values():
            points.sort()
    return curves, cohort_sizes


def _merge_reconstruction(
    destination: dict[str, dict[str, list[tuple[int, float]]]],
    destination_sizes: dict[str, int],
    incoming: dict[str, dict[str, list[tuple[int, float]]]],
    incoming_sizes: dict[str, int],
    source: Path,
) -> None:
    for policy, policy_curves in incoming.items():
        if policy in destination and destination[policy] != policy_curves:
            raise ValueError(
                f"Conflicting reconstruction curves for {policy}; completed run "
                f"{source} does not match another selected run"
            )
        if (
            policy in destination_sizes
            and destination_sizes[policy] != incoming_sizes[policy]
        ):
            raise ValueError(
                f"Conflicting reconstruction object counts for {policy} in {source}"
            )
        destination[policy] = policy_curves
        destination_sizes[policy] = incoming_sizes[policy]


def _transform(value: float, scale: str) -> float:
    if scale == "log":
        if value < 0:
            raise ValueError("Chamfer values must be non-negative")
        return math.log10(1.0 + value)
    return value


def _axis_bounds(values: list[float], scale: str) -> tuple[float, float]:
    transformed = [_transform(value, scale) for value in values]
    low, high = min(transformed), max(transformed)
    low = min(0.0, low)
    if math.isclose(low, high):
        padding = max(0.05, abs(low) * 0.05)
    else:
        padding = (high - low) * 0.06
    return (low if low == 0.0 else low - padding, high + padding)


def _tick_label(value: float, scale: str) -> str:
    if scale == "log":
        value = 10 ** value - 1.0
        return f"{value:.3g}"
    return f"{value:.2f}"


def write_plot(
    output: Path,
    curves: dict[str, dict[str, list[tuple[int, float]]]],
    cohort_sizes: dict[str, int],
    policies: tuple[tuple[str, str, str, str], ...],
    missing: list[str],
) -> None:
    metrics = tuple(
        definition
        for definition in METRICS
        if curves
        and all(
            definition[0] in policy_curves for policy_curves in curves.values()
        )
    )
    if not metrics:
        raise ValueError("No common reconstruction metrics are available to plot")
    width, height = 60 + 500 * len(metrics), 720
    legend_rows = math.ceil(len(policies) / 4)
    top = 115.0 + legend_rows * 30 + (24.0 if missing else 0.0)
    bottom, outer, gap = 78.0, 72.0, 48.0
    panel_width = (
        width - 2 * outer - (len(metrics) - 1) * gap
    ) / len(metrics)
    plot_height = height - top - bottom
    all_counts = [
        count
        for policy_curves in curves.values()
        for points in policy_curves.values()
        for count, _ in points
    ]
    x_min, x_max = min(all_counts, default=1), max(all_counts, default=10)
    cohort = next(iter(cohort_sizes.values()), 0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#536076}",
        ".axis{font-size:11px;fill:#536076}.legend{font-size:13px;font-weight:600}",
        ".panel{font-size:14px;font-weight:700}.note{font-size:13px;fill:#92400e}",
        ".grid{stroke:#d9dee8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" class="title" text-anchor="middle">All-policy reconstruction quality</text>',
        f'<text x="{width / 2}" y="64" class="subtitle" text-anchor="middle">Shared frozen VGGT backend; mean over {cohort} objects; normalized by object diameter; Chamfer axis uses log10(1+x).</text>',
    ]

    for index, (policy, label, color, _group) in enumerate(policies):
        row, column = divmod(index, 4)
        x = 72 + column * ((width - 144) / 4)
        y = 96 + row * 30
        available = policy in curves
        shown_color = color if available else "#a8afbd"
        dash = "" if available else ' stroke-dasharray="5 4"'
        suffix = f" (n={cohort_sizes[policy]})" if available else " (pending)"
        lines.extend([
            f'<line x1="{x}" y1="{y}" x2="{x + 30}" y2="{y}" stroke="{shown_color}" stroke-width="4"{dash}/>',
            f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(label + suffix)}</text>',
        ])
    if missing:
        labels = [label for policy, label, _, _ in policies if policy in missing]
        lines.append(
            f'<text x="72" y="{93 + legend_rows * 30}" class="note">Pending (no completed reconstruction data): {escape(", ".join(labels))}.</text>'
        )

    for panel_index, (metric, title, scale) in enumerate(metrics):
        left = outer + panel_index * (panel_width + gap)
        values = [value for policy in curves.values() for _, value in policy[metric]]
        y_min, y_max = _axis_bounds(values, scale)
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="{top - 14:.2f}" class="panel" text-anchor="middle">{escape(title)}</text>'
        )
        for tick_index in range(6):
            value = y_min + (y_max - y_min) * tick_index / 5
            y = _scale(value, y_min, y_max, top + plot_height, top)
            lines.extend([
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + panel_width:.2f}" y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" class="axis" text-anchor="end">{_tick_label(value, scale)}</text>',
            ])
        lines.extend([
            f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
            f'<line x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + panel_width:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
        ])
        for count in sorted(set(all_counts)):
            x = _scale(count, x_min, x_max, left, left + panel_width)
            lines.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 21:.2f}" class="axis" text-anchor="middle">{count}</text>'
            )
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="{height - 24}" class="subtitle" text-anchor="middle">Total acquired views</text>'
        )
        for policy, _label, color, group in policies:
            if policy not in curves:
                continue
            points = curves[policy][metric]
            coordinates = " ".join(
                f"{_scale(count, x_min, x_max, left, left + panel_width):.2f},"
                f"{_scale(_transform(value, scale), y_min, y_max, top + plot_height, top):.2f}"
                for count, value in points
            )
            dash = ' stroke-dasharray="9 5"' if group == "phase3" else ""
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"{dash}/>'
            )
            for count, value in points:
                x = _scale(count, x_min, x_max, left, left + panel_width)
                transformed = _transform(value, scale)
                y = _scale(transformed, y_min, y_max, top + plot_height, top)
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}"/>')

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    phase2_runs = args.phase2_run or list(DEFAULT_PHASE2_RUNS)
    curves: dict[str, dict[str, list[tuple[int, float]]]] = {}
    cohort_sizes: dict[str, int] = {}
    for phase2_run in phase2_runs:
        phase2_csv = _reconstruction_path(phase2_run)
        if not phase2_csv.is_file():
            raise FileNotFoundError(
                f"Missing required Phase 2 reconstruction data: {phase2_csv}"
            )
        phase2_curves, phase2_sizes = _read_reconstruction(phase2_csv)
        _merge_reconstruction(
            curves, cohort_sizes, phase2_curves, phase2_sizes, phase2_run
        )

    phase3_runs = args.phase3_run or _completed_phase3_runs(args.phase3_root)
    phase3_policy_names: set[str] = set()
    for phase3_run in phase3_runs:
        phase3_csv = _reconstruction_path(phase3_run)
        if not _phase3_is_complete(phase3_run) or not phase3_csv.is_file():
            print(
                f"skipping Phase 3 run without completed reconstruction data: {phase3_run}",
                file=sys.stderr,
            )
            continue
        phase3_curves, phase3_sizes = _read_reconstruction(phase3_csv)
        _merge_reconstruction(
            curves, cohort_sizes, phase3_curves, phase3_sizes, phase3_run
        )
        phase3_policy_names.update(phase3_curves)

    available_sizes = set(cohort_sizes.values())
    if len(available_sizes) != 1:
        raise ValueError(
            "Cannot combine reconstruction curves from different object cohorts: "
            + ", ".join(str(value) for value in sorted(available_sizes))
        )
    policies = PHASE2_POLICIES + _phase3_policy_specs(phase3_policy_names)
    missing = [policy for policy, _, _, _ in policies if policy not in curves]
    write_plot(args.output, curves, cohort_sizes, policies, missing)
    print(args.output)
    if missing:
        print("pending reconstruction policies: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
