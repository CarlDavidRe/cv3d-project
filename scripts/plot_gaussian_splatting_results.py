#!/usr/bin/env python3
"""Plot the matched silhouette-refined 2DGS case-study results."""

from __future__ import annotations

import argparse
import csv
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs/gaussian_splatting_per_view_budget_alignment_repair"
    / "recovered_metrics.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/all_policy_comparison"

POLICIES = (
    ("phase2_random", "P2 Random", "#64748b", "phase2"),
    ("phase2_farthest", "P2 Farthest", "#2563eb", "phase2"),
    ("phase2_pun", "P2 PUN", "#dc2626", "phase2"),
    ("phase2_vggt", "P2 VGGT", "#7c3aed", "phase2"),
    ("phase2_oracle", "P2 Oracle", "#059669", "phase2"),
    (
        "phase3_vggt_independent_history",
        "P3 independent",
        "#ea580c",
        "phase3",
    ),
    ("phase3_vggt_joint_history", "P3 joint", "#0891b2", "phase3"),
    (
        "phase3_vggt_joint_pose_deepsets",
        "P3 pose DeepSets",
        "#ca8a04",
        "phase3",
    ),
    (
        "phase3_vggt_joint_token_attention",
        "P3 token attention",
        "#db2777",
        "phase3",
    ),
)
METRICS = (
    ("chamfer_l1_normalized", "Normalized Chamfer-L1 ↓", 0.0, None),
    ("fscore_2pct", "F-score @ 2% of object diameter ↑", 0.0, 1.0),
)
OBJECT_LABELS = {
    "02691156": "Airplane",
    "02828884": "Bench",
    "02933112": "Cabinet",
    "02958343": "Car",
}

ResultKey = tuple[str, str, int]
ResultRows = dict[ResultKey, dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--object-id",
        help="object for the view-budget plot; defaults to the most complete object",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_results(path: Path) -> ResultRows:
    required = {
        "variant",
        "object_id",
        "acquired_view_count",
        "backend",
        *(metric for metric, _title, _low, _high in METRICS),
    }
    rows: ResultRows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} does not have the expected 2DGS columns")
        for row in reader:
            if row["backend"] != "2dgs":
                continue
            key = (
                row["object_id"],
                row["variant"],
                int(row["acquired_view_count"]),
            )
            if key in rows:
                raise ValueError(f"duplicate 2DGS result for {key}")
            for metric, _title, _low, _high in METRICS:
                float(row[metric])
            rows[key] = row
    if not rows:
        raise ValueError(f"{path} contains no 2DGS results")
    return rows


def complete_views(rows: Mapping[ResultKey, dict[str, str]], object_id: str) -> list[int]:
    policy_names = {policy for policy, _label, _color, _group in POLICIES}
    candidates = {
        view_count
        for row_object, _policy, view_count in rows
        if row_object == object_id
    }
    return sorted(
        view_count
        for view_count in candidates
        if all((object_id, policy, view_count) in rows for policy in policy_names)
    )


def select_primary_object(rows: Mapping[ResultKey, dict[str, str]]) -> str:
    objects = sorted({object_id for object_id, _policy, _views in rows})
    if not objects:
        raise ValueError("no objects are available")
    return max(objects, key=lambda object_id: (len(complete_views(rows, object_id)), object_id))


def select_view_budget_cohort(
    rows: Mapping[ResultKey, dict[str, str]],
) -> tuple[list[str], list[int]]:
    """Choose the largest balanced object-by-view rectangle in the aggregate."""

    objects = sorted({object_id for object_id, _policy, _views in rows})
    all_views = sorted({view for object_id in objects for view in complete_views(rows, object_id)})
    choices: list[tuple[int, int, int, tuple[int, ...], list[str]]] = []
    for count in range(2, len(all_views) + 1):
        for views in combinations(all_views, count):
            cohort = [
                object_id
                for object_id in objects
                if set(views).issubset(complete_views(rows, object_id))
            ]
            if cohort:
                choices.append((len(cohort) * count, count, len(cohort), views, cohort))
    if not choices:
        raise ValueError("no object has at least two complete view budgets")
    _cells, _view_count, _object_count, views, cohort = max(choices)
    return cohort, list(views)


def select_transfer_budget(
    rows: Mapping[ResultKey, dict[str, str]],
) -> tuple[int, list[str]]:
    objects = sorted({object_id for object_id, _policy, _views in rows})
    view_counts = sorted({views for _object, _policy, views in rows})
    choices = []
    for view_count in view_counts:
        complete_objects = [
            object_id
            for object_id in objects
            if view_count in complete_views(rows, object_id)
        ]
        if complete_objects:
            choices.append((len(complete_objects), view_count, complete_objects))
    if not choices:
        raise ValueError("no complete policy-by-object slice is available")
    _count, view_count, complete_objects = max(choices)
    return view_count, complete_objects


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def _object_label(object_id: str) -> str:
    category, object_key = object_id.split("/", 1)
    category_label = OBJECT_LABELS.get(category, category)
    return f"{category_label} ({object_key[:6]}…)"


def write_view_budget_plot(
    output: Path,
    rows: Mapping[ResultKey, dict[str, str]],
    object_ids: list[str],
) -> None:
    if not object_ids:
        raise ValueError("the view-budget plot needs at least one object")
    view_sets = [set(complete_views(rows, object_id)) for object_id in object_ids]
    views = sorted(set.intersection(*view_sets))
    if len(views) < 2:
        raise ValueError("the selected objects need at least two common view budgets")
    width, height = 1320, 680
    outer, gap, top, bottom = 82.0, 70.0, 190.0, 82.0
    panel_width = (width - 2 * outer - gap) / 2
    plot_height = height - top - bottom
    cohort_label = (
        f"{_object_label(object_ids[0])} case study"
        if len(object_ids) == 1
        else f"case study: mean over {len(object_ids)} matched objects"
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#536076}",
        ".axis{font-size:12px;fill:#536076}.legend{font-size:12px;font-weight:600}",
        ".panel{font-size:15px;font-weight:700}.grid{stroke:#d9dee8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="55" y="40" class="title">Silhouette-refined 2DGS quality by view budget</text>',
        f'<text x="55" y="66" class="subtitle">Matched nine-policy {escape(cohort_label)}; each budget is trained independently for 1,500 iterations per acquired view.</text>',
    ]
    for index, (_policy, label, color, group) in enumerate(POLICIES):
        row, column = divmod(index, 5)
        x = 62 + column * 250
        y = 104 + row * 30
        dash = ' stroke-dasharray="8 5"' if group == "phase3" else ""
        lines.extend((
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4"{dash}/>',
            f'<text x="{x + 36}" y="{y + 4}" class="legend">{escape(label)}</text>',
        ))

    for panel_index, (metric, title, fixed_low, fixed_high) in enumerate(METRICS):
        left = outer + panel_index * (panel_width + gap)
        values = [
            float(rows[(object_id, policy, view_count)][metric])
            for object_id in object_ids
            for policy, _label, _color, _group in POLICIES
            for view_count in views
        ]
        low = fixed_low
        high = fixed_high if fixed_high is not None else max(values) * 1.08
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="{top - 18}" class="panel" text-anchor="middle">{escape(title)}</text>'
        )
        for tick_index in range(6):
            value = low + (high - low) * tick_index / 5
            y = _scale(value, low, high, top + plot_height, top)
            label = f"{value:.2f}" if high <= 1 else f"{value:.1f}"
            lines.extend((
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + panel_width:.2f}" y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 10:.2f}" y="{y + 4:.2f}" class="axis" text-anchor="end">{label}</text>',
            ))
        lines.extend((
            f'<line x1="{left:.2f}" y1="{top}" x2="{left:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
            f'<line x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + panel_width:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
        ))
        for view_count in views:
            x = _scale(view_count, views[0], views[-1], left, left + panel_width)
            lines.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 24:.2f}" class="axis" text-anchor="middle">{view_count}</text>'
            )
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="{height - 24}" class="axis" text-anchor="middle">Total acquired views</text>'
        )
        for policy, _label, color, group in POLICIES:
            points = [
                (
                    view_count,
                    sum(
                        float(rows[(object_id, policy, view_count)][metric])
                        for object_id in object_ids
                    )
                    / len(object_ids),
                )
                for view_count in views
            ]
            coordinates = " ".join(
                f"{_scale(view_count, views[0], views[-1], left, left + panel_width):.2f},"
                f"{_scale(value, low, high, top + plot_height, top):.2f}"
                for view_count, value in points
            )
            dash = ' stroke-dasharray="8 5"' if group == "phase3" else ""
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round"{dash}/>'
            )
            for view_count, value in points:
                x = _scale(view_count, views[0], views[-1], left, left + panel_width)
                y = _scale(value, low, high, top + plot_height, top)
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transfer_plot(
    output: Path,
    rows: Mapping[ResultKey, dict[str, str]],
    view_count: int,
    object_ids: list[str],
) -> None:
    if len(object_ids) < 2:
        raise ValueError("the transfer plot needs at least two complete objects")
    object_colors = ("#0f766e", "#9333ea", "#b45309", "#0369a1")
    width, height = 1420, 700
    outer, gap, top, bottom = 82.0, 70.0, 145.0, 145.0
    panel_width = (width - 2 * outer - gap) / 2
    plot_height = height - top - bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#536076}",
        ".axis{font-size:11px;fill:#536076}.legend{font-size:13px;font-weight:600}",
        ".panel{font-size:15px;font-weight:700}.grid{stroke:#d9dee8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="55" y="40" class="title">{len(object_ids)}-object silhouette-refined 2DGS comparison at {view_count} views</text>',
        '<text x="55" y="66" class="subtitle">Paired object scores expose case-study sensitivity; policy positions are descriptive, not population estimates.</text>',
    ]
    for index, object_id in enumerate(object_ids):
        x = 62 + index * 270
        color = object_colors[index % len(object_colors)]
        lines.extend((
            f'<circle cx="{x}" cy="105" r="6" fill="{color}"/>',
            f'<text x="{x + 14}" y="109" class="legend">{escape(_object_label(object_id))}</text>',
        ))

    for panel_index, (metric, title, fixed_low, fixed_high) in enumerate(METRICS):
        left = outer + panel_index * (panel_width + gap)
        values = [
            float(rows[(object_id, policy, view_count)][metric])
            for object_id in object_ids
            for policy, _label, _color, _group in POLICIES
        ]
        low = fixed_low
        high = fixed_high if fixed_high is not None else max(values) * 1.08
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="{top - 17}" class="panel" text-anchor="middle">{escape(title)}</text>'
        )
        for tick_index in range(6):
            value = low + (high - low) * tick_index / 5
            y = _scale(value, low, high, top + plot_height, top)
            lines.extend((
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + panel_width:.2f}" y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 10:.2f}" y="{y + 4:.2f}" class="axis" text-anchor="end">{value:.2f}</text>',
            ))
        step = panel_width / len(POLICIES)
        for policy_index, (policy, label, _color, _group) in enumerate(POLICIES):
            x = left + step * (policy_index + 0.5)
            policy_values = [
                float(rows[(object_id, policy, view_count)][metric])
                for object_id in object_ids
            ]
            y_values = [
                _scale(value, low, high, top + plot_height, top)
                for value in policy_values
            ]
            lines.append(
                f'<line x1="{x:.2f}" y1="{min(y_values):.2f}" x2="{x:.2f}" y2="{max(y_values):.2f}" stroke="#94a3b8" stroke-width="1.5"/>'
            )
            for object_index, y in enumerate(y_values):
                color = object_colors[object_index % len(object_colors)]
                offset = (object_index - (len(object_ids) - 1) / 2) * 5
                lines.append(
                    f'<circle cx="{x + offset:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
                )
            lines.append(
                f'<text x="{x + 3:.2f}" y="{top + plot_height + 16:.2f}" class="axis" text-anchor="end" transform="rotate(-42 {x + 3:.2f} {top + plot_height + 16:.2f})">{escape(label)}</text>'
            )
        lines.extend((
            f'<line x1="{left:.2f}" y1="{top}" x2="{left:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
            f'<line x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + panel_width:.2f}" y2="{top + plot_height:.2f}" stroke="#64748b"/>',
        ))
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_results(args.input)
    if args.object_id:
        view_budget_objects = [args.object_id]
    else:
        view_budget_objects, _views = select_view_budget_cohort(rows)
    view_budget_output = args.output_dir / "gaussian_splatting_view_budget.svg"
    write_view_budget_plot(view_budget_output, rows, view_budget_objects)
    transfer_budget, transfer_objects = select_transfer_budget(rows)
    transfer_output = args.output_dir / "gaussian_splatting_two_object.svg"
    write_transfer_plot(transfer_output, rows, transfer_budget, transfer_objects)
    print(view_budget_output)
    print(transfer_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
