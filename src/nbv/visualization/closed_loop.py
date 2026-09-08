"""Dependency-free SVG summaries for closed-loop policy experiments."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_COLORS = {
    "random": "#64748b",
    "farthest": "#2563eb",
    "pun": "#dc2626",
    "vggt": "#7c3aed",
    "oracle": "#059669",
}
_FALLBACK_COLORS = ("#d97706", "#0891b2", "#be185d", "#4d7c0f")


def write_closed_loop_visualizations(
    results: Sequence[Any], destination_dir: str | Path
) -> dict[str, Path]:
    """Write coverage, per-step, and aggregate-policy figures."""

    if not results:
        raise ValueError("At least one rollout result is required")
    policies = tuple(dict.fromkeys(result.metadata["policy"] for result in results))
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": destination / "coverage_curves.svg",
        "per_step": destination / "per_step_policy_quality.svg",
        "summary": destination / "policy_summary.svg",
    }
    _write_coverage(results, policies, paths["coverage"])
    _write_per_step(results, policies, paths["per_step"])
    _write_summary(results, policies, paths["summary"])
    return paths


def _write_coverage(results: Sequence[Any], policies: tuple[str, ...], path: Path) -> None:
    series = {}
    for policy in policies:
        selected = [result for result in results if result.metadata["policy"] == policy]
        counts = sorted({int(value) for result in selected for value in result.acquired_view_counts})
        series[policy] = [
            (
                float(count),
                float(np.mean([
                    result.coverage[np.flatnonzero(result.acquired_view_counts == count)[0]]
                    for result in selected
                    if count in result.acquired_view_counts
                ])),
            )
            for count in counts
        ]
    _write_line_figure(
        path,
        title="Mean surface coverage by acquired-view count",
        subtitle=_cohort_note(results),
        policies=policies,
        panels=(("Coverage", series, (0.0, 1.0)),),
        x_label="total acquired views (initial views included)",
        width=980,
        height=590,
    )


def _write_per_step(results: Sequence[Any], policies: tuple[str, ...], path: Path) -> None:
    definitions = (
        ("Selected true gain", "selected_true_gain", None),
        ("Normalized regret ↓", "normalized_regret", (0.0, 1.0)),
        ("Spearman ↑", "spearman", (-1.0, 1.0)),
        ("NDCG@5 ↑", "ndcg_at_5", (0.0, 1.0)),
    )
    panels = []
    for title, key, bounds in definitions:
        series = {}
        for policy in policies:
            steps = [step for result in results if result.metadata["policy"] == policy for step in result.steps]
            indices = sorted({int(step["step_index"]) for step in steps})
            points = []
            for index in indices:
                values = [
                    float(step[key]) for step in steps
                    if int(step["step_index"]) == index and step[key] is not None
                ]
                if values:
                    points.append((float(index + 1), float(np.mean(values))))
            series[policy] = points
        panels.append((title, series, bounds))
    _write_line_figure(
        path,
        title="Per-step policy quality against true geometric gain",
        subtitle="Each point is a mean over available object rollouts; decision 1 follows the initial view.",
        policies=policies,
        panels=tuple(panels),
        x_label="decision number",
        width=1460,
        height=560,
    )


def _write_summary(results: Sequence[Any], policies: tuple[str, ...], path: Path) -> None:
    definitions = (
        ("Final coverage ↑", "final_coverage", (0.0, 1.0)),
        ("Coverage AUC ↑", "coverage_auc", None),
        ("Mean regret ↓", "normalized_regret_mean", (0.0, 1.0)),
        ("Mean NDCG@5 ↑", "ndcg_at_5_mean", (0.0, 1.0)),
    )
    values = {
        policy: [result.summary() for result in results if result.metadata["policy"] == policy]
        for policy in policies
    }
    panels = []
    for title, key, bounds in definitions:
        panel = {}
        for policy in policies:
            finite = [float(row[key]) for row in values[policy] if row[key] is not None]
            panel[policy] = float(np.mean(finite)) if finite else None
        panels.append((title, panel, bounds))
    _write_bar_figure(
        path,
        title="Closed-loop policy summary",
        subtitle=_cohort_note(results),
        policies=policies,
        panels=tuple(panels),
    )


def _write_line_figure(
    path: Path,
    *,
    title: str,
    subtitle: str,
    policies: tuple[str, ...],
    panels: tuple[tuple[str, dict[str, list[tuple[float, float]]], tuple[float, float] | None], ...],
    x_label: str,
    width: int,
    height: int,
) -> None:
    margin_x, gap, top, bottom = 75.0, 42.0, 125.0, 80.0
    panel_width = (width - 2 * margin_x - gap * (len(panels) - 1)) / len(panels)
    plot_height = height - top - bottom
    lines = _header(width, height, title, subtitle)
    _legend(lines, policies, width, y=92)
    for panel_index, (panel_title, series, fixed_bounds) in enumerate(panels):
        left = margin_x + panel_index * (panel_width + gap)
        points = [point for values in series.values() for point in values]
        x_values = [point[0] for point in points] or [0.0, 1.0]
        y_values = [point[1] for point in points] or [0.0, 1.0]
        x_min, x_max = _bounds(x_values, include_zero=False)
        y_min, y_max = fixed_bounds or _bounds(y_values, include_zero=True)
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="116" class="panel" '
            f'text-anchor="middle">{escape(panel_title)}</text>'
        )
        _axes(lines, left, top, panel_width, plot_height, x_min, x_max, y_min, y_max, x_label)
        for policy in policies:
            values = series.get(policy, [])
            if not values:
                continue
            scaled = [
                (
                    _scale(x, x_min, x_max, left, left + panel_width),
                    _scale(y, y_min, y_max, top + plot_height, top),
                )
                for x, y in values
            ]
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in scaled)
            color = _color(policy, policies)
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="3" stroke-linejoin="round" data-policy="{escape(policy)}"/>'
            )
            lines.extend(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" '
                f'data-policy="{escape(policy)}"/>' for x, y in scaled
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bar_figure(
    path: Path,
    *,
    title: str,
    subtitle: str,
    policies: tuple[str, ...],
    panels: tuple[tuple[str, dict[str, float | None], tuple[float, float] | None], ...],
) -> None:
    width, height = 1460, 560
    margin_x, gap, top, bottom = 70.0, 40.0, 125.0, 105.0
    panel_width = (width - 2 * margin_x - gap * (len(panels) - 1)) / len(panels)
    plot_height = height - top - bottom
    lines = _header(width, height, title, subtitle)
    for panel_index, (panel_title, values, fixed_bounds) in enumerate(panels):
        left = margin_x + panel_index * (panel_width + gap)
        finite = [value for value in values.values() if value is not None and math.isfinite(value)]
        y_min, y_max = fixed_bounds or _bounds(finite or [0.0, 1.0], include_zero=True)
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="108" class="panel" '
            f'text-anchor="middle">{escape(panel_title)}</text>'
        )
        _axes(lines, left, top, panel_width, plot_height, 0, len(policies), y_min, y_max, None, x_ticks=False)
        slot = panel_width / len(policies)
        for index, policy in enumerate(policies):
            value = values.get(policy)
            x = left + index * slot + slot * 0.18
            bar_width = slot * 0.64
            baseline = _scale(max(0.0, y_min), y_min, y_max, top + plot_height, top)
            if value is not None and math.isfinite(value):
                y = _scale(value, y_min, y_max, top + plot_height, top)
                upper, bar_height = min(y, baseline), abs(baseline - y)
                lines.append(
                    f'<rect x="{x:.2f}" y="{upper:.2f}" width="{bar_width:.2f}" '
                    f'height="{max(bar_height, 1):.2f}" fill="{_color(policy, policies)}" '
                    f'data-policy="{escape(policy)}" data-value="{value:.8g}"/>'
                )
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{upper - 7:.2f}" '
                    f'class="value" text-anchor="middle">{value:.3f}</text>'
                )
            lines.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 24:.2f}" '
                f'class="tick" text-anchor="middle">{escape(policy)}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:system-ui,sans-serif;fill:#0f172a}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;fill:#475569}.panel{font-size:14px;font-weight:650}.tick{font-size:10px;fill:#475569}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.legend{font-size:12px}.value{font-size:10px;font-weight:600}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.2f}" y="32" class="title" text-anchor="middle">{escape(title)}</text>',
        f'<text x="{width / 2:.2f}" y="54" class="subtitle" text-anchor="middle">{escape(subtitle)}</text>',
    ]


def _legend(lines: list[str], policies: tuple[str, ...], width: int, *, y: float) -> None:
    spacing = 112.0
    start = width / 2 - spacing * (len(policies) - 1) / 2
    for index, policy in enumerate(policies):
        x = start + index * spacing
        lines.append(f'<line x1="{x - 20:.2f}" y1="{y}" x2="{x:.2f}" y2="{y}" stroke="{_color(policy, policies)}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 7:.2f}" y="{y + 4:.2f}" class="legend">{escape(policy)}</text>')


def _axes(
    lines: list[str], left: float, top: float, width: float, height: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
    x_label: str | None, *, x_ticks: bool = True,
) -> None:
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = _scale(value, y_min, y_max, top + height, top)
        lines.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + width:.2f}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" class="tick" text-anchor="end">{value:.2f}</text>')
    lines.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + height:.2f}" class="axis"/>')
    lines.append(f'<line x1="{left:.2f}" y1="{top + height:.2f}" x2="{left + width:.2f}" y2="{top + height:.2f}" class="axis"/>')
    if x_ticks:
        ticks = sorted(set(np.linspace(x_min, x_max, min(6, max(2, int(x_max - x_min + 1)))).round(6)))
        for value in ticks:
            x = _scale(float(value), x_min, x_max, left, left + width)
            label = str(int(value)) if float(value).is_integer() else f"{value:.1f}"
            lines.append(f'<text x="{x:.2f}" y="{top + height + 19:.2f}" class="tick" text-anchor="middle">{label}</text>')
    if x_label:
        lines.append(f'<text x="{left + width / 2:.2f}" y="{top + height + 48:.2f}" class="subtitle" text-anchor="middle">{escape(x_label)}</text>')


def _bounds(values: Sequence[float], *, include_zero: bool) -> tuple[float, float]:
    minimum, maximum = min(values), max(values)
    if include_zero:
        minimum = min(0.0, minimum)
    if math.isclose(minimum, maximum):
        padding = max(0.05, abs(minimum) * 0.05)
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.06
    return minimum if include_zero and minimum == 0 else minimum - padding, maximum + padding


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) / (high - low) * (end - start)


def _color(policy: str, policies: tuple[str, ...]) -> str:
    if policy in _COLORS:
        return _COLORS[policy]
    unknown = [name for name in policies if name not in _COLORS]
    return _FALLBACK_COLORS[unknown.index(policy) % len(_FALLBACK_COLORS)]


def _cohort_note(results: Sequence[Any]) -> str:
    object_count = len({result.metadata["object_id"] for result in results})
    target = results[0].metadata["coverage_target"]
    return f"Mean over {object_count} object{'s' if object_count != 1 else ''}; coverage target: {target}."
