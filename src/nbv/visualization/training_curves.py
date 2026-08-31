"""Write self-contained SVG diagnostics from Phase 1 training histories."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Mapping, Sequence


HistoryRow = Mapping[str, float | int | None]
_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#475569",
    "#9333ea",
    "#0f766e",
    "#b45309",
)


def write_variant_training_curves(
    history: Sequence[HistoryRow],
    destination_dir: str | Path,
    *,
    variant_name: str,
    best_epoch: int,
    epochs_completed: int,
    stopped_early: bool,
    ndcg_k: int,
) -> tuple[Path, Path]:
    """Write loss-component and validation-metric curves for one variant."""

    if not history:
        raise ValueError("history must not be empty")
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loss_path = destination / f"{variant_name}_losses.svg"
    metric_path = destination / f"{variant_name}_validation_metrics.svg"

    _write_panel_grid(
        loss_path,
        title=f"Training diagnostics — {variant_name}",
        history=history,
        best_epoch=best_epoch,
        last_epoch=epochs_completed,
        stopped_early=stopped_early,
        panels=(
            (
                "Combined objective",
                (
                    (
                        "optimization train",
                        "optimization_train_loss",
                        "#94a3b8",
                        "5 4",
                    ),
                    ("train (eval mode)", "train_loss", "#2563eb", None),
                    ("validation", "validation_loss", "#dc2626", None),
                ),
            ),
            (
                "Huber loss",
                (
                    ("train (eval mode)", "train_huber_loss", "#2563eb", None),
                    ("validation", "validation_huber_loss", "#dc2626", None),
                ),
            ),
            (
                "Pairwise ranking loss",
                (
                    ("train (eval mode)", "train_ranking_loss", "#2563eb", None),
                    ("validation", "validation_ranking_loss", "#dc2626", None),
                ),
            ),
        ),
    )
    _write_panel_grid(
        metric_path,
        title=f"Validation ranking metrics — {variant_name}",
        history=history,
        best_epoch=best_epoch,
        last_epoch=epochs_completed,
        stopped_early=stopped_early,
        panels=(
            (
                "Normalized regret (lower is better)",
                (
                    (
                        "validation",
                        "validation_normalized_regret_mean",
                        "#7c3aed",
                        None,
                    ),
                ),
            ),
            (
                "Spearman (higher is better)",
                (("validation", "validation_spearman_mean", "#059669", None),),
            ),
            (
                f"NDCG@{ndcg_k} (higher is better)",
                (
                    (
                        "validation",
                        f"validation_ndcg_at_{ndcg_k}_mean",
                        "#d97706",
                        None,
                    ),
                ),
            ),
        ),
    )
    return loss_path, metric_path


def write_validation_loss_comparison(
    histories: Mapping[str, Sequence[HistoryRow]],
    destination: str | Path,
    *,
    best_epochs: Mapping[str, int],
) -> Path:
    """Write one validation-loss overlay for all learned variants."""

    if not histories:
        raise ValueError("histories must not be empty")
    series: list[tuple[str, list[tuple[float, float]], str]] = []
    for index, (name, history) in enumerate(histories.items()):
        points = _points(history, "validation_loss")
        if points:
            series.append((name, points, _COLORS[index % len(_COLORS)]))
    if not series:
        raise ValueError("histories contain no finite validation losses")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = max(650, 170 + 24 * len(series))
    left, right, top, bottom = 90.0, 350.0, 85.0, 75.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for _, points, _ in series for point in points]
    x_min, x_max = _bounds([point[0] for point in all_points], include_zero=True)
    y_min, y_max = _bounds([point[1] for point in all_points])
    lines = _svg_header(width, height, "Validation loss comparison")
    lines.extend(
        _axes(
            left,
            top,
            plot_width,
            plot_height,
            x_min,
            x_max,
            y_min,
            y_max,
            x_label="epoch",
            y_label="combined validation loss",
        )
    )
    for index, (name, points, color) in enumerate(series):
        lines.append(
            _polyline(
                points,
                color,
                left,
                top,
                plot_width,
                plot_height,
                x_min,
                x_max,
                y_min,
                y_max,
                data_series=name,
            )
        )
        best_epoch = best_epochs.get(name)
        best_point = next(
            (point for point in points if int(point[0]) == best_epoch), None
        )
        if best_point is not None:
            x, y = _scale_point(
                best_point,
                left,
                top,
                plot_width,
                plot_height,
                x_min,
                x_max,
                y_min,
                y_max,
            )
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" '
                f'data-best-epoch="{best_epoch}" data-variant="{escape(name)}"/>'
            )
        legend_y = top + 20 + index * 24
        legend_x = width - right + 40
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 34}" y="{legend_y + 4}" class="legend">'
            f'{escape(name)}</text>'
        )
    lines.append(
        f'<text x="{width - right + 40}" y="{height - 35}" class="note">'
        "Circles mark validation-selected best epochs. Test data are not plotted."
        "</text>"
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_panel_grid(
    destination: Path,
    *,
    title: str,
    history: Sequence[HistoryRow],
    best_epoch: int,
    last_epoch: int,
    stopped_early: bool,
    panels: Sequence[
        tuple[str, Sequence[tuple[str, str, str, str | None]]]
    ],
) -> None:
    width = 1200
    height = 470
    gap = 35.0
    outer_left = 65.0
    outer_right = 35.0
    top = 90.0
    bottom = 65.0
    panel_width = (
        width - outer_left - outer_right - gap * (len(panels) - 1)
    ) / len(panels)
    plot_height = height - top - bottom
    lines = _svg_header(width, height, title)

    for panel_index, (panel_title, definitions) in enumerate(panels):
        left = outer_left + panel_index * (panel_width + gap)
        series = [
            (label, _points(history, key), color, dash, key)
            for label, key, color, dash in definitions
        ]
        finite_points = [
            point for _, points, _, _, _ in series for point in points
        ]
        x_values = [point[0] for point in finite_points]
        y_values = [point[1] for point in finite_points]
        x_min, x_max = _bounds(x_values or [0.0, 1.0], include_zero=True)
        y_min, y_max = _bounds(y_values or [0.0, 1.0])
        lines.append(
            f'<text x="{left + panel_width / 2:.2f}" y="68" '
            f'class="panel-title" text-anchor="middle">{escape(panel_title)}</text>'
        )
        lines.extend(
            _axes(
                left,
                top,
                panel_width,
                plot_height,
                x_min,
                x_max,
                y_min,
                y_max,
                x_label="epoch",
                y_label=None,
            )
        )
        last_x = _scale(last_epoch, x_min, x_max, left, left + panel_width)
        lines.append(
            f'<line x1="{last_x:.2f}" y1="{top}" x2="{last_x:.2f}" '
            f'y2="{top + plot_height}" stroke="#64748b" stroke-width="1.5" '
            f'stroke-dasharray="1 4" data-last-epoch="{last_epoch}" '
            f'data-stopped-early="{str(stopped_early).lower()}"/>'
        )
        best_x = _scale(best_epoch, x_min, x_max, left, left + panel_width)
        lines.append(
            f'<line x1="{best_x:.2f}" y1="{top}" x2="{best_x:.2f}" '
            f'y2="{top + plot_height}" stroke="#111827" stroke-width="1.5" '
            f'stroke-dasharray="3 4" data-best-epoch="{best_epoch}"/>'
        )
        legend_y = top + 17
        for series_index, (label, points, color, dash, key) in enumerate(series):
            if points:
                lines.append(
                    _polyline(
                        points,
                        color,
                        left,
                        top,
                        panel_width,
                        plot_height,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        dash=dash,
                        data_series=key,
                    )
                )
            legend_x = left + 10
            y = legend_y + series_index * 19
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<line x1="{legend_x:.2f}" y1="{y:.2f}" '
                f'x2="{legend_x + 20:.2f}" y2="{y:.2f}" stroke="{color}" '
                f'stroke-width="2.5"{dash_attr}/>'
            )
            lines.append(
                f'<text x="{legend_x + 27:.2f}" y="{y + 4:.2f}" '
                f'class="legend">{escape(label)}</text>'
            )
        if not finite_points:
            lines.append(
                f'<text x="{left + panel_width / 2:.2f}" '
                f'y="{top + plot_height / 2:.2f}" class="note" '
                'text-anchor="middle">No finite values</text>'
            )
    last_label = "early-stop epoch" if stopped_early else "last completed epoch"
    lines.append(
        f'<text x="{width / 2:.2f}" y="{height - 18}" class="note" '
        'text-anchor="middle">Black dashed: validation-selected best epoch. '
        f'Gray dotted: {last_label}. '
        "Test data are evaluated only on the restored best checkpoint.</text>"
    )
    lines.append("</svg>")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #111827; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".panel-title { font-size: 15px; font-weight: 650; }",
        ".tick { font-size: 11px; fill: #475569; }",
        ".legend { font-size: 11px; }",
        ".note { font-size: 11px; fill: #64748b; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.2f}" y="32" class="title" '
        f'text-anchor="middle">{escape(title)}</text>',
    ]


def _axes(
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    x_label: str,
    y_label: str | None,
) -> list[str]:
    lines = [
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" fill="#f8fafc" stroke="#cbd5e1"/>'
    ]
    for index in range(6):
        fraction = index / 5
        x = left + fraction * width
        value = x_min + fraction * (x_max - x_min)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" '
            f'y2="{top + height:.2f}" stroke="#e2e8f0"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + height + 19:.2f}" class="tick" '
            f'text-anchor="middle">{value:.0f}</text>'
        )
        y = top + height - fraction * height
        y_value = y_min + fraction * (y_max - y_min)
        lines.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + width:.2f}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
        )
        lines.append(
            f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" class="tick" '
            f'text-anchor="end">{y_value:.3g}</text>'
        )
    lines.append(
        f'<text x="{left + width / 2:.2f}" y="{top + height + 42:.2f}" '
        f'class="tick" text-anchor="middle">{escape(x_label)}</text>'
    )
    if y_label:
        lines.append(
            f'<text x="18" y="{top + height / 2:.2f}" class="tick" '
            f'text-anchor="middle" transform="rotate(-90 18 {top + height / 2:.2f})">'
            f'{escape(y_label)}</text>'
        )
    return lines


def _polyline(
    points: Sequence[tuple[float, float]],
    color: str,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    dash: str | None = None,
    data_series: str,
) -> str:
    scaled = [
        _scale_point(
            point, left, top, width, height, x_min, x_max, y_min, y_max
        )
        for point in points
    ]
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in scaled)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
        f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"'
        f'{dash_attr} data-series="{escape(data_series)}"/>'
    )


def _points(
    history: Sequence[HistoryRow], key: str
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in history:
        epoch = row.get("epoch")
        value = row.get(key)
        if _finite(epoch) and _finite(value):
            points.append((float(epoch), float(value)))
    return points


def _bounds(
    values: Sequence[float], *, include_zero: bool = False
) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if include_zero:
        minimum = min(0.0, minimum)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
    else:
        padding = (maximum - minimum) * 0.05
    lower = minimum if include_zero and minimum == 0.0 else minimum - padding
    return lower, maximum + padding


def _scale_point(
    point: tuple[float, float],
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float]:
    return (
        _scale(point[0], x_min, x_max, left, left + width),
        _scale(point[1], y_min, y_max, top + height, top),
    )


def _scale(
    value: float,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
) -> float:
    fraction = (float(value) - input_min) / (input_max - input_min)
    return output_min + fraction * (output_max - output_min)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
