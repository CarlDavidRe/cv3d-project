"""Static prediction-versus-target visualization for a Phase 1 experiment."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from html import escape
from io import BytesIO
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from nbv.config import validate_artifact_path
from nbv.eval import ndcg_at_k, normalized_regret, spearman_rank
from nbv.geometry import canonical_anchors


TargetDirection = Literal["higher", "lower"]
_VIRIDIS_STOPS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


@dataclass(frozen=True, slots=True)
class Phase1PredictionDiagnostics:
    """Ranking and regression diagnostics displayed by the qualitative plot."""

    predicted_anchor_id: int
    target_anchor_id: int
    normalized_regret: float
    spearman: float | None
    ndcg_at_k: float
    mean_absolute_error: float
    top_predicted_ids: tuple[int, ...]
    top_target_ids: tuple[int, ...]


def prediction_diagnostics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    *,
    target_direction: TargetDirection,
    ndcg_k: int = 5,
) -> Phase1PredictionDiagnostics:
    """Compute the exact ranking diagnostics used by the Phase 1 evaluator."""

    target_values, prediction_values, mask = _validated_arrays(
        target, prediction, valid_mask
    )
    if target_direction not in ("higher", "lower"):
        raise ValueError("target_direction must be 'higher' or 'lower'")
    if isinstance(ndcg_k, bool) or not isinstance(ndcg_k, int) or ndcg_k < 1:
        raise ValueError("ndcg_k must be a positive integer")

    sign = 1.0 if target_direction == "higher" else -1.0
    scores = sign * prediction_values
    relevance = _target_relevance(target_values, mask, target_direction)
    valid_ids = np.flatnonzero(mask)
    predicted_order = valid_ids[np.argsort(-scores[mask], kind="stable")]
    target_order = valid_ids[np.argsort(-relevance[mask], kind="stable")]
    correlation = spearman_rank(scores, relevance, mask)
    return Phase1PredictionDiagnostics(
        predicted_anchor_id=int(predicted_order[0]),
        target_anchor_id=int(target_order[0]),
        normalized_regret=normalized_regret(scores, relevance, mask),
        spearman=(float(correlation) if math.isfinite(correlation) else None),
        ndcg_at_k=ndcg_at_k(scores, relevance, mask, k=ndcg_k),
        mean_absolute_error=float(
            np.mean(np.abs(prediction_values[mask] - target_values[mask]))
        ),
        top_predicted_ids=tuple(int(value) for value in predicted_order[:ndcg_k]),
        top_target_ids=tuple(int(value) for value in target_order[:ndcg_k]),
    )


def write_phase1_prediction_svg(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    source_image_path: str | Path,
    destination: str | Path,
    *,
    title: str,
    variant_name: str,
    sample_id: str,
    split: str,
    target_name: str,
    target_direction: TargetDirection,
    source_global_anchor_id: int,
    feature_source: str,
    ndcg_k: int = 5,
) -> Phase1PredictionDiagnostics:
    """Write one self-contained SVG comparing prediction and ground truth."""

    target_values, prediction_values, mask = _validated_arrays(
        target, prediction, valid_mask
    )
    diagnostics = prediction_diagnostics(
        target_values,
        prediction_values,
        mask,
        target_direction=target_direction,
        ndcg_k=ndcg_k,
    )
    target_utility, prediction_utility = _shared_utility_scale(
        target_values, prediction_values, mask, target_direction
    )
    utility_error = np.abs(prediction_utility - target_utility)
    target_utility[~mask] = 0.0
    prediction_utility[~mask] = 0.0
    utility_error[~mask] = 0.0

    image_uri = _source_image_data_uri(Path(source_image_path))
    target_uri = _polar_map_data_uri(target_utility, mask)
    prediction_uri = _polar_map_data_uri(prediction_utility, mask)
    error_uri = _polar_map_data_uri(utility_error, mask)

    width, height = 1680, 850
    card_y, card_size = 118.0, 350.0
    card_positions = (35.0, 440.0, 845.0, 1250.0)
    map_radius = 160.0
    map_y = card_y + 210.0
    ground_truth_marks = _anchor_marks(
        card_positions[1] + card_size / 2,
        map_y,
        map_radius,
        mask,
        highlighted_id=diagnostics.target_anchor_id,
        highlight_role="target-best",
        highlight_color="#16a34a",
    )
    prediction_marks = _anchor_marks(
        card_positions[2] + card_size / 2,
        map_y,
        map_radius,
        mask,
        highlighted_id=diagnostics.predicted_anchor_id,
        highlight_role="predicted-best",
        highlight_color="#dc2626",
    )
    error_marks = _anchor_marks(
        card_positions[3] + card_size / 2,
        map_y,
        map_radius,
        mask,
        highlighted_id=None,
        highlight_role="error",
        highlight_color="#111827",
    )
    direction_text = (
        f"higher {target_name} = higher utility"
        if target_direction == "higher"
        else f"lower {target_name} = higher utility"
    )
    spearman_text = (
        "undefined"
        if diagnostics.spearman is None
        else f"{diagnostics.spearman:.4f}"
    )
    top_rows = _top_candidate_rows(
        diagnostics,
        target_values,
        prediction_values,
        target_direction,
    )
    output = Path(destination)
    validate_artifact_path(output.parent, "prediction figure directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <linearGradient id="viridis" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#440154"/><stop offset="25%" stop-color="#3b528b"/>
    <stop offset="50%" stop-color="#21918c"/><stop offset="75%" stop-color="#5ec962"/>
    <stop offset="100%" stop-color="#fde725"/>
  </linearGradient>
  <style>
    text {{ font-family: ui-sans-serif, system-ui, sans-serif; fill: #111827; }}
    .title {{ font-size: 23px; font-weight: 700; }}
    .subtitle {{ font-size: 13px; fill: #475569; }}
    .card-title {{ font-size: 16px; font-weight: 650; }}
    .small {{ font-size: 11px; fill: #475569; }}
    .metric {{ font-size: 14px; font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="35" y="36" class="title">{escape(title)}</text>
<text x="35" y="62" class="subtitle">experiment variant: {escape(variant_name)} · split: {escape(split)} · sample: {escape(sample_id)}</text>
<text x="35" y="82" class="subtitle">source global anchor {source_global_anchor_id}; local source anchor 0 is masked · {escape(direction_text)}</text>
{_card(card_positions[0], card_y, card_size, "Input RGB image")}
<image x="55" y="158" width="310" height="310" href="{image_uri}" preserveAspectRatio="xMidYMid meet"/>
{_card(card_positions[1], card_y, card_size, "Ground-truth utility map")}
<image x="{card_positions[1] + 15:.1f}" y="{map_y - map_radius:.1f}" width="{2 * map_radius:.1f}" height="{2 * map_radius:.1f}" href="{target_uri}"/>
{ground_truth_marks}
{_card(card_positions[2], card_y, card_size, "Predicted utility map")}
<image x="{card_positions[2] + 15:.1f}" y="{map_y - map_radius:.1f}" width="{2 * map_radius:.1f}" height="{2 * map_radius:.1f}" href="{prediction_uri}"/>
{prediction_marks}
{_card(card_positions[3], card_y, card_size, "Absolute utility error")}
<image x="{card_positions[3] + 15:.1f}" y="{map_y - map_radius:.1f}" width="{2 * map_radius:.1f}" height="{2 * map_radius:.1f}" href="{error_uri}"/>
{error_marks}
<rect x="440" y="505" width="1160" height="16" fill="url(#viridis)" stroke="#64748b"/>
<text x="440" y="540" class="small">0 — low</text><text x="1020" y="540" class="small" text-anchor="middle">normalized utility / absolute normalized error</text><text x="1600" y="540" class="small" text-anchor="end">1 — high</text>
<text x="35" y="585" class="metric">Predicted top: local anchor {diagnostics.predicted_anchor_id}</text>
<text x="360" y="585" class="metric">Ground-truth top: local anchor {diagnostics.target_anchor_id}</text>
<text x="710" y="585" class="metric">Normalized regret: {diagnostics.normalized_regret:.4f}</text>
<text x="1010" y="585" class="metric">Spearman: {spearman_text}</text>
<text x="1260" y="585" class="metric">NDCG@{ndcg_k}: {diagnostics.ndcg_at_k:.4f}</text>
<text x="1480" y="585" class="metric">MAE: {diagnostics.mean_absolute_error:.4f}</text>
<rect x="35" y="615" width="1610" height="195" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
<text x="55" y="643" class="card-title">Top-{ndcg_k} candidate comparison</text>
<text x="55" y="670" class="mono">rank</text><text x="130" y="670" class="mono">predicted order</text><text x="490" y="670" class="mono">ground-truth order</text><text x="900" y="670" class="mono">raw {escape(target_name)}: prediction / target</text>
{top_rows}
<text x="55" y="820" class="small">Feature source: {escape(feature_source)} · green ring = ground-truth best · red ring = predicted best · gray crossed cell = masked source view</text>
</svg>
"""
    output.write_text(document, encoding="utf-8")
    return diagnostics


def _validated_arrays(
    target: np.ndarray, prediction: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_count = len(canonical_anchors())
    target_values = np.asarray(target, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    expected = (anchor_count,)
    if target_values.shape != expected or prediction_values.shape != expected:
        raise ValueError(f"target and prediction must have shape {expected}")
    if mask.shape != expected:
        raise ValueError(f"valid_mask must have shape {expected}")
    if not mask.any():
        raise ValueError("valid_mask must contain at least one valid anchor")
    if not np.isfinite(target_values).all() or not np.isfinite(
        prediction_values
    ).all():
        raise ValueError("target and prediction values must be finite")
    return target_values.copy(), prediction_values.copy(), mask.copy()


def _target_relevance(
    target: np.ndarray, valid_mask: np.ndarray, direction: TargetDirection
) -> np.ndarray:
    oriented = target.copy() if direction == "higher" else -target
    valid = oriented[valid_mask]
    minimum = float(valid.min())
    scale = float(valid.max()) - minimum
    if scale == 0.0:
        return np.zeros_like(oriented)
    return (oriented - minimum) / scale


def _shared_utility_scale(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    direction: TargetDirection,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if direction == "higher" else -1.0
    oriented_target = sign * target
    oriented_prediction = sign * prediction
    valid = oriented_target[valid_mask]
    minimum = float(valid.min())
    scale = float(valid.max()) - minimum
    if scale == 0.0:
        return np.zeros_like(target), np.zeros_like(prediction)
    return (
        np.clip((oriented_target - minimum) / scale, 0.0, 1.0),
        np.clip((oriented_prediction - minimum) / scale, 0.0, 1.0),
    )


def _source_image_data_uri(source: Path) -> str:
    with Image.open(source) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _polar_map_data_uri(
    values: np.ndarray, valid_mask: np.ndarray, diameter: int = 360
) -> str:
    coordinates = np.arange(diameter, dtype=np.float32) + np.float32(0.5)
    center = np.float32(diameter / 2)
    radius = np.float32(diameter / 2 - 1)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    x = (x_grid - center) / radius
    y = (center - y_grid) / radius
    radial = np.sqrt(x * x + y * y)
    inside = radial <= 1.0
    theta = np.minimum(radial[inside], 1.0) * np.float32(math.pi)
    phi = np.mod(np.arctan2(y[inside], x[inside]), np.float32(2 * math.pi))
    sin_theta = np.sin(theta)
    directions = np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta))
    ).astype(np.float32)
    nearest_ids = np.argmax(
        directions @ canonical_anchors().directions.astype(np.float32).T,
        axis=1,
    )
    colors = _viridis_rgb(values[nearest_ids])
    colors[~valid_mask[nearest_ids]] = np.asarray((203, 213, 225), dtype=np.uint8)
    rgba = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    rgba[inside, :3] = colors
    rgba[inside, 3] = 255
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _viridis_rgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    result = np.empty((*clipped.shape, 3), dtype=np.float32)
    for (left_x, left_rgb), (right_x, right_rgb) in zip(
        _VIRIDIS_STOPS[:-1], _VIRIDIS_STOPS[1:], strict=True
    ):
        selected = (clipped >= left_x) & (
            (clipped <= right_x) if right_x == 1.0 else (clipped < right_x)
        )
        weight = (clipped[selected] - left_x) / (right_x - left_x)
        result[selected] = (
            np.asarray(left_rgb, dtype=np.float32) * (1.0 - weight[:, None])
            + np.asarray(right_rgb, dtype=np.float32) * weight[:, None]
        )
    return np.rint(result).astype(np.uint8)


def _card(x: float, y: float, size: float, title: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{size}" height="{size + 30}" '
        'rx="9" fill="#ffffff" stroke="#cbd5e1"/>'
        f'<text x="{x + 15}" y="{y + 28}" class="card-title">'
        f"{escape(title)}</text>"
    )


def _anchor_marks(
    center_x: float,
    center_y: float,
    radius: float,
    valid_mask: np.ndarray,
    *,
    highlighted_id: int | None,
    highlight_role: str,
    highlight_color: str,
) -> str:
    marks: list[str] = []
    for anchor in canonical_anchors():
        radial = anchor.polar_angle_rad / math.pi * radius
        x = center_x + radial * math.cos(anchor.azimuth_rad)
        y = center_y - radial * math.sin(anchor.azimuth_rad)
        if not valid_mask[anchor.anchor_id]:
            marks.append(
                f'<path d="M {x - 6:.2f} {y - 6:.2f} L {x + 6:.2f} '
                f'{y + 6:.2f} M {x + 6:.2f} {y - 6:.2f} L {x - 6:.2f} '
                f'{y + 6:.2f}" stroke="#475569" stroke-width="3" '
                f'data-role="masked" data-anchor-id="{anchor.anchor_id}"/>'
            )
        else:
            marks.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="#111827" '
                f'stroke="#ffffff" data-role="anchor" '
                f'data-anchor-id="{anchor.anchor_id}"/>'
            )
    if highlighted_id is not None:
        anchor = canonical_anchors().by_id(highlighted_id)
        radial = anchor.polar_angle_rad / math.pi * radius
        x = center_x + radial * math.cos(anchor.azimuth_rad)
        y = center_y - radial * math.sin(anchor.azimuth_rad)
        marks.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="10" fill="none" '
            f'stroke="{highlight_color}" stroke-width="4" '
            f'data-role="{highlight_role}" data-anchor-id="{highlighted_id}"/>'
        )
    return "".join(marks)


def _top_candidate_rows(
    diagnostics: Phase1PredictionDiagnostics,
    target: np.ndarray,
    prediction: np.ndarray,
    direction: TargetDirection,
) -> str:
    rows: list[str] = []
    for rank, (predicted_id, target_id) in enumerate(
        zip(
            diagnostics.top_predicted_ids,
            diagnostics.top_target_ids,
            strict=True,
        ),
        start=1,
    ):
        y = 696 + (rank - 1) * 23
        predicted_arrow = "↑" if direction == "higher" else "↓"
        rows.append(
            f'<text x="55" y="{y}" class="mono">{rank}</text>'
            f'<text x="130" y="{y}" class="mono">anchor {predicted_id}</text>'
            f'<text x="490" y="{y}" class="mono">anchor {target_id}</text>'
            f'<text x="900" y="{y}" class="mono">anchor {predicted_id}: '
            f'{prediction[predicted_id]:.5g} / {target[predicted_id]:.5g} '
            f'({predicted_arrow} is better)</text>'
        )
    return "".join(rows)
