"""Self-contained diagnostic SVG for a replay-verified Phase 2 rollout."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import numpy as np

from nbv.data.visibility_cache import VisibilityCache
from nbv.eval.result_schema import RolloutResult
from nbv.geometry.anchors import canonical_anchors


def write_phase2_rollout_demo(
    result: RolloutResult,
    cache: VisibilityCache,
    path: str | Path,
    *,
    decision_index: int = -1,
) -> Path:
    """Show acquired history acquired RGB, score/gain maps, coverage, and seen faces."""
    if not result.steps:
        raise ValueError("A rollout demo requires at least one decision")
    decision = decision_index if decision_index >= 0 else len(result.steps) - 1
    if decision >= len(result.steps):
        raise ValueError("decision_index is out of range")
    step = result.steps[decision]
    history = tuple(step["history_anchor_ids"])
    selected = int(step["selected_anchor"])
    seen = np.any(cache.face_visibility[np.asarray(history, dtype=int)], axis=0)
    face_weights = (
        cache.face_areas / cache.face_areas.sum()
        if result.metadata["coverage_target"] == "vis_a"
        else np.full(cache.face_visibility.shape[1], 1 / cache.face_visibility.shape[1])
    )
    order = np.argsort(face_weights)[::-1]
    bins = min(600, len(order))
    chunks = np.array_split(order, bins)
    seen_bins = [float(face_weights[c][seen[c]].sum() / face_weights[c].sum()) for c in chunks]

    width, height = 1500, 890
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:system-ui,sans-serif;fill:#0f172a}.title{font-size:24px;font-weight:700}.label{font-size:14px;font-weight:650}.small{font-size:11px;fill:#475569}.axis{stroke:#94a3b8}.grid{stroke:#e2e8f0}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="750" y="34" text-anchor="middle" class="title">Phase 2 rollout: {escape(result.metadata["policy"])} — {escape(result.metadata["object_id"])}</text>',
        f'<text x="750" y="58" text-anchor="middle" class="small">decision {decision + 1}; selected anchor {selected}; {escape(result.metadata["coverage_target"])} coverage {step["coverage_before"]:.4f} → {step["coverage_after"]:.4f}</text>',
        '<text x="35" y="92" class="label">Acquired observations available to the policy</text>',
    ]
    thumb_w, thumb_h = 132, 112
    for index, (anchor, image_path) in enumerate(zip(result.acquired_anchor_ids, result.image_paths, strict=True)):
        if index >= len(history):
            break
        x = 35 + index * 145
        data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        lines.append(f'<image x="{x}" y="108" width="{thumb_w}" height="{thumb_h}" preserveAspectRatio="xMidYMid meet" href="data:image/{suffix};base64,{data}"/>')
        lines.append(f'<text x="{x + thumb_w/2}" y="236" text-anchor="middle" class="small">anchor {int(anchor)}</text>')

    _anchor_map(lines, result.scores[decision], step["valid_candidate_mask"], selected, 55, 300, "Aggregated policy scores")
    _anchor_map(lines, result.candidate_gains[decision], step["valid_candidate_mask"], selected, 545, 300, "Evaluator-only true gains")
    lines.append('<text x="1035" y="290" class="label">Accumulated visible-face weight</text>')
    for index, value in enumerate(seen_bins):
        color = f"rgb({int(241-182*value)},{int(245-114*value)},{int(249-144*value)})"
        lines.append(f'<rect x="{1035 + index * 420/bins:.3f}" y="310" width="{420/bins + .2:.3f}" height="78" fill="{color}"/>')
    lines.append(f'<text x="1035" y="410" class="small">dark = visible; weighted coverage before decision: {step["coverage_before"]:.4f}</text>')
    _coverage_curve(lines, result, 1035, 475, 420, 285)
    lines.append(f'<text x="35" y="850" class="small">Replay fingerprint: {escape(result.metadata["visibility_cache_fingerprint"][:20])}… · scores are policy scores, not predicted surface gains.</text>')
    lines.append('</svg>')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _anchor_map(lines, values, valid, selected, left, top, title):
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    directions = canonical_anchors().directions
    x = left + 220 + np.arctan2(directions[:, 1], directions[:, 0]) / np.pi * 205
    y = top + 180 - directions[:, 2] * 145
    finite = values[valid]
    low, high = float(finite.min()), float(finite.max())
    scale = np.zeros(48) if high == low else (values - low) / (high - low)
    lines.append(f'<text x="{left}" y="{top}" class="label">{escape(title)}</text>')
    lines.append(f'<rect x="{left}" y="{top+18}" width="440" height="340" fill="#f8fafc" stroke="#cbd5e1"/>')
    for index in range(48):
        color = "#cbd5e1" if not valid[index] else f"rgb({int(239-115*scale[index])},{int(246-188*scale[index])},{int(255-10*scale[index])})"
        stroke = "#f97316" if index == selected else "#475569"
        radius = 10 if index == selected else 7
        lines.append(f'<circle cx="{x[index]:.2f}" cy="{y[index]+18:.2f}" r="{radius}" fill="{color}" stroke="{stroke}" data-anchor="{index}"/>')
        lines.append(f'<text x="{x[index]:.2f}" y="{y[index]+21:.2f}" text-anchor="middle" class="small">{index}</text>')
    lines.append(f'<text x="{left}" y="{top+378}" class="small">valid range: {low:.5g} … {high:.5g}; orange ring = selected</text>')


def _coverage_curve(lines, result, left, top, width, height):
    lines.append(f'<text x="{left}" y="{top-18}" class="label">Coverage trajectory</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>')
    counts = result.acquired_view_counts.astype(float)
    coverage = result.coverage.astype(float)
    x = left + (counts-counts.min()) / max(1, counts.max()-counts.min()) * width
    y = top + height - coverage * height
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y, strict=True))
    lines.append(f'<polyline points="{points}" fill="none" stroke="#7c3aed" stroke-width="3"/>')
    for a, b in zip(x, y, strict=True):
        lines.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="4" fill="#7c3aed"/>')
