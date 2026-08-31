#!/usr/bin/env python3
"""Load one NUM sample and write a source-relative polar uncertainty SVG."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import validate_artifact_path  # noqa: E402
from nbv.data import NUMDataset  # noqa: E402
from nbv.geometry import canonical_anchors  # noqa: E402


UncertaintyDirection = Literal[
    "auto", "higher-is-uncertain", "lower-is-uncertain"
]

_LOWER_IS_UNCERTAIN_TARGETS = frozenset({"PSNR", "SSIM"})
_HIGHER_IS_UNCERTAIN_TARGETS = frozenset({"MSE", "LPIPS"})
_VIRIDIS_STOPS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one official PUN/NUM image-target pair"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="all"
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--target", default="PSNR")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--uncertainty-direction",
        choices=("auto", "higher-is-uncertain", "lower-is-uncertain"),
        default="auto",
        help=(
            "How target values map to uncertainty. 'auto' inverts PSNR/SSIM "
            "and uses MSE/LPIPS directly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/phase1/num_sample/seed_0/figures/num_sample.svg"
        ),
    )
    parser.add_argument(
        "--interactive-output",
        type=Path,
        help=(
            "Self-contained rotatable 3D sphere HTML. Defaults to "
            "<output-stem>_3d.html beside --output."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-object",
        action="store_true",
        help="Permit fewer than 48 source images (useful for a tiny fixture)",
    )
    return parser.parse_args()


def normalize_uncertainty(
    target: np.ndarray,
    target_name: str,
    direction: UncertaintyDirection = "auto",
) -> tuple[np.ndarray, Literal["higher", "lower"]]:
    """Map one raw 48-anchor target to normalized uncertainty in ``[0, 1]``."""

    values = np.asarray(target, dtype=np.float64)
    if values.shape != (len(canonical_anchors()),):
        raise ValueError(
            f"target must have shape ({len(canonical_anchors())},), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("target values must be finite")

    metric = target_name.strip().upper()
    if direction == "auto":
        if metric in _LOWER_IS_UNCERTAIN_TARGETS:
            resolved_direction: Literal["higher", "lower"] = "lower"
        elif metric in _HIGHER_IS_UNCERTAIN_TARGETS:
            resolved_direction = "higher"
        else:
            raise ValueError(
                f"Unknown uncertainty direction for target {target_name!r}; "
                "pass --uncertainty-direction explicitly"
            )
    elif direction == "higher-is-uncertain":
        resolved_direction = "higher"
    elif direction == "lower-is-uncertain":
        resolved_direction = "lower"
    else:
        raise ValueError(f"Unsupported uncertainty direction: {direction!r}")

    minimum = float(np.min(values))
    scale = float(np.max(values)) - minimum
    if scale == 0:
        uncertainty = np.zeros_like(values)
    else:
        uncertainty = (values - minimum) / scale
        if resolved_direction == "lower":
            uncertainty = 1.0 - uncertainty
    return uncertainty.astype(np.float32), resolved_direction


def _viridis_rgb(values: np.ndarray) -> np.ndarray:
    """Return an approximate viridis RGB color for values in ``[0, 1]``."""

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


def _polar_map_data_uri(uncertainty: np.ndarray, diameter: int = 500) -> str:
    """Rasterize nearest-anchor spherical cells into a transparent PNG URI."""

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
    anchor_directions = canonical_anchors().directions.astype(np.float32)
    nearest_anchor_ids = np.argmax(directions @ anchor_directions.T, axis=1)

    rgba = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    rgba[inside, :3] = _viridis_rgb(uncertainty[nearest_anchor_ids])
    rgba[inside, 3] = 255
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _source_image_data_uri(source: Path) -> str:
    """Return a normalized, self-contained PNG representation of the input."""

    with Image.open(source) as image:
        rgb = image.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_for_script(value: object) -> str:
    """Serialize JSON without permitting an embedded ``</script>`` boundary."""

    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def write_interactive_sphere_html(
    target: np.ndarray,
    source_image_path: Path,
    destination: Path,
    *,
    title: str,
    target_name: str,
    source_global_anchor_id: int,
    uncertainty_direction: UncertaintyDirection = "auto",
) -> None:
    """Write a dependency-free, rotatable 3D uncertainty sphere."""

    uncertainty, resolved_direction = normalize_uncertainty(
        target, target_name, uncertainty_direction
    )
    most_uncertain_id = int(np.argmax(uncertainty))
    payload = {
        "title": title,
        "targetName": target_name,
        "sourceGlobalAnchorId": source_global_anchor_id,
        "sourceImageUri": _source_image_data_uri(source_image_path),
        "rawMinimum": float(np.min(target)),
        "rawMaximum": float(np.max(target)),
        "resolvedDirection": resolved_direction,
        "mostUncertainId": most_uncertain_id,
        "anchors": [
            {
                "id": anchor.anchor_id,
                "direction": list(anchor.direction),
                "raw": float(raw_value),
                "uncertainty": float(normalized_value),
            }
            for anchor, raw_value, normalized_value in zip(
                canonical_anchors(), target, uncertainty, strict=True
            )
        ],
    }
    payload_json = _json_for_script(payload)
    document = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive NUM uncertainty sphere</title>
<style>
  :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f5f6f8; color: #17191d; }
  main { max-width: 1320px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 6px; font-size: clamp(19px, 2.4vw, 28px); }
  .subtitle { margin: 0 0 20px; color: #555d68; }
  .layout { display: grid; grid-template-columns: minmax(260px, 350px) minmax(500px, 1fr); gap: 20px; }
  .card { background: #fff; border: 1px solid #d8dce2; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px #1720330d; }
  h2 { margin: 0 0 12px; font-size: 17px; }
  #inputImage { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #fff; border: 1px solid #e1e4e8; }
  .meta { margin: 14px 0 0; display: grid; gap: 7px; font-size: 13px; color: #444b55; }
  .sphere-card { min-width: 0; }
  .canvas-wrap { position: relative; width: 100%; aspect-ratio: 1.22; min-height: 480px; background: radial-gradient(circle at 50% 45%, #fff 0, #f7f8fa 65%, #eef0f3 100%); border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; outline: none; }
  canvas:active { cursor: grabbing; }
  canvas:focus-visible { box-shadow: inset 0 0 0 3px #276ef1; }
  #readout { position: absolute; left: 12px; top: 12px; min-height: 44px; padding: 8px 10px; border-radius: 7px; background: #ffffffea; border: 1px solid #d8dce2; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; pointer-events: none; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 9px 14px; margin-top: 12px; }
  button { border: 1px solid #aeb5bf; border-radius: 7px; padding: 7px 11px; background: #fff; color: #20242a; cursor: pointer; }
  button:hover { background: #f1f3f5; }
  label { font-size: 13px; color: #444b55; }
  .hint { margin-left: auto; font-size: 12px; color: #68717d; }
  .legend { margin-top: 15px; }
  .bar { height: 18px; border: 1px solid #777; background: linear-gradient(90deg, #440154, #3b528b 25%, #21918c 50%, #5ec962 75%, #fde725); }
  .legend-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px; color: #444b55; }
  .best { margin-top: 10px; color: #b42318; font-size: 13px; font-weight: 600; }
  @media (max-width: 900px) {
    main { padding: 14px; }
    .layout { grid-template-columns: 1fr; }
    .source-card { max-width: 480px; }
    .canvas-wrap { min-height: 420px; }
  }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="subtitle">Source-relative UMap on a rotatable sphere. The initial front-facing point is local anchor 0 (+Z).</p>
  <div class="layout">
    <section class="card source-card">
      <h2>Input RGB image</h2>
      <img id="inputImage" alt="NUM source view">
      <div class="meta">
        <span id="sourceMeta"></span>
        <span id="rangeMeta"></span>
        <span id="directionMeta"></span>
      </div>
    </section>
    <section class="card sphere-card">
      <h2>Interactive 3D uncertainty sphere</h2>
      <div class="canvas-wrap">
        <canvas id="sphere" tabindex="0" aria-label="Rotatable three-dimensional uncertainty sphere">Canvas is required for the interactive sphere.</canvas>
        <div id="readout">Drag to rotate · wheel to zoom</div>
      </div>
      <div class="controls">
        <button id="reset" type="button">Reset view</button>
        <button id="autoRotate" type="button" aria-pressed="false">Auto-rotate</button>
        <label><input id="showAnchors" type="checkbox" checked> Show front anchors</label>
        <span class="hint">Arrow keys rotate · +/− zoom · double-click resets</span>
      </div>
      <div class="legend">
        <div class="bar"></div>
        <div class="legend-labels"><span>0 — low uncertainty</span><span>normalized uncertainty</span><span>1 — high uncertainty</span></div>
        <div class="best" id="bestMeta"></div>
      </div>
    </section>
  </div>
</main>
<script>
"use strict";
const data = __PAYLOAD__;
const canvas = document.getElementById("sphere");
const context = canvas.getContext("2d");
const readout = document.getElementById("readout");
const anchorsToggle = document.getElementById("showAnchors");
let yaw = 0;
let pitch = 0;
let zoom = 1;
let dragging = false;
let pointerX = 0;
let pointerY = 0;
let autoRotate = false;
let dirty = true;
let viewWidth = 700;
let viewHeight = 570;

document.getElementById("title").textContent = data.title;
document.getElementById("inputImage").src = data.sourceImageUri;
document.getElementById("sourceMeta").textContent = `Global source anchor: ${data.sourceGlobalAnchorId}; local UMap source: 0 (+Z)`;
document.getElementById("rangeMeta").textContent = `${data.targetName} range: ${format(data.rawMinimum)}–${format(data.rawMaximum)}`;
document.getElementById("directionMeta").textContent = data.resolvedDirection === "lower"
  ? `Low ${data.targetName} → high uncertainty`
  : `High ${data.targetName} → high uncertainty`;
const bestAnchor = data.anchors[data.mostUncertainId];
document.getElementById("bestMeta").textContent = `Red ring: most uncertain local anchor ${bestAnchor.id} (${data.targetName}=${format(bestAnchor.raw)})`;

function format(value) {
  return Number(value).toPrecision(6).replace(/\.?0+$/, "");
}

function normalize(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function spherePoint(theta, phi) {
  const sine = Math.sin(theta);
  return [sine * Math.cos(phi), sine * Math.sin(phi), Math.cos(theta)];
}

function nearestAnchorId(direction) {
  let bestId = 0;
  let bestDot = -Infinity;
  for (const anchor of data.anchors) {
    const candidate = anchor.direction;
    const dot = direction[0] * candidate[0] + direction[1] * candidate[1] + direction[2] * candidate[2];
    if (dot > bestDot) {
      bestDot = dot;
      bestId = anchor.id;
    }
  }
  return bestId;
}

function viridis(value) {
  const stops = [
    [0.00, [68, 1, 84]], [0.25, [59, 82, 139]], [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]], [1.00, [253, 231, 37]]
  ];
  const clipped = Math.max(0, Math.min(1, value));
  for (let index = 0; index < stops.length - 1; index += 1) {
    const left = stops[index];
    const right = stops[index + 1];
    if (clipped <= right[0]) {
      const weight = (clipped - left[0]) / (right[0] - left[0]);
      const rgb = left[1].map((channel, channelIndex) =>
        Math.round(channel * (1 - weight) + right[1][channelIndex] * weight));
      return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }
  }
  return "rgb(253,231,37)";
}

const mesh = [];
const latitudeSteps = 32;
const longitudeSteps = 64;
for (let latitude = 0; latitude < latitudeSteps; latitude += 1) {
  const theta0 = Math.PI * latitude / latitudeSteps;
  const theta1 = Math.PI * (latitude + 1) / latitudeSteps;
  for (let longitude = 0; longitude < longitudeSteps; longitude += 1) {
    const phi0 = 2 * Math.PI * longitude / longitudeSteps;
    const phi1 = 2 * Math.PI * (longitude + 1) / longitudeSteps;
    const a = spherePoint(theta0, phi0);
    const b = spherePoint(theta1, phi0);
    const c = spherePoint(theta1, phi1);
    const d = spherePoint(theta0, phi1);
    for (const vertices of [[a, b, c], [a, c, d]]) {
      const midpoint = normalize([
        vertices[0][0] + vertices[1][0] + vertices[2][0],
        vertices[0][1] + vertices[1][1] + vertices[2][1],
        vertices[0][2] + vertices[1][2] + vertices[2][2]
      ]);
      const anchorId = nearestAnchorId(midpoint);
      mesh.push({ vertices, color: viridis(data.anchors[anchorId].uncertainty) });
    }
  }
}

function rotate(vector) {
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const x = cosYaw * vector[0] + sinYaw * vector[2];
  const yawZ = -sinYaw * vector[0] + cosYaw * vector[2];
  const y = cosPitch * vector[1] - sinPitch * yawZ;
  const z = sinPitch * vector[1] + cosPitch * yawZ;
  return [x, y, z];
}

function resizeCanvas() {
  const rectangle = canvas.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  viewWidth = rectangle.width;
  viewHeight = rectangle.height;
  canvas.width = Math.round(viewWidth * ratio);
  canvas.height = Math.round(viewHeight * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  dirty = true;
}

function render() {
  context.clearRect(0, 0, viewWidth, viewHeight);
  const centerX = viewWidth / 2;
  const centerY = viewHeight / 2;
  const radius = Math.min(viewWidth, viewHeight) * 0.40 * zoom;
  const transformed = mesh.map(triangle => {
    const vertices = triangle.vertices.map(rotate);
    return { vertices, color: triangle.color, depth: vertices.reduce((sum, point) => sum + point[2], 0) / 3 };
  }).sort((left, right) => left.depth - right.depth);

  context.lineJoin = "round";
  for (const triangle of transformed) {
    context.beginPath();
    triangle.vertices.forEach((point, index) => {
      const x = centerX + radius * point[0];
      const y = centerY - radius * point[1];
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fillStyle = triangle.color;
    context.strokeStyle = triangle.color;
    context.lineWidth = 1.25;
    context.fill();
    context.stroke();
  }

  context.beginPath();
  context.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  context.strokeStyle = "#20242a";
  context.lineWidth = 1.5;
  context.stroke();

  if (anchorsToggle.checked) {
    const visibleAnchors = data.anchors.map(anchor => ({ anchor, point: rotate(anchor.direction) }))
      .filter(item => item.point[2] >= 0)
      .sort((left, right) => left.point[2] - right.point[2]);
    for (const item of visibleAnchors) {
      const x = centerX + radius * item.point[0];
      const y = centerY - radius * item.point[1];
      const isSource = item.anchor.id === 0;
      const isBest = item.anchor.id === data.mostUncertainId;
      context.beginPath();
      context.arc(x, y, isBest ? 8 : (isSource ? 6 : 3), 0, 2 * Math.PI);
      context.fillStyle = isSource ? "#ffffff" : "#17191d";
      context.fill();
      context.strokeStyle = isBest ? "#ff3b30" : "#ffffff";
      context.lineWidth = isBest ? 4 : 1.5;
      context.stroke();
    }
  }
  dirty = false;
}

function updateReadout(clientX, clientY) {
  const rectangle = canvas.getBoundingClientRect();
  const centerX = rectangle.width / 2;
  const centerY = rectangle.height / 2;
  const radius = Math.min(rectangle.width, rectangle.height) * 0.40 * zoom;
  let nearest = null;
  let distance = 14;
  for (const anchor of data.anchors) {
    const point = rotate(anchor.direction);
    if (point[2] < 0) continue;
    const x = centerX + radius * point[0];
    const y = centerY - radius * point[1];
    const candidateDistance = Math.hypot(clientX - rectangle.left - x, clientY - rectangle.top - y);
    if (candidateDistance < distance) {
      distance = candidateDistance;
      nearest = anchor;
    }
  }
  readout.textContent = nearest
    ? `local anchor ${nearest.id}\n${data.targetName}: ${format(nearest.raw)} · uncertainty: ${nearest.uncertainty.toFixed(4)}`
    : "Drag to rotate · wheel to zoom";
  readout.style.whiteSpace = "pre-line";
}

function resetView() {
  yaw = 0;
  pitch = 0;
  zoom = 1;
  dirty = true;
}

canvas.addEventListener("pointerdown", event => {
  dragging = true;
  pointerX = event.clientX;
  pointerY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (dragging) {
    yaw += (event.clientX - pointerX) * 0.010;
    pitch += (event.clientY - pointerY) * 0.010;
    pitch = Math.max(-Math.PI, Math.min(Math.PI, pitch));
    pointerX = event.clientX;
    pointerY = event.clientY;
    dirty = true;
  }
  updateReadout(event.clientX, event.clientY);
});
canvas.addEventListener("pointerup", event => {
  dragging = false;
  canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointercancel", () => { dragging = false; });
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  zoom = Math.max(0.65, Math.min(1.35, zoom * Math.exp(-event.deltaY * 0.001)));
  dirty = true;
}, { passive: false });
canvas.addEventListener("dblclick", resetView);
canvas.addEventListener("keydown", event => {
  const rotationStep = 0.10;
  if (event.key === "ArrowLeft") yaw -= rotationStep;
  else if (event.key === "ArrowRight") yaw += rotationStep;
  else if (event.key === "ArrowUp") pitch -= rotationStep;
  else if (event.key === "ArrowDown") pitch += rotationStep;
  else if (event.key === "+" || event.key === "=") zoom = Math.min(1.35, zoom + 0.05);
  else if (event.key === "-" || event.key === "_") zoom = Math.max(0.65, zoom - 0.05);
  else return;
  event.preventDefault();
  dirty = true;
});
document.getElementById("reset").addEventListener("click", resetView);
document.getElementById("autoRotate").addEventListener("click", event => {
  autoRotate = !autoRotate;
  event.currentTarget.setAttribute("aria-pressed", String(autoRotate));
  event.currentTarget.textContent = autoRotate ? "Stop rotation" : "Auto-rotate";
  dirty = true;
});
anchorsToggle.addEventListener("change", () => { dirty = true; });
window.addEventListener("resize", resizeCanvas);

function animationFrame() {
  if (autoRotate && !dragging) {
    yaw += 0.004;
    dirty = true;
  }
  if (dirty) render();
  window.requestAnimationFrame(animationFrame);
}

resizeCanvas();
window.requestAnimationFrame(animationFrame);
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)
    validate_artifact_path(destination.parent, "interactive output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def write_target_svg(
    target: np.ndarray,
    source_image_path: Path,
    destination: Path,
    *,
    title: str,
    target_name: str,
    source_global_anchor_id: int,
    uncertainty_direction: UncertaintyDirection = "auto",
) -> None:
    """Write an input-image and source-relative polar uncertainty-map SVG."""

    uncertainty, resolved_direction = normalize_uncertainty(
        target, target_name, uncertainty_direction
    )
    raw_minimum, raw_maximum = float(np.min(target)), float(np.max(target))
    most_uncertain_id = int(np.argmax(uncertainty))
    most_uncertain_value = float(target[most_uncertain_id])
    source_image_uri = _source_image_data_uri(source_image_path)
    polar_map_uri = _polar_map_data_uri(uncertainty)

    width, height = 1280, 730
    map_center_x, map_center_y, map_radius = 895.0, 335.0, 250.0
    map_left, map_top = map_center_x - map_radius, map_center_y - map_radius
    anchor_marks: list[str] = []
    for anchor, raw_value, normalized_value in zip(
        canonical_anchors(), target, uncertainty, strict=True
    ):
        radial = anchor.polar_angle_rad / math.pi * map_radius
        x = map_center_x + radial * math.cos(anchor.azimuth_rad)
        y = map_center_y - radial * math.sin(anchor.azimuth_rad)
        role = "source" if anchor.anchor_id == 0 else "anchor"
        radius = 7 if role == "source" else 3.5
        fill = "#ffffff" if role == "source" else "#111111"
        stroke_width = 3 if role == "source" else 1
        anchor_marks.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" '
            f'fill="{fill}" stroke="#111111" stroke-width="{stroke_width}" '
            f'data-role="{role}" data-anchor-id="{anchor.anchor_id}">'
            f'<title>local anchor {anchor.anchor_id}: {target_name} '
            f'{float(raw_value):.6g}, normalized uncertainty '
            f'{float(normalized_value):.4f}</title></circle>'
        )

    best_anchor = canonical_anchors().by_id(most_uncertain_id)
    best_radial = best_anchor.polar_angle_rad / math.pi * map_radius
    best_x = map_center_x + best_radial * math.cos(best_anchor.azimuth_rad)
    best_y = map_center_y - best_radial * math.sin(best_anchor.azimuth_rad)
    anchor_marks.append(
        f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="10" fill="none" '
        f'stroke="#ff3b30" stroke-width="4" data-role="most-uncertain" '
        f'data-anchor-id="{most_uncertain_id}"><title>most uncertain local '
        f'anchor {most_uncertain_id}</title></circle>'
    )

    grid_marks: list[str] = []
    for degrees in (45, 90, 135):
        radius = map_radius * degrees / 180
        grid_marks.append(
            f'<circle cx="{map_center_x}" cy="{map_center_y}" r="{radius:.2f}" '
            'fill="none" stroke="#ffffff" stroke-opacity="0.55" '
            'stroke-width="1"/>'
        )
        grid_marks.append(
            f'<text x="{map_center_x + 5}" y="{map_center_y - radius + 15:.2f}" '
            f'font-family="sans-serif" font-size="11" fill="#ffffff" '
            f'stroke="#111111" stroke-width="2" paint-order="stroke">'
            f'{degrees}°</text>'
        )
    for degrees, label_x, label_y, anchor in (
        (0, map_center_x + map_radius + 17, map_center_y + 4, "start"),
        (90, map_center_x, map_center_y - map_radius - 13, "middle"),
        (180, map_center_x - map_radius - 17, map_center_y + 4, "end"),
        (270, map_center_x, map_center_y + map_radius + 22, "middle"),
    ):
        grid_marks.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
            f'text-anchor="{anchor}" font-family="sans-serif" font-size="12" '
            f'fill="#333">φ={degrees}°</text>'
        )

    metric_explanation = (
        f"low {target_name} → high uncertainty"
        if resolved_direction == "lower"
        else f"high {target_name} → high uncertainty"
    )
    escaped_title = html.escape(title)
    escaped_source_path = html.escape(str(source_image_path))
    escaped_metric = html.escape(target_name)
    escaped_explanation = html.escape(metric_explanation)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <linearGradient id="viridis" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#440154"/>
    <stop offset="25%" stop-color="#3b528b"/>
    <stop offset="50%" stop-color="#21918c"/>
    <stop offset="75%" stop-color="#5ec962"/>
    <stop offset="100%" stop-color="#fde725"/>
  </linearGradient>
</defs>
<rect width="100%" height="100%" fill="#fafafa"/>
<text x="45" y="38" font-family="sans-serif" font-size="21" font-weight="600">{escaped_title}</text>
<text x="45" y="65" font-family="sans-serif" font-size="13" fill="#555">UMap coordinates are relative to the input view: center=current view, edge=opposite view.</text>

<text x="45" y="108" font-family="sans-serif" font-size="17" font-weight="600">Input RGB image</text>
<rect x="45" y="125" width="500" height="500" rx="4" fill="#fff" stroke="#bbb"/>
<image x="70" y="150" width="450" height="450" href="{source_image_uri}" preserveAspectRatio="xMidYMid meet"/>
<text x="45" y="651" font-family="sans-serif" font-size="12" fill="#444">global source anchor: {source_global_anchor_id}; local UMap source anchor: 0 (+Z)</text>
<text x="45" y="672" font-family="sans-serif" font-size="10" fill="#777">{escaped_source_path}</text>

<text x="645" y="108" font-family="sans-serif" font-size="17" font-weight="600">Polar neural uncertainty map</text>
<image x="{map_left:.2f}" y="{map_top:.2f}" width="{2 * map_radius:.2f}" height="{2 * map_radius:.2f}" href="{polar_map_uri}"/>
{''.join(grid_marks)}
{''.join(anchor_marks)}
<text x="895" y="335" dx="12" dy="5" font-family="sans-serif" font-size="11" fill="#111" stroke="#fff" stroke-width="3" paint-order="stroke">source</text>

<rect x="680" y="620" width="430" height="18" fill="url(#viridis)" stroke="#777"/>
<text x="680" y="655" text-anchor="middle" font-family="sans-serif" font-size="11">0 — low</text>
<text x="895" y="655" text-anchor="middle" font-family="sans-serif" font-size="12">normalized uncertainty</text>
<text x="1110" y="655" text-anchor="middle" font-family="sans-serif" font-size="11">1 — high</text>
<text x="680" y="680" font-family="sans-serif" font-size="12" fill="#444">{escaped_explanation}; raw range {raw_minimum:.6g}–{raw_maximum:.6g}</text>
<text x="1125" y="705" text-anchor="end" font-family="sans-serif" font-size="12" fill="#b42318">red ring: local anchor {most_uncertain_id} ({escaped_metric}={most_uncertain_value:.6g})</text>
</svg>
"""
    validate_artifact_path(destination.parent, "SVG output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset = NUMDataset(
        args.data_root,
        split=args.split,
        split_manifest=args.split_manifest,
        target_name=args.target,
        require_complete_objects=not args.allow_incomplete_object,
    )
    sample = dataset[args.index]
    title = (
        f"NUM {sample.target_name} UMap: {sample.record.sample_id} "
        f"(shape={tuple(sample.target_map.shape)})"
    )
    write_target_svg(
        sample.target_map,
        sample.record.image_path,
        args.output,
        title=title,
        target_name=sample.target_name,
        source_global_anchor_id=sample.source_anchor_id,
        uncertainty_direction=args.uncertainty_direction,
    )
    interactive_output = args.interactive_output
    if interactive_output is None:
        interactive_output = args.output.with_name(f"{args.output.stem}_3d.html")
    write_interactive_sphere_html(
        sample.target_map,
        sample.record.image_path,
        interactive_output,
        title=title,
        target_name=sample.target_name,
        source_global_anchor_id=sample.source_anchor_id,
        uncertainty_direction=args.uncertainty_direction,
    )
    uncertainty, _ = normalize_uncertainty(
        sample.target_map, sample.target_name, args.uncertainty_direction
    )
    print(f"sample: {sample.record.sample_id}")
    print(f"image: {sample.record.image_path}")
    print(f"preprocessed image shape: {getattr(sample.image, 'shape', None)}")
    print(f"target: {sample.target_name} {sample.target_map.shape}")
    print(f"most uncertain local anchor: {int(np.argmax(uncertainty))}")
    print(f"visualization: {args.output}")
    print(f"interactive 3D visualization: {interactive_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
