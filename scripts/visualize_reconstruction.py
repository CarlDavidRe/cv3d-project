#!/usr/bin/env python3
"""Render an aligned cached reconstruction beside its ground-truth point cloud."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.reconstruction import align_vggt_to_num, point_cloud_metrics
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world


GT_COLOR = (37, 99, 235)
PREDICTION_COLOR = (234, 88, 12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="reconstruction_per_object.csv")
    parser.add_argument("--object-id", required=True, help="ShapeNet category/object ID")
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--views", type=int,
        help="Acquired-view count; defaults to the largest available count",
    )
    parser.add_argument(
        "--reconstruction-cache-root", type=Path,
        default=ROOT / "data/cache/reconstruction",
    )
    parser.add_argument(
        "--visibility-cache-root", type=Path,
        default=ROOT / "data/cache/visibility",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--preview-points", type=int, default=5_000,
        help="Maximum points from each cloud in the PNG preview",
    )
    parser.add_argument(
        "--interactive-points", type=int, default=8_000,
        help="Maximum points from each cloud embedded in each interactive HTML viewer",
    )
    parser.add_argument(
        "--oracle-icp", action="store_true",
        help="Also fit a ground-truth-assisted Sim(3) ICP diagnostic",
    )
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-points", type=int, default=3_000)
    args = parser.parse_args()
    try:
        row = select_row(args.metrics, args.object_id, args.policy, args.views)
        object_id = validate_object_id(args.object_id)
        prediction_path = cached_path(
            args.reconstruction_cache_root, "predictions", object_id,
            row["prediction_cache_path"],
        )
        metric_path = cached_path(
            args.reconstruction_cache_root, "metrics", object_id,
            row["metric_cache_path"],
        )
        points, predicted_cameras, prediction_identity = load_prediction(prediction_path)
        history = parse_history(row["history_anchor_ids"])
        if prediction_identity.get("object_id") != object_id:
            raise ValueError("prediction cache object_id does not match the selected row")
        if prediction_identity.get("history_anchor_ids") != history:
            raise ValueError("prediction cache history does not match the selected row")

        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        target_id = str(metric["identity"]["target_id"])
        target_path, target_points = find_ground_truth(
            args.reconstruction_cache_root, object_id, target_id
        )
        visibility_path = args.visibility_cache_root.joinpath(
            *object_id.split("/")[:-1], f"{object_id.split('/')[-1]}.npz"
        )
        visibility = load_visibility_cache(visibility_path)
        radius = float(visibility.metadata["camera_radius"])
        convention = str(visibility.metadata.get("camera_convention", CAMERA_CONVENTION))
        anchors = canonical_anchors()
        known_cameras = np.stack([
            anchor_camera_to_world(anchors.by_id(anchor_id), radius, convention=convention)
            for anchor_id in history
        ])
        transform, alignment = align_vggt_to_num(
            points, predicted_cameras, known_cameras, target_points
        )
        aligned = points @ transform[:3, :3].T + transform[:3, 3]

        view_count = int(row["acquired_view_count"])
        destination = args.output_dir or args.metrics.parent / (
            "reconstruction_visualizations/"
            f"{object_id.replace('/', '_')}_{args.policy}_{view_count}views"
        )
        destination.mkdir(parents=True, exist_ok=True)
        write_ply(destination / "prediction_aligned.ply", aligned, PREDICTION_COLOR)
        write_ply(destination / "ground_truth.ply", target_points, GT_COLOR)
        write_ply(
            destination / "comparison.ply",
            np.concatenate((target_points, aligned)),
            (GT_COLOR,) * len(target_points) + (PREDICTION_COLOR,) * len(aligned),
        )
        clipped = write_preview(
            destination / "comparison.png", target_points, aligned,
            max_points=args.preview_points,
        )
        write_interactive_html(
            destination / "comparison_interactive.html",
            target_points,
            aligned,
            max_points=args.interactive_points,
            title=f"{object_id} · {args.policy} · {view_count} views · camera alignment",
        )
        metadata = {
            "object_id": object_id,
            "policy": args.policy,
            "acquired_view_count": view_count,
            "history_anchor_ids": history,
            "prediction_cache": str(prediction_path.resolve()),
            "ground_truth_cache": str(target_path.resolve()),
            "visibility_cache": str(visibility_path.resolve()),
            "alignment": alignment,
            "alignment_transform": transform.tolist(),
            "preview_prediction_points_outside_ground_truth_frame": clipped,
            "colors": {"ground_truth": GT_COLOR, "prediction": PREDICTION_COLOR},
        }
        if args.oracle_icp:
            oracle_aligned, oracle_transform, oracle_info = similarity_icp(
                aligned, target_points,
                iterations=args.icp_iterations,
                sample_count=args.icp_points,
            )
            write_ply(
                destination / "prediction_oracle_icp.ply",
                oracle_aligned,
                PREDICTION_COLOR,
            )
            write_ply(
                destination / "comparison_oracle_icp.ply",
                np.concatenate((target_points, oracle_aligned)),
                (GT_COLOR,) * len(target_points)
                + (PREDICTION_COLOR,) * len(oracle_aligned),
            )
            oracle_clipped = write_preview(
                destination / "comparison_oracle_icp.png",
                target_points,
                oracle_aligned,
                max_points=args.preview_points,
                title=(
                    "Oracle diagnostic: ground truth (blue) vs "
                    "GT-assisted Sim(3) ICP (orange)"
                ),
            )
            oracle_metrics = point_cloud_metrics(
                oracle_aligned,
                target_points,
                chunk_size=2_000,
                fscore_thresholds=(0.01, 0.02, 0.10),
            )
            write_interactive_html(
                destination / "comparison_oracle_icp_interactive.html",
                target_points,
                oracle_aligned,
                max_points=args.interactive_points,
                title=f"{object_id} · {args.policy} · {view_count} views · oracle ICP",
            )
            metadata["oracle_icp_diagnostic"] = {
                **oracle_info,
                "transform_after_camera_alignment": oracle_transform.tolist(),
                "combined_transform_from_vggt_frame": (
                    oracle_transform @ transform
                ).tolist(),
                "metrics": oracle_metrics,
                "prediction_points_outside_ground_truth_frame": oracle_clipped,
                "warning": (
                    "Ground truth was used for alignment. These values are an "
                    "oracle structural diagnostic, not official evaluation metrics."
                ),
            }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Reconstruction comparison: {destination / 'comparison.png'}")
        print(f"Interactive-viewer point cloud: {destination / 'comparison.ply'}")
        print(f"Interactive browser viewer: {destination / 'comparison_interactive.html'}")
        if args.oracle_icp:
            print(f"Oracle ICP diagnostic: {destination / 'comparison_oracle_icp.png'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Reconstruction visualization error: {exc}", file=sys.stderr)
        return 1


def select_row(
    path: Path, object_id: str, policy: str, views: int | None
) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("object_id") == object_id and row.get("policy") == policy
        ]
    if not rows:
        raise ValueError(f"no reconstruction row for object={object_id!r}, policy={policy!r}")
    if views is None:
        views = max(int(row["acquired_view_count"]) for row in rows)
    selected = [row for row in rows if int(row["acquired_view_count"]) == views]
    if len(selected) != 1:
        raise ValueError(
            f"expected one reconstruction row at {views} views, found {len(selected)}"
        )
    return selected[0]


def validate_object_id(object_id: str) -> str:
    parts = object_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("object-id must have safe 'category/object' form")
    return object_id


def cached_path(root: Path, layer: str, object_id: str, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    local = root / layer / object_id / path.name
    if not local.is_file():
        raise ValueError(f"cache file does not exist locally: {local}")
    return local


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        points = payload["points"].copy()
        cameras = payload["camera_to_world"].copy()
        identity = json.loads(str(payload["identity_json"].item()))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"invalid prediction points in {path}")
    if cameras.ndim != 3 or cameras.shape[1:] != (4, 4):
        raise ValueError(f"invalid predicted cameras in {path}")
    return points, cameras, identity


def parse_history(value: str) -> list[int]:
    history = json.loads(value)
    if not isinstance(history, list) or not history or not all(type(x) is int for x in history):
        raise ValueError("history_anchor_ids must be a non-empty JSON integer list")
    return history


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def find_ground_truth(root: Path, object_id: str, target_id: str) -> tuple[Path, np.ndarray]:
    directory = root / "ground_truth" / object_id
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            identity = json.loads(str(payload["identity_json"].item()))
            if mapping_sha256(identity) == target_id:
                points = payload["points"].copy()
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError(f"invalid ground-truth points in {path}")
                return path, points
    raise ValueError(f"no ground-truth cache matching metric target {target_id} in {directory}")


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: Sequence[Sequence[int]] | Sequence[int],
) -> None:
    color_array = np.asarray(colors, dtype=np.uint8)
    if color_array.shape == (3,):
        color_array = np.broadcast_to(color_array, (len(points), 3))
    if color_array.shape != (len(points), 3):
        raise ValueError("PLY colors must contain one RGB triplet per point")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(points, color_array):
            handle.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_preview(
    path: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    max_points: int,
    title: str = "Ground truth (blue) vs aligned VGGT reconstruction (orange)",
) -> int:
    if max_points <= 0:
        raise ValueError("preview-points must be positive")
    size, margin, header = 430, 24, 54
    image = Image.new("RGB", (3 * size, size + header), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 14), title, fill=(15, 23, 42, 255))
    bounds_min = ground_truth.min(axis=0)
    bounds_max = ground_truth.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = max(float(np.max(bounds_max - bounds_min)), 1e-12) * 1.08
    lower, upper = center - extent / 2, center + extent / 2
    outside = int(np.sum(np.any((prediction < lower) | (prediction > upper), axis=1)))
    axes = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    gt = deterministic_subset(ground_truth, max_points)
    pred = deterministic_subset(prediction, max_points)
    for panel, (horizontal, vertical, title) in enumerate(axes):
        left, top = panel * size, header
        draw.rectangle((left, top, left + size - 1, top + size - 1), outline=(203, 213, 225, 255))
        draw.text((left + 10, top + 8), title, fill=(15, 23, 42, 255))
        for cloud, color in ((gt, GT_COLOR), (pred, PREDICTION_COLOR)):
            x = left + margin + (cloud[:, horizontal] - lower[horizontal]) / extent * (size - 2 * margin)
            y = top + size - margin - (cloud[:, vertical] - lower[vertical]) / extent * (size - 2 * margin)
            visible = (x >= left) & (x < left + size) & (y >= top) & (y < top + size)
            for px, py in zip(x[visible], y[visible]):
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(*color, 145))
    image.save(path)
    return outside


def deterministic_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[np.linspace(0, len(points) - 1, count, dtype=np.int64)]


def write_interactive_html(
    path: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    max_points: int,
    title: str,
) -> None:
    """Write a dependency-free, Phase-1-style orbitable point-cloud viewer."""

    if max_points <= 0:
        raise ValueError("interactive-points must be positive")
    gt = deterministic_subset(np.asarray(ground_truth, dtype=np.float64), max_points)
    pred = deterministic_subset(np.asarray(prediction, dtype=np.float64), max_points)
    lower, upper = ground_truth.min(axis=0), ground_truth.max(axis=0)
    center = 0.5 * (lower + upper)
    extent = max(float(np.max(upper - lower)), 1e-12)
    payload = {
        "title": title,
        "groundTruth": np.round((gt - center) / extent, 6).tolist(),
        "prediction": np.round((pred - center) / extent, 6).tolist(),
        "groundTruthCount": len(ground_truth),
        "predictionCount": len(prediction),
        "embeddedGroundTruthCount": len(gt),
        "embeddedPredictionCount": len(pred),
        "bounds": {
            "lower": np.round((lower - center) / extent, 6).tolist(),
            "upper": np.round((upper - center) / extent, 6).tolist(),
        },
        "worldBounds": {
            "lower": np.round(lower, 6).tolist(),
            "upper": np.round(upper, 6).tolist(),
            "dimensions": np.round(upper - lower, 6).tolist(),
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    escaped_title = html.escape(title)
    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reconstruction comparison</title>
<style>
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #f8fafc; color: #0f172a; overflow: hidden; }
header { height: 76px; padding: 10px 18px; background: white; border-bottom: 1px solid #cbd5e1;
  display: flex; align-items: center; justify-content: space-between; gap: 18px; }
h1 { margin: 0 0 4px; font-size: 16px; font-weight: 650; }
.subtitle { color: #64748b; font-size: 12px; }
.controls { display: flex; align-items: center; flex-wrap: wrap; justify-content: end; gap: 10px; font-size: 13px; }
button { border: 1px solid #94a3b8; border-radius: 5px; padding: 5px 9px; background: white; color: #0f172a; cursor: pointer; }
button:hover { background: #f1f5f9; }
label { white-space: nowrap; }
.blue { color: #2563eb; font-weight: 600; } .orange { color: #ea580c; font-weight: 600; }
#viewer { display: block; width: 100vw; height: calc(100vh - 76px); cursor: grab; outline: none; }
#viewer.dragging { cursor: grabbing; }
#readout { position: fixed; left: 14px; bottom: 12px; padding: 7px 10px; border-radius: 5px;
  background: rgba(255,255,255,.86); border: 1px solid #cbd5e1; color: #475569; font-size: 12px; pointer-events: none; }
#dimensions { position: fixed; right: 14px; bottom: 12px; padding: 7px 10px; border-radius: 5px;
  background: rgba(255,255,255,.86); border: 1px solid #cbd5e1; color: #334155; font-size: 12px; pointer-events: none; }
</style>
</head>
<body>
<header>
  <div><h1>__TITLE__</h1><div class="subtitle">Drag to rotate · wheel to zoom · double-click to reset</div></div>
  <div class="controls">
    <label class="blue"><input id="showGt" type="checkbox" checked> Ground truth</label>
    <label class="orange"><input id="showPrediction" type="checkbox" checked> Prediction</label>
    <label><input id="showBox" type="checkbox" checked> 3D world box</label>
    <label>Layout <select id="layout"><option value="overlay">Overlay (on top)</option><option value="side-by-side">Side by side</option></select></label>
    <label>Point size <input id="pointSize" type="range" min="1" max="6" step="0.5" value="2"></label>
    <button id="xy">XY</button><button id="xz">XZ</button><button id="yz">YZ</button>
    <button id="autoRotate" aria-pressed="false">Auto-rotate</button>
    <button id="reset">Reset</button>
  </div>
</header>
<canvas id="viewer" tabindex="0" aria-label="Interactive reconstruction point-cloud comparison"></canvas>
<div id="readout"></div>
<div id="dimensions"></div>
<script>
const data = __PAYLOAD__;
const canvas = document.getElementById("viewer");
const context = canvas.getContext("2d");
const showGt = document.getElementById("showGt");
const showPrediction = document.getElementById("showPrediction");
const showBox = document.getElementById("showBox");
const layout = document.getElementById("layout");
const pointSize = document.getElementById("pointSize");
const readout = document.getElementById("readout");
let width = 1, height = 1, yaw = -0.55, pitch = -0.35, zoom = 1;
let dragging = false, pointerX = 0, pointerY = 0, autoRotate = false, dirty = true;
readout.textContent = `Embedded ${data.embeddedGroundTruthCount.toLocaleString()} / ${data.groundTruthCount.toLocaleString()} GT points and ${data.embeddedPredictionCount.toLocaleString()} / ${data.predictionCount.toLocaleString()} prediction points`;
const dimensions = data.worldBounds.dimensions;
document.getElementById("dimensions").textContent = `GT world dimensions — X: ${dimensions[0].toFixed(4)} · Y: ${dimensions[1].toFixed(4)} · Z: ${dimensions[2].toFixed(4)}`;

function resize() {
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  width = bounds.width; height = bounds.height;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0); dirty = true;
}
function rotate(point) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = cy * point[0] + sy * point[2];
  const z0 = -sy * point[0] + cy * point[2];
  return [x, cp * point[1] - sp * z0, sp * point[1] + cp * z0];
}
function render() {
  context.clearRect(0, 0, width, height);
  const points = [];
  function add(cloud, color, group) {
    for (const original of cloud) { const p = rotate(original); points.push([p[2], p[0], p[1], color, group]); }
  }
  if (showGt.checked) add(data.groundTruth, "rgba(37,99,235,0.72)", -1);
  if (showPrediction.checked) add(data.prediction, "rgba(234,88,12,0.72)", 1);
  points.sort((a, b) => a[0] - b[0]);
  const separated = layout.value === "side-by-side";
  const scale = Math.min(width, height) * (separated ? 0.46 : 0.78) * zoom;
  const cx = width / 2, cy = height / 2, radius = Number(pointSize.value);
  function project(original, group) {
    const point = rotate(original);
    const offset = separated ? group * width * 0.25 : 0;
    return [cx + offset + scale * point[0], cy - scale * point[1], point[2]];
  }
  function drawBox(group) {
    const low=data.bounds.lower, high=data.bounds.upper;
    const corners=[
      [low[0],low[1],low[2]], [high[0],low[1],low[2]],
      [low[0],high[1],low[2]], [low[0],low[1],high[2]],
      [high[0],high[1],low[2]], [high[0],low[1],high[2]],
      [low[0],high[1],high[2]], [high[0],high[1],high[2]]
    ];
    const projected=corners.map(point => project(point, group));
    const edges=[[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
    context.lineWidth=1; context.strokeStyle="rgba(71,85,105,.55)";
    for(const edge of edges){ context.beginPath(); context.moveTo(projected[edge[0]][0],projected[edge[0]][1]); context.lineTo(projected[edge[1]][0],projected[edge[1]][1]); context.stroke(); }
    const axes=[[1,"#dc2626","X"],[2,"#16a34a","Y"],[3,"#2563eb","Z"]];
    context.font="600 12px system-ui, sans-serif";
    for(const axis of axes){ const end=projected[axis[0]]; context.strokeStyle=axis[1]; context.lineWidth=2.5; context.beginPath(); context.moveTo(projected[0][0],projected[0][1]); context.lineTo(end[0],end[1]); context.stroke(); context.fillStyle=axis[1]; context.fillText(axis[2],end[0]+4,end[1]-4); }
  }
  if(showBox.checked){
    if(separated){ if(showGt.checked)drawBox(-1); if(showPrediction.checked)drawBox(1); }
    else drawBox(0);
  }
  for (const point of points) {
    const offset = separated ? point[4] * width * 0.25 : 0;
    context.beginPath(); context.arc(cx + offset + scale * point[1], cy - scale * point[2], radius, 0, Math.PI * 2);
    context.fillStyle = point[3]; context.fill();
  }
  if (separated) {
    context.font = "600 13px system-ui, sans-serif"; context.textAlign = "center";
    context.fillStyle = "#2563eb"; context.fillText("Ground truth", width * .25, 25);
    context.fillStyle = "#ea580c"; context.fillText("Prediction", width * .75, 25);
    context.strokeStyle = "rgba(148,163,184,.45)"; context.beginPath();
    context.moveTo(width / 2, 12); context.lineTo(width / 2, height - 12); context.stroke();
  }
  context.strokeStyle = "rgba(15,23,42,.18)"; context.lineWidth = 1;
  context.beginPath(); context.moveTo(cx - 8, cy); context.lineTo(cx + 8, cy);
  context.moveTo(cx, cy - 8); context.lineTo(cx, cy + 8); context.stroke();
  dirty = false;
}
function reset() { yaw = -0.55; pitch = -0.35; zoom = 1; dirty = true; }
function setView(nextYaw, nextPitch) { yaw = nextYaw; pitch = nextPitch; dirty = true; }
canvas.addEventListener("pointerdown", event => { dragging = true; pointerX = event.clientX; pointerY = event.clientY; canvas.classList.add("dragging"); canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener("pointermove", event => { if (!dragging) return; yaw += (event.clientX - pointerX) * .009; pitch += (event.clientY - pointerY) * .009; pitch = Math.max(-Math.PI, Math.min(Math.PI, pitch)); pointerX = event.clientX; pointerY = event.clientY; dirty = true; });
canvas.addEventListener("pointerup", event => { dragging = false; canvas.classList.remove("dragging"); canvas.releasePointerCapture(event.pointerId); });
canvas.addEventListener("pointercancel", () => { dragging = false; canvas.classList.remove("dragging"); });
canvas.addEventListener("wheel", event => { event.preventDefault(); zoom = Math.max(.25, Math.min(4, zoom * Math.exp(-event.deltaY * .001))); dirty = true; }, {passive:false});
canvas.addEventListener("dblclick", reset);
canvas.addEventListener("keydown", event => { const step=.1; if(event.key==="ArrowLeft")yaw-=step; else if(event.key==="ArrowRight")yaw+=step; else if(event.key==="ArrowUp")pitch-=step; else if(event.key==="ArrowDown")pitch+=step; else return; event.preventDefault(); dirty=true; });
showGt.addEventListener("change", () => dirty = true); showPrediction.addEventListener("change", () => dirty = true); showBox.addEventListener("change", () => dirty = true); layout.addEventListener("change", () => dirty = true); pointSize.addEventListener("input", () => dirty = true);
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("xy").addEventListener("click", () => setView(0, 0));
document.getElementById("xz").addEventListener("click", () => setView(0, -Math.PI/2));
document.getElementById("yz").addEventListener("click", () => setView(Math.PI/2, 0));
document.getElementById("autoRotate").addEventListener("click", event => { autoRotate=!autoRotate; event.currentTarget.setAttribute("aria-pressed", String(autoRotate)); event.currentTarget.textContent=autoRotate?"Stop rotation":"Auto-rotate"; dirty=true; });
window.addEventListener("resize", resize);
function frame() { if(autoRotate && !dragging){ yaw += .004; dirty=true; } if(dirty)render(); requestAnimationFrame(frame); }
resize(); requestAnimationFrame(frame);
</script>
</body>
</html>'''.replace("__TITLE__", escaped_title).replace("__PAYLOAD__", payload_json)
    path.write_text(document, encoding="utf-8")


def similarity_icp(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 30,
    sample_count: int = 3_000,
    trim_fraction: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a symmetric, trimmed Sim(3) ICP using ground-truth correspondences."""

    if iterations <= 0 or sample_count < 3:
        raise ValueError("ICP iterations must be positive and icp-points must be at least 3")
    if not 0.5 <= trim_fraction <= 1.0:
        raise ValueError("ICP trim_fraction must lie in [0.5, 1]")
    source_sample = deterministic_subset(np.asarray(prediction, dtype=np.float64), sample_count)
    target_sample = deterministic_subset(np.asarray(target, dtype=np.float64), sample_count)
    source_center = source_sample.mean(axis=0)
    target_center = target_sample.mean(axis=0)
    source_spread = float(np.sqrt(np.mean(np.sum((source_sample - source_center) ** 2, axis=1))))
    target_spread = float(np.sqrt(np.mean(np.sum((target_sample - target_center) ** 2, axis=1))))
    if source_spread <= 1e-12 or target_spread <= 1e-12:
        raise ValueError("ICP point clouds must have positive spread")
    initial_scale = target_spread / source_spread
    cumulative = np.eye(4, dtype=np.float64)
    cumulative[:3, :3] *= initial_scale
    cumulative[:3, 3] = target_center - initial_scale * source_center
    previous_error = float("inf")
    completed = 0
    for completed in range(1, iterations + 1):
        moving = apply_transform(source_sample, cumulative)
        forward_distance, forward_index = nearest_neighbors(moving, target_sample)
        reverse_distance, reverse_index = nearest_neighbors(target_sample, moving)
        forward_keep = forward_distance <= np.quantile(forward_distance, trim_fraction)
        reverse_keep = reverse_distance <= np.quantile(reverse_distance, trim_fraction)
        paired_source = np.concatenate((
            moving[forward_keep],
            moving[reverse_index[reverse_keep]],
        ))
        paired_target = np.concatenate((
            target_sample[forward_index[forward_keep]],
            target_sample[reverse_keep],
        ))
        delta = estimate_similarity(paired_source, paired_target)
        cumulative = delta @ cumulative
        error = 0.5 * (
            float(np.mean(forward_distance[forward_keep]))
            + float(np.mean(reverse_distance[reverse_keep]))
        )
        if np.isfinite(previous_error) and (
            previous_error - error <= 1e-7 * max(previous_error, 1.0)
        ):
            previous_error = error
            break
        previous_error = error
    aligned = apply_transform(np.asarray(prediction, dtype=np.float64), cumulative)
    scale = float(np.cbrt(np.linalg.det(cumulative[:3, :3])))
    return aligned, cumulative, {
        "method": "symmetric_trimmed_similarity_icp_v1",
        "iterations_completed": completed,
        "sample_count_per_cloud": min(sample_count, len(source_sample), len(target_sample)),
        "trim_fraction": trim_fraction,
        "additional_scale": scale,
        "sample_symmetric_trimmed_distance": previous_error,
    }


def nearest_neighbors(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    target_norm = np.sum(target * target, axis=1)
    for start in range(0, len(source), 512):
        chunk = source[start : start + 512]
        squared = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + target_norm[None, :]
            - 2.0 * chunk @ target.T
        )
        nearest = np.argmin(squared, axis=1)
        indices.append(nearest)
        distances.append(np.sqrt(np.maximum(squared[np.arange(len(chunk)), nearest], 0.0)))
    return np.concatenate(distances), np.concatenate(indices)


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-12:
        raise ValueError("ICP correspondences have zero source variance")
    scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("ICP produced a non-positive scale")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - scale * (rotation @ source_center)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


if __name__ == "__main__":
    raise SystemExit(main())
