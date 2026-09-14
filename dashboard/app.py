"""Interactive experiment dashboard for the CV3D next-best-view project."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


REPO_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_ROOT = Path(
    os.environ.get("CV3D_DASHBOARD_ROOT", str(REPO_ROOT))
).resolve()

PHASE1_ROOT = DASHBOARD_ROOT / "outputs" / "phase1" / "backbone_sweep"

PHASE2_CLOSED_LOOP_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "phase2"
    / "phase2_closed_loop_reconstruction"
    / "seed_0"
)

PHASE3_CLOSED_LOOP_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "phase3"
    / "controlled_history_comparison_reconstruction"
    / "seed_0"
)

PHASE3_POSE_DEEPSETS_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "phase3"
    / "controlled_pose_deepsets"
    / "seed_0"
)

PHASE3_TOKEN_ATTENTION_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "phase3"
    / "controlled_token_attention"
    / "seed_0"
)

GAUSSIAN_SPLATTING_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "gaussian_splatting_variant_comparison"
)

GAUSSIAN_SPLATTING_CPU_RECOVERY_ROOT = (
    DASHBOARD_ROOT
    / "outputs"
    / "gaussian_splatting_cpu_recovery"
)

NUM_ROOT = DASHBOARD_ROOT / "data" / "NUM"

SHAPENET_ROOT = DASHBOARD_ROOT / "data" / "ShapeNetCore.v2"

SPLIT_MANIFEST = (
    DASHBOARD_ROOT
    / "data"
    / "splits"
    / "num_v1.json"
)

ANCHOR_PATH = (
    DASHBOARD_ROOT
    / "src"
    / "nbv"
    / "geometry"
    / "anchors_v1.csv"
)

PREDICTION_DATA = (
    DASHBOARD_ROOT
    / "dashboard"
    / "data"
    / "phase1_test_predictions.npz"
)

DASHBOARD_RENDERING_OBJECTS_README = (
    DASHBOARD_ROOT
    / "dashboard_rendering_objects"
    / "README.md"
)

CATEGORY_NAMES = {
    "02691156": "Airplane",
    "02828884": "Bench",
    "02933112": "Cabinet",
    "02958343": "Car",
    "03001627": "Chair",
    "03211117": "Display",
    "03636649": "Lamp",
    "03691459": "Loudspeaker",
    "04090263": "Rifle",
    "04256520": "Sofa",
    "04379243": "Table",
    "04401088": "Telephone",
    "04530566": "Watercraft",
}

SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {"train": "#34d399", "val": "#f59e0b", "test": "#60a5fa"}

METRICS = {
    "normalized_regret_mean": {
        "label": "Normalized regret",
        "short": "Regret",
        "direction": "lower",
        "description": "Gap between the selected view and the oracle, normalized per sample.",
    },
    "spearman_mean": {
        "label": "Spearman correlation",
        "short": "Spearman",
        "direction": "higher",
        "description": "Rank correlation between predicted and target view scores.",
    },
    "ndcg_at_5_mean": {
        "label": "NDCG @ 5",
        "short": "NDCG@5",
        "direction": "higher",
        "description": "Ranking quality among the five highest-value candidate views.",
    },
    "huber_loss": {
        "label": "Huber loss",
        "short": "Huber",
        "direction": "lower",
        "description": "Robust regression loss on predicted view scores.",
    },
    "ranking_loss": {
        "label": "Ranking loss",
        "short": "Ranking loss",
        "direction": "lower",
        "description": "Pairwise ranking component of the training objective.",
    },
    "loss": {
        "label": "Combined loss",
        "short": "Loss",
        "direction": "lower",
        "description": "Full evaluation objective: Huber loss plus weighted ranking loss.",
    },
}

BACKBONE_LABELS = {
    "pun_upnet": "PUN / UPNet",
    "vggt": "VGGT",
    "dinov2": "DINOv2",
    "imagenet_vit": "ImageNet ViT",
    "raw_rgb": "Raw RGB",
    "none": "Baseline",
}

BACKBONE_COLORS = {
    "PUN / UPNet": "#f59e0b",
    "VGGT": "#8b5cf6",
    "DINOv2": "#06b6d4",
    "ImageNet ViT": "#3b82f6",
    "Raw RGB": "#ec4899",
    "Baseline": "#64748b",
}

CLOSED_LOOP_POLICY_LABELS = {
    "random": "Random",
    "farthest": "Farthest view",
    "pun": "PUN",
    "vggt": "VGGT · single image",
    "oracle": "Oracle",
    "vggt_independent_history": "VGGT · independent history",
    "vggt_joint_history": "VGGT · joint history",
    "vggt_joint_pose_deepsets": "VGGT · pose DeepSets",
    "vggt_joint_token_attention": "VGGT · token attention",
}

CLOSED_LOOP_POLICY_COLORS = {
    "Random": "#94a3b8",
    "Farthest view": "#3b82f6",
    "PUN": "#f97316",
    "VGGT · single image": "#8b5cf6",
    "Oracle": "#34d399",
    "VGGT · independent history": "#22d3ee",
    "VGGT · joint history": "#ec4899",
    "VGGT · pose DeepSets": "#ef4444",
    "VGGT · token attention": "#facc15",
}

RECONSTRUCTION_METRICS = {
    "chamfer_l1_normalized": "Normalized Chamfer-L1 ↓",
    "accuracy_normalized": "Normalized accuracy ↓",
    "completeness_normalized": "Normalized completeness ↓",
    "fscore_1pct": "F-score @ 1% ↑",
    "fscore_2pct": "F-score @ 2% ↑",
    "fscore_10pct": "F-score @ 10% ↑",
    "precision_1pct": "Precision @ 1% ↑",
    "precision_2pct": "Precision @ 2% ↑",
    "precision_10pct": "Precision @ 10% ↑",
    "recall_1pct": "Recall @ 1% ↑",
    "recall_2pct": "Recall @ 2% ↑",
    "recall_10pct": "Recall @ 10% ↑",
}

VARIANT_DESCRIPTIONS = {
    "pun_upnet": "Official pretrained PUN model using a ViT-based UPNet to predict view uncertainty.",
    "train_mean_map": "Image-free baseline that predicts the training set's mean uncertainty for each view.",
    "raw_rgb_16x16_mlp": "MLP trained on a coarse 16×16 grid of raw RGB pixels.",
    "imagenet_vit_pooled_patch": "Probe trained on the mean of ImageNet ViT's spatial patch tokens.",
    "imagenet_vit_cls_token": "Probe trained on ImageNet ViT's global classification token.",
    "dinov2_pooled_patch": "Probe trained on the mean of DINOv2's spatial patch tokens.",
    "dinov2_cls_token": "Probe trained on DINOv2's global classification token.",
    "vggt_pooled_patch": "Probe trained on the mean of VGGT's spatial patch tokens.",
    "vggt_max_pooled_patch": "Probe trained on max-pooled VGGT patch tokens to retain strong local evidence.",
    "vggt_camera_token": "Probe trained on VGGT's global camera-aware representation.",
    "vggt_pooled_register": "Probe trained on the mean of VGGT's learned register tokens.",
    "vggt_camera_patch": "Probe combining VGGT's camera representation with mean-pooled patch tokens.",
}


def load_dashboard_rendering_object_keys(path: str | Path) -> frozenset[str]:
    """Read the rendering-object allowlist maintained in its Markdown README."""
    category_id: str | None = None
    object_keys: set[str] = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            parts = line.split("`")
            category_id = parts[1] if len(parts) >= 3 else None
        elif category_id is not None and line.startswith("- `") and line.endswith("`"):
            object_id = line[3:-1]
            object_keys.add(f"{category_id}/{object_id}")

    if not object_keys:
        raise ValueError(f"No rendering objects are defined in {path}.")
    return frozenset(object_keys)


DASHBOARD_RENDERING_OBJECT_KEYS = load_dashboard_rendering_object_keys(
    DASHBOARD_RENDERING_OBJECTS_README
)


def pretty_variant(value: str) -> str:
    """Turn a persisted variant identifier into a compact display label."""
    aliases = {
        "pun_upnet": "PUN / UPNet",
        "train_mean_map": "Train mean map",
        "raw_rgb_16x16_mlp": "Raw RGB · 16×16 MLP",
        "imagenet_vit_pooled_patch": "ImageNet ViT · pooled patch",
        "imagenet_vit_cls_token": "ImageNet ViT · CLS token",
        "dinov2_pooled_patch": "DINOv2 · pooled patch",
        "dinov2_cls_token": "DINOv2 · CLS token",
        "vggt_pooled_patch": "VGGT · pooled patch",
        "vggt_max_pooled_patch": "VGGT · max pooled patch",
        "vggt_camera_token": "VGGT · camera token",
        "vggt_pooled_register": "VGGT · pooled register",
        "vggt_camera_patch": "VGGT · camera + patch",
    }
    if value in aliases.values():
        return value
    return aliases.get(value, value.replace("_", " ").title())


def pretty_anchor(value: int | str) -> str:
    """Format an anchor ID while remaining stable across widget reruns."""
    label = str(value)
    return label if label.startswith("Anchor ") else f"Anchor {label}"


@st.cache_data(show_spinner=False)
def load_dataset_index(path: str) -> pd.DataFrame:
    """Load the frozen object-level split manifest with readable categories."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    records = []
    for split, object_keys in manifest["splits"].items():
        for object_key in object_keys:
            category_id, object_id = object_key.split("/", maxsplit=1)
            records.append(
                {
                    "category_id": category_id,
                    "category": CATEGORY_NAMES.get(category_id, category_id),
                    "split": split,
                    "split_label": SPLIT_LABELS[split],
                    "object_id": object_id,
                }
            )
    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False)
def load_anchor_directions(path: str) -> pd.DataFrame:
    """Load the canonical PUN/NUM camera directions."""
    anchors = pd.read_csv(path)
    expected = {"anchor_id", "direction_x", "direction_y", "direction_z"}
    if not expected.issubset(anchors.columns) or len(anchors) != 48:
        raise ValueError("The canonical anchor file must contain 48 XYZ directions.")
    return anchors


@st.cache_data(show_spinner=False)
def load_prediction_maps(path: str) -> dict[str, np.ndarray]:
    """Load the compact, inference-free Phase 1 dashboard export."""
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    if int(payload["schema_version"]) != 2:
        raise ValueError("Unsupported dashboard prediction schema.")
    object_count = len(payload["object_keys"])
    variant_count = len(payload["variant_names"])
    if payload["targets"].shape != (object_count, 48, 48):
        raise ValueError(
            "Dashboard ground-truth maps must have shape [objects, sources, anchors]."
        )
    if payload["predictions"].shape != (variant_count, object_count, 48, 48):
        raise ValueError(
            "Dashboard predictions must have shape [variants, objects, sources, anchors]."
        )
    if payload["available_source_anchors"].shape != (variant_count, 48):
        raise ValueError("Dashboard source-anchor availability has an invalid shape.")
    if payload["target_best_ids"].shape != (object_count, 48):
        raise ValueError("Dashboard ground-truth best-anchor IDs have an invalid shape.")
    if payload["predicted_best_ids"].shape != (variant_count, object_count, 48):
        raise ValueError("Dashboard predicted best-anchor IDs have an invalid shape.")
    return payload


@st.cache_data(show_spinner=False)
def load_object_mesh(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a compact ASCII PLY mesh without adding a geometry dependency."""
    source = Path(path)
    if not source.is_file():
        return None

    with source.open(encoding="utf-8") as handle:
        if handle.readline().strip() != "ply":
            return None
        vertex_count = face_count = 0
        while True:
            line = handle.readline()
            if not line:
                return None
            fields = line.split()
            if fields[:2] == ["format", "ascii"]:
                pass
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
            elif fields[0] == "end_header":
                break

        vertices = np.asarray(
            [[float(value) for value in handle.readline().split()[:3]] for _ in range(vertex_count)],
            dtype=np.float32,
        )
        triangles: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            values = [int(value) for value in handle.readline().split()]
            indices = values[1 : values[0] + 1]
            triangles.extend(
                (indices[0], indices[index], indices[index + 1])
                for index in range(1, len(indices) - 1)
            )

    vertices -= vertices.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(vertices, axis=1).max())
    if scale > 0:
        vertices *= 0.72 / scale
    return vertices, np.asarray(triangles, dtype=np.int32)


def build_anchor_sphere(
    anchors: pd.DataFrame, selected_anchor: int, mesh: tuple[np.ndarray, np.ndarray] | None
) -> go.Figure:
    """Build a rotatable object-centred view of the 48 camera anchors."""
    figure = go.Figure()
    sphere_radius = 1.35

    line_color = "rgba(100, 116, 139, .28)"
    longitude = np.linspace(0, 2 * math.pi, 100)
    for elevation in np.linspace(-math.pi / 3, math.pi / 3, 5):
        radius = sphere_radius * math.cos(elevation)
        figure.add_trace(
            go.Scatter3d(
                x=radius * np.cos(longitude),
                y=radius * np.sin(longitude),
                z=np.full_like(longitude, sphere_radius * math.sin(elevation)),
                mode="lines",
                line={"color": line_color, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    latitude = np.linspace(-math.pi / 2, math.pi / 2, 80)
    for azimuth in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        figure.add_trace(
            go.Scatter3d(
                x=sphere_radius * np.cos(latitude) * math.cos(azimuth),
                y=sphere_radius * np.cos(latitude) * math.sin(azimuth),
                z=sphere_radius * np.sin(latitude),
                mode="lines",
                line={"color": line_color, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if mesh is not None:
        vertices, faces = mesh
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#22d3ee",
                opacity=0.86,
                flatshading=False,
                lighting={"ambient": 0.45, "diffuse": 0.75, "roughness": 0.65},
                hoverinfo="skip",
                name="Object",
            )
        )
    else:
        figure.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers", marker={"size": 24, "color": "#22d3ee"},
                hovertemplate="Selected object<extra></extra>", showlegend=False
            )
        )

    directions = anchors[["direction_x", "direction_y", "direction_z"]].to_numpy()
    positions = directions * sphere_radius
    ids = anchors["anchor_id"].astype(int).to_numpy()
    selected = ids == selected_anchor
    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker={
                "size": np.where(selected, 10, 5),
                "color": np.where(selected, "#f59e0b", "#e2e8f0"),
                "line": {"color": "#07101f", "width": 1},
            },
            customdata=np.column_stack(
                (ids, np.degrees(anchors["azimuth_rad"]), np.degrees(anchors["elevation_rad"]))
            ),
            hovertemplate=(
                "Anchor %{customdata[0]:.0f}<br>Azimuth %{customdata[1]:.1f}°"
                "<br>Elevation %{customdata[2]:.1f}°<extra></extra>"
            ),
            showlegend=False,
        )
    )
    chosen = positions[selected][0]
    camera_direction = directions[selected][0]
    preferred_up = np.array([0.0, 0.0, 1.0])
    camera_right = np.cross(preferred_up, camera_direction)
    if np.linalg.norm(camera_right) < 1e-9:
        preferred_up = np.array([0.0, 1.0, 0.0])
        camera_right = np.cross(preferred_up, camera_direction)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_eye = camera_direction * 2.15
    figure.add_trace(
        go.Scatter3d(
            x=[0, chosen[0]], y=[0, chosen[1]], z=[0, chosen[2]], mode="lines",
            line={"color": "#f59e0b", "width": 5}, hoverinfo="skip", showlegend=False
        )
    )
    figure.update_layout(
        height=590,
        margin={"l": 0, "r": 0, "t": 18, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "aspectmode": "cube",
            "camera": {
                "eye": {
                    "x": float(camera_eye[0]),
                    "y": float(camera_eye[1]),
                    "z": float(camera_eye[2]),
                },
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "up": {
                    "x": float(camera_up[0]),
                    "y": float(camera_up[1]),
                    "z": float(camera_up[2]),
                },
            },
            "xaxis": {"visible": False, "range": [-1.55, 1.55]},
            "yaxis": {"visible": False, "range": [-1.55, 1.55]},
            "zaxis": {"visible": False, "range": [-1.55, 1.55]},
        },
        dragmode="orbit",
        showlegend=False,
    )
    return figure


def build_prediction_map_chart(
    anchors: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    variant_label: str,
    target_best_id: int,
    predicted_best_id: int,
) -> go.Figure:
    """Compare ground-truth and predicted NUM maps on a shared polar scale."""
    target_values = np.asarray(target, dtype=np.float32)
    prediction_values = np.asarray(prediction, dtype=np.float32)
    valid = anchors["anchor_id"].to_numpy(dtype=int) != 0
    oriented_target = -target_values
    oriented_prediction = -prediction_values
    minimum = float(oriented_target[valid].min())
    scale = float(oriented_target[valid].max()) - minimum
    if scale == 0:
        target_uncertainty = np.zeros(48, dtype=np.float32)
        predicted_uncertainty = np.zeros(48, dtype=np.float32)
    else:
        target_uncertainty = np.clip((oriented_target - minimum) / scale, 0, 1)
        predicted_uncertainty = np.clip(
            (oriented_prediction - minimum) / scale, 0, 1
        )

    diameter = 181
    coordinates = np.linspace(-1.0, 1.0, diameter, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radial = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    inside = radial <= 1.0
    theta = np.minimum(radial[inside], 1.0) * np.float32(math.pi)
    phi = np.mod(np.arctan2(y_grid[inside], x_grid[inside]), 2 * math.pi)
    sin_theta = np.sin(theta)
    sample_directions = np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta))
    )
    anchor_directions = anchors[
        ["direction_x", "direction_y", "direction_z"]
    ].to_numpy(dtype=np.float32)
    nearest_ids = np.argmax(sample_directions @ anchor_directions.T, axis=1)

    def map_grid(values: np.ndarray) -> np.ndarray:
        grid = np.full((diameter, diameter), np.nan, dtype=np.float32)
        mapped = values[nearest_ids]
        mapped[nearest_ids == 0] = np.nan
        grid[inside] = mapped
        return grid

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Ground truth", f"Prediction · {variant_label}"),
        horizontal_spacing=0.08,
    )
    for column, values, raw_values, highlight_color, highlight_label, best_id in (
        (
            1,
            target_uncertainty,
            target_values,
            "#34d399",
            "Ground truth",
            target_best_id,
        ),
        (
            2,
            predicted_uncertainty,
            prediction_values,
            "#f59e0b",
            "Predicted",
            predicted_best_id,
        ),
    ):
        figure.add_trace(
            go.Heatmap(
                x=coordinates,
                y=coordinates,
                z=map_grid(values),
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                showscale=column == 2,
                colorbar={"title": {"text": "Uncertainty"}, "thickness": 12},
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        polar_radius = (
            (math.pi / 2 - anchors["elevation_rad"].to_numpy()) / math.pi
        )
        anchor_x = polar_radius * np.cos(anchors["azimuth_rad"].to_numpy())
        anchor_y = polar_radius * np.sin(anchors["azimuth_rad"].to_numpy())
        ids = anchors["anchor_id"].to_numpy(dtype=int)
        figure.add_trace(
            go.Scatter(
                x=anchor_x,
                y=anchor_y,
                mode="markers",
                marker={
                    "size": 5,
                    "color": "#f8fafc",
                    "line": {"color": "#07101f", "width": 1},
                },
                customdata=np.column_stack((ids, raw_values, values)),
                hovertemplate=(
                    "Anchor %{customdata[0]:.0f}<br>PSNR %{customdata[1]:.3f}"
                    "<br>Normalized uncertainty %{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[anchor_x[best_id]],
                y=[anchor_y[best_id]],
                mode="markers+text",
                marker={
                    "size": 19 if column == 2 else 15,
                    "color": highlight_color,
                    "symbol": "diamond" if column == 2 else "circle",
                    "line": {"color": "#f8fafc", "width": 2},
                },
                text=[f"{highlight_label} anchor {best_id}"],
                textposition="top center",
                textfont={"color": highlight_color, "size": 12},
                hovertemplate=(
                    f"{highlight_label} most uncertain: anchor {best_id}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker={
                    "size": 10,
                    "color": "#94a3b8",
                    "symbol": "x",
                    "line": {"width": 2},
                },
                hovertemplate="Source anchor 0 · excluded<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )

    figure.update_xaxes(visible=False, range=[-1.04, 1.04], constrain="domain")
    figure.update_yaxes(
        visible=False, range=[-1.04, 1.04], scaleanchor="x", scaleratio=1, row=1, col=1
    )
    figure.update_yaxes(
        visible=False,
        range=[-1.04, 1.04],
        scaleanchor="x2",
        scaleratio=1,
        row=1,
        col=2,
    )
    figure.update_layout(
        height=540,
        margin={"l": 15, "r": 65, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.42)",
        font={"color": "#cbd5e1"},
    )
    return figure


@st.cache_data(show_spinner=False)
def load_phase1_results(root: str) -> pd.DataFrame:
    """Load all Phase 1 variant summaries into one tidy, seed-level table."""
    records: list[dict[str, object]] = []
    for path in sorted(Path(root).glob("seed_*/variants/*/summary.json")):
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        seed_name = path.parents[2].name
        try:
            seed = int(seed_name.removeprefix("seed_"))
        except ValueError:
            continue

        base = {
            "seed": seed,
            "variant": summary.get("variant", path.parent.name),
            "backbone": summary.get("backbone", "unknown"),
            "feature": summary.get("feature", "unknown"),
            "input_dim": summary.get("input_dim"),
            "trainable_parameters": summary.get("trainable_parameters"),
            "best_epoch": summary.get("best_epoch"),
        }
        for split in ("test", "validation"):
            split_metrics = summary.get(split, {})
            record = {**base, "split": split}
            record.update({metric: split_metrics.get(metric) for metric in METRICS})
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["variant_label"] = frame["variant"].map(pretty_variant)
    frame["variant_description"] = frame["variant"].map(
        lambda value: VARIANT_DESCRIPTIONS.get(
            value, "Experiment variant using the recorded backbone and feature configuration."
        )
    )
    frame["backbone_label"] = frame["backbone"].map(
        lambda value: BACKBONE_LABELS.get(value, str(value).replace("_", " ").title())
    )
    return frame


def build_metric_chart(data: pd.DataFrame, metric: str) -> alt.LayerChart:
    """Build the Phase 1 ranking chart with seed variation and rich tooltips."""
    metadata = METRICS[metric]
    ascending = metadata["direction"] == "lower"
    order = (
        data.groupby("variant_label", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=ascending)["variant_label"]
        .tolist()
    )

    grouped = (
        data.groupby(
            ["variant", "variant_label", "variant_description", "backbone_label"],
            as_index=False,
        )
        .agg(
            mean=(metric, "mean"),
            minimum=(metric, "min"),
            maximum=(metric, "max"),
            seeds=("seed", "nunique"),
        )
    )

    y = alt.Y(
        "variant_label:N",
        title=None,
        sort=order,
        axis=alt.Axis(labelLimit=230, labelPadding=10, ticks=False, domain=False),
    )
    color = alt.Color(
        "backbone_label:N",
        title=None,
        scale=alt.Scale(
            domain=list(BACKBONE_COLORS), range=list(BACKBONE_COLORS.values())
        ),
        legend=alt.Legend(orient="top", direction="horizontal", symbolType="circle"),
    )
    click = alt.selection_point(fields=["variant"], on="click", clear="dblclick")
    opacity = alt.condition(click, alt.value(1.0), alt.value(0.28))

    bars = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusEnd=5, height=18)
        .encode(
            x=alt.X(
                "mean:Q",
                title=metadata["label"],
                scale=alt.Scale(zero=False, padding=18),
                axis=alt.Axis(format=".3f", gridColor="#263247", tickColor="#475569"),
            ),
            y=y,
            color=color,
            opacity=opacity,
            tooltip=[
                alt.Tooltip("variant_label:N", title="Variant"),
                alt.Tooltip("variant_description:N", title="About"),
                alt.Tooltip("backbone_label:N", title="Backbone"),
                alt.Tooltip("mean:Q", title=f"Mean {metadata['short']}", format=".4f"),
                alt.Tooltip("minimum:Q", title="Seed minimum", format=".4f"),
                alt.Tooltip("maximum:Q", title="Seed maximum", format=".4f"),
                alt.Tooltip("seeds:Q", title="Seeds"),
            ],
        )
        .add_params(click)
    )

    ranges = (
        alt.Chart(grouped[grouped["seeds"] > 1])
        .mark_rule(color="#e2e8f0", strokeWidth=2.2, opacity=0.9)
        .encode(x="minimum:Q", x2="maximum:Q", y=y, opacity=opacity)
    )
    seed_points = (
        alt.Chart(data)
        .mark_circle(size=40, fill="#0f172a", stroke="#f8fafc", strokeWidth=1.2)
        .encode(
            x=alt.X(f"{metric}:Q"),
            y=y,
            opacity=opacity,
            tooltip=[
                alt.Tooltip("variant_label:N", title="Variant"),
                alt.Tooltip("variant_description:N", title="About"),
                alt.Tooltip("seed:O", title="Seed"),
                alt.Tooltip(f"{metric}:Q", title=metadata["short"], format=".4f"),
            ],
        )
    )

    return (
        alt.layer(bars, ranges, seed_points)
        .properties(height=max(430, len(order) * 38))
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#e2e8f0",
            titleFontSize=12,
            labelFontSize=12,
        )
        .configure_legend(labelColor="#cbd5e1", labelFontSize=11, padding=8)
    )


def pretty_policy(value: str) -> str:
    """Return a paper-friendly label for a closed-loop policy identifier."""
    return CLOSED_LOOP_POLICY_LABELS.get(value, value.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def load_closed_loop_tables(
    phase2_root: str, phase3_root: str
) -> dict[str, pd.DataFrame]:
    """Load the canonical Phase 2 and Phase 3 evaluation tables."""
    specifications = (
        ("Phase 2", Path(phase2_root), "comparison.csv", None),
        ("Phase 3", Path(phase3_root), "closed_loop_comparison.csv", None),
        (
            "Phase 3",
            PHASE3_POSE_DEEPSETS_ROOT,
            "closed_loop_comparison.csv",
            {"vggt_joint_pose_deepsets"},
        ),
        (
            "Phase 3",
            PHASE3_TOKEN_ATTENTION_ROOT,
            "closed_loop_comparison.csv",
            {"vggt_joint_token_attention"},
        ),
    )
    grouped: dict[str, list[pd.DataFrame]] = {
        "comparison": [],
        "coverage": [],
        "per_step": [],
        "vggt_reconstruction": [],
    }
    for phase, root, comparison_name, included_policies in specifications:
        paths = {
            "comparison": root / "metrics" / comparison_name,
            "coverage": root / "metrics" / "coverage.csv",
            "per_step": root / "metrics" / "per_step.csv",
            "vggt_reconstruction": root / "metrics" / "reconstruction_curves.csv",
        }
        for table_name, path in paths.items():
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            if included_policies is not None:
                frame = frame[frame["policy"].isin(included_policies)].copy()
            frame["phase"] = phase
            frame["policy_label"] = frame["policy"].map(pretty_policy)
            grouped[table_name].append(frame)
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in grouped.items()
    }


@st.cache_data(show_spinner=False)
def load_gaussian_splatting_metrics(root: str) -> pd.DataFrame:
    """Aggregate verified CPU-recovered 2DGS evaluations by policy and view count."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(root).glob("*/*/*views/phase*/2dgs/metrics.csv")):
        summary_path = path.with_name("summary.json")
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "cpu_recovery" not in summary:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        variant = path.parents[1].name
        frame["phase"] = "Phase 2" if variant.startswith("phase2_") else "Phase 3"
        frame["policy_label"] = frame["policy"].map(pretty_policy)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    metric_columns = [
        column
        for column in combined.columns
        if column.endswith("_normalized")
        or column.startswith(("precision_", "recall_", "fscore_"))
    ]
    aggregations = {column: "mean" for column in metric_columns}
    aggregations["object_id"] = "nunique"
    return (
        combined.groupby(
            ["phase", "policy", "policy_label", "acquired_view_count"],
            as_index=False,
        )
        .agg(aggregations)
        .rename(columns={"object_id": "object_count"})
    )


@st.cache_data(show_spinner=False)
def load_3dgs_render_catalog(root: str) -> pd.DataFrame:
    """Index 48-anchor 3DGS/ground-truth viewers by policy and view count."""
    records: list[dict[str, object]] = []
    root_path = Path(root)
    for path in sorted(
        root_path.glob("*/*/*views/phase*/3dgs/ground_truth_comparison.html")
    ):
        relative = path.relative_to(root_path)
        category_id, object_id, view_directory, variant, _, _ = relative.parts
        object_key = f"{category_id}/{object_id}"
        if object_key not in DASHBOARD_RENDERING_OBJECT_KEYS:
            continue
        phase_token, policy = variant.split("_", maxsplit=1)
        records.append(
            {
                "phase": "Phase 2" if phase_token == "phase2" else "Phase 3",
                "policy": policy,
                "policy_label": pretty_policy(policy),
                "category_id": category_id,
                "object_id": object_id,
                "object_key": object_key,
                "acquired_view_count": int(view_directory.removesuffix("views")),
                "path": str(path),
            }
        )
    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False)
def load_vggt_render_catalog(
    phase2_root: str, phase3_root: str
) -> pd.DataFrame:
    """Index oracle-ICP-aligned VGGT/ground-truth point-cloud comparisons."""
    records: list[dict[str, object]] = []
    sources = (
        ("Phase 2", Path(phase2_root), None),
        ("Phase 3", Path(phase3_root), None),
        ("Phase 3", PHASE3_POSE_DEEPSETS_ROOT, {"vggt_joint_pose_deepsets"}),
        ("Phase 3", PHASE3_TOKEN_ATTENTION_ROOT, {"vggt_joint_token_attention"}),
    )
    for phase, root, included_policies in sources:
        visualization_root = root / "metrics" / "reconstruction_visualizations"
        for metadata_path in sorted(visualization_root.glob("*/metadata.json")):
            comparison_path = metadata_path.with_name(
                "comparison_oracle_icp_interactive.html"
            )
            if not comparison_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            policy = str(metadata["policy"])
            if included_policies is not None and policy not in included_policies:
                continue
            object_key = str(metadata["object_id"])
            if object_key not in DASHBOARD_RENDERING_OBJECT_KEYS:
                continue
            category_id, object_id = object_key.split("/", maxsplit=1)
            records.append(
                {
                    "phase": phase,
                    "policy": policy,
                    "policy_label": pretty_policy(policy),
                    "category_id": category_id,
                    "object_id": object_id,
                    "object_key": object_key,
                    "acquired_view_count": int(metadata["acquired_view_count"]),
                    "path": str(comparison_path),
                }
            )
    return pd.DataFrame.from_records(records)


def make_compact_3dgs_view(document: str, *, ground_truth: bool = False) -> str:
    """Show the 3DGS frame nearest to the synchronized point-cloud viewpoint."""
    hidden_panel = ":last-child" if ground_truth else ":first-child"
    compact_styles = f"""
<style>
h1 {{ display: none; }}
.stage {{ height: 100vh; padding: 0; gap: 0; align-items: center; }}
.panel {{ width: 100%; height: auto; max-height: 100%; aspect-ratio: 1; }}
#side .panel{hidden_panel} {{ display: none; }}
footer {{ display: none; }}
.panel img {{ pointer-events: none; }}
</style>
"""
    document = document.replace("</head>", f"{compact_styles}</head>")
    document = document.replace("let index=0,playing=true;", "let index=0;")
    automatic_controls = (
        "document.getElementById('play').onclick=e=>{playing=!playing;"
        "e.target.textContent=playing?'Pause':'Play'};\n"
        "setInterval(()=>{if(playing){index=(index+1)%data.references.length;show()}},350);"
    )
    anchor_directions = json.dumps(
        load_anchor_directions(str(ANCHOR_PATH))
        .sort_values("anchor_id")[["direction_x", "direction_y", "direction_z"]]
        .to_numpy(dtype=float)
        .tolist(),
        separators=(",", ":"),
    )
    synchronized_controls = (
        f"const anchorDirections={anchor_directions};"
        "const rotationChannel=('BroadcastChannel' in window)?"
        "new BroadcastChannel('cv3d-reconstruction-rotation'):null;"
        "function closestAnchor(yaw,pitch){const cp=Math.cos(pitch);"
        "const direction=[-cp*Math.sin(yaw),Math.sin(pitch),cp*Math.cos(yaw)];"
        "let best=0,bestDot=-Infinity;"
        "const count=Math.min(anchorDirections.length,data.references.length);"
        "for(let candidate=0;candidate<count;candidate++){const anchor=anchorDirections[candidate];"
        "const dot=direction[0]*anchor[0]+direction[1]*anchor[1]+direction[2]*anchor[2];"
        "if(dot>bestDot){bestDot=dot;best=candidate}}return best}"
        "index=closestAnchor(-.55,-.35);"
        "if(rotationChannel){rotationChannel.onmessage=e=>{"
        "if(typeof e.data?.yaw!=='number'||typeof e.data?.pitch!=='number')return;"
        "index=closestAnchor(e.data.yaw,e.data.pitch);show()}}"
    )
    return document.replace(automatic_controls, synchronized_controls)


def make_compact_point_cloud_view(document: str, *, ground_truth: bool) -> str:
    """Reduce a point-cloud comparison to one compact interactive cloud."""
    compact_styles = """
<style>
header, #readout, #dimensions { display: none !important; }
#viewer { height: 100vh !important; }
</style>
"""
    document = document.replace("</head>", f"{compact_styles}</head>")
    initial_state = (
        f'showGt.checked = {str(ground_truth).lower()}; '
        f'showPrediction.checked = {str(not ground_truth).lower()}; '
        'showBox.checked = false; layout.value = "overlay";\n'
    )
    document = document.replace(
        'const pointSize = document.getElementById("pointSize");',
        'const pointSize = document.getElementById("pointSize");\n' + initial_state,
    )
    synchronized_state = """
const rotationChannel = ("BroadcastChannel" in window)
  ? new BroadcastChannel("cv3d-reconstruction-rotation") : null;
function broadcastRotation() {
  rotationChannel?.postMessage({yaw, pitch, zoom});
}
if (rotationChannel) rotationChannel.onmessage = event => {
  if (typeof event.data?.yaw !== "number" || typeof event.data?.pitch !== "number") return;
  yaw = event.data.yaw;
  pitch = event.data.pitch;
  if (typeof event.data.zoom === "number") zoom = event.data.zoom;
  dirty = true;
};
"""
    document = document.replace(
        "let dragging = false, pointerX = 0, pointerY = 0, autoRotate = false, dirty = true;",
        "let dragging = false, pointerX = 0, pointerY = 0, autoRotate = false, dirty = true;\n"
        + synchronized_state,
    )
    document = document.replace(
        "pointerX = event.clientX; pointerY = event.clientY; dirty = true; });",
        "pointerX = event.clientX; pointerY = event.clientY; dirty = true; "
        "broadcastRotation(); });",
    )
    document = document.replace(
        "zoom = Math.max(.25, Math.min(4, zoom * Math.exp(-event.deltaY * .001))); "
        "dirty = true; }, {passive:false});",
        "zoom = Math.max(.25, Math.min(4, zoom * Math.exp(-event.deltaY * .001))); "
        "dirty = true; broadcastRotation(); }, {passive:false});",
    )
    document = document.replace(
        "function reset() { yaw = -0.55; pitch = -0.35; zoom = 1; dirty = true; }",
        "function reset() { yaw = -0.55; pitch = -0.35; zoom = 1; dirty = true; "
        "broadcastRotation(); }",
    )
    document = document.replace(
        "function setView(nextYaw, nextPitch) { yaw = nextYaw; pitch = nextPitch; dirty = true; }",
        "function setView(nextYaw, nextPitch) { yaw = nextYaw; pitch = nextPitch; "
        "dirty = true; broadcastRotation(); }",
    )
    document = document.replace(
        "event.preventDefault(); dirty=true; });",
        "event.preventDefault(); dirty=true; broadcastRotation(); });",
    )
    return document


def render_reconstruction_comparison_row(
    policy: str,
    category_id: str,
    object_id: str,
    acquired_view_count: int,
    render_catalog: pd.DataFrame,
    vggt_render_catalog: pd.DataFrame,
) -> None:
    """Render one compact policy row beneath the gallery's shared header."""
    filters = (
        ("policy", policy),
        ("category_id", category_id),
        ("object_id", object_id),
        ("acquired_view_count", acquired_view_count),
    )

    def matching_rows(catalog: pd.DataFrame) -> pd.DataFrame:
        if catalog.empty:
            return pd.DataFrame()
        selected = catalog
        for column, value in filters:
            selected = selected[selected[column] == value]
        return selected

    matching_3dgs = matching_rows(render_catalog)
    matching_vggt = matching_rows(vggt_render_catalog)
    comparison_3dgs_path = (
        Path(matching_3dgs["path"].iloc[0]) if not matching_3dgs.empty else None
    )
    comparison_vggt_path = (
        Path(matching_vggt["path"].iloc[0]) if not matching_vggt.empty else None
    )
    comparison_2dgs_path = None
    if comparison_3dgs_path is not None:
        comparison_2dgs_path = (
            GAUSSIAN_SPLATTING_CPU_RECOVERY_ROOT
            / comparison_3dgs_path.parent.parent.relative_to(GAUSSIAN_SPLATTING_ROOT)
            / "2dgs"
            / "comparison_interactive.html"
        )
    ground_truth_cloud_path = comparison_vggt_path
    if ground_truth_cloud_path is None and (
        comparison_2dgs_path is not None and comparison_2dgs_path.is_file()
    ):
        ground_truth_cloud_path = comparison_2dgs_path

    (
        policy_column,
        render_3dgs_column,
        surface_2dgs_column,
        vggt_column,
        ground_truth_column,
    ) = st.columns([0.9, 1, 1, 1, 1], gap="small", vertical_alignment="center")
    with policy_column:
        st.markdown(
            f'<div class="matrix-row-label">{pretty_policy(policy)}</div>',
            unsafe_allow_html=True,
        )
    with render_3dgs_column:
        if comparison_3dgs_path is None:
            st.info("Not available.")
        else:
            components.html(
                make_compact_3dgs_view(
                    comparison_3dgs_path.read_text(encoding="utf-8")
                ),
                height=240,
                scrolling=False,
            )
    with surface_2dgs_column:
        if comparison_2dgs_path is None or not comparison_2dgs_path.is_file():
            st.info("Not available.")
        else:
            components.html(
                make_compact_point_cloud_view(
                    comparison_2dgs_path.read_text(encoding="utf-8"),
                    ground_truth=False,
                ),
                height=240,
                scrolling=False,
            )
    with vggt_column:
        if comparison_vggt_path is None:
            st.info("Not available.")
        else:
            components.html(
                make_compact_point_cloud_view(
                    comparison_vggt_path.read_text(encoding="utf-8"),
                    ground_truth=False,
                ),
                height=240,
                scrolling=False,
            )
    with ground_truth_column:
        if ground_truth_cloud_path is not None:
            components.html(
                make_compact_point_cloud_view(
                    ground_truth_cloud_path.read_text(encoding="utf-8"),
                    ground_truth=True,
                ),
                height=240,
                scrolling=False,
            )
        elif comparison_3dgs_path is not None:
            components.html(
                make_compact_3dgs_view(
                    comparison_3dgs_path.read_text(encoding="utf-8"),
                    ground_truth=True,
                ),
                height=240,
                scrolling=False,
            )
        else:
            st.info("Not available.")


@st.cache_data(show_spinner=False)
def load_rollout_catalog(phase2_root: str, phase3_root: str) -> pd.DataFrame:
    """Index replay files without loading their large score arrays."""
    records: list[dict[str, str]] = []
    sources = (
        ("Phase 2", Path(phase2_root), None),
        ("Phase 3", Path(phase3_root), None),
        ("Phase 3", PHASE3_POSE_DEEPSETS_ROOT, {"vggt_joint_pose_deepsets"}),
        ("Phase 3", PHASE3_TOKEN_ATTENTION_ROOT, {"vggt_joint_token_attention"}),
    )
    for phase, root, included_policies in sources:
        rollout_root = root / "rollouts"
        for path in sorted(rollout_root.glob("*/*/*.npz")):
            relative = path.relative_to(rollout_root)
            policy, category_id, filename = relative.parts
            if included_policies is not None and policy not in included_policies:
                continue
            records.append(
                {
                    "phase": phase,
                    "policy": policy,
                    "policy_label": pretty_policy(policy),
                    "category_id": category_id,
                    "object_id": Path(filename).stem,
                    "object_key": f"{category_id}/{Path(filename).stem}",
                    "path": str(path),
                }
            )
    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False)
def load_dashboard_rollout(path: str) -> dict[str, object]:
    """Load only the replay fields used by the interactive dashboard."""
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        steps = json.loads(str(payload["steps_json"].item()))
        return {
            "metadata": metadata,
            "steps": steps,
            "acquired_anchor_ids": payload["acquired_anchor_ids"].copy(),
            "acquired_view_counts": payload["acquired_view_counts"].copy(),
            "coverage": payload["coverage"].copy(),
        }


def build_closed_loop_curve(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    y_domain: tuple[float, float] | None = None,
) -> alt.LayerChart:
    """Build a consistent multi-policy closed-loop trajectory chart."""
    domain = [
        label
        for label in CLOSED_LOOP_POLICY_COLORS
        if label in set(data["policy_label"])
    ]
    color = alt.Color(
        "policy_label:N",
        title=None,
        scale=alt.Scale(
            domain=domain,
            range=[CLOSED_LOOP_POLICY_COLORS[label] for label in domain],
        ),
        legend=alt.Legend(orient="top", direction="horizontal", columns=4),
    )
    phase_dash = alt.StrokeDash(
        "phase:N",
        scale=alt.Scale(domain=["Phase 2", "Phase 3"], range=[[1, 0], [7, 4]]),
        legend=None,
    )
    selection = alt.selection_point(fields=["policy_label"], bind="legend")
    opacity = alt.condition(selection, alt.value(1.0), alt.value(0.14))
    tooltips = [
        alt.Tooltip("phase:N", title="Evaluation"),
        alt.Tooltip("policy_label:N", title="Policy"),
        alt.Tooltip(f"{x}:Q", title=x_title),
        alt.Tooltip(f"{y}:Q", title=y_title, format=".4f"),
    ]
    if "object_count" in data.columns:
        tooltips.append(alt.Tooltip("object_count:Q", title="Objects"))
    y_scale = (
        alt.Scale(domain=list(y_domain), nice=False)
        if y_domain is not None
        else alt.Scale(zero=False)
    )
    encoding = {
        "x": alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(tickMinStep=1)),
        "y": alt.Y(f"{y}:Q", title=y_title, scale=y_scale),
        "color": color,
        "strokeDash": phase_dash,
        "opacity": opacity,
        "tooltip": tooltips,
    }
    line = alt.Chart(data).mark_line(point=False, strokeWidth=3).encode(**encoding)
    points = alt.Chart(data).mark_circle(size=48).encode(**encoding)
    return (
        alt.layer(line, points)
        .add_params(selection)
        .properties(height=430)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1", titleColor="#e2e8f0", gridColor="#263247"
        )
        .configure_legend(labelColor="#cbd5e1", titleColor="#e2e8f0")
    )


def build_reconstruction_comparison_chart(
    vggt_data: pd.DataFrame,
    gaussian_data: pd.DataFrame,
    *,
    vggt_metric: str,
    gaussian_metric: str,
    y_title: str,
    y_domain: tuple[float, float],
) -> alt.FacetChart:
    """Draw separate VGGT and 2DGS plots with one shared policy legend."""
    direction_symbol = (
        "↓" if y_title.endswith("↓")
        else "↑" if y_title.endswith("↑")
        else ""
    )

    method_order = [
        f"VGGT reconstruction {direction_symbol}",
        f"2D Gaussian Splatting {direction_symbol}",
    ]
    frames: list[pd.DataFrame] = []
    for data, metric, method in (
        (vggt_data, vggt_metric, method_order[0]),
        (gaussian_data, gaussian_metric, method_order[1]),
    ):
        frame = data.copy()
        frame["metric_value"] = frame[metric]
        frame["reconstruction_method"] = method
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    domain = [
        label
        for label in CLOSED_LOOP_POLICY_COLORS
        if label in set(combined["policy_label"])
    ]
    selection = alt.selection_point(fields=["policy_label"], bind="legend")
    opacity = alt.condition(selection, alt.value(1.0), alt.value(0.14))
    encoding = {
        "x": alt.X(
            "acquired_view_count:Q",
            title="Acquired views",
            axis=alt.Axis(tickMinStep=1),
        ),
        "y": alt.Y(
            "metric_value:Q",
            title=y_title,
            scale=alt.Scale(domain=list(y_domain), nice=False),
        ),
        "color": alt.Color(
            "policy_label:N",
            title=None,
            scale=alt.Scale(
                domain=domain,
                range=[CLOSED_LOOP_POLICY_COLORS[label] for label in domain],
            ),
            legend=None,
        ),
        "strokeDash": alt.StrokeDash(
            "phase:N",
            scale=alt.Scale(domain=["Phase 2", "Phase 3"], range=[[1, 0], [7, 4]]),
            legend=None,
        ),
        "opacity": opacity,
        "tooltip": [
            alt.Tooltip("reconstruction_method:N", title="Reconstruction"),
            alt.Tooltip("phase:N", title="Evaluation"),
            alt.Tooltip("policy_label:N", title="Policy"),
            alt.Tooltip("acquired_view_count:Q", title="Acquired views"),
            alt.Tooltip("metric_value:Q", title=y_title, format=".4f"),
            alt.Tooltip("object_count:Q", title="Objects"),
        ],
    }
    line = alt.Chart(combined).mark_line(point=False, strokeWidth=3).encode(**encoding)
    points = alt.Chart(combined).mark_circle(size=48).encode(**encoding)
    return (
        alt.layer(line, points)
        .add_params(selection)
        .properties(width=500, height=430)
        .facet(
            column=alt.Column(
                "reconstruction_method:N",
                title=None,
                sort=method_order,
                header=alt.Header(
                    labelColor="#e2e8f0", labelFontSize=16, labelFontWeight=600
                ),
            )
        )
        .resolve_scale(color="shared", strokeDash="shared", y="shared")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1", titleColor="#e2e8f0", gridColor="#263247"
        )
        .configure_legend(labelColor="#cbd5e1", titleColor="#e2e8f0")
    )


def build_rollout_trajectory(
    anchors: pd.DataFrame,
    acquired_anchor_ids: np.ndarray,
    mesh: tuple[np.ndarray, np.ndarray] | None,
) -> go.Figure:
    """Show the ordered camera trajectory around the selected object."""
    figure = build_anchor_sphere(anchors, int(acquired_anchor_ids[-1]), mesh)
    selected = anchors.set_index("anchor_id").loc[acquired_anchor_ids.astype(int)]
    radius = 1.35
    x = selected["direction_x"].to_numpy() * radius
    y = selected["direction_y"].to_numpy() * radius
    z = selected["direction_z"].to_numpy() * radius
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers+text",
            line={"color": "#22d3ee", "width": 7},
            marker={"color": "#22d3ee", "size": 7},
            text=[str(index + 1) for index in range(len(x))],
            textposition="top center",
            textfont={"color": "#f8fafc", "size": 12},
            hovertemplate=(
                "View %{text}<br>Anchor %{customdata}<extra>Acquisition order</extra>"
            ),
            customdata=acquired_anchor_ids,
            name="Acquisition order",
        )
    )
    figure.update_layout(
        height=560, 
        showlegend=False, 
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
    return figure


st.set_page_config(
    page_title="CV3D · Experiment dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #07101f; color: #e2e8f0; }
      [data-testid="stHeader"] { background: rgba(7, 16, 31, .82); backdrop-filter: blur(14px); border-bottom: 1px solid rgba(148, 163, 184, 0.12);}
      [data-testid="stMainBlockContainer"],
      .block-container {
            width: 100% !important;
            max-width: none !important;
            padding-top: 4rem !important;
            padding-bottom: 4rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            }
      @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
}
      .eyebrow { color: #22d3ee; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
      .hero-title { color: #f8fafc; font-size: clamp(2.2rem, 5vw, 4.2rem); font-weight: 680; line-height: .98; letter-spacing: -.055em; margin: .65rem 0 .8rem; }
      .hero-copy { color: #94a3b8; font-size: 1rem; max-width: 680px; line-height: 1.65; }
      .rollout-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2rem;
            padding: 0 0 1.25rem;
            margin-bottom: 1.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }

        .rollout-header-main {
            max-width: 820px;
        }

        .rollout-title {
            color: #f8fafc;
            font-size: clamp(2rem, 2.8vw, 2.9rem);
            font-weight: 680;
            line-height: 1.03;
            letter-spacing: -0.045em;
            margin: 0.45rem 0 0.7rem;
        }

        .rollout-copy {
            color: #94a3b8;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 760px;
            margin: 0;
        }

        .panel-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin: 0 0 0.55rem;
        }

        .viz-title {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 650;
            letter-spacing: -.015em;
            margin-bottom: .2rem;
        }

        .viz-copy {
            color: #64748b;
            font-size: .8rem;
            line-height: 1.5;
            margin-bottom: .65rem;
        }

        .observation-empty {
            min-height: 350px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2rem;
        }

        .observation-empty-icon {
            color: #22d3ee;
            font-size: 2rem;
            margin-bottom: .9rem;
        }

        .observation-empty-title {
            color: #e2e8f0;
            font-size: .95rem;
            font-weight: 650;
            margin-bottom: .45rem;
        }

        .observation-empty-copy {
            color: #64748b;
            font-size: .8rem;
            line-height: 1.6;
            max-width: 260px;
        }
        .research-kpi {
            position: relative;
            min-height: 118px;
            padding: 1rem 1.15rem;
            background:
                linear-gradient(135deg, rgba(34,211,238,.045), transparent 55%),
                rgba(15, 23, 42, .72);
            border: 1px solid #1e293b;
            border-radius: 16px;
            overflow: hidden;
        }

        .research-kpi::before {
            content: "";
            position: absolute;
            left: 0;
            top: 18px;
            bottom: 18px;
            width: 3px;
            border-radius: 3px;
            background: #22d3ee;
        }

        .research-kpi-label {
            color: #94a3b8;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .10em;
            text-transform: uppercase;
            margin-bottom: .6rem;
        }

        .research-kpi-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 650;
            line-height: 1;
            letter-spacing: -.035em;
        }

        .research-kpi-value span {
            color: #64748b;
            font-size: 1rem;
            font-weight: 500;
        }

        .research-kpi-meta {
            color: #64748b;
            font-size: .78rem;
            margin-top: .55rem;
        }

        .dataset-kpi {
            position: relative;
            min-height: 118px;
            padding: 1rem 1.15rem;
            background:
                linear-gradient(135deg, rgba(34,211,238,.035), transparent 55%),
                rgba(15, 23, 42, .72);
            border: 1px solid #1e293b;
            border-radius: 16px;
            overflow: hidden;
        }

        .dataset-kpi::before {
            content: "";
            position: absolute;
            left: 0;
            top: 18px;
            bottom: 18px;
            width: 3px;
            border-radius: 3px;
            background: var(--accent, #22d3ee);
        }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, .38);
        border-color: #1e293b !important;
        border-radius: 16px !important;
    }


        
        @media (max-width: 900px) {
            .rollout-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
      .run-pill { display: inline-flex; align-items: center; gap: .5rem; color: #a7f3d0; background: rgba(16,185,129,.1); border: 1px solid rgba(52,211,153,.24); border-radius: 999px; padding: .38rem .7rem; font-size: .75rem; }
      .run-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 12px #34d399; }
      .section-kicker { color: #64748b; font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-top: 2rem; }
      .section-title { color: #f8fafc; font-size: 1.65rem; font-weight: 650; letter-spacing: -.025em; margin: .25rem 0 0; }
      .matrix-header { color: #94a3b8; border-bottom: 1px solid #334155; padding: .7rem .25rem .55rem; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-align: center; text-transform: uppercase; }
      .matrix-row-label { color: #f8fafc; font-size: .9rem; font-weight: 650; line-height: 1.35; padding-right: .65rem; }
      .matrix-footer { color: #94a3b8; border-top: 1px solid #334155; margin-top: .15rem; padding: .7rem .25rem 0; font-size: .78rem; line-height: 1.55; }
      .metric-note { color: #94a3b8; font-size: .84rem; padding-top: .25rem; }
      .split-badge { display: inline-flex; align-items: center; gap: .45rem; border: 1px solid currentColor; border-radius: 999px; padding: .28rem .62rem; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .split-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
      .object-id { color: #64748b; font-family: monospace; font-size: .75rem; overflow-wrap: anywhere; margin: .55rem 0 1rem; }
      [data-testid="stMetric"] { background: rgba(15, 23, 42, .7); border: 1px solid #1e293b; border-radius: 12px; padding: .9rem 1rem; }
      [data-testid="stMetricLabel"] { color: #94a3b8; }
      [data-testid="stMetricValue"] { color: #f8fafc; font-size: 1.55rem; }
      [data-testid="stSelectbox"], [data-testid="stMultiSelect"] { margin-top: .2rem; }
      [data-testid="stVegaLiteChart"] { background: rgba(15,23,42,.42); border: 1px solid #1e293b; border-radius: 16px; padding: 14px 16px 8px; }
      hr { border-color: #1e293b !important; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_dataset_page() -> None:
    """Render the dataset and camera-anchor explorer page."""
    dataset = load_dataset_index(str(SPLIT_MANIFEST))
    anchors = load_anchor_directions(str(ANCHOR_PATH))

    st.markdown(
        f"""<div class="rollout-header">
<div class="rollout-header-main">
<div class="eyebrow">CV3D / DATASET EXPLORER</div>
<div class="rollout-title">Explore the observation space.</div>
<p class="rollout-copy">Inspect ShapeNet objects across the 48 canonical camera anchors used throughout the next-best-view experiments.</p>
</div>
<div class="run-pill"><span class="run-dot"></span>{len(dataset):,} objects · {len(anchors)} anchors</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-label">Dataset overview</div>',
        unsafe_allow_html=True,
    )

    split_columns = st.columns(3)

    split_descriptions = {
        "train": "training split",
        "val": "model selection",
        "test": "held-out evaluation",
    }

    for column, split_name in zip(
        split_columns, ("train", "val", "test"), strict=True
    ):
        split_count = int((dataset["split"] == split_name).sum())
        split_color = SPLIT_COLORS[split_name]

        with column:
            st.markdown(
                f'<div class="dataset-kpi" style="--accent:{split_color};">'
                f'<div class="research-kpi-label">{SPLIT_LABELS[split_name]} objects</div>'
                f'<div class="research-kpi-value">{split_count:,}</div>'
                f'<div class="research-kpi-meta">{split_descriptions[split_name]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="panel-label" style="margin-top:1.4rem;">Explore an object</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        category_options = sorted(dataset["category"].unique())
        selector_a, selector_b, selector_c, selector_d = st.columns([1.1, 1, 2.2, 1])

        with selector_a:
            selected_category = st.selectbox("Category", category_options)

        category_rows = dataset[dataset["category"] == selected_category]

        available_splits = [
            split_name
            for split_name in ("train", "val", "test")
            if split_name in set(category_rows["split"])
        ]

        with selector_b:
            selected_split = st.selectbox(
                "Object split",
                available_splits,
                format_func=lambda value: SPLIT_LABELS[value],
            )

        object_options = sorted(
            category_rows.loc[
                category_rows["split"] == selected_split,
                "object_id",
            ].tolist()
        )

        with selector_c:
            selected_object = st.selectbox(
                "Object",
                object_options,
            )

        with selector_d:
            selected_anchor = st.selectbox(
                "Camera anchor",
                anchors["anchor_id"].astype(int).tolist(),
                format_func=pretty_anchor,
            )

    category_id = str(category_rows.iloc[0]["category_id"])
    object_key = f"{category_id}/{selected_object}"
    mesh_path = SHAPENET_ROOT / object_key / "models" / "model_normalized.ply"
    image_path = (
        NUM_ROOT
        / object_key
        / "images"
        / f"viewpoint_{selected_anchor}_offset_phi_0.png"
    )
    mesh = load_object_mesh(str(mesh_path))

    
    selected_row = anchors.loc[
        anchors["anchor_id"] == selected_anchor
    ].iloc[0]

    st.markdown(
        '<div class="panel-label" style="margin-top:1.5rem;">Object inspection</div>',
        unsafe_allow_html=True,
    )

    sphere_column, preview_column = st.columns(
        [2.1, 1],
        vertical_alignment="top",
    )

    with sphere_column:
        st.markdown(
            '<div class="viz-title">Camera geometry</div>'
            '<div class="viz-copy">Orbit the object and inspect the 48 canonical camera anchors.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.plotly_chart(
                build_anchor_sphere(
                    anchors,
                    selected_anchor,
                    mesh,
                ),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
            )

    with preview_column:
        st.markdown(
            '<div class="viz-title">Rendered observation</div>'
            f'<div class="viz-copy">Anchor {selected_anchor} · '
            f'{CATEGORY_NAMES.get(category_id, category_id)}</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            split_color = SPLIT_COLORS[selected_split]

            st.markdown(
                f'<div class="split-badge" style="color:{split_color}">'
                f'<span class="split-dot"></span>{SPLIT_LABELS[selected_split]}</div>'
                f'<div class="object-id">{object_key}</div>',
                unsafe_allow_html=True,
            )

            if image_path.is_file():
                st.image(
                    str(image_path),
                    caption=f"Rendered observation from anchor {selected_anchor}",
                    width="stretch",
                )
            else:
                st.markdown(
                    '<div class="observation-empty">'
                    '<div class="observation-empty-icon">◈</div>'
                    '<div class="observation-empty-title">Observation preview unavailable</div>'
                    '<div class="observation-empty-copy">'
                    'The lightweight dashboard keeps dataset metadata and camera geometry '
                    'but omits the large NUM image dataset.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="metric-note">'
                f'<strong>Selected camera</strong><br>'
                f'Azimuth {math.degrees(float(selected_row["azimuth_rad"])):.1f}° · '
                f'Elevation {math.degrees(float(selected_row["elevation_rad"])):.1f}°'
                '</div>',
                unsafe_allow_html=True,
            )


def render_representation_page() -> None:
    """Render the Phase 1 representation-comparison page."""
    results = load_phase1_results(str(PHASE1_ROOT))
    seed_count = results["seed"].nunique() if not results.empty else 0
    variant_count = results["variant"].nunique() if not results.empty else 0

    st.markdown(
        f"""<div class="rollout-header">
<div class="rollout-header-main">
<div class="eyebrow">CV3D / PHASE 1 · REPRESENTATION SWEEP</div>
<div class="rollout-title">Which frozen representation predicts the next view best?</div>
<p class="rollout-copy">Compare frozen feature probes and baselines on single-image prediction of view uncertainty across the 48 candidate camera anchors.</p>
</div>
<div class="run-pill"><span class="run-dot"></span>{variant_count} variants · {seed_count} seeds</div>
</div>""",
        unsafe_allow_html=True,
    )

    if results.empty:
        st.error(f"No Phase 1 summaries found under `{PHASE1_ROOT}`.")
        return

    st.markdown(
        '<div class="panel-label">Representation comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">How well does each representation predict view '
        'uncertainty?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-label" style="margin-top:1.1rem;">Compare representations</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        control_a, control_b, control_c = st.columns([1.15, 1.7, 2.6])

        with control_a:
            split = st.selectbox(
                "Evaluation split",
                ["test", "validation"],
                format_func=str.title,
            )

        with control_b:
            metric = st.selectbox(
                "Metric",
                list(METRICS),
                format_func=lambda value: METRICS[value]["label"],
            )

        with control_c:
            available_backbones = [
                label
                for label in BACKBONE_COLORS
                if label in set(results["backbone_label"])
            ]

            selected_backbones = st.multiselect(
                "Backbone families",
                available_backbones,
                default=available_backbones,
            )

    filtered = results[
        (results["split"] == split)
        & results["backbone_label"].isin(selected_backbones)
        & results[metric].notna()
    ].copy()

    if filtered.empty:
        st.warning("No results match the current filters.")
        return

    metric_meta = METRICS[metric]
    variant_means = filtered.groupby(["variant", "variant_label"])[metric].mean()
    best_index = (
        variant_means.idxmin()
        if metric_meta["direction"] == "lower"
        else variant_means.idxmax()
    )
    _, best_label = best_index

    best_value = float(variant_means.loc[best_index])

    kpi_variant, kpi_score, kpi_count = st.columns(3)

    with kpi_variant:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Leading variant</div>'
            f'<div class="research-kpi-value" style="font-size:1.45rem;">{best_label}</div>'
            f'<div class="research-kpi-meta">{split.title()} split</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_score:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">{metric_meta["short"]}</div>'
            f'<div class="research-kpi-value">{best_value:.4f}</div>'
            f'<div class="research-kpi-meta">'
            f'{"lower is better ↓" if metric_meta["direction"] == "lower" else "higher is better ↑"}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_count:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Variants in view</div>'
            f'<div class="research-kpi-value">{filtered["variant"].nunique()}</div>'
            f'<div class="research-kpi-meta">filtered comparison</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    direction_symbol = (
        "↓ lower is better"
        if metric_meta["direction"] == "lower"
        else "↑ higher is better"
    )
    st.markdown(
        f'<div class="metric-note">{metric_meta["description"]} &nbsp;·&nbsp; '
        f'<strong>{direction_symbol}</strong> &nbsp;·&nbsp; Bars show seed means; '
        'dots show individual seeds. Click to focus, double-click to reset.</div>',
        unsafe_allow_html=True,
    )

    st.altair_chart(build_metric_chart(filtered, metric), width="stretch")

    prediction_data = load_prediction_maps(str(PREDICTION_DATA))
    object_keys = [str(value) for value in prediction_data["object_keys"]]
    variant_names = [str(value) for value in prediction_data["variant_names"]]
    category_ids = sorted(
        {key.split("/", maxsplit=1)[0] for key in object_keys},
        key=lambda value: CATEGORY_NAMES.get(value, value),
    )

    st.markdown(
        '<div class="section-kicker">Prediction inspection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Compare a predicted uncertainty map with its '
        'ground truth.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="metric-note">'
        'Compare the ground-truth uncertainty map with the prediction from a frozen '
        'representation. Brighter regions indicate more uncertain candidate views.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to read the uncertainty maps"):
        st.markdown(
            """
Each map is expressed in **source-relative coordinates**:

- the selected input view is local **+Z** at the center;
- the equator lies halfway to the rim;
- **−Z** lies at the outer rim;
- each colored region corresponds to its nearest camera anchor;
- purple indicates lower uncertainty and yellow indicates higher uncertainty;
- white dots mark the 48 canonical anchors;
- the center **×** marks the excluded source view;
- the orange diamond marks the predicted most-uncertain anchor.
"""
        )

    st.markdown(
        '<div class="panel-label" style="margin-top:1.1rem;">Inspect a prediction</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        map_control_a, map_control_b, map_control_c, map_control_d = st.columns(
            [1.1, 2.2, 1.8, 1.1]
        )

        with map_control_a:
            map_category_id = st.selectbox(
                "Test category",
                category_ids,
                format_func=lambda value: CATEGORY_NAMES.get(value, value),
            )

        category_object_keys = [
            key for key in object_keys if key.startswith(f"{map_category_id}/")
        ]

        with map_control_b:
            map_object_key = st.selectbox(
                "Test object",
                category_object_keys,
                format_func=lambda value: value.split("/", maxsplit=1)[1],
            )

        with map_control_c:
            map_variant = st.selectbox(
                "Prediction variant",
                variant_names,
                format_func=pretty_variant,
            )

        object_index = object_keys.index(map_object_key)
        variant_index = variant_names.index(map_variant)

        available_source_anchors = np.flatnonzero(
            prediction_data["available_source_anchors"][variant_index]
        ).tolist()

        with map_control_d:
            source_anchor = st.selectbox(
                "Input anchor",
                available_source_anchors,
                format_func=pretty_anchor,
            )

        if len(available_source_anchors) == 1:
            st.caption(
                "Only source anchor 0 was retained for the official PUN baseline; "
                "the locally trained probes support all 48 input anchors."
            )

    target_map = prediction_data["targets"][object_index, source_anchor]
    predicted_map = prediction_data["predictions"][
        variant_index, object_index, source_anchor
    ]
    map_anchors = load_anchor_directions(str(ANCHOR_PATH))
    st.plotly_chart(
        build_prediction_map_chart(
            map_anchors,
            target_map,
            predicted_map,
            pretty_variant(map_variant),
            int(prediction_data["target_best_ids"][object_index, source_anchor]),
            int(
                prediction_data["predicted_best_ids"][
                    variant_index, object_index, source_anchor
                ]
            ),
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    source_image = (
        NUM_ROOT
        / map_object_key
        / "images"
        / f"viewpoint_{source_anchor}_offset_phi_0.png"
    )
    if source_image.is_file():
        st.image(
            str(source_image),
            caption=f"Input observation · {map_object_key} · source anchor {source_anchor}",
            width=180,
        )

    with st.expander("Inspect the underlying metrics"):
        table_columns = [
            "variant_label",
            "backbone_label",
            "seed",
            *METRICS.keys(),
            "trainable_parameters",
            "best_epoch",
        ]
        st.dataframe(
            filtered[table_columns].sort_values(["variant_label", "seed"]),
            column_config={
                "variant_label": "Variant",
                "backbone_label": "Backbone",
                "seed": "Seed",
                "normalized_regret_mean": st.column_config.NumberColumn(
                    "Regret", format="%.4f"
                ),
                "spearman_mean": st.column_config.NumberColumn(
                    "Spearman", format="%.4f"
                ),
                "ndcg_at_5_mean": st.column_config.NumberColumn(
                    "NDCG@5", format="%.4f"
                ),
                "huber_loss": st.column_config.NumberColumn(
                    "Huber", format="%.4f"
                ),
                "ranking_loss": st.column_config.NumberColumn(
                    "Ranking loss", format="%.4f"
                ),
                "loss": st.column_config.NumberColumn(
                    "Combined loss", format="%.4f"
                ),
                "trainable_parameters": st.column_config.NumberColumn(
                    "Trainable params", format="localized"
                ),
                "best_epoch": "Best epoch",
            },
            hide_index=True,
            width="stretch",
        )


def render_closed_loop_page() -> None:
    """Render the Phase 2 and Phase 3 closed-loop evaluation page."""
    tables = load_closed_loop_tables(
        str(PHASE2_CLOSED_LOOP_ROOT), str(PHASE3_CLOSED_LOOP_ROOT)
    )
    comparison = tables["comparison"]

    policy_count = comparison["policy"].nunique() if not comparison.empty else 0

    st.markdown(
        f"""<div class="rollout-header">
<div class="rollout-header-main">
<div class="eyebrow">CV3D / PHASE 2–3 · CLOSED-LOOP EVALUATION</div>
<div class="rollout-title">Which policy explores the object most effectively?</div>
<p class="rollout-copy">Compare single-image baselines and history-conditioned policies over the same ten-view closed-loop evaluation protocol.</p>
</div>
<div class="run-pill"><span class="run-dot"></span>{policy_count} policies · 2 phases</div>
</div>""",
        unsafe_allow_html=True,
    )

    if comparison.empty or tables["coverage"].empty:
        st.error(
            "Closed-loop result tables were not found under the configured Phase 2 and "
            "Phase 3 output directories."
        )
        return

    st.markdown(
        '<div class="panel-label">Policy comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Which policy leaves the most surface observed?</div>',
        unsafe_allow_html=True,
    )
    summary_metrics = {
        "final_coverage_mean": ("Final absolute coverage", "↑ higher is better"),
        "coverage_auc_mean": ("Coverage AUC", "↑ higher is better"),
        "normalized_regret_mean": ("Mean normalized regret", "↓ lower is better"),
        "ndcg_at_5_mean": ("Mean NDCG @ 5", "↑ higher is better"),
        "median_policy_ms": ("Median decision time (ms)", "↓ lower is better"),
    }
    st.markdown(
        '<div class="panel-label" style="margin-top:1rem;">Evaluation metric</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        summary_metric = st.selectbox(
            "Policy summary metric",
            list(summary_metrics),
            format_func=lambda value: summary_metrics[value][0],
            label_visibility="collapsed",
        )

    metric_label, direction = summary_metrics[summary_metric]

    oracle_rows = comparison[comparison["policy"] == "oracle"]
    non_oracle = comparison[comparison["policy"] != "oracle"]

    oracle_row = oracle_rows.iloc[0]

    best_non_oracle_row = non_oracle.loc[
        non_oracle[summary_metric].idxmin()
        if direction.startswith("↓")
        else non_oracle[summary_metric].idxmax()
    ]

    oracle_value = float(oracle_row[summary_metric])
    best_non_oracle_value = float(best_non_oracle_row[summary_metric])

    if summary_metric == "final_coverage_mean":
        gap_value = abs(oracle_value - best_non_oracle_value) * 100.0
        oracle_display = f"{oracle_value * 100:.1f}%"
        best_display = f"{best_non_oracle_value * 100:.1f}%"
        gap_display = f"{gap_value:.1f} pp"
    else:
        gap_value = abs(oracle_value - best_non_oracle_value)
        oracle_display = f"{oracle_value:.4f}"
        best_display = f"{best_non_oracle_value:.4f}"
        gap_display = f"{gap_value:.4f}"

    kpi_oracle, kpi_policy, kpi_gap = st.columns(3)

    with kpi_oracle:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Oracle reference</div>'
            f'<div class="research-kpi-value">{oracle_display}</div>'
            f'<div class="research-kpi-meta">{metric_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_policy:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Best non-oracle policy</div>'
            f'<div class="research-kpi-value" style="font-size:1.35rem;">'
            f'{best_non_oracle_row["policy_label"]}</div>'
            f'<div class="research-kpi-meta">{best_display} · {metric_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_gap:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Gap to oracle</div>'
            f'<div class="research-kpi-value">{gap_display}</div>'
            f'<div class="research-kpi-meta">best non-oracle vs oracle</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="metric-note"><strong>{direction}</strong> · Solid lines are Phase 2; '
        'dashed lines are Phase 3. Select a policy in the legend to isolate it.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-kicker">Coverage × reconstruction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Does observing more surface produce better '
        'VGGT and 2DGS reconstructions?</div>',
        unsafe_allow_html=True,
    )
    vggt_reconstruction = tables["vggt_reconstruction"]
    gaussian_splatting = load_gaussian_splatting_metrics(
        str(GAUSSIAN_SPLATTING_CPU_RECOVERY_ROOT)
    )
    available_reconstruction_metrics = [
        metric
        for metric in RECONSTRUCTION_METRICS
        if metric in gaussian_splatting.columns
        or f"{metric}_mean" in vggt_reconstruction.columns
    ]
    coverage_policy_labels = set(tables["coverage"]["policy_label"])
    vggt_reconstruction_policy_labels = (
        set(vggt_reconstruction["policy_label"])
        if not vggt_reconstruction.empty
        else set()
    )
    gs_policy_labels = (
        set(gaussian_splatting["policy_label"])
        if not gaussian_splatting.empty
        else set()
    )
    comparable_policies = [
        label
        for label in CLOSED_LOOP_POLICY_COLORS
        if label in coverage_policy_labels
        and label in vggt_reconstruction_policy_labels.union(gs_policy_labels)
    ]

    policy_control, reconstruction_metric_control = st.columns(2)
    st.markdown(
        '<div class="panel-label" style="margin-top:1rem;">Compare coverage and reconstruction</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        policy_control, reconstruction_metric_control = st.columns([2.2, 1.4])

        default_policy_labels = [
            "Oracle",
            "Farthest view",
            "VGGT · single image",
            "VGGT · independent history",
            "VGGT · token attention",
        ]

        default_selected_policies = [
            label
            for label in default_policy_labels
            if label in comparable_policies
        ]

        selected_policies = st.multiselect(
            "Policies to compare",
            comparable_policies,
            default=default_selected_policies,
            key="closed_loop_policy_comparison_v2",
        )

        with reconstruction_metric_control:
            reconstruction_metric = st.selectbox(
                "Reconstruction metric",
                available_reconstruction_metrics,
                format_func=lambda value: RECONSTRUCTION_METRICS[value],
                disabled=not available_reconstruction_metrics,
            )

    if not selected_policies:
        st.info("Select at least one policy to draw the comparison graphs.")
    else:
        coverage_for_chart = tables["coverage"][
            tables["coverage"]["policy_label"].isin(selected_policies)
        ].dropna(subset=["coverage_mean"])
        vggt_metric = f"{reconstruction_metric}_mean" if reconstruction_metric else ""
        vggt_for_chart = pd.DataFrame()
        if not vggt_reconstruction.empty and vggt_metric in vggt_reconstruction:
            vggt_for_chart = vggt_reconstruction[
                vggt_reconstruction["policy_label"].isin(selected_policies)
            ].dropna(subset=[vggt_metric])
            vggt_for_chart = vggt_for_chart.loc[
                ~(
                    (vggt_for_chart["policy"] == "farthest")
                    & (vggt_for_chart["acquired_view_count"] == 2)
                )
            ].copy()
        gs_for_chart = pd.DataFrame()
        if not gaussian_splatting.empty and reconstruction_metric:
            gs_for_chart = gaussian_splatting[
                gaussian_splatting["policy_label"].isin(selected_policies)
            ].dropna(subset=[reconstruction_metric])
        coverage_values = coverage_for_chart["coverage_mean"].to_numpy(dtype=float)
        coverage_values = coverage_values[np.isfinite(coverage_values)]
        coverage_max = float(coverage_values.max()) if coverage_values.size else 1.0
        coverage_y_domain = (0.0, coverage_max * 1.05 if coverage_max > 0 else 1.0)

        reconstruction_values: list[np.ndarray] = []
        if not vggt_for_chart.empty:
            vggt_values = vggt_for_chart[vggt_metric].to_numpy(dtype=float)
            vggt_values = vggt_values[np.isfinite(vggt_values)]
            if vggt_values.size:
                reconstruction_values.append(vggt_values)
        if not gs_for_chart.empty:
            gs_values = gs_for_chart[reconstruction_metric].to_numpy(dtype=float)
            gs_values = gs_values[np.isfinite(gs_values)]
            if gs_values.size:
                reconstruction_values.append(gs_values)
        reconstruction_max = (
            float(np.concatenate(reconstruction_values).max())
            if reconstruction_values
            else 1.0
        )
        reconstruction_y_domain = (
            0.0,
            reconstruction_max * 1.05 if reconstruction_max > 0 else 1.0,
        )

        st.markdown(
            '<div class="viz-title">Surface coverage ↑</div>'
            '<div class="viz-copy">'
            'More acquired views should reveal more of the object · higher is better.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(
            build_closed_loop_curve(
                coverage_for_chart,
                x="acquired_view_count",
                y="coverage_mean",
                x_title="Acquired views",
                y_title="Absolute surface coverage",
                y_domain=coverage_y_domain,
            ),
            width="stretch",
        )

    if vggt_for_chart.empty or not reconstruction_metric:
        st.info("No completed VGGT reconstruction evaluations were found.")
    elif gs_for_chart.empty:
        st.info("No completed 2DGS evaluations were found.")
    else:
        reconstruction_label = RECONSTRUCTION_METRICS[reconstruction_metric]
        reconstruction_lower_is_better = reconstruction_label.endswith("↓")

        reconstruction_direction = (
            "lower is better"
            if reconstruction_lower_is_better
            else "higher is better"
        )

        st.markdown(
            '<div class="viz-title" style="margin-top:1.3rem;">'
            'Reconstruction quality'
            '</div>'
            f'<div class="viz-copy">'
            f'As more views are acquired, we test whether reconstruction improves. '
            f'The selected metric is <strong>{reconstruction_label}</strong> · '
            f'{reconstruction_direction}.'
            f'</div>',
            unsafe_allow_html=True,
        )

        
        st.altair_chart(
                build_reconstruction_comparison_chart(
                    vggt_for_chart,
                    gs_for_chart,
                    vggt_metric=vggt_metric,
                    gaussian_metric=reconstruction_metric,
                    y_title=RECONSTRUCTION_METRICS[reconstruction_metric],
                    y_domain=reconstruction_y_domain,
                ),
                width="stretch",
            )
    st.caption(
        "Use the shared policy and metric controls to compare the same variants across "
        "all three graphs. "
        "Click a legend entry to isolate a curve and hover over a point for its value "
        "and cohort size. All y-axes start at zero, and the VGGT and 2DGS charts share "
        "the same scale. The anomalous Phase 2 farthest-view VGGT point at two inputs "
        "is omitted from these plots and their y-axis range only; source metrics remain "
        "unchanged. Coverage and VGGT reconstruction use the full 300-object test cohort; "
        "2DGS means use only the completed runs currently available."
    )

    st.markdown(
        '<div class="section-kicker">Decision quality</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-title">How does ranking quality change as history grows?</div>',
        unsafe_allow_html=True,
    )
    quality_metrics = {
        "normalized_regret": "Normalized regret ↓",
        "selected_true_gain": "Selected true surface gain ↑",
        "spearman": "Spearman correlation ↑",
        "ndcg_at_5": "NDCG @ 5 ↑",
    }
    st.markdown(
        '<div class="panel-label" style="margin-top:1rem;">Decision metric</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        quality_metric = st.selectbox(
            "Per-decision metric",
            list(quality_metrics),
            format_func=lambda value: quality_metrics[value],
            label_visibility="collapsed",
        )
    quality = tables["per_step"].dropna(subset=[quality_metric]).copy()
    quality["decision_number"] = quality["step_index"].astype(int) + 1
    quality = (
        quality.groupby(
            ["phase", "policy", "policy_label", "decision_number"], as_index=False
        )[quality_metric]
        .mean()
    )

    st.markdown(
        '<div class="viz-title">Decision quality across the rollout</div>'
        '<div class="viz-copy">'
        'Track how policy ranking quality changes as more observations are acquired.'
        '</div>',
        unsafe_allow_html=True,
        )
    st.altair_chart(
        build_closed_loop_curve(
            quality,
            x="decision_number",
            y=quality_metric,
            x_title="Decision number",
            y_title=quality_metrics[quality_metric],
        ),
        width="stretch",
    )

    with st.expander("Inspect aggregate closed-loop metrics"):
        display_columns = [
            "phase",
            "policy_label",
            "object_count",
            "final_coverage_mean",
            "coverage_auc_mean",
            "normalized_regret_mean",
            "spearman_mean",
            "ndcg_at_5_mean",
            "median_policy_ms",
        ]
        st.dataframe(
            comparison[display_columns].sort_values(["phase", "policy_label"]),
            hide_index=True,
            width="stretch",
        )


def render_rollout_inspection_page() -> None:
    """Render object-level, view-by-view closed-loop rollout inspection."""
    catalog = load_rollout_catalog(
        str(PHASE2_CLOSED_LOOP_ROOT),
        str(PHASE3_CLOSED_LOOP_ROOT),
    )

    st.markdown(
        f"""<div class="rollout-header">
<div class="rollout-header-main">
<div class="eyebrow">CV3D / POLICY ROLLOUT EXPLORER</div>
<div class="rollout-title">Follow the policy, one view at a time.</div>
<p class="rollout-copy">Explore how a next-best-view policy moves around an object, acquires observations, and increases visible surface coverage.</p>
</div>
<div class="run-pill"><span class="run-dot"></span>{len(catalog):,} rollouts loaded</div>
</div>""",
        unsafe_allow_html=True,
    )

    if catalog.empty:
        st.info("No replayable rollout files are available for object-level inspection.")
        return

    st.markdown(
        '<div class="panel-label">Explore a rollout</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        rollout_a, rollout_b, rollout_c = st.columns([1.5, 1.05, 1.8])

        with rollout_a:
            rollout_policy = st.selectbox(
                "Policy",
                sorted(catalog["policy"].unique(), key=pretty_policy),
                format_func=pretty_policy,
            )

        policy_catalog = catalog[catalog["policy"] == rollout_policy]

        category_options = sorted(
            policy_catalog["category_id"].unique(),
            key=lambda value: CATEGORY_NAMES.get(value, value),
        )

        with rollout_b:
            rollout_category = st.selectbox(
                "Category",
                category_options,
                format_func=lambda value: CATEGORY_NAMES.get(value, value),
            )

        category_catalog = policy_catalog[
            policy_catalog["category_id"] == rollout_category
        ]
        object_options = sorted(category_catalog["object_id"].unique())

        with rollout_c:
            rollout_object = st.selectbox(
                "Object",
                object_options,
            )

        object_catalog = category_catalog[
            category_catalog["object_id"] == rollout_object
        ]
        rollout_path = object_catalog["path"].iloc[0]
        rollout = load_dashboard_rollout(str(rollout_path))

        acquired = np.asarray(rollout["acquired_anchor_ids"], dtype=int)
        counts = np.asarray(rollout["acquired_view_counts"], dtype=int)
        coverage = np.asarray(rollout["coverage"], dtype=float)
        steps = rollout["steps"]

        st.markdown(
            '<div class="panel-label" style="margin-top:.7rem;">Acquisition progress</div>',
            unsafe_allow_html=True,
        )

        acquired_count = st.segmented_control(
            "Views acquired",
            options=counts.astype(int).tolist(),
            default=int(counts[-1]),
            required=True,
            width="stretch",
            label_visibility="collapsed",
        )

    current_index = int(np.flatnonzero(counts == acquired_count)[0])
    current_anchor = int(acquired[current_index])
    current_step = steps[current_index - 1] if current_index > 0 else None
    object_key = f"{rollout_category}/{rollout_object}"

    coverage_percent = coverage[current_index] * 100.0

    if current_index > 0:
        coverage_gain_pp = (
            coverage[current_index] - coverage[current_index - 1]
        ) * 100.0
        coverage_gain_label = f"{coverage_gain_pp:+.1f} pp"
        coverage_gain_meta = "from previous view"
    else:
        coverage_gain_label = "—"
        coverage_gain_meta = "initial observation"

    regret_label = (
        f'{float(current_step["normalized_regret"]):.3f}'
        if current_step and current_step.get("normalized_regret") is not None
        else "—"
    )

    st.markdown(
        '<div class="panel-label" style="margin-top:1.4rem;">Current state</div>',
        unsafe_allow_html=True,
    )

    kpi_view, kpi_coverage, kpi_gain, kpi_regret = st.columns(4)

    with kpi_view:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Current view</div>'
            f'<div class="research-kpi-value">{int(acquired_count)} '
            f'<span>/ {int(counts[-1])}</span></div>'
            f'<div class="research-kpi-meta">Anchor {current_anchor}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_coverage:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Surface coverage</div>'
            f'<div class="research-kpi-value">{coverage_percent:.1f}%</div>'
            f'<div class="research-kpi-meta">visible surface</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_gain:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Coverage gain</div>'
            f'<div class="research-kpi-value">{coverage_gain_label}</div>'
            f'<div class="research-kpi-meta">{coverage_gain_meta}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_regret:
        st.markdown(
            f'<div class="research-kpi">'
            f'<div class="research-kpi-label">Decision regret</div>'
            f'<div class="research-kpi-value">{regret_label}</div>'
            f'<div class="research-kpi-meta">lower is better ↓</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    mesh_path = SHAPENET_ROOT / object_key / "models" / "model_normalized.ply"

    st.markdown(
        '<div class="panel-label" style="margin-top:1.5rem;">Rollout inspection</div>',
        unsafe_allow_html=True,
    )

    trajectory_column, observation_column = st.columns(
        [2.1, 1],
        vertical_alignment="top",
    )

    with trajectory_column:
        st.markdown(
            '<div class="viz-title">Camera trajectory</div>'
            '<div class="viz-copy">Acquisition order across the 48 canonical camera anchors.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.plotly_chart(
                build_rollout_trajectory(
                    load_anchor_directions(str(ANCHOR_PATH)),
                    acquired[: current_index + 1],
                    load_object_mesh(str(mesh_path)),
                ),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
            )

    with observation_column:
        st.markdown(
            '<div class="viz-title">Current observation</div>'
            f'<div class="viz-copy">View {int(acquired_count)} · anchor {current_anchor}</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            observation_path = (
                NUM_ROOT
                / object_key
                / "images"
                / f"viewpoint_{current_anchor}_offset_phi_0.png"
            )

            if observation_path.is_file():
                st.image(
                    str(observation_path),
                    caption=f"{pretty_policy(rollout_policy)} · anchor {current_anchor}",
                    width="stretch",
                )
            else:
                st.markdown(
                    '<div class="observation-empty">'
                    '<div class="observation-empty-icon">◈</div>'
                    '<div class="observation-empty-title">Observation preview unavailable</div>'
                    '<div class="observation-empty-copy">'
                    'The lightweight dashboard keeps rollout metrics but omits the large '
                    'NUM image dataset.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    trajectory = pd.DataFrame(
        {
            "Acquired views": counts[: current_index + 1],
            "Absolute coverage": coverage[: current_index + 1],
        }
    ).set_index("Acquired views")

    st.markdown(
        '<div class="viz-title" style="margin-top:1.2rem;">Coverage progression</div>'
        '<div class="viz-copy">Visible surface accumulated after each acquired view.</div>',
        unsafe_allow_html=True,
    )

    coverage_chart_data = trajectory.reset_index().copy()
    coverage_chart_data["Coverage (%)"] = (
        coverage_chart_data["Absolute coverage"] * 100.0
    )

    coverage_line = (
        alt.Chart(coverage_chart_data)
        .mark_line(
            color="#38bdf8",
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "Acquired views:Q",
                title="Acquired views",
                axis=alt.Axis(
                    tickMinStep=1,
                    grid=False,
                ),
                scale=alt.Scale(domain=[1, int(counts[-1])]),
            ),
            y=alt.Y(
                "Coverage (%):Q",
                title="Visible surface",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f"),
            ),
            tooltip=[
                alt.Tooltip("Acquired views:Q", title="View", format=".0f"),
                alt.Tooltip("Coverage (%):Q", title="Coverage", format=".1f"),
            ],
        )
    )

    coverage_area = (
        alt.Chart(coverage_chart_data)
        .mark_area(
            color="#38bdf8",
            opacity=0.08,
        )
        .encode(
            x="Acquired views:Q",
            y=alt.Y(
                "Coverage (%):Q",
                scale=alt.Scale(domain=[0, 100]),
            ),
        )
    )

    coverage_points = (
        alt.Chart(coverage_chart_data)
        .mark_circle(
            size=65,
            color="#38bdf8",
            stroke="#07101f",
            strokeWidth=1.5,
        )
        .encode(
            x="Acquired views:Q",
            y=alt.Y(
                "Coverage (%):Q",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("Acquired views:Q", title="View", format=".0f"),
                alt.Tooltip("Coverage (%):Q", title="Coverage", format=".1f"),
            ],
        )
    )

    current_chart_data = coverage_chart_data[
        coverage_chart_data["Acquired views"] == acquired_count
    ]

    current_point = (
        alt.Chart(current_chart_data)
        .mark_circle(
            size=180,
            color="#22d3ee",
            stroke="#f8fafc",
            strokeWidth=2,
        )
        .encode(
            x="Acquired views:Q",
            y=alt.Y(
                "Coverage (%):Q",
                scale=alt.Scale(domain=[0, 100]),
            ),
        )
    )

    coverage_chart = (
        alt.layer(
            coverage_area,
            coverage_line,
            coverage_points,
            current_point,
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#94a3b8",
            titleColor="#94a3b8",
            gridColor="#1e293b",
            domainColor="#334155",
            tickColor="#334155",
        )
    )

    st.altair_chart(
        coverage_chart,
        width="stretch",
    )


def render_reconstruction_gallery_page() -> None:
    """Render object-level comparisons across reconstruction methods."""
    render_catalog = load_3dgs_render_catalog(str(GAUSSIAN_SPLATTING_ROOT))
    vggt_render_catalog = load_vggt_render_catalog(
        str(PHASE2_CLOSED_LOOP_ROOT), str(PHASE3_CLOSED_LOOP_ROOT)
    )
    available_catalogs = [
        frame for frame in (render_catalog, vggt_render_catalog) if not frame.empty
    ]
    all_renders = (
        pd.concat(available_catalogs, ignore_index=True).drop_duplicates(
            ["policy", "category_id", "object_id", "acquired_view_count"]
        )
        if available_catalogs
        else pd.DataFrame()
    )

    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="eyebrow">CV3D / reconstruction gallery</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-title">Compare reconstructions<br>from every angle.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-copy">Inspect VGGT point clouds, 3DGS novel-view renders, '
            'and reconstructed 2DGS surfaces alongside ground truth.</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f'<div class="run-pill"><span class="run-dot"></span>{len(all_renders):,} '
            'comparisons loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-kicker">Reconstruction comparisons</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Inspect VGGT point clouds, 3DGS renders, and the '
        'reconstructed 2DGS surface against ground truth.</div>',
        unsafe_allow_html=True,
    )
    if not available_catalogs:
        st.info("No completed reconstruction viewer is available yet.")
    else:
        reconstruction_policies = sorted(
            all_renders["policy"].unique(), key=pretty_policy
        )
        default_reconstruction_policies = [reconstruction_policies[0]]
        (
            render_policy_control,
            render_category_control,
            render_object_control,
        ) = st.columns([2.2, 1, 2])
        with render_policy_control:
            selected_reconstruction_policies = st.multiselect(
                "Reconstruction policies",
                reconstruction_policies,
                default=default_reconstruction_policies,
                format_func=pretty_policy,
            )
        if not selected_reconstruction_policies:
            with render_category_control:
                st.selectbox("Reconstruction category", [], disabled=True)
            with render_object_control:
                st.selectbox("Reconstruction object", [], disabled=True)
            st.info("Select at least one reconstruction policy to display its row.")
        else:
            policy_renders = all_renders[
                all_renders["policy"].isin(selected_reconstruction_policies)
            ]
            render_categories = sorted(
                policy_renders["category_id"].unique(),
                key=lambda value: CATEGORY_NAMES.get(value, value),
            )
            with render_category_control:
                render_category = st.selectbox(
                    "Reconstruction category",
                    render_categories,
                    format_func=lambda value: CATEGORY_NAMES.get(value, value),
                )
            category_renders = policy_renders[
                policy_renders["category_id"] == render_category
            ]
            with render_object_control:
                render_object = st.selectbox(
                    "Reconstruction object",
                    sorted(category_renders["object_id"].unique()),
                )
            selected_renders = category_renders[
                category_renders["object_id"] == render_object
            ].sort_values("acquired_view_count")
            render_view_counts = sorted(
                selected_renders["acquired_view_count"].astype(int).unique().tolist()
            )
            render_view_count = st.segmented_control(
                "Reconstruction input views",
                options=render_view_counts,
                default=render_view_counts[-1],
                required=True,
                width="stretch",
            )
            matrix_headers = st.columns(
                [0.9, 1, 1, 1, 1], gap="small", vertical_alignment="bottom"
            )
            for column, label in zip(
                matrix_headers,
                ("Policy", "3DGS render", "2DGS surface", "VGGT cloud", "Ground truth"),
            ):
                with column:
                    st.markdown(
                        f'<div class="matrix-header">{label}</div>',
                        unsafe_allow_html=True,
                    )
            for reconstruction_policy in selected_reconstruction_policies:
                render_reconstruction_comparison_row(
                    reconstruction_policy,
                    render_category,
                    render_object,
                    render_view_count,
                    render_catalog,
                    vggt_render_catalog,
                )
            selection_label = (
                f"{CATEGORY_NAMES.get(render_category, render_category)} / "
                f"{render_object} · {render_view_count} acquired views"
            )
            st.markdown(
                f'<div class="matrix-footer"><strong>{selection_label}</strong> · '
                "Drag any point-cloud cell to rotate the entire matrix; zoom is shared. "
                "Each 3DGS cell follows the nearest rendered anchor. VGGT is oracle-ICP "
                "aligned.</div>",
                unsafe_allow_html=True,
            )


navigation = st.navigation(
    [
        st.Page(render_dataset_page, title="Dataset", url_path="dataset", default=True),
        st.Page(
            render_representation_page,
            title="Phase 1 · Representations",
            url_path="representations",
        ),
        st.Page(
            render_closed_loop_page,
            title="Phase 2–3 · Closed loop",
            url_path="closed-loop",
        ),
        st.Page(
            render_rollout_inspection_page,
            title="Policy rollouts",
            url_path="rollout-inspection",
        ),
        st.Page(
            render_reconstruction_gallery_page,
            title="3D reconstruction",
            url_path="reconstruction-gallery",
        ),
    ],
    position="top",
)
navigation.run()
