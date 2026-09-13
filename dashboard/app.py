"""Interactive experiment dashboard for the CV3D next-best-view project."""

from __future__ import annotations

import json
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = REPO_ROOT / "outputs" / "phase1" / "backbone_sweep"
PHASE2_CLOSED_LOOP_ROOT = (
    REPO_ROOT / "outputs" / "phase2" / "phase2_closed_loop_reconstruction" / "seed_0"
)
PHASE3_CLOSED_LOOP_ROOT = (
    REPO_ROOT
    / "outputs"
    / "phase3"
    / "controlled_history_comparison_reconstruction"
    / "seed_0"
)
PHASE3_POSE_DEEPSETS_ROOT = (
    REPO_ROOT / "outputs" / "phase3" / "controlled_pose_deepsets" / "seed_0"
)
PHASE3_TOKEN_ATTENTION_ROOT = (
    REPO_ROOT / "outputs" / "phase3" / "controlled_token_attention" / "seed_0"
)
GAUSSIAN_SPLATTING_ROOT = (
    REPO_ROOT / "outputs" / "gaussian_splatting_variant_comparison"
)
GAUSSIAN_SPLATTING_CPU_RECOVERY_ROOT = (
    REPO_ROOT / "outputs" / "gaussian_splatting_cpu_recovery"
)
NUM_ROOT = REPO_ROOT / "data" / "NUM"
SHAPENET_ROOT = REPO_ROOT / "data" / "ShapeNetCore.v2"
SPLIT_MANIFEST = REPO_ROOT / "data" / "splits" / "num_v1.json"
ANCHOR_PATH = REPO_ROOT / "src" / "nbv" / "geometry" / "anchors_v1.csv"
PREDICTION_DATA = REPO_ROOT / "dashboard" / "data" / "phase1_test_predictions.npz"

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
    "Random": "#64748b",
    "Farthest view": "#3b82f6",
    "PUN": "#f59e0b",
    "VGGT · single image": "#8b5cf6",
    "Oracle": "#34d399",
    "VGGT · independent history": "#06b6d4",
    "VGGT · joint history": "#ec4899",
    "VGGT · pose DeepSets": "#ef4444",
    "VGGT · token attention": "#84cc16",
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
            "bgcolor": "rgba(15,23,42,.42)",
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
        phase_token, policy = variant.split("_", maxsplit=1)
        records.append(
            {
                "phase": "Phase 2" if phase_token == "phase2" else "Phase 3",
                "policy": policy,
                "policy_label": pretty_policy(policy),
                "category_id": category_id,
                "object_id": object_id,
                "object_key": f"{category_id}/{object_id}",
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


def make_manual_3dgs_comparison(document: str) -> str:
    """Lock side-by-side mode and add manual drag across all 48 anchors."""
    document = document.replace(
        "<button id='play'>Pause</button>",
        "<span>Drag the 3DGS rendering or use the anchor slider</span>",
    )
    document = document.replace(
        "<label>Layout <select id='layout'><option value='side'>Side by side</option>"
        "<option value='overlay'>Overlay</option></select></label>",
        "<select id='layout' hidden><option value='side' selected>Side by side</option>"
        "</select>",
    )
    document = document.replace(
        "<label>Overlay opacity <input id='opacity' type='range' min='0' max='1' "
        "step='.05' value='.5'></label>",
        "<input id='opacity' type='range' value='.5' hidden>",
    )
    document = document.replace("let index=0,playing=true;", "let index=0;")
    automatic_controls = (
        "document.getElementById('play').onclick=e=>{playing=!playing;"
        "e.target.textContent=playing?'Pause':'Play'};\n"
        "setInterval(()=>{if(playing){index=(index+1)%data.references.length;show()}},350);"
    )
    manual_controls = (
        "function installDragRotation(surface){let dragging=false;"
        "surface.draggable=false;surface.style.cursor='grab';"
        "surface.style.touchAction='none';"
        "function selectAnchor(e){const bounds=surface.getBoundingClientRect();"
        "const fraction=Math.max(0,Math.min(1,(e.clientX-bounds.left)/bounds.width));"
        "index=Math.round(fraction*(data.references.length-1));show()}"
        "surface.onpointerdown=e=>{dragging=true;surface.setPointerCapture(e.pointerId);"
        "surface.style.cursor='grabbing';selectAnchor(e)};"
        "surface.onpointermove=e=>{if(dragging)selectAnchor(e)};"
        "surface.onpointerup=surface.onpointercancel=e=>{dragging=false;"
        "surface.style.cursor='grab'}}"
        "installDragRotation(document.getElementById('predictionSide'));"
        "installDragRotation(document.getElementById('prediction'));"
    )
    return document.replace(automatic_controls, manual_controls)


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
        title="Evaluation",
        scale=alt.Scale(domain=["Phase 2", "Phase 3"], range=[[1, 0], [7, 4]]),
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
    figure.update_layout(height=560, showlegend=False)
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
      [data-testid="stHeader"] { background: rgba(7, 16, 31, .82); }
      .block-container { max-width: 1480px; padding-top: 2.2rem; padding-bottom: 4rem; }
      .eyebrow { color: #22d3ee; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
      .hero-title { color: #f8fafc; font-size: clamp(2.2rem, 5vw, 4.2rem); font-weight: 680; line-height: .98; letter-spacing: -.055em; margin: .65rem 0 .8rem; }
      .hero-copy { color: #94a3b8; font-size: 1rem; max-width: 680px; line-height: 1.65; }
      .run-pill { display: inline-flex; align-items: center; gap: .5rem; color: #a7f3d0; background: rgba(16,185,129,.1); border: 1px solid rgba(52,211,153,.24); border-radius: 999px; padding: .38rem .7rem; font-size: .75rem; }
      .run-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 12px #34d399; }
      .section-kicker { color: #64748b; font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-top: 2rem; }
      .section-title { color: #f8fafc; font-size: 1.65rem; font-weight: 650; letter-spacing: -.025em; margin: .25rem 0 0; }
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
        '<div class="eyebrow">CV3D / dataset</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-title">Explore the objects<br>and camera sphere.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">The NUM dataset contains 13 ShapeNet categories observed '
        'from 48 canonical camera anchors. Every object belongs exclusively to train, '
        'validation, or test.</div>',
        unsafe_allow_html=True,
    )

    split_columns = st.columns(3)
    for column, split_name in zip(
        split_columns, ("train", "val", "test"), strict=True
    ):
        split_count = int((dataset["split"] == split_name).sum())
        with column:
            st.metric(f"{SPLIT_LABELS[split_name]} objects", f"{split_count:,}")

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
            category_rows["split"] == selected_split, "object_id"
        ].tolist()
    )
    with selector_c:
        selected_object = st.selectbox("Object", object_options)
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

    sphere_column, preview_column = st.columns(
        [2.15, 1], vertical_alignment="center"
    )
    with sphere_column:
        st.plotly_chart(
            build_anchor_sphere(anchors, selected_anchor, mesh),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
        )
    with preview_column:
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
            st.info(
                "The split and anchor explorer is available, but this rendered "
                "view requires the local `data/NUM` dataset."
            )
        selected_row = anchors.loc[
            anchors["anchor_id"] == selected_anchor
        ].iloc[0]
        st.markdown(
            f'<div class="metric-note"><strong>Selected camera</strong><br>'
            f'Azimuth {math.degrees(float(selected_row["azimuth_rad"])):.1f}° · '
            f'Elevation {math.degrees(float(selected_row["elevation_rad"])):.1f}°'
            '<br><br>Selecting an anchor aligns the camera to its view. Drag the '
            'sphere to orbit; hover over a marker to inspect its anchor ID and pose.</div>',
            unsafe_allow_html=True,
        )


def render_representation_page() -> None:
    """Render the Phase 1 representation-comparison page."""
    results = load_phase1_results(str(PHASE1_ROOT))
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="eyebrow">CV3D / representation sweep</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-title">See what predicts<br>view uncertainty.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-copy">Compare frozen feature probes and baselines on '
            'single-image prediction of the Neural Uncertainty Map across 48 candidate '
            'views.</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        seed_count = results["seed"].nunique() if not results.empty else 0
        st.markdown(
            f'<div class="run-pill"><span class="run-dot"></span>{seed_count} seeds '
            'loaded · live outputs</div>',
            unsafe_allow_html=True,
        )

    if results.empty:
        st.error(f"No Phase 1 summaries found under `{PHASE1_ROOT}`.")
        return

    st.markdown(
        '<div class="section-title">How well does each representation predict view '
        'uncertainty?</div>',
        unsafe_allow_html=True,
    )

    control_a, control_b, control_c = st.columns([1.15, 1.7, 2.6])
    with control_a:
        split = st.selectbox(
            "Evaluation split", ["test", "validation"], format_func=str.title
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

    kpi_a, kpi_d = st.columns(2)
    kpi_a.metric("Leading variant", best_label)
    kpi_d.metric("Variants in view", filtered["variant"].nunique())

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
        '<div class="metric-note">Choose a test object, input anchor, and Phase 1 variant. '
        'Each map is rotated into source-relative coordinates, making the selected input view '
        'local +Z. The sphere is then flattened with an azimuthal-equidistant projection: +Z '
        'is at the center, the equator is halfway to the rim, −Z is at the rim, and azimuth '
        'determines the direction around the circle. Each colored region belongs to its nearest anchor. '
        'Purple means low uncertainty and yellow means high uncertainty; white dots mark all '
        '48 anchors, the center × is the excluded source view, and the orange diamond marks '
        'the predicted most-uncertain anchor.</div>',
        unsafe_allow_html=True,
    )

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
            "Prediction variant", variant_names, format_func=pretty_variant
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
            "Only source anchor 0 was retained for the official PUN baseline; the locally "
            "trained probes support all 48 input anchors."
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

    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="eyebrow">CV3D / closed-loop evaluation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-title">Follow the policy<br>around the object.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-copy">Phase 2 compares single-image policies and geometric '
            'baselines. Phase 3 tests whether independent or joint observation histories '
            'make better next-view decisions over the same ten-view test protocol.</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        policy_count = comparison["policy"].nunique() if not comparison.empty else 0
        st.markdown(
            f'<div class="run-pill"><span class="run-dot"></span>{policy_count} policies '
            'loaded · 2 phases</div>',
            unsafe_allow_html=True,
        )

    if comparison.empty or tables["coverage"].empty:
        st.error(
            "Closed-loop result tables were not found under the configured Phase 2 and "
            "Phase 3 output directories."
        )
        return

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
    summary_metric = st.selectbox(
        "Policy summary metric",
        list(summary_metrics),
        format_func=lambda value: summary_metrics[value][0],
    )

    metric_label, direction = summary_metrics[summary_metric]
    leading_row = comparison.loc[
        comparison[summary_metric].idxmin()
        if direction.startswith("↓")
        else comparison[summary_metric].idxmax()
    ]
    best_a, best_b, best_c = st.columns(3)
    best_a.metric("Leading policy", leading_row["policy_label"])
    best_b.metric(metric_label, f'{float(leading_row[summary_metric]):.4f}')
    best_c.metric("Evaluation cohort", f'{int(leading_row["object_count"]):,} objects')
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
    with policy_control:
        selected_policies = st.multiselect(
            "Policies to compare",
            comparable_policies,
            default=comparable_policies,
        )
    with reconstruction_metric_control:
        reconstruction_metric = st.selectbox(
            "Reconstruction evaluation metric",
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

        st.markdown("#### Absolute surface coverage")
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

        vggt_column, gs_column = st.columns(2)
        with vggt_column:
            st.markdown("#### VGGT reconstruction")
            if vggt_for_chart.empty or not reconstruction_metric:
                st.info("No completed VGGT reconstruction evaluations were found.")
            else:
                st.altair_chart(
                    build_closed_loop_curve(
                        vggt_for_chart,
                        x="acquired_view_count",
                        y=vggt_metric,
                        x_title="Acquired views",
                        y_title=RECONSTRUCTION_METRICS[reconstruction_metric],
                        y_domain=reconstruction_y_domain,
                    ),
                    width="stretch",
                )
        with gs_column:
            st.markdown("#### 2D Gaussian Splatting reconstruction")
            if gs_for_chart.empty or not reconstruction_metric:
                st.info("No completed 2DGS evaluations were found.")
            else:
                st.altair_chart(
                    build_closed_loop_curve(
                        gs_for_chart,
                        x="acquired_view_count",
                        y=reconstruction_metric,
                        x_title="Acquired views",
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
        "the same scale. Coverage and VGGT reconstruction use the full 300-object test "
        "cohort; 2DGS means use only the completed runs currently available."
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
    quality_metric = st.selectbox(
        "Per-decision metric",
        list(quality_metrics),
        format_func=lambda value: quality_metrics[value],
    )
    quality = tables["per_step"].dropna(subset=[quality_metric]).copy()
    quality["decision_number"] = quality["step_index"].astype(int) + 1
    quality = (
        quality.groupby(
            ["phase", "policy", "policy_label", "decision_number"], as_index=False
        )[quality_metric]
        .mean()
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
        str(PHASE2_CLOSED_LOOP_ROOT), str(PHASE3_CLOSED_LOOP_ROOT)
    )

    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="eyebrow">CV3D / rollout inspection</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-title">Inspect a policy<br>view by view.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-copy">Choose a policy and object to follow its camera '
            'trajectory, absolute surface coverage, and VGGT or Gaussian reconstruction '
            'through the acquisition sequence.</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f'<div class="run-pill"><span class="run-dot"></span>{len(catalog):,} rollouts '
            'loaded</div>',
            unsafe_allow_html=True,
        )

    if catalog.empty:
        st.info("No replayable rollout files are available for object-level inspection.")
        return

    render_catalog = load_3dgs_render_catalog(str(GAUSSIAN_SPLATTING_ROOT))
    vggt_render_catalog = load_vggt_render_catalog(
        str(PHASE2_CLOSED_LOOP_ROOT), str(PHASE3_CLOSED_LOOP_ROOT)
    )
    rollout_a, rollout_b, rollout_c = st.columns([1.8, 1.25, 2.1])
    with rollout_a:
        rollout_policy = st.selectbox(
            "Rollout policy",
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
            "Rollout category",
            category_options,
            format_func=lambda value: CATEGORY_NAMES.get(value, value),
        )
    category_catalog = policy_catalog[policy_catalog["category_id"] == rollout_category]
    rendered_objects: set[str] = set()
    for available_renders in (render_catalog, vggt_render_catalog):
        if not available_renders.empty:
            rendered_objects.update(
                available_renders.loc[
                    (available_renders["policy"] == rollout_policy)
                    & (available_renders["category_id"] == rollout_category),
                    "object_id",
                ]
            )
    object_options = sorted(
        category_catalog["object_id"].unique(),
        key=lambda value: (value not in rendered_objects, value),
    )
    with rollout_c:
        rollout_object = st.selectbox("Rollout object", object_options)
    object_catalog = category_catalog[category_catalog["object_id"] == rollout_object]
    rollout_path = object_catalog["path"].iloc[0]
    rollout = load_dashboard_rollout(str(rollout_path))
    acquired = np.asarray(rollout["acquired_anchor_ids"], dtype=int)
    counts = np.asarray(rollout["acquired_view_counts"], dtype=int)
    coverage = np.asarray(rollout["coverage"], dtype=float)
    steps = rollout["steps"]

    acquired_count = st.segmented_control(
        "Views acquired",
        options=counts.astype(int).tolist(),
        default=int(counts[-1]),
        required=True,
        width="stretch",
    )
    current_index = int(np.flatnonzero(counts == acquired_count)[0])
    current_anchor = int(acquired[current_index])
    current_step = steps[current_index - 1] if current_index > 0 else None
    object_key = f"{rollout_category}/{rollout_object}"

    rollout_kpis = st.columns(3)
    rollout_kpis[0].metric("Current anchor", current_anchor)
    rollout_kpis[1].metric("Absolute coverage", f"{coverage[current_index]:.3f}")
    rollout_kpis[2].metric(
        "Decision regret",
        f'{float(current_step["normalized_regret"]):.3f}'
        if current_step and current_step.get("normalized_regret") is not None
        else "Initial view",
    )

    mesh_path = SHAPENET_ROOT / object_key / "models" / "model_normalized.ply"
    trajectory_column, observation_column = st.columns([2.1, 1], vertical_alignment="center")
    with trajectory_column:
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
        observation_path = (
            NUM_ROOT
            / object_key
            / "images"
            / f"viewpoint_{current_anchor}_offset_phi_0.png"
        )
        if observation_path.is_file():
            st.image(
                str(observation_path),
                caption=(
                    f"Acquired view {acquired_count} · anchor {current_anchor} · "
                    f"{pretty_policy(rollout_policy)}"
                ),
                width="stretch",
            )
        else:
            st.info(
                "The rollout metrics are available, but observation previews require "
                "the local `data/NUM` dataset."
            )
        trajectory = pd.DataFrame(
            {
                "Acquired views": counts[: current_index + 1],
                "Absolute coverage": coverage[: current_index + 1],
            }
        ).set_index("Acquired views")
        st.line_chart(trajectory, height=210)

    st.markdown(
        '<div class="section-kicker">Reconstruction comparisons</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Inspect VGGT point clouds, 3DGS renders, and the '
        'reconstructed 2DGS surface against ground truth.</div>',
        unsafe_allow_html=True,
    )
    available_catalogs = [
        frame
        for frame in (render_catalog, vggt_render_catalog)
        if not frame.empty
    ]
    if not available_catalogs:
        st.info(
            "No completed reconstruction viewer is available yet."
        )
    else:
        all_renders = (
            pd.concat(available_catalogs, ignore_index=True)
            .drop_duplicates(
                ["policy", "category_id", "object_id", "acquired_view_count"]
            )
        )
        reconstruction_policies = sorted(
            all_renders["policy"].unique(), key=pretty_policy
        )
        default_reconstruction_policy = (
            reconstruction_policies.index(rollout_policy)
            if rollout_policy in reconstruction_policies
            else 0
        )
        (
            render_policy_control,
            render_category_control,
            render_object_control,
        ) = st.columns([1.25, 1, 2])
        with render_policy_control:
            reconstruction_policy = st.selectbox(
                "Reconstruction policy",
                reconstruction_policies,
                index=default_reconstruction_policy,
                format_func=pretty_policy,
                help=(
                    "Reconstruction artifacts are generated per policy and may not "
                    "exist for the rollout policy selected above."
                ),
            )
        policy_3dgs_renders = (
            render_catalog[render_catalog["policy"] == reconstruction_policy]
            if not render_catalog.empty
            else pd.DataFrame()
        )
        policy_vggt_renders = (
            vggt_render_catalog[
                vggt_render_catalog["policy"] == reconstruction_policy
            ]
            if not vggt_render_catalog.empty
            else pd.DataFrame()
        )
        policy_renders = all_renders[
            all_renders["policy"] == reconstruction_policy
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
        matching_3dgs = (
            policy_3dgs_renders.loc[
                (policy_3dgs_renders["category_id"] == render_category)
                & (policy_3dgs_renders["object_id"] == render_object)
                & (policy_3dgs_renders["acquired_view_count"] == render_view_count)
            ]
            if not policy_3dgs_renders.empty
            else pd.DataFrame()
        )
        matching_vggt = (
            policy_vggt_renders.loc[
                (policy_vggt_renders["category_id"] == render_category)
                & (policy_vggt_renders["object_id"] == render_object)
                & (policy_vggt_renders["acquired_view_count"] == render_view_count)
            ]
            if not policy_vggt_renders.empty
            else pd.DataFrame()
        )
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
                / comparison_3dgs_path.parent.parent.relative_to(
                    GAUSSIAN_SPLATTING_ROOT
                )
                / "2dgs"
                / "comparison_interactive.html"
            )
        render_3dgs_tab, surface_2dgs_tab, vggt_tab = st.tabs(
            [
                "3DGS render vs. RGB",
                "2DGS surface vs. ground truth",
                "VGGT oracle-ICP vs. ground truth",
            ]
        )
        with render_3dgs_tab:
            if comparison_3dgs_path is None:
                st.info("No 3DGS render is available for this selection yet.")
            else:
                components.html(
                    make_manual_3dgs_comparison(
                        comparison_3dgs_path.read_text(encoding="utf-8")
                    ),
                    height=680,
                    scrolling=False,
                )
                st.caption(
                    "Drag the 3DGS panel from edge to edge to inspect anchors 0–47; "
                    "the matching ground-truth NUM image stays beside it."
                )
        with surface_2dgs_tab:
            comparison_2dgs_image_path = (
                comparison_2dgs_path.with_name("comparison.png")
                if comparison_2dgs_path is not None
                else None
            )
            if (
                comparison_2dgs_image_path is not None
                and comparison_2dgs_image_path.is_file()
            ):
                st.image(
                    str(comparison_2dgs_image_path),
                    caption=(
                        "2DGS surface (orange) vs. ground truth (blue) · "
                        "XY, XZ, and YZ projections"
                    ),
                    width="stretch",
                )
                st.caption(
                    "Hover over the image and click the fullscreen icon to enlarge it."
                )
            if comparison_2dgs_path is not None and comparison_2dgs_path.is_file():
                components.html(
                    comparison_2dgs_path.read_text(encoding="utf-8"),
                    height=720,
                    scrolling=False,
                )
                st.caption(
                    "Drag to rotate the 2DGS surface comparison. Toggle ground truth "
                    "and prediction, or switch between overlay and side-by-side layouts."
                )
            else:
                st.info(
                    "No interactive 2DGS surface comparison is available for this "
                    "selection yet."
                )
        with vggt_tab:
            comparison_vggt_image_path = (
                comparison_vggt_path.with_name("comparison_oracle_icp.png")
                if comparison_vggt_path is not None
                else None
            )
            if (
                comparison_vggt_image_path is not None
                and comparison_vggt_image_path.is_file()
            ):
                st.image(
                    str(comparison_vggt_image_path),
                    caption=(
                        "Oracle-ICP-aligned VGGT prediction (orange) vs. ground truth "
                        "(blue) · XY, XZ, and YZ projections"
                    ),
                    width="stretch",
                )
                st.caption(
                    "Hover over the image and click the fullscreen icon to enlarge it."
                )
            if comparison_vggt_path is not None:
                components.html(
                    comparison_vggt_path.read_text(encoding="utf-8"),
                    height=720,
                    scrolling=False,
                )
                st.caption(
                    "Drag to rotate and use the controls to toggle the VGGT prediction "
                    "and ground truth or switch between overlay and side-by-side views. "
                    "This oracle ICP alignment uses ground truth and is a structural "
                    "diagnostic, not an official evaluation result."
                )
            else:
                available_vggt_policies = []
                if not vggt_render_catalog.empty:
                    available_vggt_policies = sorted(
                        vggt_render_catalog.loc[
                            (vggt_render_catalog["category_id"] == render_category)
                            & (vggt_render_catalog["object_id"] == render_object)
                            & (
                                vggt_render_catalog["acquired_view_count"]
                                == render_view_count
                            ),
                            "policy",
                        ].unique(),
                        key=pretty_policy,
                    )
                availability_hint = (
                    " Choose "
                    + ", ".join(
                        f"'{pretty_policy(policy)}'"
                        for policy in available_vggt_policies
                    )
                    + " under Reconstruction policy above."
                    if available_vggt_policies
                    else ""
                )
                st.info(
                    "No interactive oracle-ICP VGGT point-cloud comparison is available "
                    f"for this selection yet.{availability_hint}"
                )
        st.caption(
            f"{pretty_policy(reconstruction_policy)} · "
            f"{CATEGORY_NAMES.get(render_category, render_category)} / {render_object} · "
            f"{render_view_count} acquired views"
        )


navigation = st.navigation(
    [
        st.Page(render_dataset_page, title="Dataset", url_path="dataset", default=True),
        st.Page(
            render_representation_page,
            title="Representation sweep",
            url_path="representations",
        ),
        st.Page(
            render_closed_loop_page,
            title="Closed-loop evaluation",
            url_path="closed-loop",
        ),
        st.Page(
            render_rollout_inspection_page,
            title="Rollout inspection",
            url_path="rollout-inspection",
        ),
    ],
    position="top",
)
navigation.run()
