"""Interactive local browser for experiment and reconstruction results."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import (  # noqa: E402
    ExperimentRun,
    ReconstructionVisualization,
    discover_all_visualizations,
    discover_runs,
    discover_visualizations,
    filter_rows,
    load_csv,
    load_json,
    matching_visualization,
    summary_policies,
)


COLORS = ["#2dd4bf", "#60a5fa", "#f59e0b", "#f472b6", "#a78bfa", "#fb7185"]
METRIC_LABELS = {
    "coverage_mean": "Absolute coverage",
    "reachable_normalized_coverage_mean": "Reachable-normalized coverage",
    "chamfer_l1_normalized_mean": "Chamfer-L1 (normalized)",
    "accuracy_normalized_mean": "Accuracy (normalized)",
    "completeness_normalized_mean": "Completeness (normalized)",
    "fscore_1pct_mean": "F-score @ 1%",
    "fscore_2pct_mean": "F-score @ 2%",
    "fscore_10pct_mean": "F-score @ 10%",
}


@st.cache_data(show_spinner=False)
def cached_runs(outputs_root: str, revision: int) -> list[ExperimentRun]:
    del revision
    return discover_runs(Path(outputs_root))


@st.cache_data(show_spinner=False)
def cached_json(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return load_json(Path(path))


@st.cache_data(show_spinner=False)
def cached_csv(path: str, modified_ns: int) -> list[dict[str, str]]:
    del modified_ns
    return load_csv(Path(path))


def read_json(path: Path) -> dict[str, Any]:
    return cached_json(str(path), path.stat().st_mtime_ns)


def read_csv(path: Path) -> list[dict[str, str]]:
    return cached_csv(str(path), path.stat().st_mtime_ns)


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_value(value: Any, *, percent: bool = False) -> str:
    number = as_float(value)
    if number is None:
        return "—"
    if percent:
        return f"{number:.1%}"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.4f}"


def render_header(run: ExperimentRun, summary: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{run.phase.upper()} · {run.seed.replace('_', ' ').upper()}</div>
          <h1>{run.experiment.split('/')[-1].replace('_', ' ')}</h1>
          <p>Local experiment results and reconstruction diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    state = "Complete" if (
        summary.get("complete_fixed_split")
        or summary.get("complete_requested_cohort")
        or summary.get("cohort", {}).get("complete_requested_cohort")
    ) else "Available"
    st.caption(f"{state} · `{run.root.relative_to(ROOT)}`")


def render_overview(run: ExperimentRun, summary: dict[str, Any]) -> None:
    if run.phase == "phase1":
        render_phase1_overview(run)
        return
    policies = summary_policies(summary)
    object_count = summary.get("evaluated_object_count")
    if object_count is None and policies:
        object_count = policies[0].get("object_count")
    if object_count is None:
        object_count = summary.get("cohort", {}).get("evaluated_object_count")

    cards = st.columns(4)
    cards[0].metric("Objects", format_value(object_count))
    cards[1].metric("Policies", str(len(policies)) if policies else "—")
    cards[2].metric(
        "Best final coverage",
        format_value(max((as_float(row.get("final_coverage_mean")) or 0 for row in policies), default=None), percent=True),
    )
    chamfers = [
        number for row in policies
        if (number := as_float(row.get("final_chamfer_l1_normalized_mean"))) is not None
    ]
    cards[3].metric("Best final Chamfer ↓", format_value(min(chamfers) if chamfers else None))

    st.subheader("Policy summary")
    if not policies:
        st.info("This run has no closed-loop policy summary.")
        st.json(summary, expanded=False)
        return
    table = []
    for row in policies:
        table.append({
            "Policy": row.get("policy", "unknown"),
            "Objects": row.get("object_count"),
            "Final coverage": as_float(row.get("final_coverage_mean")),
            "Coverage AUC": as_float(row.get("coverage_auc_mean")),
            "Final Chamfer ↓": as_float(row.get("final_chamfer_l1_normalized_mean")),
            "F-score @ 1%": as_float(row.get("final_fscore_1pct_mean")),
            "Latency (ms)": as_float(row.get("median_policy_ms")),
        })
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Final coverage": st.column_config.NumberColumn(format="%.3f"),
            "Coverage AUC": st.column_config.NumberColumn(format="%.3f"),
            "Final Chamfer ↓": st.column_config.NumberColumn(format="%.4f"),
            "F-score @ 1%": st.column_config.NumberColumn(format="%.4f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    conclusion = summary.get("conclusion")
    if isinstance(conclusion, dict) and conclusion.get("statement"):
        st.subheader("Recorded conclusion")
        st.info(str(conclusion["statement"]))


def render_phase1_overview(run: ExperimentRun) -> None:
    comparison_path = run.metrics_dir / "comparison.csv"
    if comparison_path.is_file():
        rows = read_csv(comparison_path)
        hubers = [number for row in rows if (number := as_float(row.get("huber_loss"))) is not None]
        regrets = [number for row in rows if (number := as_float(row.get("normalized_regret_mean"))) is not None]
        ndcgs = [number for row in rows if (number := as_float(row.get("ndcg_at_5_mean"))) is not None]
        cards = st.columns(4)
        cards[0].metric("Variants", str(len(rows)))
        cards[1].metric("Best Huber ↓", format_value(min(hubers) if hubers else None))
        cards[2].metric("Best regret ↓", format_value(min(regrets) if regrets else None))
        cards[3].metric("Best NDCG@5 ↑", format_value(max(ndcgs) if ndcgs else None))
        st.subheader("Backbone and feature comparison")
        table = [{
            "Variant": row.get("variant"),
            "Backbone": row.get("backbone"),
            "Feature": row.get("feature"),
            "Huber ↓": as_float(row.get("huber_loss")),
            "Regret ↓": as_float(row.get("normalized_regret_mean")),
            "Spearman ↑": as_float(row.get("spearman_mean")),
            "NDCG@5 ↑": as_float(row.get("ndcg_at_5_mean")),
        } for row in rows]
        st.dataframe(table, width="stretch", hide_index=True)
        return

    metric_files = sorted(run.metrics_dir.glob("*.json"))
    cards = st.columns(2)
    cards[0].metric("Metric artifacts", str(len(metric_files)))
    cards[1].metric("Generated figures", str(sum(1 for _ in (run.root / "figures").rglob("*.svg"))) if (run.root / "figures").is_dir() else "0")
    if not metric_files:
        st.info("This Phase 1 run has no JSON metric artifacts yet.")
        return
    selected = st.selectbox("Metric artifact", metric_files, format_func=lambda path: path.name)
    st.json(read_json(selected), expanded=True)


def curve_figure(rows: list[dict[str, str]], metric: str) -> go.Figure:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        x = as_float(row.get("acquired_view_count"))
        y = as_float(row.get(metric))
        if x is None or y is None:
            continue
        grouped.setdefault(row.get("policy", "unknown"), []).append((x, y))

    figure = go.Figure()
    for index, (policy, points) in enumerate(sorted(grouped.items())):
        points.sort()
        figure.add_trace(go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines+markers",
            name=policy,
            line={"width": 3, "color": COLORS[index % len(COLORS)]},
            marker={"size": 7},
        ))
    figure.update_layout(
        template="plotly_dark",
        height=510,
        margin={"l": 16, "r": 16, "t": 32, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.45)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis_title="Acquired views",
        yaxis_title=METRIC_LABELS.get(metric, metric),
        hovermode="x unified",
    )
    return figure


def render_curves(run: ExperimentRun) -> None:
    curve_files = {
        "Coverage": run.metrics_dir / "coverage.csv",
        "Reconstruction": run.metrics_dir / "reconstruction_curves.csv",
    }
    available = {name: path for name, path in curve_files.items() if path.is_file()}
    if not available:
        st.info("No coverage or reconstruction curve artifacts are available for this run.")
        return

    family = st.segmented_control("Curve family", list(available), default=next(iter(available)))
    family = family or next(iter(available))
    rows = read_csv(available[family])
    columns = list(rows[0]) if rows else []
    candidates = [column for column in METRIC_LABELS if column in columns]
    if not candidates:
        st.warning(f"No recognized metrics found in `{available[family].name}`.")
        return
    preferred = "coverage_mean" if family == "Coverage" else "chamfer_l1_normalized_mean"
    default_index = candidates.index(preferred) if preferred in candidates else 0
    metric = st.selectbox(
        "Metric",
        candidates,
        index=default_index,
        format_func=lambda key: METRIC_LABELS.get(key, key),
    )
    st.plotly_chart(curve_figure(rows, metric), width="stretch")
    with st.expander("View source data"):
        st.dataframe(rows, width="stretch", hide_index=True)


def render_downloads(visualization: ReconstructionVisualization) -> None:
    available = [
        ("Comparison PLY", visualization.artifact("comparison.ply")),
        ("Aligned prediction PLY", visualization.artifact("prediction_aligned.ply")),
        ("Ground truth PLY", visualization.artifact("ground_truth.ply")),
        ("Oracle ICP PLY", visualization.artifact("comparison_oracle_icp.ply")),
        ("Metadata JSON", visualization.artifact("metadata.json")),
    ]
    columns = st.columns(3)
    button_index = 0
    for label, path in available:
        if path is None:
            continue
        columns[button_index % len(columns)].download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream" if path.suffix == ".ply" else "application/json",
            width="stretch",
        )
        button_index += 1


def render_viewer(visualization: ReconstructionVisualization) -> None:
    standard = visualization.artifact("comparison_interactive.html")
    oracle = visualization.artifact("comparison_oracle_icp_interactive.html")
    choices = {"Camera alignment": standard, "Oracle ICP diagnostic": oracle}
    choices = {label: path for label, path in choices.items() if path is not None}
    if choices:
        selected = st.segmented_control("Alignment", list(choices), default=next(iter(choices)))
        selected = selected or next(iter(choices))
        if selected.startswith("Oracle"):
            st.warning("Oracle ICP uses ground truth for alignment and is diagnostic only.")
        st.iframe(choices[selected], height=720)
    else:
        preview = visualization.artifact("comparison.png")
        if preview:
            st.image(str(preview), width="stretch")
        else:
            st.info("This visualization has no browser viewer or preview image.")
    render_downloads(visualization)
    with st.expander("Alignment metadata"):
        st.json(visualization.metadata, expanded=False)


def render_reconstructions(run: ExperimentRun) -> None:
    metrics_path = run.metrics_dir / "reconstruction_per_object.csv"
    visualizations = discover_visualizations(run)
    if not visualizations:
        visualizations = discover_all_visualizations(ROOT / "outputs")
    if not metrics_path.is_file() and not visualizations:
        st.info("This run does not contain reconstruction metrics or generated viewers.")
        return
    if not metrics_path.is_file():
        selection = st.selectbox("Generated visualization", visualizations, format_func=lambda item: item.label)
        render_viewer(selection)
        return

    rows = read_csv(metrics_path)
    object_ids = sorted({row["object_id"] for row in rows if row.get("object_id")})
    first_visualized = visualizations[0].object_id if visualizations else None
    initial_object = object_ids.index(first_visualized) if first_visualized in object_ids else 0
    col_object, col_policy, col_views = st.columns([2.2, 1, 1])
    object_id = col_object.selectbox("Object", object_ids, index=initial_object)
    object_rows = filter_rows(rows, object_id=object_id)
    policies = sorted({row["policy"] for row in object_rows})
    visualized_policies = [item.policy for item in visualizations if item.object_id == object_id]
    initial_policy = policies.index(visualized_policies[0]) if visualized_policies and visualized_policies[0] in policies else 0
    policy = col_policy.selectbox("Policy", policies, index=initial_policy)
    policy_rows = filter_rows(object_rows, policy=policy)
    views = sorted({int(row["acquired_view_count"]) for row in policy_rows})
    visualized_views = [
        item.acquired_view_count for item in visualizations
        if item.object_id == object_id and item.policy == policy
    ]
    initial_views = views.index(visualized_views[0]) if visualized_views and visualized_views[0] in views else len(views) - 1
    view_count = col_views.selectbox("Views", views, index=initial_views)
    selected_rows = filter_rows(policy_rows, acquired_view_count=view_count)
    selected_row = selected_rows[0] if selected_rows else {}

    metrics = st.columns(4)
    metrics[0].metric("Chamfer-L1 ↓", format_value(selected_row.get("chamfer_l1_normalized")))
    metrics[1].metric("Accuracy ↓", format_value(selected_row.get("accuracy_normalized")))
    metrics[2].metric("Completeness ↓", format_value(selected_row.get("completeness_normalized")))
    metrics[3].metric("F-score @ 1% ↑", format_value(selected_row.get("fscore_1pct"), percent=True))

    visualization = matching_visualization(
        visualizations,
        object_id=object_id,
        policy=policy,
        acquired_view_count=view_count,
    )
    if visualization:
        render_viewer(visualization)
    else:
        st.info("Metrics are available, but an interactive viewer has not been generated for this selection.")
        command = (
            "python3 scripts/visualize_reconstruction.py "
            f"{metrics_path.relative_to(ROOT)} --object-id {object_id} "
            f"--policy {policy} --views {view_count} --oracle-icp"
        )
        st.code(command, language="bash")

    with st.expander("Per-object metric row"):
        st.json(selected_row, expanded=False)


def render_artifacts(run: ExperimentRun, summary: dict[str, Any]) -> None:
    st.subheader("Run artifacts")
    artifacts = []
    for path in sorted(run.root.rglob("*")):
        if not path.is_file():
            continue
        artifacts.append({
            "Artifact": str(path.relative_to(run.root)),
            "Size": f"{path.stat().st_size / 1024:.1f} KiB",
        })
    st.dataframe(artifacts, width="stretch", hide_index=True)
    with st.expander("Full summary JSON"):
        st.json(summary, expanded=False)


def render_phase1_figures(run: ExperimentRun) -> None:
    figure_root = run.root / "figures"
    figures = sorted(figure_root.rglob("*.svg")) if figure_root.is_dir() else []
    if not figures:
        st.info("No generated Phase 1 figures are available for this run.")
        return
    groups = sorted({path.relative_to(figure_root).parts[0] for path in figures})
    group = st.segmented_control("Figure group", groups, default=groups[0])
    group = group or groups[0]
    matching = [path for path in figures if path.relative_to(figure_root).parts[0] == group]
    selected = st.selectbox(
        "Figure",
        matching,
        format_func=lambda path: str(path.relative_to(figure_root)),
    )
    st.image(str(selected), width="stretch")


def main() -> None:
    st.set_page_config(page_title="NBV Experiment Explorer", page_icon="◉", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 75% -20%, #17314f 0, #0b1220 37%, #070b12 72%); }
    [data-testid="stSidebar"] { background: rgba(8, 15, 27, .96); border-right: 1px solid #1e293b; }
    .hero { padding: 1.2rem 0 .4rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: -.045em; margin: .15rem 0; text-transform: capitalize; }
    .hero p { color: #94a3b8; font-size: 1.05rem; margin: 0; }
    .eyebrow { color: #2dd4bf; font-size: .72rem; font-weight: 750; letter-spacing: .16em; }
    [data-testid="stMetric"] { background: rgba(15,23,42,.7); border: 1px solid #243247; border-radius: 12px; padding: 1rem; }
    [data-testid="stMetricValue"] { color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #243247; }
    .stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
    iframe { border: 1px solid #243247 !important; border-radius: 12px; background: #f8fafc; }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## ◉ NBV Explorer")
    outputs_root = ROOT / "outputs"
    if st.sidebar.button("Refresh outputs", width="stretch"):
        st.session_state["outputs_revision"] = st.session_state.get("outputs_revision", 0) + 1
        st.cache_data.clear()
    runs = cached_runs(str(outputs_root), st.session_state.get("outputs_revision", 0))
    if not runs:
        st.error(f"No experiment summaries found beneath `{outputs_root}`.")
        st.stop()

    run = st.sidebar.selectbox("Experiment run", runs, format_func=lambda item: item.label)
    st.sidebar.caption(f"{len(runs)} runs discovered locally")
    summary_path = run.metrics_dir / "summary.json"
    metadata_path = run.root / "metadata.json"
    summary = read_json(summary_path) if summary_path.is_file() else (
        read_json(metadata_path) if metadata_path.is_file() else {}
    )
    render_header(run, summary)

    third_tab = "Phase 1 figures" if run.phase == "phase1" else "Reconstructions"
    overview, curves, third, artifacts = st.tabs(["Overview", "Curves", third_tab, "Artifacts"])
    with overview:
        render_overview(run, summary)
    with curves:
        render_curves(run)
    with third:
        if run.phase == "phase1":
            render_phase1_figures(run)
        else:
            render_reconstructions(run)
    with artifacts:
        render_artifacts(run, summary)


if __name__ == "__main__":
    main()
