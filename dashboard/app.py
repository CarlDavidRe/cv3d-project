"""Interactive experiment dashboard for the CV3D next-best-view project."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = REPO_ROOT / "outputs" / "phase1" / "backbone_sweep"

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
    return aliases.get(value, value.replace("_", " ").title())


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
        data.groupby(["variant", "variant_label", "backbone_label"], as_index=False)
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


def format_score(value: float) -> str:
    return f"{value:.4f}"


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

results = load_phase1_results(str(PHASE1_ROOT))

header_left, header_right = st.columns([4, 1], vertical_alignment="center")
with header_left:
    st.markdown('<div class="eyebrow">CV3D / experiment intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">See what learns<br>the next best view.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">An interactive readout of representation quality across the '
        'Phase 1 backbone sweep. Compare ranking behavior, regression error, and stability across seeds.</div>',
        unsafe_allow_html=True,
    )
with header_right:
    seed_count = results["seed"].nunique() if not results.empty else 0
    st.markdown(
        f'<div class="run-pill"><span class="run-dot"></span>{seed_count} seeds loaded · live outputs</div>',
        unsafe_allow_html=True,
    )

if results.empty:
    st.error(f"No Phase 1 summaries found under `{PHASE1_ROOT}`.")
    st.stop()

st.markdown('<div class="section-kicker">01 / Representation sweep</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Which variant ranks views best?</div>', unsafe_allow_html=True)

control_a, control_b, control_c = st.columns([1.15, 1.7, 2.6])
with control_a:
    split = st.selectbox("Evaluation split", ["test", "validation"], format_func=str.title)
with control_b:
    metric = st.selectbox(
        "Metric",
        list(METRICS),
        format_func=lambda value: METRICS[value]["label"],
    )
with control_c:
    available_backbones = [
        label for label in BACKBONE_COLORS if label in set(results["backbone_label"])
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
    st.stop()

metric_meta = METRICS[metric]
variant_means = filtered.groupby(["variant", "variant_label"])[metric].mean()
best_index = variant_means.idxmin() if metric_meta["direction"] == "lower" else variant_means.idxmax()
best_variant, best_label = best_index
best_value = float(variant_means.loc[best_index])
best_seed_values = filtered.loc[filtered["variant"] == best_variant, metric]

kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
kpi_a.metric("Leading variant", best_label)
kpi_b.metric(f"Mean {metric_meta['short']}", format_score(best_value))
kpi_c.metric("Seed range", f"{best_seed_values.min():.4f}–{best_seed_values.max():.4f}")
kpi_d.metric("Variants in view", filtered["variant"].nunique())

direction_symbol = "↓ lower is better" if metric_meta["direction"] == "lower" else "↑ higher is better"
st.markdown(
    f'<div class="metric-note">{metric_meta["description"]} &nbsp;·&nbsp; '
    f'<strong>{direction_symbol}</strong> &nbsp;·&nbsp; Bars show seed means; dots show individual seeds. '
    'Click to focus, double-click to reset.</div>',
    unsafe_allow_html=True,
)

st.altair_chart(build_metric_chart(filtered, metric), width="stretch")

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
            "normalized_regret_mean": st.column_config.NumberColumn("Regret", format="%.4f"),
            "spearman_mean": st.column_config.NumberColumn("Spearman", format="%.4f"),
            "ndcg_at_5_mean": st.column_config.NumberColumn("NDCG@5", format="%.4f"),
            "huber_loss": st.column_config.NumberColumn("Huber", format="%.4f"),
            "ranking_loss": st.column_config.NumberColumn("Ranking loss", format="%.4f"),
            "loss": st.column_config.NumberColumn("Combined loss", format="%.4f"),
            "trainable_parameters": st.column_config.NumberColumn("Trainable params", format="localized"),
            "best_epoch": "Best epoch",
        },
        hide_index=True,
        width="stretch",
    )
