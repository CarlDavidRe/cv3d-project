import json
from pathlib import Path

from dashboard.data import (
    discover_runs,
    discover_visualizations,
    filter_rows,
    matching_visualization,
    summary_policies,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discovers_nested_runs_from_summaries(tmp_path: Path) -> None:
    summary = tmp_path / "phase3" / "group" / "experiment" / "seed_2" / "metrics" / "summary.json"
    write_json(summary, {"closed_loop": []})
    write_json(summary.parent / "phase3_completion.json", {"status": "complete"})
    (summary.parent / "coverage.csv").write_text("policy,acquired_view_count,coverage_mean,object_count\n", encoding="utf-8")
    write_json(tmp_path / "phase2" / "unfinished" / "seed_0" / "metadata.json", {})

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].phase == "phase3"
    assert runs[0].experiment == "group/experiment"
    assert runs[0].seed == "seed_2"
    assert runs[0].label == "phase3 / group/experiment / seed_2"


def test_scope_includes_phase1_and_named_phase2_runs_only(tmp_path: Path) -> None:
    (tmp_path / "phase1" / "probe" / "seed_0" / "metrics").mkdir(parents=True)
    write_json(tmp_path / "phase2" / "phase2_vggt" / "seed_0" / "metrics" / "summary.json", {})
    write_json(tmp_path / "phase2" / "smoke" / "seed_0" / "metrics" / "summary.json", {})

    labels = [run.label for run in discover_runs(tmp_path)]

    assert labels == [
        "phase1 / probe / seed_0",
        "phase2 / phase2_vggt / seed_0",
    ]


def test_normalizes_phase2_and_phase3_policy_summaries() -> None:
    phase2 = [{"policy": "random"}]
    phase3 = [{"policy": "joint"}]

    assert summary_policies({"policies": phase2}) == phase2
    assert summary_policies({"closed_loop": phase3}) == phase3
    assert summary_policies({"other": []}) == []


def test_filters_rows_by_reconstruction_identity() -> None:
    rows = [
        {"object_id": "cat/a", "policy": "vggt", "acquired_view_count": "5"},
        {"object_id": "cat/a", "policy": "vggt", "acquired_view_count": "10"},
        {"object_id": "cat/b", "policy": "random", "acquired_view_count": "10"},
    ]

    assert filter_rows(rows, object_id="cat/a", policy="vggt", acquired_view_count=10) == [rows[1]]


def test_discovers_and_matches_generated_visualization(tmp_path: Path) -> None:
    run_summary = tmp_path / "phase2" / "phase2_vggt" / "seed_0" / "metrics" / "summary.json"
    write_json(run_summary, {"policies": []})
    run = discover_runs(tmp_path)[0]
    viewer = run.metrics_dir / "reconstruction_visualizations" / "sample"
    write_json(viewer / "metadata.json", {
        "object_id": "0269/object",
        "policy": "vggt",
        "acquired_view_count": 10,
    })
    (viewer / "comparison_interactive.html").write_text("<html></html>", encoding="utf-8")

    visualizations = discover_visualizations(run)
    match = matching_visualization(
        visualizations,
        object_id="0269/object",
        policy="vggt",
        acquired_view_count=10,
    )

    assert match is not None
    assert match.label == "0269/object · vggt · 10 views"
    assert match.artifact("comparison_interactive.html") is not None
