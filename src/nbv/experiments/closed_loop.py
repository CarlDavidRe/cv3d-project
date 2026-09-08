"""Config-driven Random/Oracle evaluation over a fixed object split."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nbv.config import load_config
from nbv.data.num_dataset import load_object_split
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache, visibility_cache_path
from nbv.eval.closed_loop import RolloutConfig, replay_rollout, run_rollout
from nbv.eval.result_schema import save_rollout, write_csv, write_json
from nbv.geometry import CAMERA_CONVENTION, CANONICAL_ORDERING, FACE_VISIBILITY_RENDERER
from nbv.geometry.mesh import MESH_CENTERING, sha256_file
from nbv.logging_utils import configure_logging
from nbv.policies import OraclePolicy, RandomPolicy
from nbv.reproducibility import initialize_run, resolve_run_directory


def run_closed_loop_experiment(config: Mapping[str, Any], repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    settings = config["phase2"]["evaluation"]
    if config["experiment"]["phase"] != "phase2":
        raise ValueError("Closed-loop experiments must use phase2")
    rollout_config = RolloutConfig(**{
        key: settings[key] for key in (
            "max_acquired_views", "initial_anchor_ids", "invalid_anchor_ids", "coverage_target"
        )
    }, seed=config["experiment"]["seed"])
    policies = settings["policies"]
    if not isinstance(policies, list) or not policies or len(set(policies)) != len(policies) or set(policies) - {"random", "oracle"}:
        raise ValueError("policies must be a non-empty unique list of random/oracle")
    if type(settings["skip_missing_caches"]) is not bool:
        raise ValueError("skip_missing_caches must be boolean")
    split = settings["split"]
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    split_path = resolve(settings["split_manifest"])
    split_objects = sorted("/".join(key) for key in load_object_split(split_path)[split])
    explicit = settings["object_ids"]
    if not isinstance(explicit, list) or any(not isinstance(item, str) for item in explicit):
        raise ValueError("object_ids must be a list of category/object strings")
    if len(set(explicit)) != len(explicit) or set(explicit) - set(split_objects):
        raise ValueError("Explicit objects must be unique members of the selected split")
    requested = sorted(explicit) if explicit else split_objects
    limit = settings["limit"]
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer or null")
        requested = requested[:limit]
    if not requested:
        raise ValueError("No evaluation objects selected")

    geometry = config["phase2"].get("resolved_visibility")
    if geometry is None:
        geometry_config = load_config(resolve(config["phase2"]["visibility_config"]))
        geometry = geometry_config["phase2"]["visibility"]
    expected_geometry = {key: geometry[key] for key in (
        "mesh_scale", "mesh_centering", "camera_radius", "horizontal_fov_degrees", "render_resolution",
        "near", "far", "cull_backfaces", "mesh_relative_path",
    )}
    if expected_geometry["mesh_centering"] != MESH_CENTERING:
        raise ValueError(f"mesh_centering must be {MESH_CENTERING!r}")
    expected_geometry.update({
        "schema_version": 2, "anchor_count": 48, "anchor_ordering": CANONICAL_ORDERING,
        "renderer": FACE_VISIBILITY_RENDERER, "camera_convention": CAMERA_CONVENTION,
        "visibility_definition": "pun_unoccluded_rasterized_mesh_faces_v1",
        "available_visibility_targets": ["vis", "vis_a"],
    })
    # Both metrics use identical cached geometry. Deliberately do not compare
    # visibility_target (the producer's default) to the explicit evaluator target.
    resolved = dict(config)
    resolved["phase2"] = dict(config["phase2"], resolved_visibility=geometry)
    run_dir = resolve_run_directory(resolved, root)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory is not empty: {run_dir}; choose another experiment.name")
    run = initialize_run(resolved, root)
    logger = configure_logging(run.log_path)
    cache_root = resolve(config["paths"]["visibility_cache_root"])
    missing = [oid for oid in requested if not visibility_cache_path(cache_root, oid).is_file()]
    # Snapshot availability once, so caches completing mid-run do not change the cohort.
    available = [oid for oid in requested if oid not in set(missing)]
    manifest = {
        "split": split, "split_manifest": str(split_path), "split_manifest_sha256": sha256_file(split_path),
        "split_object_count": len(split_objects), "requested_object_ids": requested,
        "missing_object_ids": missing, "evaluated_object_ids": [], "failures": [],
        "expected_geometry": expected_geometry, "coverage_target": rollout_config.coverage_target,
        "geometry_validation_status": settings["geometry_validation_status"],
        "cache_provenance": {}, "complete_requested_cohort": False, "complete_fixed_split": False,
    }
    manifest_path = run.metrics_dir / "visibility_cache_manifest.json"
    write_json(manifest, manifest_path)
    if missing and not settings["skip_missing_caches"]:
        raise ValueError(f"{len(missing)} visibility caches are missing; see {manifest_path}. Use --skip-missing-caches for an explicitly partial run.")
    if not available:
        raise ValueError(f"No visibility caches available; see {manifest_path}")
    logger.info("Requested %d objects; available %d; missing %d", len(requested), len(available), len(missing))
    all_results = []
    for oid in available:
        try:
            cache_path = visibility_cache_path(cache_root, oid)
            cache = load_visibility_cache(cache_path, expected_metadata=dict(expected_geometry, object_id=oid))
            store = ObservationStore.from_num_object(resolve(config["paths"]["data_root"]), oid)
            object_results = []
            for name in policies:
                policy = RandomPolicy() if name == "random" else OraclePolicy()
                result = run_rollout(cache, store, policy, rollout_config)
                replay_rollout(result, cache, store)
                result.metadata.update({
                    "geometry_validation_status": settings["geometry_validation_status"],
                    "split": split, "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "visibility_cache_path": str(cache_path.resolve()),
                    "visibility_cache_sha256": sha256_file(cache_path),
                })
                object_results.append(result)
            for result in object_results:
                destination = run.run_dir / "rollouts" / result.metadata["policy"] / f"{oid}.npz"
                save_rollout(result, destination)
            all_results.extend(object_results)
            manifest["evaluated_object_ids"].append(oid)
            manifest["cache_provenance"][oid] = {
                "path": str(cache_path.resolve()), "sha256": sha256_file(cache_path), "metadata": cache.metadata,
            }
            logger.info("Evaluated and replayed %s (%d/%d)", oid, len(manifest["evaluated_object_ids"]), len(available))
        except (OSError, ValueError) as exc:
            manifest["failures"].append({"object_id": oid, "error": str(exc)})
            logger.error("%s: %s", oid, exc)
        write_json(manifest, manifest_path)
    manifest["complete_requested_cohort"] = not missing and not manifest["failures"]
    manifest["complete_fixed_split"] = manifest["complete_requested_cohort"] and requested == split_objects
    write_json(manifest, manifest_path)
    per_object = [result.summary() for result in all_results]
    write_csv(per_object, run.metrics_dir / "per_object.csv")
    write_csv([step for result in all_results for step in result.steps], run.metrics_dir / "per_step.csv")
    comparisons = []
    curves = []
    for policy in policies:
        results = [result for result in all_results if result.metadata["policy"] == policy]
        rows = [result.summary() for result in results]
        comparison = {"policy": policy, "object_count": len(rows)}
        for metric in ("final_coverage", "coverage_auc"):
            comparison[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows])) if rows else None
        for metric in ("normalized_regret", "spearman", "ndcg_at_5"):
            values = [step[metric] for result in results for step in result.steps if step[metric] is not None]
            comparison[f"{metric}_mean"] = float(np.mean(values)) if values else None
            comparison[f"{metric}_valid_count"] = len(values)
        comparisons.append(comparison)
        for count in sorted({int(c) for result in results for c in result.acquired_view_counts}):
            values = [float(result.coverage[np.flatnonzero(result.acquired_view_counts == count)[0]]) for result in results if count in result.acquired_view_counts]
            curves.append({"policy": policy, "acquired_view_count": count, "coverage_mean": float(np.mean(values)), "object_count": len(values)})
    summary = {
        "schema_version": 1, "phase": "phase2", "coverage_target": rollout_config.coverage_target,
        "training_target_semantics": "none", "policies": comparisons,
        "geometry_validation_status": settings["geometry_validation_status"],
        "requested_object_count": len(requested), "evaluated_object_count": len(manifest["evaluated_object_ids"]),
        "complete_requested_cohort": manifest["complete_requested_cohort"], "complete_fixed_split": manifest["complete_fixed_split"],
        "visibility_cache_manifest": str(manifest_path),
        "ranking_aggregation": "pooled_per_step_mean_excluding_null_with_valid_counts",
        "coverage_aggregation": "per_object_mean; AUC is unnormalized over recorded acquired-view counts",
        "timing_protocol": "CPU scoring only, excludes geometry and RGB; no live model or memory profiling in Step 10",
    }
    write_json(summary, run.metrics_dir / "summary.json")
    write_csv(comparisons, run.metrics_dir / "comparison.csv")
    write_csv(curves, run.metrics_dir / "coverage.csv")
    if manifest["failures"]:
        raise ValueError(f"Evaluation failed for {len(manifest['failures'])} objects; see {manifest_path}")
    return run.run_dir
