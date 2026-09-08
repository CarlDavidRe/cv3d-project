"""Config-driven Phase 2 policy evaluation over a fixed object split."""

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
from nbv.models import (
    PUN_CHECKPOINT_SHA256,
    PUN_CHECKPOINT_URL,
    PUN_REFERENCE_COMMIT,
    PUN_RELEASE_NAME,
    PUN_REPOSITORY,
    PUNUPNet,
    create_pun_transform,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
    resolve_pun_data_config,
)
from nbv.policies import FarthestPolicy, OraclePolicy, PUNPolicy, RandomPolicy
from nbv.reproducibility import initialize_run, resolve_run_directory, seed_everything


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
    policy_factories = {
        "random": RandomPolicy,
        "farthest": FarthestPolicy,
        "oracle": OraclePolicy,
    }
    if (
        not isinstance(policies, list)
        or not policies
        or len(set(policies)) != len(policies)
        or set(policies) - (policy_factories.keys() | {"pun"})
    ):
        raise ValueError(
            "policies must be a non-empty unique list drawn from "
            "random/farthest/pun/oracle"
        )
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
    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"]["deterministic"]),
    )
    pun_components = (
        _load_pun_components(config, root) if "pun" in policies else None
    )
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
                policy = (
                    PUNPolicy(**pun_components)
                    if name == "pun" and pun_components is not None
                    else policy_factories[name]()
                )
                result = run_rollout(cache, store, policy, rollout_config)
                result.metadata.update({
                    "geometry_validation_status": settings["geometry_validation_status"],
                    "split": split, "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "visibility_cache_path": str(cache_path.resolve()),
                    "visibility_cache_sha256": sha256_file(cache_path),
                })
                if isinstance(policy, PUNPolicy):
                    prediction_path = (
                        run.run_dir / "prediction_maps" / "pun" / f"{oid}.npz"
                    )
                    policy.save_prediction_cache(prediction_path)
                    result.metadata["raw_prediction_cache_path"] = str(
                        prediction_path.resolve()
                    )
                    result.metadata["raw_prediction_cache_sha256"] = sha256_file(
                        prediction_path
                    )
                replay_rollout(result, cache, store)
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
        comparison = {
            "policy": policy,
            "object_count": len(rows),
            "policy_score_semantics": (
                results[0].metadata["policy_score_semantics"] if results else None
            ),
            "training_target_semantics": (
                results[0].metadata["training_target_semantics"] if results else None
            ),
        }
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
        "training_target_semantics": "policy_specific; see policy rows and rollouts",
        "policies": comparisons,
        "policy_provenance": {
            policy: _summarize_policy_provenance(
                [
                    result for result in all_results
                    if result.metadata["policy"] == policy
                ]
            )
            for policy in policies
        },
        "geometry_validation_status": settings["geometry_validation_status"],
        "requested_object_count": len(requested), "evaluated_object_count": len(manifest["evaluated_object_ids"]),
        "complete_requested_cohort": manifest["complete_requested_cohort"], "complete_fixed_split": manifest["complete_fixed_split"],
        "visibility_cache_manifest": str(manifest_path),
        "ranking_aggregation": "pooled_per_step_mean_excluding_null_with_valid_counts",
        "coverage_aggregation": "per_object_mean; AUC is unnormalized over recorded acquired-view counts",
        "timing_protocol": "Policy scoring includes PUN preprocessing, uncached image inference, and aggregation; excludes geometry and evaluator RGB loading. No warm-up, synchronization, or memory profiling yet.",
    }
    write_json(summary, run.metrics_dir / "summary.json")
    write_csv(comparisons, run.metrics_dir / "comparison.csv")
    write_csv(curves, run.metrics_dir / "coverage.csv")
    if manifest["failures"]:
        raise ValueError(f"Evaluation failed for {len(manifest['failures'])} objects; see {manifest_path}")
    return run.run_dir


def _load_pun_components(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Load the checksum-pinned inference-only UPNet once per experiment."""

    settings = config["phase2"].get("pun")
    if not isinstance(settings, Mapping):
        raise ValueError("phase2.pun settings are required when pun is selected")
    pinned = {
        "source_repository": PUN_REPOSITORY,
        "source_commit": PUN_REFERENCE_COMMIT,
        "release_name": PUN_RELEASE_NAME,
    }
    for key, expected in pinned.items():
        if settings.get(key) != expected:
            raise ValueError(f"phase2.pun.{key} must equal the pinned value {expected!r}")
    if settings.get("target_name") != "PSNR":
        raise ValueError("phase2.pun.target_name must be PSNR")
    if settings.get("model_name") != "vit_small_patch16_224":
        raise ValueError("phase2.pun.model_name must be vit_small_patch16_224")
    if settings.get("selection_method") != "all" or settings.get("delete_method") != "small":
        raise ValueError("Official PUN requires selection_method=all and delete_method=small")
    if settings.get("interpolation_degrees") != 30.0:
        raise ValueError("Official PUN requires interpolation_degrees=30.0")
    if settings.get("suppression_threshold") != 0.1:
        raise ValueError("Official PUN requires suppression_threshold=0.1")
    checkpoint_settings = settings.get("checkpoint")
    if not isinstance(checkpoint_settings, Mapping):
        raise ValueError("phase2.pun.checkpoint must be a mapping")
    if checkpoint_settings.get("sha256") != PUN_CHECKPOINT_SHA256:
        raise ValueError("phase2.pun.checkpoint.sha256 does not match the release")
    if checkpoint_settings.get("download_url") != PUN_CHECKPOINT_URL:
        raise ValueError("phase2.pun.checkpoint.download_url does not match the release")
    if type(checkpoint_settings.get("download_if_missing")) is not bool:
        raise ValueError("phase2.pun.checkpoint.download_if_missing must be boolean")
    expected_source_hashes = {
        "source_policy_file_sha256": "09a7ad1f0cc2e54c8032dc38169995c75707601c68cace6a013c2e9bb085445c",
        "source_viewpoint_file_sha256": "2f0abec6c5405f3d21cd70f0fa0b628f5560200341bff338573a65b2c9c696f7",
    }
    for key, expected in expected_source_hashes.items():
        if settings.get(key) != expected:
            raise ValueError(f"phase2.pun.{key} does not match the pinned source")

    checkpoint_path = Path(checkpoint_settings["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint = ensure_pun_checkpoint(
        checkpoint_path,
        expected_sha256=PUN_CHECKPOINT_SHA256,
        download_url=str(checkpoint_settings["download_url"]),
        download_if_missing=bool(checkpoint_settings["download_if_missing"]),
    )
    model = PUNUPNet(
        model_name=str(settings["model_name"]),
        num_anchors=48,
        pretrained=False,
        model_cache_root=root / config["paths"]["model_cache_root"],
    )
    load_pun_checkpoint(model, checkpoint, expected_sha256=PUN_CHECKPOINT_SHA256)
    data_config = resolve_pun_data_config(model)
    transform = create_pun_transform(data_config)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = {
        "baseline": "official_pretrained_pun_upnet",
        "source_repository": PUN_REPOSITORY,
        "source_commit": PUN_REFERENCE_COMMIT,
        "source_policy_file": "fep_nbv/baseline/our_policy_single.py",
        "source_policy_file_sha256": settings["source_policy_file_sha256"],
        "source_viewpoint_file": "fep_nbv/utils/generate_viewpoints.py",
        "source_viewpoint_file_sha256": settings["source_viewpoint_file_sha256"],
        "release_name": PUN_RELEASE_NAME,
        "model_name": settings["model_name"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": PUN_CHECKPOINT_SHA256,
        "preprocessing": data_config,
        "locally_trained": False,
        "frozen_parameter_count": total_parameters,
        "candidate_set_adaptation": "official fresh 512-point spherical samples replaced by evaluator canonical 48 anchors",
        "candidate_mask_adaptation": "shared acquired/invalid anchor mask applied by common evaluator",
        "tie_breaking_adaptation": "lowest valid canonical anchor ID",
        "empty_small_filter_fallback": "ignore small suppression when it removes every valid canonical candidate",
        "official_antipodal_rotation_behavior_retained": True,
        "training_overlap_limitation": "official release has no machine-readable training manifest; overlap with test objects/categories cannot be ruled out",
    }
    return {
        "model": model,
        "transform": transform,
        "device": settings["device"],
        "inference_batch_size": int(settings["inference_batch_size"]),
        "interpolation_degrees": float(settings["interpolation_degrees"]),
        "suppression_threshold": float(settings["suppression_threshold"]),
        "provenance": provenance,
    }


def _summarize_policy_provenance(results: list[Any]) -> dict[str, Any]:
    if not results:
        return {}
    provenance = dict(results[0].metadata.get("policy_provenance", {}))
    if results[0].metadata["policy"] != "pun":
        return provenance
    fallbacks = {
        result.metadata["object_id"]: result.metadata["policy_provenance"][
            "filter_fallback_steps"
        ]
        for result in results
        if result.metadata["policy_provenance"]["filter_fallback_steps"]
    }
    filtered_counts = [
        count
        for result in results
        for count in result.metadata["policy_provenance"][
            "filtered_candidate_counts"
        ]
    ]
    provenance.pop("filter_fallback_steps", None)
    provenance.pop("filtered_candidate_counts", None)
    provenance.update({
        "filter_fallback_count": sum(len(steps) for steps in fallbacks.values()),
        "filter_fallback_object_steps": fallbacks,
        "filtered_candidate_count_mean": (
            float(np.mean(filtered_counts)) if filtered_counts else None
        ),
        "filtered_candidate_count_valid_steps": len(filtered_counts),
    })
    return provenance
