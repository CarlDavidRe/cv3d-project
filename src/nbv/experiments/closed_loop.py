"""Config-driven Phase 2 policy evaluation over a fixed object split."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

import numpy as np
import torch

from nbv.config import load_config
from nbv.data.num_dataset import load_object_split
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache, visibility_cache_path
from nbv.features import create_feature_extractor, load_feature_cache
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
    LightweightProbeHead,
    create_pun_transform,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
    resolve_pun_data_config,
)
from nbv.policies import (
    FarthestPolicy,
    OraclePolicy,
    PUNPolicy,
    RandomPolicy,
    VGGTPolicy,
)
from nbv.reproducibility import initialize_run, resolve_run_directory, seed_everything
from nbv.visualization import write_closed_loop_visualizations, write_phase2_rollout_demo


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
        or set(policies) - (policy_factories.keys() | {"pun", "vggt"})
    ):
        raise ValueError(
            "policies must be a non-empty unique list drawn from "
            "random/farthest/pun/vggt/oracle"
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
    vggt_components = (
        _load_vggt_components(config, root) if "vggt" in policies else None
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
                    else VGGTPolicy(**vggt_components)
                    if name == "vggt" and vggt_components is not None
                    else policy_factories[name]()
                )
                result = run_rollout(cache, store, policy, rollout_config)
                result.metadata.update({
                    "geometry_validation_status": settings["geometry_validation_status"],
                    "split": split, "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "visibility_cache_path": str(cache_path.resolve()),
                    "visibility_cache_sha256": sha256_file(cache_path),
                })
                if isinstance(policy, (PUNPolicy, VGGTPolicy)):
                    prediction_path = (
                        run.run_dir / "prediction_maps" / policy.name / f"{oid}.npz"
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
        policy_times = [step["policy_ms"] for result in results for step in result.steps]
        comparison["median_policy_ms"] = (
            float(np.median(policy_times)) if policy_times else None
        )
        comparison["profiling_modes"] = sorted({
            step.get("profiling_mode", "unspecified")
            for result in results for step in result.steps
        })
        for key in ("process_rss_bytes", "process_rss_delta_bytes", "peak_cuda_allocated_bytes"):
            values = [step.get(key) for result in results for step in result.steps if step.get(key) is not None]
            comparison[f"peak_{key}"] = max(values) if values else None
        provenance = results[0].metadata.get("policy_provenance", {}) if results else {}
        comparison["trainable_parameter_count"] = provenance.get("trainable_parameter_count", 0)
        comparison["frozen_parameter_count"] = provenance.get(
            "frozen_parameter_count", provenance.get("frozen_parameter_count_unavailable")
        )
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
        "timing_protocol": "Policy scoring includes cache lookup or preprocessing/inference plus aggregation; excludes geometry and evaluator RGB loading. CUDA is synchronized around scoring; medians include the cold first decision.",
        "memory_protocol": "Per-step process RSS plus RSS delta and CUDA peak allocated bytes; CUDA peak stats reset immediately before each score call.",
        "original_num_target_results": {
            "namespace": "phase1_num_target_metrics",
            "vggt_summary_path": config.get("phase2", {}).get("vggt", {}).get("summary", {}).get("path"),
            "note": "Original NUM-target metrics remain in the pinned Phase 1 summary and are not geometric gain metrics.",
        },
    }
    if all_results:
        figure_paths = write_closed_loop_visualizations(
            all_results, run.figure_dir / "closed_loop"
        )
        summary["figures"] = {
            name: str(path.relative_to(run.run_dir))
            for name, path in figure_paths.items()
        }
        demo_result = next(
            (result for result in all_results if result.metadata["policy"] == "vggt"),
            all_results[0],
        )
        demo_cache = load_visibility_cache(
            demo_result.metadata["visibility_cache_path"],
            expected_metadata=dict(
                expected_geometry, object_id=demo_result.metadata["object_id"]
            ),
        )
        demo_path = write_phase2_rollout_demo(
            demo_result,
            demo_cache,
            run.figure_dir / "closed_loop" / "rollout_demo.svg",
        )
        summary["figures"]["rollout_demo"] = str(demo_path.relative_to(run.run_dir))
    write_json(summary, run.metrics_dir / "summary.json")
    write_csv(comparisons, run.metrics_dir / "comparison.csv")
    write_csv(curves, run.metrics_dir / "coverage.csv")
    required_policies = {"random", "farthest", "pun", "vggt", "oracle"}
    profile_modes = {
        step.get("profiling_mode")
        for result in all_results for step in result.steps
    }
    pun_provenance = summary["policy_provenance"].get("pun", {})
    vggt_provenance = summary["policy_provenance"].get("vggt", {})
    checks = {
        "complete_fixed_test_split": bool(manifest["complete_fixed_split"] and split == "test"),
        "five_core_policies": set(policies) == required_policies,
        "all_rollouts_replay_verified": len(all_results) == len(manifest["evaluated_object_ids"]) * len(policies),
        "per_object_and_per_step_exports": bool(per_object) and bool([s for r in all_results for s in r.steps]),
        "profiling_records": all(
            "policy_ms" in step and "process_rss_bytes" in step
            for result in all_results for step in result.steps
        ),
        "live_and_cached_profile_modes": (
            "live_model_inference_incremental" in profile_modes
            and "cached_features_plus_live_head" in profile_modes
        ),
        "figures_generated": bool(all_results),
        "replayable_demo": bool(
            all_results
            and (run.figure_dir / "closed_loop" / "rollout_demo.svg").is_file()
        ),
        "pinned_learned_policy_references": bool(
            pun_provenance.get("checkpoint_sha256")
            and pun_provenance.get("source_commit")
            and vggt_provenance.get("checkpoint_sha256")
            and vggt_provenance.get("feature_cache_sha256")
        ),
    }
    completion = {
        "schema_version": 1,
        "phase": "phase2",
        "status": "complete" if all(checks.values()) else "pending",
        "checks": checks,
        "result_summary": str(run.metrics_dir / "summary.json"),
        "resolved_config": str(run.run_dir / "config.yaml"),
        "note": "Only a complete fixed test-split five-policy run may freeze Phase 2.",
    }
    write_json(completion, run.metrics_dir / "phase2_completion.json")
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
        "trainable_parameter_count": 0,
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


def _load_vggt_components(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Load the pinned Phase 1 VGGT head and compatible fixed-vector cache."""

    settings = config["phase2"].get("vggt")
    if not isinstance(settings, Mapping):
        raise ValueError("phase2.vggt settings are required when vggt is selected")
    if settings.get("variant") != "vggt_max_pooled_patch":
        raise ValueError("The validation-selected Phase 2 variant is vggt_max_pooled_patch")
    if settings.get("target_name") != "PSNR" or settings.get("target_direction") != "lower":
        raise ValueError("Phase 2 VGGT must retain the lower-is-better PSNR target")
    components = settings.get("feature_components")
    if components != ["max_pooled_patch"]:
        raise ValueError("phase2.vggt.feature_components must be [max_pooled_patch]")
    if settings.get("selection_split") != "val":
        raise ValueError("Phase 2 VGGT selection_split must be val")
    expected_selection = "best_vggt_validation_ranking_across_regret_spearman_ndcg_at_5"
    if settings.get("selection_metric") != expected_selection:
        raise ValueError(f"phase2.vggt.selection_metric must be {expected_selection}")
    if settings.get("map_frame") != "source_relative_canonical_anchor_zero":
        raise ValueError("Phase 2 VGGT must retain the Phase 1 source-relative map frame")
    if type(settings.get("allow_live_extraction")) is not bool:
        raise ValueError("phase2.vggt.allow_live_extraction must be boolean")

    def pinned_path(section: Mapping[str, Any], label: str) -> tuple[Path, str]:
        path = Path(section.get("path", ""))
        if not path.is_absolute():
            path = root / path
        digest = section.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"phase2.vggt.{label}.sha256 must be a SHA-256 digest")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"phase2.vggt.{label} is missing or its checksum differs")
        return path, digest

    checkpoint_settings = settings.get("checkpoint")
    summary_settings = settings.get("summary")
    cache_settings = settings.get("feature_cache")
    phase1_config_settings = settings.get("phase1_config")
    if not all(isinstance(item, Mapping) for item in (
        checkpoint_settings, summary_settings, cache_settings, phase1_config_settings
    )):
        raise ValueError(
            "phase2.vggt checkpoint, summary, phase1_config, and feature_cache are required"
        )
    checkpoint_path, checkpoint_sha = pinned_path(checkpoint_settings, "checkpoint")
    summary_path, summary_sha = pinned_path(summary_settings, "summary")
    cache_path, cache_sha = pinned_path(cache_settings, "feature_cache")
    phase1_config_path, phase1_config_sha = pinned_path(
        phase1_config_settings, "phase1_config"
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("variant") != settings["variant"]
        or summary.get("feature_components") != components
        or summary.get("target_name") != "PSNR"
        or summary.get("target_direction") != "lower"
    ):
        raise ValueError("Pinned Phase 1 summary does not match the VGGT policy")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        head_config = checkpoint["head"]
        variant_config = checkpoint["variant"]
        head = LightweightProbeHead(**head_config)
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Phase 1 VGGT checkpoint: {exc}") from exc
    if (
        variant_config.get("name") != settings["variant"]
        or variant_config.get("backbone") != "vggt"
        or variant_config.get("feature_components") != components
    ):
        raise ValueError("Pinned checkpoint variant does not match phase2.vggt")

    feature_cache = load_feature_cache(cache_path)
    metadata = dict(feature_cache.metadata)
    expected_cache_fields = {
        "backbone": "vggt",
        "variant": settings["variant"],
        "feature_components": components,
        "split": config["phase2"]["evaluation"]["split"],
        "target_name": "PSNR",
        "num_anchors": 48,
        "history_mode": "single_image",
    }
    for key, expected in expected_cache_fields.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Pinned VGGT feature cache has incompatible {key}")
    split_manifest = root / config["phase2"]["evaluation"]["split_manifest"]
    if metadata.get("split_manifest_sha256") != sha256_file(split_manifest):
        raise ValueError("Pinned VGGT feature cache uses a different split manifest")
    if feature_cache.features.shape[1] != head.input_dim:
        raise ValueError("VGGT feature-cache dimension does not match the Phase 1 head")
    cached_features = dict(zip(feature_cache.sample_ids, feature_cache.features, strict=True))

    model_settings = settings.get("model")
    if not isinstance(model_settings, Mapping):
        raise ValueError("phase2.vggt.model must be a mapping")
    allowed_model_fields = {"model_id", "image_size", "layer_index"}
    if set(model_settings) != allowed_model_fields:
        raise ValueError(f"phase2.vggt.model must contain {sorted(allowed_model_fields)}")
    extractor_factory = None
    if settings["allow_live_extraction"]:
        extractor_factory = lambda: create_feature_extractor(
            "vggt",
            device=settings["device"],
            model_cache_root=root / config["paths"]["model_cache_root"],
            **model_settings,
        )
    provenance = {
        "predictor": "phase1_validation_selected_frozen_vggt_num_head",
        "variant": settings["variant"],
        "selection_split": "val",
        "selection_metric": settings.get("selection_metric"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "phase1_summary_path": str(summary_path.resolve()),
        "phase1_summary_sha256": summary_sha,
        "phase1_config_path": str(phase1_config_path.resolve()),
        "phase1_config_sha256": phase1_config_sha,
        "feature_cache_path": str(cache_path.resolve()),
        "feature_cache_sha256": cache_sha,
        "feature_cache_metadata": metadata,
        "model": dict(model_settings),
        "allow_live_extraction": settings["allow_live_extraction"],
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "frozen_parameter_count": None,
        "frozen_parameter_count_unavailable": "VGGT is not loaded in cache-only evaluation; profile a live run to record its instantiated aggregator count",
        "true_surface_gain_supervision": False,
        "map_frame": settings["map_frame"],
    }
    return {
        "head": head,
        "feature_components": tuple(components),
        "cached_features": cached_features,
        "extractor_factory": extractor_factory,
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
    if results[0].metadata["policy"] not in {"pun", "vggt"}:
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
