"""Step 18 paired one-step and closed-loop Phase 3 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nbv.config import load_config
from nbv.data import (
    FrozenFeatureLookup,
    HistoryDataset,
    HistoryFeatureDataset,
    relocated_history_manifest_identity,
)
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import (
    load_visibility_cache,
    visibility_cache_fingerprint,
    visibility_cache_path,
)
from nbv.eval.closed_loop import RolloutConfig, replay_rollout, run_rollout
from nbv.eval.result_schema import load_rollout, save_rollout, write_csv, write_json
from nbv.experiments.phase3_independent import load_independent_history_checkpoint
from nbv.experiments.phase3_joint import (
    _materialize_joint_features,
    load_joint_history_checkpoint,
)
from nbv.features import VGGTJointExtractor
from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT
from nbv.models import count_trainable_parameters
from nbv.policies import IndependentHistoryPolicy, JointHistoryPolicy
from nbv.reproducibility import initialize_run, resolve_run_directory, seed_everything
from nbv.training import HistoryEvaluationResult, evaluate_history_model
from nbv.visualization import write_closed_loop_visualizations


POLICIES = ("vggt_independent_history", "vggt_joint_history")


@dataclass(frozen=True, slots=True)
class Phase3ControlledSettings:
    history_manifest: Path
    coverage_target: str
    device: str
    independent_checkpoint: Path
    independent_checkpoint_sha256: str
    independent_test_features: Path
    joint_checkpoint: Path
    joint_checkpoint_sha256: str
    joint_feature_batch_size: int
    joint_feature_num_workers: int
    joint_feature_pin_memory: bool
    joint_feature_cache_every_batches: int
    evaluation_batch_size: int
    huber_delta: float
    ranking_weight: float
    ranking_margin: float
    ndcg_k: int
    object_ids: tuple[str, ...]
    limit: int | None
    skip_missing_caches: bool
    initial_anchor_ids: tuple[int, ...]
    max_acquired_views: int
    external_phase2_summary: Path | None
    data_root: Path
    visibility_cache_root: Path
    model_cache_root: Path


def parse_phase3_controlled_settings(
    config: Mapping[str, Any], repository_root: str | Path
) -> Phase3ControlledSettings:
    root = Path(repository_root).resolve()
    if config.get("experiment", {}).get("phase") != "phase3":
        raise ValueError("controlled history evaluation requires experiment.phase=phase3")
    paths = _mapping(config, "paths")
    controlled = _mapping(_mapping(config, "phase3"), "controlled")
    independent = _mapping(controlled, "independent")
    joint = _mapping(controlled, "joint")
    feature_cache = _mapping(joint, "test_feature_cache")
    one_step = _mapping(controlled, "one_step")
    closed_loop = _mapping(controlled, "closed_loop")
    external = controlled.get("external_references", {})
    if not isinstance(external, Mapping):
        raise TypeError("external_references must be a mapping")
    object_ids = closed_loop.get("object_ids")
    if not isinstance(object_ids, list) or any(
        not isinstance(value, str) for value in object_ids
    ):
        raise TypeError("closed_loop.object_ids must be a string list")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("closed_loop.object_ids must be unique")
    if closed_loop.get("split") != "test":
        raise ValueError("closed_loop.split must be test")
    limit = closed_loop.get("limit")
    if limit is not None:
        limit = _positive_integer(limit, "closed_loop.limit")
    initial = closed_loop.get("initial_anchor_ids")
    if not isinstance(initial, list) or not initial:
        raise ValueError("closed_loop.initial_anchor_ids must be a non-empty list")
    if any(type(value) is not int for value in initial):
        raise TypeError("closed_loop.initial_anchor_ids must contain integers")
    if len(initial) != len(set(initial)) or any(
        value < 0 or value >= CANONICAL_ANCHOR_COUNT for value in initial
    ):
        raise ValueError("closed_loop.initial_anchor_ids are invalid or duplicated")
    max_acquired_views = _positive_integer(
        closed_loop.get("max_acquired_views"), "closed_loop.max_acquired_views"
    )
    if max_acquired_views <= len(initial):
        raise ValueError("closed_loop.max_acquired_views must exceed the initial history")
    for key in ("pin_memory",):
        if type(feature_cache.get(key)) is not bool:
            raise TypeError(f"joint.test_feature_cache.{key} must be boolean")
    if type(closed_loop.get("skip_missing_caches")) is not bool:
        raise TypeError("closed_loop.skip_missing_caches must be boolean")
    phase2_path = external.get("phase2_summary")
    ndcg_k = _positive_integer(one_step.get("ndcg_k"), "one_step.ndcg_k")
    if ndcg_k != 5:
        raise ValueError("Step 18 requires one_step.ndcg_k=5")
    return Phase3ControlledSettings(
        history_manifest=_rooted(root, controlled.get("history_manifest"), "history_manifest"),
        coverage_target=_string(controlled.get("coverage_target"), "coverage_target"),
        device=_string(controlled.get("device"), "device"),
        independent_checkpoint=_rooted(
            root, independent.get("checkpoint"), "independent.checkpoint"
        ),
        independent_checkpoint_sha256=_digest(
            independent.get("checkpoint_sha256"), "independent.checkpoint_sha256"
        ),
        independent_test_features=_rooted(
            root, independent.get("test_feature_cache"), "independent.test_feature_cache"
        ),
        joint_checkpoint=_rooted(root, joint.get("checkpoint"), "joint.checkpoint"),
        joint_checkpoint_sha256=_digest(
            joint.get("checkpoint_sha256"), "joint.checkpoint_sha256"
        ),
        joint_feature_batch_size=_positive_integer(
            feature_cache.get("batch_size"), "joint.test_feature_cache.batch_size"
        ),
        joint_feature_num_workers=_nonnegative_integer(
            feature_cache.get("num_workers"), "joint.test_feature_cache.num_workers"
        ),
        joint_feature_pin_memory=bool(feature_cache["pin_memory"]),
        joint_feature_cache_every_batches=_positive_integer(
            feature_cache.get("cache_every_batches"),
            "joint.test_feature_cache.cache_every_batches",
        ),
        evaluation_batch_size=_positive_integer(
            one_step.get("batch_size"), "one_step.batch_size"
        ),
        huber_delta=_positive_float(one_step.get("huber_delta"), "one_step.huber_delta"),
        ranking_weight=_nonnegative_float(
            one_step.get("ranking_weight"), "one_step.ranking_weight"
        ),
        ranking_margin=_nonnegative_float(
            one_step.get("ranking_margin"), "one_step.ranking_margin"
        ),
        ndcg_k=ndcg_k,
        object_ids=tuple(object_ids),
        limit=limit,
        skip_missing_caches=bool(closed_loop["skip_missing_caches"]),
        initial_anchor_ids=tuple(initial),
        max_acquired_views=max_acquired_views,
        external_phase2_summary=(
            None
            if phase2_path is None
            else _rooted(root, phase2_path, "external_references.phase2_summary")
        ),
        data_root=_rooted(root, paths.get("data_root"), "paths.data_root"),
        visibility_cache_root=_rooted(
            root, paths.get("visibility_cache_root"), "paths.visibility_cache_root"
        ),
        model_cache_root=_rooted(
            root, paths.get("model_cache_root"), "paths.model_cache_root"
        ),
    )


def run_phase3_controlled(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    logger: logging.Logger | None = None,
    joint_extractor: object | None = None,
    resume: bool = False,
) -> Path:
    """Evaluate the validation-selected controls on paired test inputs."""

    root = Path(repository_root).resolve()
    settings = parse_phase3_controlled_settings(config, root)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    run_dir = resolve_run_directory(config, root)
    existing_outputs = (
        [path for path in run_dir.iterdir() if path.name != "run.log"]
        if run_dir.exists()
        else []
    )
    if existing_outputs:
        if not resume:
            raise ValueError(f"Run directory is not empty: {run_dir}; use --resume")
        saved_config = run_dir / "config.yaml"
        if not saved_config.is_file() or load_config(saved_config) != dict(config):
            raise ValueError("resumed Step 18 run configuration differs from config.yaml")
    context = initialize_run(config, root)
    active_logger = logger or logging.getLogger(__name__)
    manifest = json.loads(settings.history_manifest.read_text(encoding="utf-8"))
    _validate_manifest(manifest, settings)
    histories_path_only = HistoryDataset(
        settings.history_manifest, split="test", load_images=False
    )

    independent_payload = _load_checkpoint_payload(
        settings.independent_checkpoint,
        settings.independent_checkpoint_sha256,
        "phase3_independent_history_gain",
    )
    joint_payload = _load_checkpoint_payload(
        settings.joint_checkpoint,
        settings.joint_checkpoint_sha256,
        "phase3_joint_history_gain",
    )
    identity = _validate_checkpoint_history_identity(
        manifest,
        settings.history_manifest,
        root,
        (independent_payload, joint_payload),
    )
    controls = _validate_control_pair(independent_payload, joint_payload, settings)

    independent_model, _ = load_independent_history_checkpoint(
        settings.independent_checkpoint, device=settings.device
    )
    expected_feature_sha = independent_payload["feature"]["cache_sha256"]["test"]
    if _sha256(settings.independent_test_features) != expected_feature_sha:
        raise ValueError("independent test feature cache checksum differs from checkpoint")
    lookup = FrozenFeatureLookup.load(
        settings.independent_test_features,
        split="test",
        expected_metadata={
            "backbone": "vggt",
            "feature_components": independent_payload["feature"]["components"],
            "history_mode": "single_image",
        },
    )
    if lookup.metadata.get("split_manifest_sha256") != manifest["split_manifest_sha256"]:
        raise ValueError("independent test features and history object split differ")
    independent_data = HistoryFeatureDataset(histories_path_only, lookup)

    if joint_extractor is None:
        backbone = joint_payload["backbone"]
        fields = joint_payload["model"]
        active_logger.info("Step 18: loading frozen joint %s", backbone["model_id"])
        joint_extractor = VGGTJointExtractor(
            feature_components=backbone["feature_components"],
            model_id=backbone["model_id"],
            image_size=int(backbone["image_size"]),
            layer_index=int(backbone["layer_index"]),
            expected_feature_dim=int(fields["feature_dim"]),
            device=settings.device,
            model_cache_root=settings.model_cache_root,
        )
    joint_model, _ = load_joint_history_checkpoint(
        settings.joint_checkpoint,
        extractor=joint_extractor,
        device=settings.device,
    )
    histories_rgb = HistoryDataset(
        settings.history_manifest,
        split="test",
        data_root=settings.data_root,
        load_images=True,
    )

    active_logger.info("Step 18: materializing/restoring paired joint test features")
    joint_cache_dir = context.run_dir / "joint_feature_cache" / "test"
    cached_shards_before = len(tuple(joint_cache_dir.glob("*.pt")))
    precompute_started = time.perf_counter()
    joint_data = _materialize_joint_features(
        joint_model,
        histories_rgb,
        batch_size=settings.joint_feature_batch_size,
        num_workers=settings.joint_feature_num_workers,
        pin_memory=settings.joint_feature_pin_memory,
        logger=active_logger,
        progress_label="Step 18 joint test feature precompute",
        cache_dir=joint_cache_dir,
        cache_every_batches=settings.joint_feature_cache_every_batches,
        resume=resume,
        cache_identity={
            "model_type": "phase3_joint_history_gain",
            "checkpoint_sha256": settings.joint_checkpoint_sha256,
            "history_data_fingerprint": identity["history_data_fingerprint"],
            "split": "test",
        },
    )
    precompute_seconds_this_invocation = time.perf_counter() - precompute_started
    feature_equivalence = _length_one_feature_equivalence(
        independent_data, joint_data
    )
    if feature_equivalence["cosine_similarity_min"] < 0.999:
        raise ValueError(
            "length-1 joint and independent VGGT features are not equivalent; "
            "the controls may use different weights or preprocessing"
        )

    evaluation_kwargs = {
        "batch_size": settings.evaluation_batch_size,
        "huber_delta": controls["huber_delta"],
        "ranking_weight": controls["ranking_weight"],
        "ranking_margin": controls["ranking_margin"],
        "ndcg_k": settings.ndcg_k,
        "device": settings.device,
    }
    active_logger.info("Step 18: evaluating independent control on held-out histories")
    independent_eval, independent_seconds, independent_peak = _profile_evaluation(
        independent_model, independent_data, evaluation_kwargs, active_logger,
        "Step 18 independent held-out test",
    )
    active_logger.info("Step 18: evaluating joint control on identical held-out histories")
    joint_eval, joint_seconds, joint_peak = _profile_evaluation(
        joint_model, joint_data, evaluation_kwargs, active_logger,
        "Step 18 joint held-out test",
    )
    paired_rows, one_step_rows, length_rows = _paired_one_step_rows(
        independent_eval,
        joint_eval,
        histories_path_only,
        huber_delta=controls["huber_delta"],
        ndcg_k=settings.ndcg_k,
    )
    write_csv(paired_rows, context.metrics_dir / "one_step_paired_samples.csv")
    write_csv(one_step_rows, context.metrics_dir / "one_step_comparison.csv")
    write_csv(length_rows, context.metrics_dir / "one_step_by_history_length.csv")
    write_json(
        {"schema_version": 1, "models": one_step_rows, "by_history_length": length_rows},
        context.metrics_dir / "one_step_comparison.json",
    )

    active_logger.info("Step 18: running both controls through the unchanged evaluator")
    rollouts, cohort = _run_closed_loop_comparison(
        settings,
        manifest,
        independent_model,
        joint_model,
        lookup,
        independent_payload,
        joint_payload,
        context.run_dir,
        seed,
        resume=resume,
        logger=active_logger,
    )
    per_object = [result.summary() for result in rollouts]
    per_step = [step for result in rollouts for step in result.steps]
    closed_loop_rows, coverage_rows = _closed_loop_rows(rollouts)
    write_csv(per_object, context.metrics_dir / "per_object.csv")
    write_csv(per_step, context.metrics_dir / "per_step.csv")
    write_csv(closed_loop_rows, context.metrics_dir / "closed_loop_comparison.csv")
    write_csv(coverage_rows, context.metrics_dir / "coverage.csv")
    figure_paths = (
        write_closed_loop_visualizations(rollouts, context.figure_dir / "closed_loop")
        if rollouts
        else {}
    )

    external_references = _load_external_references(settings.external_phase2_summary)
    profiling = {
        "device": str(next(joint_model.parameters()).device),
        "one_step": {
            "independent_cached_seconds": independent_seconds,
            "joint_cached_seconds": joint_seconds,
            "sample_count": len(histories_path_only),
            "independent_peak_cuda_allocated_bytes": independent_peak,
            "joint_peak_cuda_allocated_bytes": joint_peak,
        },
        "joint_test_feature_precompute": {
            "seconds_this_invocation": precompute_seconds_this_invocation,
            "invocation_kind": (
                "live_complete_precompute"
                if cached_shards_before == 0
                else "resumed_cache_restore_or_live_completion"
            ),
            "cached_shards_before_invocation": cached_shards_before,
            "shard_count": len(tuple(joint_cache_dir.glob("*.pt"))),
            "storage_bytes": joint_data.storage_bytes,
            "note": (
                "Do not interpret a resumed cache-restore duration as live VGGT "
                "precompute time."
            ),
        },
        "closed_loop": {
            row["policy"]: {
                "median_policy_ms": row["median_policy_ms"],
                "peak_process_rss_bytes": row["peak_process_rss_bytes"],
                "peak_process_rss_delta_bytes": row["peak_process_rss_delta_bytes"],
                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                "profiling_modes": row["profiling_modes"],
            }
            for row in closed_loop_rows
        },
        "parameters": {
            # Policy adapters freeze their model objects for inference, so use
            # the validated checkpoint states for architectural capacity.
            "independent_trainable": controls["trainable_parameters_each"],
            "joint_trainable": controls["trainable_parameters_each"],
            "joint_frozen_backbone": sum(
                parameter.numel() for parameter in joint_extractor.model.parameters()
            ) if hasattr(joint_extractor, "model") else None,
            "independent_frozen_backbone_reference": sum(
                parameter.numel() for parameter in joint_extractor.model.parameters()
            ) if hasattr(joint_extractor, "model") else None,
        },
        "timing_protocol": (
            "One-step cached-head evaluation is timed separately from joint feature "
            "materialization. Closed-loop policy timing synchronizes CUDA and excludes "
            "evaluator geometry and RGB loading; independent uses cached per-image "
            "features while joint recomputes the complete live history."
        ),
    }
    write_json(profiling, context.metrics_dir / "profiling.json")
    write_json(external_references, context.metrics_dir / "external_references.json")

    conclusion = _conclusion(one_step_rows, closed_loop_rows, cohort)
    summary = {
        "schema_version": 1,
        "phase": "phase3",
        "step": 18,
        "coverage_target": settings.coverage_target,
        "history_identity": identity,
        "controlled_match": controls,
        "length_one_feature_equivalence": feature_equivalence,
        "one_step": one_step_rows,
        "one_step_by_history_length": length_rows,
        "closed_loop": closed_loop_rows,
        "cohort": cohort,
        "profiling": profiling,
        "external_references": external_references,
        "figures": {
            key: str(path.relative_to(context.run_dir)) for key, path in figure_paths.items()
        },
        "conclusion": conclusion,
        "negative_or_inconclusive_results_reportable": True,
    }
    write_json(summary, context.metrics_dir / "summary.json")
    _write_markdown_report(summary, context.metrics_dir / "report.md", settings.ndcg_k)
    completion = _completion(summary, len(histories_path_only), context.run_dir)
    write_json(completion, context.metrics_dir / "phase3_completion.json")
    if cohort["failures"]:
        raise ValueError(
            f"Step 18 failed for {len(cohort['failures'])} objects; see phase3_completion.json"
        )
    return context.run_dir


def _profile_evaluation(
    model: torch.nn.Module,
    dataset: HistoryFeatureDataset,
    kwargs: Mapping[str, Any],
    logger: logging.Logger,
    label: str,
) -> tuple[HistoryEvaluationResult, float, int | None]:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = evaluate_history_model(
        model, dataset, logger=logger, progress_label=label, **kwargs
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = None
    return result, time.perf_counter() - started, peak


def _paired_one_step_rows(
    independent: HistoryEvaluationResult,
    joint: HistoryEvaluationResult,
    histories: HistoryDataset,
    *,
    huber_delta: float,
    ndcg_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    independent_rows = list(independent.per_sample)
    joint_rows = list(joint.per_sample)
    if [row["sample_id"] for row in independent_rows] != [
        row["sample_id"] for row in joint_rows
    ]:
        raise ValueError("control evaluations do not contain identical ordered histories")
    targets = torch.from_numpy(histories._arrays["target_surface_gain"].copy()).float()
    masks = torch.from_numpy(histories._arrays["valid_candidate_mask"].copy()).bool()
    independent_huber = _masked_huber_per_sample(
        independent.predictions.float(), targets, masks, huber_delta
    )
    joint_huber = _masked_huber_per_sample(
        joint.predictions.float(), targets, masks, huber_delta
    )
    paired = []
    for index, (left, right) in enumerate(zip(independent_rows, joint_rows, strict=True)):
        if (
            left["history_anchor_ids"] != right["history_anchor_ids"]
            or left["object_id"] != right["object_id"]
        ):
            raise ValueError("paired control history metadata differs")
        row = {
            "sample_id": left["sample_id"],
            "object_id": left["object_id"],
            "history_length": left["history_length"],
            "history_anchor_ids": left["history_anchor_ids"],
            "independent_huber": float(independent_huber[index]),
            "joint_huber": float(joint_huber[index]),
        }
        for metric in ("normalized_regret", "spearman", f"ndcg_at_{ndcg_k}"):
            row[f"independent_{metric}"] = left[metric]
            row[f"joint_{metric}"] = right[metric]
        paired.append(row)
    models = []
    for name, result, huber in (
        (POLICIES[0], independent, independent_huber),
        (POLICIES[1], joint, joint_huber),
    ):
        models.append({
            "policy": name,
            "sample_count": len(paired),
            "huber_loss": float(huber.mean()),
            "normalized_regret_mean": result.summary["normalized_regret_mean"],
            "spearman_mean": result.summary["spearman_mean"],
            "spearman_valid_count": result.summary["spearman_valid_count"],
            f"ndcg_at_{ndcg_k}_mean": result.summary[f"ndcg_at_{ndcg_k}_mean"],
            f"ndcg_at_{ndcg_k}_valid_count": result.summary[f"ndcg_at_{ndcg_k}_valid_count"],
        })
    lengths = sorted({int(row["history_length"]) for row in paired})
    by_length = []
    for length in lengths:
        selected = [row for row in paired if int(row["history_length"]) == length]
        for prefix, name in (("independent", POLICIES[0]), ("joint", POLICIES[1])):
            by_length.append({
                "policy": name,
                "history_length": length,
                "sample_count": len(selected),
                "huber_loss": _mean(row[f"{prefix}_huber"] for row in selected),
                "normalized_regret_mean": _mean(
                    row[f"{prefix}_normalized_regret"] for row in selected
                ),
                "spearman_mean": _mean(
                    row[f"{prefix}_spearman"] for row in selected
                ),
                f"ndcg_at_{ndcg_k}_mean": _mean(
                    row[f"{prefix}_ndcg_at_{ndcg_k}"] for row in selected
                ),
            })
    return paired, models, by_length


def _run_closed_loop_comparison(
    settings: Phase3ControlledSettings,
    manifest: Mapping[str, Any],
    independent_model: torch.nn.Module,
    joint_model: torch.nn.Module,
    lookup: FrozenFeatureLookup,
    independent_payload: Mapping[str, Any],
    joint_payload: Mapping[str, Any],
    run_dir: Path,
    seed: int,
    *,
    resume: bool,
    logger: logging.Logger,
) -> tuple[list[Any], dict[str, Any]]:
    split_objects = tuple(manifest["splits"]["test"]["object_ids"])
    if settings.object_ids:
        if set(settings.object_ids) - set(split_objects):
            raise ValueError("closed-loop objects must belong to the test split")
        requested = tuple(sorted(settings.object_ids))
    else:
        requested = tuple(sorted(split_objects))
    if settings.limit is not None:
        requested = requested[: settings.limit]
    missing = tuple(
        object_id for object_id in requested
        if not visibility_cache_path(settings.visibility_cache_root, object_id).is_file()
    )
    if missing and not settings.skip_missing_caches:
        raise ValueError(f"{len(missing)} closed-loop visibility caches are missing")
    available = tuple(object_id for object_id in requested if object_id not in set(missing))
    rollout_config = RolloutConfig(
        max_acquired_views=settings.max_acquired_views,
        initial_anchor_ids=settings.initial_anchor_ids,
        invalid_anchor_ids=tuple(manifest["invalid_anchor_ids"]),
        coverage_target=settings.coverage_target,
        seed=seed,
    )
    checkpoint_info = {
        POLICIES[0]: {
            "checkpoint_sha256": settings.independent_checkpoint_sha256,
            "checkpoint_selection": independent_payload["checkpoint_selection"],
            "history_dataset_id": independent_payload["supervision"]["history_dataset_id"],
            "trainable_parameter_count": count_trainable_parameters(independent_model),
        },
        POLICIES[1]: {
            "checkpoint_sha256": settings.joint_checkpoint_sha256,
            "checkpoint_selection": joint_payload["checkpoint_selection"],
            "history_dataset_id": joint_payload["supervision"]["history_dataset_id"],
            "trainable_parameter_count": count_trainable_parameters(joint_model),
        },
    }
    results = []
    failures: list[dict[str, str]] = []
    for object_index, object_id in enumerate(available, start=1):
        try:
            cache_path = visibility_cache_path(settings.visibility_cache_root, object_id)
            cache = load_visibility_cache(cache_path)
            expected_cache_id = manifest["visibility_cache_ids"][object_id]
            if visibility_cache_fingerprint(cache) != expected_cache_id:
                raise ValueError("visibility cache differs from direct-gain labels")
            for policy_name in POLICIES:
                destination = run_dir / "rollouts" / policy_name / f"{object_id}.npz"
                if resume and destination.is_file():
                    result = load_rollout(destination)
                    replay_rollout(
                        result,
                        cache,
                        ObservationStore.from_num_object(settings.data_root, object_id),
                    )
                    results.append(result)
                    continue
                policy = (
                    IndependentHistoryPolicy(
                        independent_model,
                        independent_payload["feature"]["components"],
                        feature_lookup=lookup,
                        device=settings.device,
                        provenance=checkpoint_info[policy_name],
                    )
                    if policy_name == POLICIES[0]
                    else JointHistoryPolicy(
                        joint_model,
                        device=settings.device,
                        provenance=checkpoint_info[policy_name],
                    )
                )
                result = run_rollout(
                    cache,
                    ObservationStore.from_num_object(settings.data_root, object_id),
                    policy,
                    rollout_config,
                )
                result.metadata.update({
                    "experiment_phase": "phase3",
                    "step": 18,
                    "history_dataset_id": checkpoint_info[policy_name]["history_dataset_id"],
                    "visibility_cache_path": str(cache_path.resolve()),
                })
                replay_rollout(
                    result,
                    cache,
                    ObservationStore.from_num_object(settings.data_root, object_id),
                )
                save_rollout(result, destination)
                results.append(result)
            logger.info(
                "Step 18 closed loop: %s (%d/%d)", object_id, object_index, len(available)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append({"object_id": object_id, "error": str(exc)})
            logger.error("Step 18 %s: %s", object_id, exc)
    return results, {
        "split": "test",
        "split_object_count": len(split_objects),
        "requested_object_count": len(requested),
        "evaluated_object_count": len({r.metadata["object_id"] for r in results}),
        "missing_object_ids": list(missing),
        "failures": failures,
        "complete_requested_cohort": not missing and not failures,
        "complete_fixed_test_split": (
            not missing and not failures and set(requested) == set(split_objects)
        ),
        "initial_anchor_ids": list(settings.initial_anchor_ids),
        "max_acquired_views": settings.max_acquired_views,
        "same_evaluator": "nbv.eval.closed_loop.run_rollout",
    }


def _closed_loop_rows(
    rollouts: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison = []
    curves = []
    for policy in POLICIES:
        selected = [result for result in rollouts if result.metadata["policy"] == policy]
        summaries = [result.summary() for result in selected]
        steps = [step for result in selected for step in result.steps]
        row: dict[str, Any] = {"policy": policy, "object_count": len(selected)}
        for metric in (
            "final_coverage", "coverage_auc", "reachable_coverage_ceiling",
            "final_reachable_normalized_coverage", "reachable_normalized_coverage_auc",
        ):
            row[f"{metric}_mean"] = _mean(item[metric] for item in summaries)
        for metric in ("normalized_regret", "spearman", "ndcg_at_5"):
            values = [step[metric] for step in steps if step[metric] is not None]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_valid_count"] = len(values)
        row["median_policy_ms"] = (
            float(np.median([step["policy_ms"] for step in steps])) if steps else None
        )
        row["profiling_modes"] = sorted({step["profiling_mode"] for step in steps})
        for source, destination in (
            ("process_rss_bytes", "peak_process_rss_bytes"),
            ("process_rss_delta_bytes", "peak_process_rss_delta_bytes"),
            ("peak_cuda_allocated_bytes", "peak_cuda_allocated_bytes"),
        ):
            values = [step[source] for step in steps if step[source] is not None]
            row[destination] = max(values) if values else None
        comparison.append(row)
        counts = sorted({
            int(value)
            for result in selected
            for value in result.acquired_view_counts
        })
        for count in counts:
            values = [
                float(result.coverage[np.flatnonzero(result.acquired_view_counts == count)[0]])
                for result in selected if count in result.acquired_view_counts
            ]
            reachable = [
                float(result.reachable_normalized_coverage[
                    np.flatnonzero(result.acquired_view_counts == count)[0]
                ])
                for result in selected
                if count in result.acquired_view_counts
                and result.reachable_normalized_coverage is not None
            ]
            curves.append({
                "policy": policy,
                "acquired_view_count": count,
                "coverage_mean": _mean(values),
                "reachable_normalized_coverage_mean": _mean(reachable),
                "object_count": len(values),
            })
    return comparison, curves


def _validate_checkpoint_history_identity(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    repository_root: Path,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = {
        (
            payload["supervision"]["history_dataset_id"],
            payload["supervision"]["history_manifest_sha256"],
        )
        for payload in payloads
    }
    if len(identities) != 1:
        raise ValueError("independent and joint checkpoints use different history datasets")
    expected_id, expected_sha = identities.pop()
    current_sha = _sha256(manifest_path)
    if manifest.get("dataset_id") == expected_id and current_sha == expected_sha:
        mode = "exact"
        artifact_root = str(repository_root)
    else:
        artifact_manifest = Path(payloads[0]["supervision"]["history_manifest"])
        relative = manifest_path.resolve().relative_to(repository_root)
        artifact_root_path = artifact_manifest.parents[len(relative.parts) - 1]
        relocated_id, relocated_sha = relocated_history_manifest_identity(
            manifest,
            current_repository_root=repository_root,
            artifact_repository_root=artifact_root_path,
        )
        if (relocated_id, relocated_sha) != (expected_id, expected_sha):
            raise ValueError("checkpoint and local history manifest differ beyond relocation")
        mode = "repository_root_relocation_only"
        artifact_root = str(artifact_root_path)
    fingerprint_fields = {
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "coverage_target": manifest["coverage_target"],
        "visibility_definition": manifest["visibility_definition"],
        "sampling": manifest["sampling"],
        "rotation_metadata": manifest["rotation_metadata"],
        "invalid_anchor_ids": manifest["invalid_anchor_ids"],
        "split_sha256": {
            split: manifest["splits"][split]["sha256"]
            for split in ("train", "val", "test")
        },
        "visibility_cache_ids": manifest["visibility_cache_ids"],
    }
    return {
        "match": True,
        "mode": mode,
        "checkpoint_dataset_id": expected_id,
        "checkpoint_manifest_sha256": expected_sha,
        "local_dataset_id": manifest["dataset_id"],
        "local_manifest_sha256": current_sha,
        "artifact_repository_root": artifact_root,
        "history_data_fingerprint": _mapping_sha256(fingerprint_fields),
    }


def _validate_control_pair(
    independent: Mapping[str, Any],
    joint: Mapping[str, Any],
    settings: Phase3ControlledSettings,
) -> dict[str, Any]:
    left = independent["model"]
    right = joint["model"]
    independent_backbone = independent["feature"]["metadata"]["test"][
        "backbone_configuration"
    ]
    joint_backbone = joint["backbone"]
    model_fields = (
        "feature_dim", "hidden_dim", "num_anchors", "dropout",
        "include_anchor_directions", "aggregation", "anchor_ordering",
    )
    checks = {field: left.get(field) == right.get(field) for field in model_fields}
    checks.update({
        "coverage_target": (
            independent["supervision"]["coverage_target"]
            == joint["supervision"]["coverage_target"]
            == settings.coverage_target
        ),
        "target": independent["supervision"]["target"] == joint["supervision"]["target"],
        "trainable_capacity": (
            _state_trainable_count(independent["state_dict"])
            == _state_trainable_count(joint["state_dict"])
        ),
        "feature_components": (
            independent["feature"]["components"] == joint["backbone"]["feature_components"]
        ),
        "vggt_model_id": (
            independent_backbone["model_id"] == joint_backbone["model_id"]
        ),
        "vggt_image_size": (
            independent_backbone["image_size"] == joint_backbone["image_size"]
        ),
        "vggt_layer_index": (
            independent_backbone["layer_index"] == joint_backbone["layer_index"]
        ),
        "vggt_preprocessing": (
            independent_backbone["preprocessing"] == "square_resize_rgb_0_1"
        ),
        "joint_backbone_frozen": joint_backbone["frozen"] is True,
    })
    if not all(checks.values()):
        raise ValueError(
            "Step 18 control mismatch: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    return {
        "all_required_fields_match": True,
        "field_checks": checks,
        "independent_checkpoint_sha256": settings.independent_checkpoint_sha256,
        "joint_checkpoint_sha256": settings.joint_checkpoint_sha256,
        "trainable_parameters_each": _state_trainable_count(independent["state_dict"]),
        "huber_delta": settings.huber_delta,
        "ranking_weight": settings.ranking_weight,
        "ranking_margin": settings.ranking_margin,
        "training_configuration_source": "validation-selected Step 16/17 checkpoint artifacts",
    }


def _completion(summary: Mapping[str, Any], test_count: int, run_dir: Path) -> dict[str, Any]:
    one_step = summary["one_step"]
    closed_loop = summary["closed_loop"]
    cohort = summary["cohort"]
    checks = {
        "paired_held_out_histories": (
            len(one_step) == 2
            and all(row["sample_count"] == test_count for row in one_step)
        ),
        "one_step_metrics_reported": all(
            all(row.get(key) is not None for key in (
                "huber_loss", "normalized_regret_mean", "spearman_mean", "ndcg_at_5_mean"
            )) for row in one_step
        ),
        "unchanged_evaluator": cohort["same_evaluator"] == "nbv.eval.closed_loop.run_rollout",
        "complete_fixed_test_split": bool(cohort["complete_fixed_test_split"]),
        "two_control_policies": {row["policy"] for row in closed_loop} == set(POLICIES),
        "coverage_and_ranking_reported": all(
            row.get("final_coverage_mean") is not None
            and row.get("coverage_auc_mean") is not None
            and row.get("normalized_regret_mean") is not None
            for row in closed_loop
        ),
        "runtime_memory_capacity_reported": (
            (run_dir / "metrics/profiling.json").is_file()
            and summary["profiling"]["parameters"]["independent_trainable"]
            == summary["profiling"]["parameters"]["joint_trainable"]
        ),
        "tables_and_figures_generated": bool(summary["figures"]),
        "conclusion_recorded": bool(summary["conclusion"]),
        "negative_results_reportable": bool(
            summary["negative_or_inconclusive_results_reportable"]
        ),
        "length_one_backbone_equivalence": (
            summary["length_one_feature_equivalence"]["cosine_similarity_min"] >= 0.999
        ),
        "external_phase2_references_recorded": (
            summary["external_references"].get("status") == "loaded"
            and {
                row.get("policy")
                for row in summary["external_references"].get("policies", [])
            } == {"pun", "vggt"}
        ),
    }
    return {
        "schema_version": 1,
        "phase": "phase3",
        "step": 18,
        "status": "complete" if all(checks.values()) else "pending",
        "checks": checks,
        "note": "Only a complete paired fixed-test-split run may freeze Phase 3.",
    }


def _conclusion(
    one_step: Sequence[Mapping[str, Any]],
    closed_loop: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    one = {row["policy"]: row for row in one_step}
    loop = {row["policy"]: row for row in closed_loop}
    if not cohort["complete_fixed_test_split"]:
        return {
            "status": "inconclusive_partial_cohort",
            "statement": "The controlled run is partial; no final H4 conclusion is permitted.",
        }
    joint = POLICIES[1]
    independent = POLICIES[0]
    indicators = {
        "one_step_huber_improved": one[joint]["huber_loss"] < one[independent]["huber_loss"],
        "one_step_regret_improved": (
            one[joint]["normalized_regret_mean"]
            < one[independent]["normalized_regret_mean"]
        ),
        "one_step_spearman_improved": (
            one[joint]["spearman_mean"] > one[independent]["spearman_mean"]
        ),
        "one_step_ndcg_improved": (
            one[joint]["ndcg_at_5_mean"] > one[independent]["ndcg_at_5_mean"]
        ),
        "coverage_auc_improved": (
            loop[joint]["coverage_auc_mean"] > loop[independent]["coverage_auc_mean"]
        ),
        "final_coverage_improved": (
            loop[joint]["final_coverage_mean"]
            > loop[independent]["final_coverage_mean"]
        ),
    }
    wins = sum(indicators.values())
    status = "supports_h4" if wins >= 5 else "does_not_support_h4" if wins <= 1 else "mixed"
    return {
        "status": status,
        "indicators": indicators,
        "statement": (
            "Joint processing improves a clear majority of the prespecified outcomes."
            if status == "supports_h4"
            else "Joint processing does not improve a clear majority of the prespecified outcomes."
            if status == "does_not_support_h4"
            else "Joint processing produces mixed one-step and closed-loop outcomes."
        ),
        "caution": "This descriptive rule is not a statistical significance test.",
    }


def _write_markdown_report(summary: Mapping[str, Any], path: Path, ndcg_k: int) -> None:
    lines = [
        "# Phase 3 controlled comparison",
        "",
        f"Coverage target: `{summary['coverage_target']}`.",
        "",
        "## Held-out fixed-history results",
        "",
        f"| Policy | Samples | Huber | Regret | Spearman | NDCG@{ndcg_k} |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["one_step"]:
        lines.append(
            f"| {row['policy']} | {row['sample_count']} | {_format_metric(row['huber_loss'])} | "
            f"{_format_metric(row['normalized_regret_mean'])} | "
            f"{_format_metric(row['spearman_mean'])} | "
            f"{_format_metric(row[f'ndcg_at_{ndcg_k}_mean'])} |"
        )
    lines.extend([
        "",
        "## Closed-loop results",
        "",
        "| Policy | Objects | Coverage AUC | Final coverage | Regret | Spearman | "
        "NDCG@5 | Median policy ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["closed_loop"]:
        lines.append(
            f"| {row['policy']} | {row['object_count']} | "
            f"{_format_metric(row['coverage_auc_mean'])} | "
            f"{_format_metric(row['final_coverage_mean'])} | "
            f"{_format_metric(row['normalized_regret_mean'])} | "
            f"{_format_metric(row['spearman_mean'])} | "
            f"{_format_metric(row['ndcg_at_5_mean'])} | "
            f"{_format_metric(row['median_policy_ms'], digits=3)} |"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        summary["conclusion"]["statement"],
        "",
        "Official PUN and Phase 2 VGGT remain external references; they retain "
        "original NUM supervision.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_external_references(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_configured"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        row for row in payload.get("policies", []) if row.get("policy") in {"pun", "vggt"}
    ]
    return {
        "status": "loaded",
        "source": str(path),
        "source_sha256": _sha256(path),
        "note": (
            "External Phase 2 references retain original NUM supervision and "
            "are not matched Phase 3 controls."
        ),
        "policies": selected,
    }


def _length_one_feature_equivalence(
    independent: HistoryFeatureDataset,
    joint: HistoryFeatureDataset,
) -> dict[str, Any]:
    """Empirically guard the frozen-backbone control on one-view histories."""

    differences = []
    cosines = []
    for index in range(len(independent)):
        left = independent[index]
        if left.history_features.shape[0] != 1:
            continue
        right = joint[index]
        if (
            left.sample_id != right.sample_id
            or left.history_anchor_ids.tolist() != right.history_anchor_ids.tolist()
            or right.history_features.shape != left.history_features.shape
        ):
            raise ValueError("length-1 control features are not aligned")
        left_vector = left.history_features[0].float()
        right_vector = right.history_features[0].float()
        difference = left_vector - right_vector
        differences.append(difference)
        cosine = torch.nn.functional.cosine_similarity(
            left_vector.unsqueeze(0), right_vector.unsqueeze(0), dim=1, eps=1e-12
        )
        cosines.append(float(cosine.item()))
    if not differences:
        raise ValueError("Step 18 requires length-1 test histories")
    flattened = torch.cat(differences)
    return {
        "sample_count": len(differences),
        "mean_absolute_difference": float(flattened.abs().mean()),
        "root_mean_squared_difference": float(flattened.square().mean().sqrt()),
        "maximum_absolute_difference": float(flattened.abs().max()),
        "cosine_similarity_mean": float(np.mean(cosines)),
        "cosine_similarity_min": float(np.min(cosines)),
        "acceptance_threshold_min_cosine": 0.999,
        "interpretation": (
            "Length-1 equivalence checks the effective frozen VGGT weights and "
            "preprocessing without assuming an unavailable backbone-file digest."
        ),
    }


def _masked_huber_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    difference = (prediction - target).abs()
    values = torch.where(
        difference <= delta,
        0.5 * difference.square(),
        delta * (difference - 0.5 * delta),
    )
    return (values * mask).sum(dim=1) / mask.sum(dim=1)


def _load_checkpoint_payload(path: Path, digest: str, model_type: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    actual = _sha256(path)
    if actual != digest:
        raise ValueError(f"checkpoint checksum differs: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("model_type") != model_type:
        raise ValueError(f"unsupported checkpoint: {path}")
    return payload


def _validate_manifest(manifest: Mapping[str, Any], settings: Phase3ControlledSettings) -> None:
    if manifest.get("dataset_type") != "phase3_direct_surface_gain_histories":
        raise ValueError("Step 18 requires a Phase 3 direct-gain history dataset")
    if manifest.get("coverage_target") != settings.coverage_target:
        raise ValueError("history and Step 18 coverage targets differ")
    if not manifest.get("object_disjoint") or "test" not in manifest.get("splits", {}):
        raise ValueError("Step 18 requires an object-disjoint test split")
    if set(settings.initial_anchor_ids) & set(manifest.get("invalid_anchor_ids", [])):
        raise ValueError("closed-loop initial anchors include a dataset-invalid anchor")


def _state_trainable_count(state: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() for key, value in state.items() if key.startswith("head."))


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _rooted(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    value = _string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mean(values: Any) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _format_metric(value: Any, *, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


__all__ = [
    "Phase3ControlledSettings",
    "parse_phase3_controlled_settings",
    "run_phase3_controlled",
]
