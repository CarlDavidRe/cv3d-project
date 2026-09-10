"""Reproducible Phase 3 independent-history direct-gain experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from nbv.config import validate_artifact_path
from nbv.data import FrozenFeatureLookup, HistoryDataset, HistoryFeatureDataset
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import load_visibility_cache, visibility_cache_path
from nbv.eval.closed_loop import RolloutConfig, replay_rollout, run_rollout
from nbv.eval.result_schema import save_rollout, write_csv, write_json
from nbv.features.selection import validate_feature_selection
from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, CANONICAL_ORDERING
from nbv.models import IndependentHistoryGainModel, count_trainable_parameters
from nbv.policies import IndependentHistoryPolicy
from nbv.reproducibility import RunContext, initialize_run, seed_everything
from nbv.training import HistoryFitResult, evaluate_history_model, fit_history_model
from nbv.visualization import (
    write_validation_loss_comparison,
    write_variant_training_curves,
)


@dataclass(frozen=True, slots=True)
class Phase3IndependentSettings:
    history_manifest: Path
    feature_cache_paths: Mapping[str, Path]
    feature_variant: str
    feature_components: tuple[str, ...]
    coverage_target: str
    device: str
    hidden_dim: int
    dropout: float
    include_anchor_directions: bool
    training: Mapping[str, Any]
    evaluation_batch_size: int
    ndcg_k: int
    smoke: Mapping[str, Any]
    data_root: Path
    visibility_cache_root: Path


def parse_phase3_independent_settings(
    config: Mapping[str, Any], repository_root: str | Path
) -> Phase3IndependentSettings:
    root = Path(repository_root).resolve()
    if config.get("experiment", {}).get("phase") != "phase3":
        raise ValueError("independent history training requires experiment.phase=phase3")
    paths = _mapping(config, "paths")
    phase3 = _mapping(config, "phase3")
    control = _mapping(phase3, "independent_control")
    features = _mapping(control, "features")
    model = _mapping(control, "model")
    training = _mapping(control, "training")
    evaluation = _mapping(control, "evaluation")
    smoke = _mapping(control, "closed_loop_smoke")
    cache_raw = _mapping(features, "cache_paths")
    if set(cache_raw) != {"train", "val", "test"}:
        raise ValueError("feature cache_paths must contain exactly train, val, and test")
    cache_paths = {
        split: _rooted(root, cache_raw[split], f"cache_paths.{split}")
        for split in ("train", "val", "test")
    }
    components = features.get("components")
    if not isinstance(components, list) or not components or any(
        not isinstance(value, str) for value in components
    ):
        raise ValueError("features.components must be a non-empty string list")
    components = list(validate_feature_selection("vggt", components))
    variant = _string(features.get("variant"), "features.variant")
    if model.get("aggregation") != "masked_mean":
        raise ValueError("Step 16 model.aggregation must be masked_mean")
    hidden_dim = _positive_integer(model.get("hidden_dim"), "model.hidden_dim")
    dropout = _finite(model.get("dropout"), "model.dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("model.dropout must be below 1")
    include_directions = model.get("include_anchor_directions")
    if not isinstance(include_directions, bool):
        raise TypeError("model.include_anchor_directions must be a boolean")
    required_training = {
        "epochs", "batch_size", "learning_rate", "weight_decay", "huber_delta",
        "ranking_weight", "ranking_margin", "patience",
    }
    if set(training) != required_training:
        raise ValueError(f"training must contain exactly {sorted(required_training)}")
    _positive_integer(training["epochs"], "training.epochs")
    _positive_integer(training["batch_size"], "training.batch_size")
    _positive_integer(training["patience"], "training.patience")
    _finite(training["learning_rate"], "training.learning_rate", minimum=0.0, strict=True)
    _finite(training["weight_decay"], "training.weight_decay", minimum=0.0)
    _finite(training["huber_delta"], "training.huber_delta", minimum=0.0, strict=True)
    _finite(training["ranking_weight"], "training.ranking_weight", minimum=0.0)
    _finite(training["ranking_margin"], "training.ranking_margin", minimum=0.0)
    if set(evaluation) != {"batch_size", "ndcg_k"}:
        raise ValueError("evaluation must contain exactly batch_size and ndcg_k")
    evaluation_batch_size = _positive_integer(evaluation["batch_size"], "evaluation.batch_size")
    ndcg_k = _positive_integer(evaluation["ndcg_k"], "evaluation.ndcg_k")
    _validate_smoke(smoke)
    coverage_target = _string(control.get("coverage_target"), "coverage_target")
    device = _string(control.get("device"), "device")
    return Phase3IndependentSettings(
        history_manifest=_rooted(root, control.get("history_manifest"), "history_manifest"),
        feature_cache_paths=cache_paths,
        feature_variant=variant,
        feature_components=tuple(components),
        coverage_target=coverage_target,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        include_anchor_directions=include_directions,
        training=dict(training),
        evaluation_batch_size=evaluation_batch_size,
        ndcg_k=ndcg_k,
        smoke=dict(smoke),
        data_root=_rooted(root, paths.get("data_root"), "paths.data_root"),
        visibility_cache_root=_rooted(
            root, paths.get("visibility_cache_root"), "paths.visibility_cache_root"
        ),
    )


def run_phase3_independent(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> Path:
    """Train, evaluate, checkpoint, and optionally smoke-test the control."""

    root = Path(repository_root).resolve()
    settings = parse_phase3_independent_settings(config, root)
    active_logger = logger or logging.getLogger(__name__)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))

    active_logger.info("Phase 3 independent: loading train/validation/test histories")
    histories = {
        split: HistoryDataset(settings.history_manifest, split=split, load_images=False)
        for split in ("train", "val", "test")
    }
    manifest = histories["train"].manifest
    _validate_manifest(manifest, settings)
    expected_cache = {
        "backbone": "vggt",
        "variant": settings.feature_variant,
        "feature_components": list(settings.feature_components),
        "history_mode": "single_image",
    }
    active_logger.info("Phase 3 independent: loading frozen feature caches")
    lookups = {
        split: FrozenFeatureLookup.load(
            settings.feature_cache_paths[split],
            split=split,
            expected_metadata=expected_cache,
        )
        for split in ("train", "val", "test")
    }
    for split, lookup in lookups.items():
        if lookup.metadata.get("split_manifest_sha256") != manifest["split_manifest_sha256"]:
            raise ValueError(f"{split} feature cache and history object split differ")
    dimensions = {lookup.feature_dim for lookup in lookups.values()}
    if len(dimensions) != 1:
        raise ValueError("feature dimensions differ across splits")
    datasets = {
        split: HistoryFeatureDataset(histories[split], lookups[split])
        for split in ("train", "val", "test")
    }
    feature_dim = dimensions.pop()
    model = IndependentHistoryGainModel(
        feature_dim,
        hidden_dim=settings.hidden_dim,
        dropout=settings.dropout,
        include_anchor_directions=settings.include_anchor_directions,
    )
    context = initialize_run(config, root)
    active_logger.info(
        "Phase 3 independent ready: %d train, %d validation, %d test histories; "
        "%d-D features; %d trainable parameters",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        feature_dim,
        count_trainable_parameters(model),
    )
    fit = fit_history_model(
        model,
        datasets["train"],
        datasets["val"],
        device=settings.device,
        seed=seed,
        ndcg_k=settings.ndcg_k,
        logger=active_logger,
        **settings.training,
    )
    evaluation_kwargs = {
        "batch_size": settings.evaluation_batch_size,
        "huber_delta": settings.training["huber_delta"],
        "ranking_weight": settings.training["ranking_weight"],
        "ranking_margin": settings.training["ranking_margin"],
        "ndcg_k": settings.ndcg_k,
        "device": settings.device,
    }
    validation = evaluate_history_model(
        model,
        datasets["val"],
        logger=active_logger,
        progress_label="Phase 3 independent final validation",
        **evaluation_kwargs,
    )
    test = evaluate_history_model(
        model,
        datasets["test"],
        logger=active_logger,
        progress_label="Phase 3 independent final test",
        **evaluation_kwargs,
    )
    checkpoint_path = context.checkpoint_dir / "best.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": "phase3_independent_history_gain",
        "model": {
            "feature_dim": feature_dim,
            "hidden_dim": settings.hidden_dim,
            "num_anchors": CANONICAL_ANCHOR_COUNT,
            "dropout": settings.dropout,
            "include_anchor_directions": settings.include_anchor_directions,
            "aggregation": model.aggregation,
            "backbone_history_mode": model.backbone_history_mode,
            "anchor_ordering": CANONICAL_ORDERING,
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature": {
            "backbone": "vggt",
            "variant": settings.feature_variant,
            "components": list(settings.feature_components),
            "cache_paths": {
                split: str(path.resolve()) for split, path in settings.feature_cache_paths.items()
            },
            "cache_sha256": {
                split: _sha256(path) for split, path in settings.feature_cache_paths.items()
            },
            "metadata": {split: dict(lookup.metadata) for split, lookup in lookups.items()},
        },
        "supervision": {
            "target": "target_surface_gain",
            "coverage_target": settings.coverage_target,
            "target_direction": "higher",
            "history_dataset_id": manifest["dataset_id"],
            "history_manifest": str(settings.history_manifest.resolve()),
            "history_manifest_sha256": _sha256(settings.history_manifest),
        },
        "checkpoint_selection": {
            "split": "val",
            "metric": "total_loss",
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    write_json(list(fit.history), context.metrics_dir / "training_history.json")
    write_csv(list(validation.per_sample), context.metrics_dir / "validation_per_sample.csv")
    write_csv(list(test.per_sample), context.metrics_dir / "test_per_sample.csv")
    summary: dict[str, Any] = {
        "model": checkpoint["model"],
        "feature_variant": settings.feature_variant,
        "feature_components": list(settings.feature_components),
        "trainable_parameters": count_trainable_parameters(model),
        "frozen_backbone_parameters": None,
        "frozen_backbone_parameters_note": (
            "VGGT is not instantiated during cache-backed training; all cached vectors "
            "were produced by independent one-image forwards"
        ),
        "history_dataset_id": manifest["dataset_id"],
        "coverage_target": settings.coverage_target,
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "history_length_counts": {
            split: histories[split].manifest["splits"][split]["history_length_counts"]
            for split in ("train", "val", "test")
        },
        "best_epoch": fit.best_epoch,
        "epochs_completed": fit.epochs_completed,
        "best_validation_loss": fit.best_validation_loss,
        "validation": dict(validation.summary),
        "test": dict(test.summary),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "closed_loop_smoke": None,
    }
    if settings.smoke["enabled"]:
        summary["closed_loop_smoke"] = _run_smoke(
            model, lookups, histories, settings, context.run_dir, checkpoint, seed
        )
    write_json(summary, context.metrics_dir / "summary.json")
    _write_phase1_style_reports(
        context,
        summary,
        fit,
        variant_name=str(config["experiment"]["name"]),
        epochs_requested=int(settings.training["epochs"]),
        ndcg_k=settings.ndcg_k,
    )
    active_logger.info(
        "Phase 3 independent validation: Huber %.6f, regret %.4f, NDCG@%d %.4f",
        validation.summary["huber_loss"],
        validation.summary["normalized_regret_mean"],
        settings.ndcg_k,
        validation.summary[f"ndcg_at_{settings.ndcg_k}_mean"],
    )
    active_logger.info(
        "Phase 3 independent test: Huber %.6f, regret %.4f, NDCG@%d %.4f",
        test.summary["huber_loss"],
        test.summary["normalized_regret_mean"],
        settings.ndcg_k,
        test.summary[f"ndcg_at_{settings.ndcg_k}_mean"],
    )
    return context.run_dir


def _write_phase1_style_reports(
    context: RunContext,
    summary: Mapping[str, Any],
    fit: HistoryFitResult,
    *,
    variant_name: str,
    epochs_requested: int,
    ndcg_k: int,
) -> None:
    """Emit the training figures and comparison tables used by Phase 1."""

    write_variant_training_curves(
        fit.history,
        context.figure_dir / "training",
        variant_name=variant_name,
        best_epoch=fit.best_epoch,
        epochs_completed=fit.epochs_completed,
        stopped_early=fit.epochs_completed < epochs_requested,
        ndcg_k=ndcg_k,
    )
    write_validation_loss_comparison(
        {variant_name: fit.history},
        context.figure_dir / "training" / "validation_loss_comparison.svg",
        best_epochs={variant_name: fit.best_epoch},
    )

    test = summary["test"]
    row = {
        "variant": variant_name,
        "backbone": "vggt",
        "feature": summary["feature_variant"],
        "input_dim": summary["model"]["feature_dim"] + (
            3 if summary["model"]["include_anchor_directions"] else 0
        ),
        "trainable_parameters": summary["trainable_parameters"],
        "best_epoch": summary["best_epoch"],
        "huber_loss": test["huber_loss"],
        "normalized_regret_mean": test["normalized_regret_mean"],
        "spearman_mean": test["spearman_mean"],
        f"ndcg_at_{ndcg_k}_mean": test[f"ndcg_at_{ndcg_k}_mean"],
    }
    columns = (
        "variant",
        "backbone",
        "feature",
        "input_dim",
        "trainable_parameters",
        "best_epoch",
        "huber_loss",
        "official_unmasked_mse_loss",
        "normalized_regret_mean",
        "spearman_mean",
        f"ndcg_at_{ndcg_k}_mean",
    )
    with (context.metrics_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: row.get(column) for column in columns})
    write_json(
        {"schema_version": 1, "results": [row]},
        context.metrics_dir / "comparison.json",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    values = "| " + " | ".join(
        _format_table_value(row.get(column)) for column in columns
    ) + " |"
    (context.metrics_dir / "comparison.md").write_text(
        "\n".join((header, separator, values)) + "\n", encoding="utf-8"
    )


def _format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _run_smoke(
    model: IndependentHistoryGainModel,
    lookups: Mapping[str, FrozenFeatureLookup],
    histories: Mapping[str, HistoryDataset],
    settings: Phase3IndependentSettings,
    run_dir: Path,
    checkpoint: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    smoke = settings.smoke
    split = str(smoke["split"])
    split_objects = histories[split].manifest["splits"][split]["object_ids"]
    object_id = smoke["object_id"] or split_objects[0]
    if object_id not in split_objects:
        raise ValueError("closed-loop smoke object is outside its configured history split")
    lookup = lookups[split]
    missing = [anchor for anchor in range(48) if not lookup.contains(object_id, anchor)]
    if missing:
        raise ValueError(f"closed-loop smoke feature cache lacks anchors {missing}")
    cache_path = visibility_cache_path(settings.visibility_cache_root, object_id)
    cache = load_visibility_cache(cache_path)
    observations = ObservationStore.from_num_object(settings.data_root, object_id)
    policy = IndependentHistoryPolicy(
        model,
        settings.feature_components,
        feature_lookup=lookup,
        device=settings.device,
        provenance={
            "checkpoint_selection": checkpoint["checkpoint_selection"],
            "history_dataset_id": checkpoint["supervision"]["history_dataset_id"],
            "coverage_target": settings.coverage_target,
            "trainable_parameter_count": count_trainable_parameters(model),
        },
    )
    rollout_config = RolloutConfig(
        max_acquired_views=int(smoke["max_acquired_views"]),
        initial_anchor_ids=tuple(smoke["initial_anchor_ids"]),
        invalid_anchor_ids=tuple(histories[split].manifest["invalid_anchor_ids"]),
        coverage_target=settings.coverage_target,
        seed=seed,
    )
    result = run_rollout(cache, observations, policy, rollout_config)
    replay_rollout(result, cache, ObservationStore.from_num_object(settings.data_root, object_id))
    path = save_rollout(result, run_dir / "closed_loop_smoke" / "rollout.npz")
    smoke_summary = {
        **result.summary(),
        "split": split,
        "rollout_path": str(path.resolve()),
        "replay_verified": True,
    }
    write_json(smoke_summary, run_dir / "closed_loop_smoke" / "summary.json")
    return smoke_summary


def load_independent_history_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[IndependentHistoryGainModel, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("model_type") != "phase3_independent_history_gain":
        raise ValueError("unsupported Phase 3 independent checkpoint")
    fields = payload["model"]
    if (
        fields.get("aggregation") != "masked_mean"
        or fields.get("backbone_history_mode") != "independent_single_image"
        or fields.get("anchor_ordering") != CANONICAL_ORDERING
    ):
        raise ValueError("checkpoint does not describe the Step 16 independent contract")
    model = IndependentHistoryGainModel(
        int(fields["feature_dim"]),
        hidden_dim=int(fields["hidden_dim"]),
        num_anchors=int(fields["num_anchors"]),
        dropout=float(fields["dropout"]),
        include_anchor_directions=bool(fields["include_anchor_directions"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def _validate_manifest(manifest: Mapping[str, Any], settings: Phase3IndependentSettings) -> None:
    if manifest.get("dataset_type") != "phase3_direct_surface_gain_histories":
        raise ValueError("history manifest does not contain direct surface-gain labels")
    if manifest.get("coverage_target") != settings.coverage_target:
        raise ValueError("history and experiment coverage targets differ")
    if manifest.get("anchor_ordering") != CANONICAL_ORDERING:
        raise ValueError("history dataset uses a different anchor ordering")
    if manifest.get("object_disjoint") is not True:
        raise ValueError("Phase 3 histories must be object-disjoint")
    if manifest.get("rotation_metadata", {}).get("enabled") is not False:
        raise ValueError("Step 16 supports the verified non-rotated dataset only")
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("history manifest must contain train, val, and test")


def _validate_smoke(smoke: Mapping[str, Any]) -> None:
    required = {"enabled", "split", "object_id", "initial_anchor_ids", "max_acquired_views"}
    if set(smoke) != required:
        raise ValueError(f"closed_loop_smoke must contain exactly {sorted(required)}")
    if not isinstance(smoke["enabled"], bool):
        raise TypeError("closed_loop_smoke.enabled must be boolean")
    if smoke["split"] not in {"train", "val"}:
        raise ValueError("closed-loop smoke must use train or val, never test")
    if smoke["object_id"] is not None and not isinstance(smoke["object_id"], str):
        raise TypeError("closed_loop_smoke.object_id must be a string or null")
    ids = smoke["initial_anchor_ids"]
    if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
        raise ValueError("closed_loop_smoke.initial_anchor_ids must be a non-empty integer list")
    if len(set(ids)) != len(ids) or any(value < 0 or value >= 48 for value in ids):
        raise ValueError("closed_loop_smoke.initial_anchor_ids contains invalid/duplicate anchors")
    budget = _positive_integer(smoke["max_acquired_views"], "closed_loop_smoke.max_acquired_views")
    if len(ids) > budget:
        raise ValueError("closed-loop smoke budget must include its initial anchors")


def _mapping(owner: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = owner.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _rooted(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    validate_artifact_path(value, name)
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ValueError(f"{name} has an invalid value")
    return number


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "Phase3IndependentSettings",
    "load_independent_history_checkpoint",
    "parse_phase3_independent_settings",
    "run_phase3_independent",
]
