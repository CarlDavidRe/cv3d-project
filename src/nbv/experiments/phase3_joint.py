"""Step 17 joint frozen-VGGT direct-gain control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from nbv.config import load_config, validate_artifact_path
from nbv.data import (
    HistoryDataset,
    HistoryFeatureSample,
    MaterializedHistoryFeatureDataset,
    collate_history_samples,
    relocated_history_manifest_identity,
)
from nbv.eval.result_schema import write_csv, write_json
from nbv.experiments.phase3_independent import parse_phase3_independent_settings
from nbv.features import VGGTJointExtractor, load_feature_cache
from nbv.features.selection import validate_feature_selection
from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, CANONICAL_ORDERING
from nbv.models import (
    IndependentHistoryGainModel,
    JointHistoryGainModel,
    PoseConditionedDeepSetsHistoryGainModel,
    TokenCandidateAttentionHistoryGainModel,
    count_trainable_parameters,
)
from nbv.reproducibility import initialize_run, resolve_run_directory, seed_everything
from nbv.training import evaluate_history_model, fit_history_model
from nbv.visualization import (
    write_validation_loss_comparison,
    write_variant_training_curves,
)


@dataclass(frozen=True, slots=True)
class Phase3JointSettings:
    history_manifest: Path
    coverage_target: str
    device: str
    matched_independent_config: Path
    model_id: str
    image_size: int
    layer_index: int
    expected_feature_dim: int
    feature_components: tuple[str, ...]
    backbone_representation: str
    architecture: str
    hidden_dim: int
    element_dim: int | None
    attention_dim: int | None
    attention_heads: int | None
    token_grid_size: int | None
    dropout: float
    include_anchor_directions: bool
    training: Mapping[str, Any]
    checkpoint_every_batches: int
    evaluation_batch_size: int
    ndcg_k: int
    preflight_enabled: bool
    preflight_history_lengths: tuple[int, ...]
    feature_precompute_enabled: bool
    feature_precompute_batch_size: int
    feature_precompute_num_workers: int
    feature_precompute_pin_memory: bool
    feature_precompute_cache_every_batches: int
    data_root: Path
    model_cache_root: Path


def parse_phase3_joint_settings(
    config: Mapping[str, Any], repository_root: str | Path
) -> Phase3JointSettings:
    root = Path(repository_root).resolve()
    if config.get("experiment", {}).get("phase") != "phase3":
        raise ValueError("joint history training requires experiment.phase=phase3")
    paths = _mapping(config, "paths")
    phase3 = _mapping(config, "phase3")
    control = _mapping(phase3, "joint_control")
    backbone = _mapping(control, "backbone")
    model = _mapping(control, "model")
    training = _mapping(control, "training")
    evaluation = _mapping(control, "evaluation")
    preflight = _mapping(control, "memory_preflight")
    precompute = control.get("frozen_feature_precompute", {
        "enabled": True,
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "cache_every_batches": 64,
    })
    if not isinstance(precompute, Mapping):
        raise TypeError("frozen_feature_precompute must be a mapping")
    components = backbone.get("components")
    if not isinstance(components, list) or not components or any(
        not isinstance(value, str) for value in components
    ):
        raise ValueError("backbone.components must be a non-empty string list")
    components = validate_feature_selection("vggt", components)
    architecture = model.get("architecture", "masked_mean")
    if architecture not in {
        "masked_mean", "pose_deepsets", "token_candidate_attention"
    }:
        raise ValueError(
            "model.architecture must be masked_mean, pose_deepsets, or "
            "token_candidate_attention"
        )
    expected_aggregation = {
        "masked_mean": "masked_mean",
        "pose_deepsets": "pose_conditioned_deepsets_mean",
        "token_candidate_attention": "candidate_cross_attention",
    }[architecture]
    if model.get("aggregation") != expected_aggregation:
        raise ValueError(
            f"model.aggregation must be {expected_aggregation} for {architecture}"
        )
    backbone_representation = backbone.get(
        "representation", "pooled_view_components"
    )
    expected_representation = (
        "spatial_patch_tokens"
        if architecture == "token_candidate_attention"
        else "pooled_view_components"
    )
    if backbone_representation != expected_representation:
        raise ValueError(
            f"backbone.representation must be {expected_representation} for "
            f"{architecture}"
        )
    include_directions = model.get("include_anchor_directions")
    if not isinstance(include_directions, bool):
        raise TypeError("model.include_anchor_directions must be a boolean")
    if architecture == "pose_deepsets":
        element_dim = _positive_integer(model.get("element_dim"), "model.element_dim")
        hidden_dim = _positive_integer(model.get("hidden_dim"), "model.hidden_dim")
        attention_dim = attention_heads = token_grid_size = None
    elif architecture == "token_candidate_attention":
        attention_dim = _positive_integer(
            model.get("attention_dim"), "model.attention_dim"
        )
        attention_heads = _positive_integer(
            model.get("attention_heads"), "model.attention_heads"
        )
        if attention_dim % attention_heads:
            raise ValueError("model.attention_heads must divide model.attention_dim")
        token_grid_size = _positive_integer(
            model.get("token_grid_size"), "model.token_grid_size"
        )
        hidden_dim = _positive_integer(
            model.get("score_hidden_dim"), "model.score_hidden_dim"
        )
        element_dim = None
    else:
        hidden_dim = _positive_integer(model.get("hidden_dim"), "model.hidden_dim")
        element_dim = attention_dim = attention_heads = token_grid_size = None
    dropout = _finite(model.get("dropout"), "model.dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("model.dropout must be below 1")
    required_training = {
        "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
        "weight_decay", "huber_delta", "ranking_weight", "ranking_margin", "patience",
        "checkpoint_every_batches",
    }
    if set(training) != required_training:
        raise ValueError(f"training must contain exactly {sorted(required_training)}")
    for key in (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "patience",
        "checkpoint_every_batches",
    ):
        _positive_integer(training[key], f"training.{key}")
    _finite(training["learning_rate"], "training.learning_rate", minimum=0.0, strict=True)
    _finite(training["weight_decay"], "training.weight_decay", minimum=0.0)
    _finite(training["huber_delta"], "training.huber_delta", minimum=0.0, strict=True)
    _finite(training["ranking_weight"], "training.ranking_weight", minimum=0.0)
    _finite(training["ranking_margin"], "training.ranking_margin", minimum=0.0)
    if set(evaluation) != {"batch_size", "ndcg_k"}:
        raise ValueError("evaluation must contain exactly batch_size and ndcg_k")
    if set(preflight) != {"enabled", "history_lengths"}:
        raise ValueError("memory_preflight must contain exactly enabled and history_lengths")
    if not isinstance(preflight["enabled"], bool):
        raise TypeError("memory_preflight.enabled must be boolean")
    lengths = preflight["history_lengths"]
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("memory_preflight.history_lengths must be a non-empty list")
    for value in lengths:
        _positive_integer(value, "memory_preflight.history_lengths item")
    if len(set(lengths)) != len(lengths):
        raise ValueError("memory_preflight.history_lengths must be unique")
    if set(precompute) != {
        "enabled", "batch_size", "num_workers", "pin_memory", "cache_every_batches"
    }:
        raise ValueError(
            "frozen_feature_precompute must contain exactly enabled, batch_size, "
            "num_workers, pin_memory, and cache_every_batches"
        )
    if not isinstance(precompute["enabled"], bool):
        raise TypeError("frozen_feature_precompute.enabled must be boolean")
    if not isinstance(precompute["pin_memory"], bool):
        raise TypeError("frozen_feature_precompute.pin_memory must be boolean")
    return Phase3JointSettings(
        history_manifest=_rooted(root, control.get("history_manifest"), "history_manifest"),
        coverage_target=_string(control.get("coverage_target"), "coverage_target"),
        device=_string(control.get("device"), "device"),
        matched_independent_config=_rooted(
            root, control.get("matched_independent_config"), "matched_independent_config"
        ),
        model_id=_string(backbone.get("model_id"), "backbone.model_id"),
        image_size=_positive_integer(backbone.get("image_size"), "backbone.image_size"),
        layer_index=_integer(backbone.get("layer_index"), "backbone.layer_index"),
        expected_feature_dim=_positive_integer(
            backbone.get("expected_feature_dim"), "backbone.expected_feature_dim"
        ),
        feature_components=tuple(components),
        backbone_representation=str(backbone_representation),
        architecture=str(architecture),
        hidden_dim=hidden_dim,
        element_dim=element_dim,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        token_grid_size=token_grid_size,
        dropout=dropout,
        include_anchor_directions=include_directions,
        training=dict(training),
        checkpoint_every_batches=int(training["checkpoint_every_batches"]),
        evaluation_batch_size=_positive_integer(
            evaluation["batch_size"], "evaluation.batch_size"
        ),
        ndcg_k=_positive_integer(evaluation["ndcg_k"], "evaluation.ndcg_k"),
        preflight_enabled=bool(preflight["enabled"]),
        preflight_history_lengths=tuple(int(value) for value in lengths),
        feature_precompute_enabled=bool(precompute["enabled"]),
        feature_precompute_batch_size=_positive_integer(
            precompute["batch_size"], "frozen_feature_precompute.batch_size"
        ),
        feature_precompute_num_workers=_nonnegative_integer(
            precompute["num_workers"], "frozen_feature_precompute.num_workers"
        ),
        feature_precompute_pin_memory=bool(precompute["pin_memory"]),
        feature_precompute_cache_every_batches=_positive_integer(
            precompute["cache_every_batches"],
            "frozen_feature_precompute.cache_every_batches",
        ),
        data_root=_rooted(root, paths.get("data_root"), "paths.data_root"),
        model_cache_root=_rooted(
            root, paths.get("model_cache_root"), "paths.model_cache_root"
        ),
    )


def run_phase3_joint(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    logger: logging.Logger | None = None,
    extractor: VGGTJointExtractor | object | None = None,
    resume: bool = False,
) -> Path:
    """Train the Step 17 control; held-out comparison remains Step 18."""

    root = Path(repository_root).resolve()
    settings = parse_phase3_joint_settings(config, root)
    active_logger = logger or logging.getLogger(__name__)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    run_dir = resolve_run_directory(config, root)
    training_checkpoint_path = run_dir / "checkpoints" / "training_state.pt"
    joint_feature_cache_root = run_dir / "joint_feature_cache"
    cached_feature_shards = tuple(joint_feature_cache_root.glob("*/*.pt"))
    if (training_checkpoint_path.exists() or cached_feature_shards) and not resume:
        raise ValueError(
            f"resumable joint state already exists under {run_dir}; "
            "use --resume or choose a new experiment.name"
        )
    if resume:
        saved_config_path = run_dir / "config.yaml"
        if not saved_config_path.is_file():
            raise FileNotFoundError(
                f"resumed joint run lacks its saved config: {saved_config_path}"
            )
        if load_config(saved_config_path) != dict(config):
            raise ValueError("resumed joint run configuration differs from config.yaml")
    active_logger.info("Phase 3 joint: validating the matched independent control")
    match = _validate_matched_control(settings, root, config)
    active_logger.info("Phase 3 joint: loading train/validation histories with RGB images")
    histories = {
        split: HistoryDataset(
            settings.history_manifest,
            split=split,
            data_root=settings.data_root,
            load_images=True,
        )
        for split in ("train", "val")
    }
    manifest = histories["train"].manifest
    _validate_manifest(manifest, settings)
    if extractor is None:
        active_logger.info(
            "Phase 3 joint: initializing frozen %s backbone", settings.model_id
        )
        extractor = VGGTJointExtractor(
            feature_components=settings.feature_components,
            model_id=settings.model_id,
            image_size=settings.image_size,
            layer_index=settings.layer_index,
            expected_feature_dim=settings.expected_feature_dim,
            spatial_token_grid_size=settings.token_grid_size,
            device=settings.device,
            model_cache_root=settings.model_cache_root,
        )
    model = _create_joint_model(extractor, settings)
    resolved_device = _resolve_device(settings.device)
    extractor_device = torch.device(getattr(extractor, "device", resolved_device))
    if extractor_device != resolved_device:
        raise ValueError("joint extractor and experiment devices differ")
    model.to(resolved_device)
    independent_shape = IndependentHistoryGainModel(
        settings.expected_feature_dim,
        hidden_dim=settings.hidden_dim,
        dropout=settings.dropout,
        include_anchor_directions=settings.include_anchor_directions,
    )
    joint_parameters = count_trainable_parameters(model)
    independent_parameters = count_trainable_parameters(independent_shape)
    if settings.architecture == "masked_mean" and joint_parameters != independent_parameters:
        raise ValueError("joint and independent trainable head capacities differ")
    context = initialize_run(config, root)
    run_identity = {
        "model_type": "phase3_joint_history_gain",
        "architecture": settings.architecture,
        "experiment_config_sha256": _mapping_sha256(config),
        "history_dataset_id": manifest["dataset_id"],
        "history_manifest_sha256": _sha256(settings.history_manifest),
    }
    if resume:
        run_identity = _resolve_resumed_run_identity(
            run_identity,
            manifest,
            repository_root=root,
            cached_feature_shards=cached_feature_shards,
            training_checkpoint_path=training_checkpoint_path,
            logger=active_logger,
        )
    active_logger.info(
        "Phase 3 joint memory preflight: %s",
        (
            "disabled"
            if not settings.preflight_enabled
            else "history lengths "
            + ", ".join(str(value) for value in settings.preflight_history_lengths)
        ),
    )
    preflight = _memory_preflight(
        model, histories["train"], settings, logger=active_logger
    )
    write_json(preflight, context.metrics_dir / "memory_preflight.json")
    active_logger.info(
        "Phase 3 joint ready: %d train, %d validation histories; %d-D features; "
        "%d trainable parameters",
        len(histories["train"]),
        len(histories["val"]),
        settings.expected_feature_dim,
        joint_parameters,
    )
    if settings.feature_precompute_enabled:
        active_logger.info(
            "Phase 3 joint: materializing frozen joint features once "
            "(batch size %d, workers %d)",
            settings.feature_precompute_batch_size,
            settings.feature_precompute_num_workers,
        )
        materialized_started = time.perf_counter()
        training_data = _materialize_joint_features(
            model,
            histories["train"],
            batch_size=settings.feature_precompute_batch_size,
            num_workers=settings.feature_precompute_num_workers,
            pin_memory=settings.feature_precompute_pin_memory,
            logger=active_logger,
            progress_label="Phase 3 joint train feature precompute",
            cache_dir=joint_feature_cache_root / "train",
            cache_every_batches=settings.feature_precompute_cache_every_batches,
            resume=resume,
            cache_identity={**run_identity, "split": "train"},
        )
        validation_data = _materialize_joint_features(
            model,
            histories["val"],
            batch_size=settings.feature_precompute_batch_size,
            num_workers=settings.feature_precompute_num_workers,
            pin_memory=settings.feature_precompute_pin_memory,
            logger=active_logger,
            progress_label="Phase 3 joint validation feature precompute",
            cache_dir=joint_feature_cache_root / "val",
            cache_every_batches=settings.feature_precompute_cache_every_batches,
            resume=resume,
            cache_identity={**run_identity, "split": "val"},
        )
        feature_precompute = {
            "enabled": True,
            "storage": "system_memory_and_resumable_disk_shards",
            "backbone_forwards_per_history": 1,
            "configured_batch_size": settings.feature_precompute_batch_size,
            "num_workers": settings.feature_precompute_num_workers,
            "pin_memory": settings.feature_precompute_pin_memory,
            "cache_every_batches": settings.feature_precompute_cache_every_batches,
            "cache_root": str(joint_feature_cache_root.resolve()),
            "cache_shards": len(tuple(joint_feature_cache_root.glob("*/*.pt"))),
            "train_bytes": training_data.storage_bytes,
            "validation_bytes": validation_data.storage_bytes,
            "elapsed_seconds": time.perf_counter() - materialized_started,
        }
        active_logger.info(
            "Phase 3 joint: feature precompute complete in %.1f seconds; "
            "%s retained in system memory",
            feature_precompute["elapsed_seconds"],
            _format_bytes(
                feature_precompute["train_bytes"]
                + feature_precompute["validation_bytes"]
            ),
        )
    else:
        training_data = histories["train"]
        validation_data = histories["val"]
        feature_precompute = {
            "enabled": False,
            "storage": None,
            "backbone_forwards_per_history": None,
        }
    training = dict(settings.training)
    accumulation = int(training.pop("gradient_accumulation_steps"))
    checkpoint_every_batches = int(training.pop("checkpoint_every_batches"))
    resume_training = resume and training_checkpoint_path.is_file()
    if resume and not resume_training:
        active_logger.info(
            "No head-training checkpoint exists yet; starting head training "
            "after restoring the available joint-feature shards"
        )
    fit = fit_history_model(
        model,
        training_data,
        validation_data,
        device=settings.device,
        seed=seed,
        ndcg_k=settings.ndcg_k,
        logger=active_logger,
        gradient_accumulation_steps=accumulation,
        training_checkpoint_path=training_checkpoint_path,
        checkpoint_every_batches=checkpoint_every_batches,
        resume=resume_training,
        checkpoint_identity=run_identity,
        **training,
    )
    validation = evaluate_history_model(
        model,
        validation_data,
        batch_size=settings.evaluation_batch_size,
        huber_delta=settings.training["huber_delta"],
        ranking_weight=settings.training["ranking_weight"],
        ranking_margin=settings.training["ranking_margin"],
        ndcg_k=settings.ndcg_k,
        device=settings.device,
        logger=active_logger,
        progress_label="Phase 3 joint final validation",
    )
    checkpoint_path = context.checkpoint_dir / "best.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": "phase3_joint_history_gain",
        "model": {
            "architecture": settings.architecture,
            "feature_dim": settings.expected_feature_dim,
            "hidden_dim": settings.hidden_dim,
            "num_anchors": CANONICAL_ANCHOR_COUNT,
            "dropout": settings.dropout,
            "include_anchor_directions": settings.include_anchor_directions,
            "aggregation": model.aggregation,
            "backbone_history_mode": model.backbone_history_mode,
            "anchor_ordering": CANONICAL_ORDERING,
            **(
                {"element_dim": settings.element_dim}
                if settings.architecture == "pose_deepsets"
                else {}
            ),
            **(
                {
                    "attention_dim": settings.attention_dim,
                    "attention_heads": settings.attention_heads,
                    "score_hidden_dim": settings.hidden_dim,
                    "token_grid_size": settings.token_grid_size,
                }
                if settings.architecture == "token_candidate_attention"
                else {}
            ),
        },
        "backbone": {
            "name": "vggt",
            "model_id": settings.model_id,
            "image_size": settings.image_size,
            "layer_index": settings.layer_index,
            "feature_components": list(settings.feature_components),
            "representation": settings.backbone_representation,
            "spatial_token_grid_size": settings.token_grid_size,
            "history_mode": "joint_multiview",
            "padding_strategy": "group_by_real_history_length",
            "training_feature_strategy": (
                "materialize_joint_features_once_in_system_memory"
                if settings.feature_precompute_enabled
                else "live_joint_forward_per_head_batch"
            ),
            "frozen": True,
        },
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "supervision": {
            "target": "target_surface_gain",
            "coverage_target": settings.coverage_target,
            "target_direction": "higher",
            "history_dataset_id": manifest["dataset_id"],
            "history_manifest": str(settings.history_manifest.resolve()),
            "history_manifest_sha256": _sha256(settings.history_manifest),
        },
        "control_match": match,
        "checkpoint_selection": {
            "split": "val",
            "metric": "total_loss",
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    write_json(list(fit.history), context.metrics_dir / "training_history.json")
    write_csv(
        list(validation.per_sample), context.metrics_dir / "validation_per_sample.csv"
    )
    frozen_model = getattr(extractor, "model", None)
    frozen_parameters = (
        None
        if frozen_model is None
        else sum(parameter.numel() for parameter in frozen_model.parameters())
    )
    if frozen_model is not None and any(
        parameter.requires_grad for parameter in frozen_model.parameters()
    ):
        raise ValueError("joint VGGT backbone must remain frozen")
    summary = {
        "step": 17,
        "scope": "joint_control_training_and_short_history_preflight",
        "held_out_test_and_closed_loop_comparison": "deferred_to_step_18",
        "model": checkpoint["model"],
        "backbone": checkpoint["backbone"],
        "trainable_parameters": joint_parameters,
        "matched_independent_trainable_parameters": independent_parameters,
        "frozen_backbone_parameters": frozen_parameters,
        "control_match": match,
        "history_dataset_id": manifest["dataset_id"],
        "coverage_target": settings.coverage_target,
        "train_samples": len(histories["train"]),
        "validation_samples": len(histories["val"]),
        "best_epoch": fit.best_epoch,
        "epochs_completed": fit.epochs_completed,
        "best_validation_loss": fit.best_validation_loss,
        "validation": dict(validation.summary),
        "memory_preflight": preflight,
        "frozen_feature_precompute": feature_precompute,
        "resumed": resume,
        "training_resumed": resume_training,
        "training_checkpoint": str(training_checkpoint_path.resolve()),
        "training_checkpoint_sha256": _sha256(training_checkpoint_path),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    write_json(summary, context.metrics_dir / "summary.json")
    variant = str(config["experiment"]["name"])
    write_variant_training_curves(
        fit.history,
        context.figure_dir / "training",
        variant_name=variant,
        best_epoch=fit.best_epoch,
        epochs_completed=fit.epochs_completed,
        stopped_early=fit.epochs_completed < int(settings.training["epochs"]),
        ndcg_k=settings.ndcg_k,
    )
    write_validation_loss_comparison(
        {variant: fit.history},
        context.figure_dir / "training" / "validation_loss_comparison.svg",
        best_epochs={variant: fit.best_epoch},
    )
    active_logger.info(
        "Phase 3 joint validation: Huber %.6f, regret %.4f, NDCG@%d %.4f",
        validation.summary["huber_loss"],
        validation.summary["normalized_regret_mean"],
        settings.ndcg_k,
        validation.summary[f"ndcg_at_{settings.ndcg_k}_mean"],
    )
    return context.run_dir


def load_joint_history_checkpoint(
    path: str | Path,
    *,
    extractor: object | None = None,
    device: str | torch.device = "cpu",
    model_cache_root: str | Path | None = None,
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("model_type") != "phase3_joint_history_gain":
        raise ValueError("unsupported Phase 3 joint checkpoint")
    fields = payload["model"]
    backbone = payload["backbone"]
    architecture = fields.get("architecture", "masked_mean")
    if (
        architecture not in {"masked_mean", "pose_deepsets", "token_candidate_attention"}
        or fields.get("anchor_ordering") != CANONICAL_ORDERING
        or backbone.get("padding_strategy") != "group_by_real_history_length"
    ):
        raise ValueError("checkpoint does not describe the Step 17 joint contract")
    expected_aggregation = {
        "masked_mean": "masked_mean",
        "pose_deepsets": "pose_conditioned_deepsets_mean",
        "token_candidate_attention": "candidate_cross_attention",
    }[architecture]
    if fields.get("aggregation") != expected_aggregation:
        raise ValueError("checkpoint aggregation disagrees with its architecture")
    expected_representation = (
        "spatial_patch_tokens"
        if architecture == "token_candidate_attention"
        else "pooled_view_components"
    )
    if backbone.get("representation", "pooled_view_components") != expected_representation:
        raise ValueError("checkpoint backbone representation disagrees with its architecture")
    if extractor is None:
        extractor = VGGTJointExtractor(
            feature_components=backbone["feature_components"],
            model_id=backbone["model_id"],
            image_size=int(backbone["image_size"]),
            layer_index=int(backbone["layer_index"]),
            expected_feature_dim=int(fields["feature_dim"]),
            spatial_token_grid_size=(
                int(fields["token_grid_size"])
                if architecture == "token_candidate_attention"
                else None
            ),
            device=device,
            model_cache_root=model_cache_root,
        )
    common = {
        "num_anchors": int(fields["num_anchors"]),
        "dropout": float(fields["dropout"]),
        "include_anchor_directions": bool(fields["include_anchor_directions"]),
    }
    if architecture == "pose_deepsets":
        model = PoseConditionedDeepSetsHistoryGainModel(
            extractor,
            int(fields["feature_dim"]),
            element_dim=int(fields["element_dim"]),
            hidden_dim=int(fields["hidden_dim"]),
            **common,
        )
    elif architecture == "token_candidate_attention":
        model = TokenCandidateAttentionHistoryGainModel(
            extractor,
            int(fields["feature_dim"]),
            attention_dim=int(fields["attention_dim"]),
            attention_heads=int(fields["attention_heads"]),
            score_hidden_dim=int(fields["score_hidden_dim"]),
            **common,
        )
    else:
        model = JointHistoryGainModel(
            extractor,
            int(fields["feature_dim"]),
            hidden_dim=int(fields["hidden_dim"]),
            **common,
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    resolved_device = _resolve_device(device)
    extractor_device = torch.device(getattr(extractor, "device", resolved_device))
    if extractor_device != resolved_device:
        raise ValueError("joint extractor and checkpoint devices differ")
    model.to(resolved_device).eval()
    return model, payload


def _validate_matched_control(
    settings: Phase3JointSettings,
    root: Path,
    joint_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    independent_config = load_config(settings.matched_independent_config)
    independent = parse_phase3_independent_settings(independent_config, root)
    comparisons = {
        "seed": int(independent_config["experiment"]["seed"])
        == int(joint_config["experiment"]["seed"]),
        "deterministic_mode": bool(independent_config["experiment"]["deterministic"])
        == bool(joint_config["experiment"]["deterministic"]),
        "history_manifest": independent.history_manifest == settings.history_manifest,
        "coverage_target": independent.coverage_target == settings.coverage_target,
        "feature_components": independent.feature_components == settings.feature_components,
        "hidden_dim": independent.hidden_dim == settings.hidden_dim,
        "dropout": independent.dropout == settings.dropout,
        "include_anchor_directions": (
            independent.include_anchor_directions == settings.include_anchor_directions
        ),
    }
    shared_training_keys = (
        "epochs", "learning_rate", "weight_decay", "huber_delta",
        "ranking_weight", "ranking_margin", "patience",
    )
    comparisons["optimizer_loss_and_budget"] = all(
        independent.training[key] == settings.training[key]
        for key in shared_training_keys
    )
    effective_batch = int(settings.training["batch_size"]) * int(
        settings.training["gradient_accumulation_steps"]
    )
    comparisons["effective_batch_size"] = (
        effective_batch == int(independent.training["batch_size"])
    )
    if not all(comparisons.values()):
        failed = [key for key, value in comparisons.items() if not value]
        raise ValueError(
            "joint config does not match the independent control: " + ", ".join(failed)
        )
    cache_path = independent.feature_cache_paths["train"]
    cache = load_feature_cache(cache_path)
    metadata = cache.metadata
    backbone = metadata.get("backbone_configuration", {})
    history_manifest = json.loads(settings.history_manifest.read_text(encoding="utf-8"))
    checkpoint_match = (
        metadata.get("backbone") == "vggt"
        and backbone.get("model_id") == settings.model_id
        and backbone.get("image_size") == settings.image_size
        and backbone.get("layer_index") == settings.layer_index
        and backbone.get("input_resolution") == [settings.image_size, settings.image_size]
        and backbone.get("preprocessing") == "square_resize_rgb_0_1"
        and backbone.get("history_mode") == "single_image"
        and metadata.get("feature_components") == list(settings.feature_components)
        and metadata.get("split_manifest_sha256")
        == history_manifest.get("split_manifest_sha256")
        and int(cache.features.shape[1]) == settings.expected_feature_dim
    )
    if not checkpoint_match:
        raise ValueError(
            "joint VGGT settings do not match the independent feature-cache backbone"
        )
    return {
        "matched_independent_config": str(settings.matched_independent_config),
        "matched_independent_config_sha256": _sha256(settings.matched_independent_config),
        "all_required_fields_match": True,
        "field_checks": comparisons,
        "same_vggt_model_id_and_feature_selection": (
            None if settings.architecture == "token_candidate_attention" else True
        ),
        "same_vggt_model_id_layer_and_preprocessing": True,
        "same_patch_token_source": True,
        "joint_backbone_representation": settings.backbone_representation,
        "checkpoint_identity_note": (
            "The Step 16 feature cache records facebook/VGGT-1B but no weight "
            "checksum; model ID, preprocessing, layer, and source patch-token "
            "family are matched. The token-attention follow-up intentionally "
            "retains a spatial grid instead of applying the independent max pool."
            if settings.architecture == "token_candidate_attention"
            else "The Step 16 feature cache records facebook/VGGT-1B but no weight "
            "checksum; model ID, preprocessing, layer, and selected component "
            "are matched exactly."
        ),
        "independent_feature_cache": str(cache_path),
        "independent_feature_cache_sha256": _sha256(cache_path),
        "independent_microbatch_size": int(independent.training["batch_size"]),
        "joint_head_microbatch_size": int(settings.training["batch_size"]),
        "joint_gradient_accumulation_steps": int(
            settings.training["gradient_accumulation_steps"]
        ),
        "joint_backbone_precompute_batch_size": (
            settings.feature_precompute_batch_size
            if settings.feature_precompute_enabled
            else None
        ),
        "effective_batch_size": effective_batch,
        "optimizer": "AdamW",
        "capacity_match": (
            "exact"
            if settings.architecture == "masked_mean"
            else "not_required_for_expressive_follow_up_variant"
        ),
    }


def _create_joint_model(
    extractor: object,
    settings: Phase3JointSettings,
) -> torch.nn.Module:
    common = {
        "dropout": settings.dropout,
        "include_anchor_directions": settings.include_anchor_directions,
    }
    if settings.architecture == "pose_deepsets":
        return PoseConditionedDeepSetsHistoryGainModel(
            extractor,
            settings.expected_feature_dim,
            element_dim=int(settings.element_dim),
            hidden_dim=settings.hidden_dim,
            **common,
        )
    if settings.architecture == "token_candidate_attention":
        return TokenCandidateAttentionHistoryGainModel(
            extractor,
            settings.expected_feature_dim,
            attention_dim=int(settings.attention_dim),
            attention_heads=int(settings.attention_heads),
            score_hidden_dim=settings.hidden_dim,
            **common,
        )
    return JointHistoryGainModel(
        extractor,
        settings.expected_feature_dim,
        hidden_dim=settings.hidden_dim,
        **common,
    )


def _memory_preflight(
    model: torch.nn.Module,
    dataset: HistoryDataset,
    settings: Phase3JointSettings,
    *,
    logger: logging.Logger | None = None,
) -> Mapping[str, Any]:
    if not settings.preflight_enabled:
        return {"enabled": False, "measurements": []}
    from nbv.data import collate_history_samples

    by_length: dict[int, int] = {}
    lengths = dataset._arrays["history_lengths"].tolist()
    for index, value in enumerate(lengths):
        length = int(value)
        if length in settings.preflight_history_lengths and length not in by_length:
            by_length[length] = index
    missing = set(settings.preflight_history_lengths) - set(by_length)
    if missing:
        raise ValueError(f"memory preflight history lengths are absent: {sorted(missing)}")
    device = next(model.parameters()).device
    measurements = []
    was_training = model.training
    model.eval()
    for length in settings.preflight_history_lengths:
        if logger is not None:
            logger.info(
                "Phase 3 joint memory preflight running: history length %d", length
            )
        batch = collate_history_samples([dataset[by_length[length]]]).to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(
                batch.history_images,
                batch.history_anchor_ids,
                batch.history_padding_mask,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = None
            peak_reserved = None
        measurement = {
            "history_length": length,
            "batch_size": 1,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "prediction_shape": list(prediction.shape),
        }
        measurements.append(measurement)
        if logger is not None:
            logger.info(
                "Phase 3 joint memory preflight complete: history length %d, "
                "%.3f seconds, peak allocated %s, peak reserved %s",
                length,
                measurement["elapsed_seconds"],
                _format_bytes(peak_allocated),
                _format_bytes(peak_reserved),
            )
    model.train(was_training)
    return {
        "enabled": True,
        "device": str(device),
        "measurement_scope": "frozen_joint_vggt_plus_trainable_head_forward",
        "cpu_memory_note": (
            None if device.type == "cuda" else "PyTorch peak accelerator memory is unavailable on CPU"
        ),
        "measurements": measurements,
    }


def _materialize_joint_features(
    model: torch.nn.Module,
    dataset: HistoryDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    logger: logging.Logger | None,
    progress_label: str,
    cache_dir: Path,
    cache_every_batches: int,
    resume: bool,
    cache_identity: Mapping[str, Any],
) -> MaterializedHistoryFeatureDataset:
    """Cache frozen joint vectors in atomic shards and retain them in memory."""

    lengths = [int(value) for value in dataset._arrays["history_lengths"]]
    batches: list[list[int]] = []
    # Same-length batches avoid extra VGGT calls because the backbone has no
    # padding-mask input. Long histories run first so an oversized batch fails
    # (and triggers the OOM backoff) early rather than late in a long precompute.
    for length in sorted(set(lengths), reverse=True):
        indices = [index for index, value in enumerate(lengths) if value == length]
        batches.extend(
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    sample_indices = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset._arrays["sample_ids"])
    }
    materialized: list[HistoryFeatureSample | None] = [None] * len(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_shards = math.ceil(len(batches) / cache_every_batches)
    first_missing_shard = total_shards
    completed = 0
    for shard_index in range(total_shards):
        shard_path = cache_dir / f"shard_{shard_index:05d}.pt"
        if not shard_path.is_file():
            first_missing_shard = shard_index
            break
        if not resume:
            raise ValueError(
                f"joint feature cache already exists: {shard_path}; use --resume"
            )
        start = shard_index * cache_every_batches
        end = min(start + cache_every_batches, len(batches))
        expected_ids = _batch_sample_ids(dataset, batches[start:end])
        cached = _load_joint_feature_shard(
            shard_path,
            expected_sample_ids=expected_ids,
            cache_identity=cache_identity,
        )
        for sample in cached:
            materialized[sample_indices[sample.sample_id]] = sample
        completed += len(cached)
    if any(
        (cache_dir / f"shard_{index:05d}.pt").exists()
        for index in range(first_missing_shard + 1, total_shards)
    ):
        raise ValueError("joint feature cache has a gap between completed shards")
    if logger is not None and completed:
        logger.info(
            "%s: restored %d/%d histories from %d disk shards",
            progress_label,
            completed,
            len(dataset),
            first_missing_shard,
        )

    pending_batches = batches[first_missing_shard * cache_every_batches :]
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": pending_batches,
        "num_workers": num_workers,
        "collate_fn": collate_history_samples,
        "pin_memory": pin_memory and next(model.parameters()).device.type == "cuda",
    }
    if num_workers:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    safe_batch_sizes: dict[int, int] = {}
    shard_samples: list[HistoryFeatureSample] = []
    shard_index = first_missing_shard
    batches_in_shard = 0
    last_progress_log = time.monotonic()
    for cpu_batch in loader:
        feature_rows = _extract_with_oom_backoff(
            model.extractor,
            cpu_batch.history_images,
            cpu_batch.history_padding_mask,
            logger=logger,
            safe_batch_sizes=safe_batch_sizes,
        )
        for row, sample_id in enumerate(cpu_batch.sample_ids):
            length = int(cpu_batch.history_lengths[row])
            index = sample_indices[sample_id]
            sample = HistoryFeatureSample(
                sample_id=sample_id,
                object_id=cpu_batch.object_ids[row],
                history_features=feature_rows[row, :length].contiguous(),
                history_anchor_ids=cpu_batch.history_anchor_ids[row, :length].clone(),
                target_surface_gain=cpu_batch.target_surface_gain[row].clone(),
                valid_candidate_mask=cpu_batch.valid_candidate_mask[row].clone(),
                visibility_cache_id=cpu_batch.visibility_cache_ids[row],
            )
            materialized[index] = sample
            shard_samples.append(sample)
        completed += len(cpu_batch.sample_ids)
        batches_in_shard += 1
        if batches_in_shard == cache_every_batches or completed == len(dataset):
            shard_path = cache_dir / f"shard_{shard_index:05d}.pt"
            _save_joint_feature_shard(
                shard_path,
                shard_samples,
                cache_identity=cache_identity,
            )
            if logger is not None:
                logger.info(
                    "%s: cached shard %d/%d (%d/%d histories)",
                    progress_label,
                    shard_index + 1,
                    total_shards,
                    completed,
                    len(dataset),
                )
            shard_index += 1
            batches_in_shard = 0
            shard_samples = []
        now = time.monotonic()
        if logger is not None and (
            completed == len(dataset) or now - last_progress_log >= 30.0
        ):
            logger.info(
                "%s: %d/%d histories (%.1f%%)",
                progress_label,
                completed,
                len(dataset),
                100.0 * completed / len(dataset),
            )
            last_progress_log = now
    if any(sample is None for sample in materialized):
        raise RuntimeError("joint feature precompute did not visit every history")
    return MaterializedHistoryFeatureDataset(
        dataset,
        [sample for sample in materialized if sample is not None],
    )


def _batch_sample_ids(
    dataset: HistoryDataset, batches: list[list[int]]
) -> tuple[str, ...]:
    sample_ids = dataset._arrays["sample_ids"]
    return tuple(str(sample_ids[index]) for batch in batches for index in batch)


def _save_joint_feature_shard(
    path: Path,
    samples: list[HistoryFeatureSample],
    *,
    cache_identity: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "cache_type": "phase3_joint_materialized_features",
        "cache_identity": dict(cache_identity),
        "sample_ids": [sample.sample_id for sample in samples],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "object_id": sample.object_id,
                "history_features": sample.history_features,
                "history_anchor_ids": sample.history_anchor_ids,
                "target_surface_gain": sample.target_surface_gain,
                "valid_candidate_mask": sample.valid_candidate_mask,
                "visibility_cache_id": sample.visibility_cache_id,
            }
            for sample in samples
        ],
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_joint_feature_shard(
    path: Path,
    *,
    expected_sample_ids: tuple[str, ...],
    cache_identity: Mapping[str, Any],
) -> list[HistoryFeatureSample]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("cache_type") != "phase3_joint_materialized_features"
    ):
        raise ValueError(f"unsupported joint feature-cache shard: {path}")
    if payload.get("cache_identity") != dict(cache_identity):
        raise ValueError(f"joint feature-cache identity differs: {path}")
    if tuple(payload.get("sample_ids", ())) != expected_sample_ids:
        raise ValueError(f"joint feature-cache sample order differs: {path}")
    samples = [HistoryFeatureSample(**row) for row in payload["samples"]]
    if tuple(sample.sample_id for sample in samples) != expected_sample_ids:
        raise ValueError(f"joint feature-cache payload is inconsistent: {path}")
    return samples


def _resolve_resumed_run_identity(
    current_identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    cached_feature_shards: tuple[Path, ...],
    training_checkpoint_path: Path,
    logger: logging.Logger | None,
) -> dict[str, Any]:
    """Accept an otherwise exact resume identity after repository relocation."""

    artifact_identities: list[Mapping[str, Any]] = []
    if cached_feature_shards:
        payload = torch.load(
            sorted(cached_feature_shards)[0], map_location="cpu", weights_only=True
        )
        if (
            payload.get("schema_version") == 1
            and payload.get("cache_type") == "phase3_joint_materialized_features"
            and isinstance(payload.get("cache_identity"), Mapping)
        ):
            cache_identity = dict(payload["cache_identity"])
            cache_identity.pop("split", None)
            artifact_identities.append(cache_identity)
    if training_checkpoint_path.is_file():
        payload = torch.load(
            training_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            payload.get("schema_version") == 1
            and payload.get("checkpoint_type") == "phase3_history_training_state"
            and isinstance(payload.get("checkpoint_identity"), Mapping)
        ):
            artifact_identities.append(payload["checkpoint_identity"])
    if not artifact_identities:
        return dict(current_identity)
    artifact_identity = dict(artifact_identities[0])
    if any(
        dict(identity) != artifact_identity for identity in artifact_identities[1:]
    ):
        raise ValueError("resumable joint artifacts use different experiment identities")
    if artifact_identity == dict(current_identity):
        return artifact_identity

    embedded_root = _history_manifest_repository_root(manifest)
    if embedded_root is not None and embedded_root != repository_root:
        dataset_id, manifest_sha256 = relocated_history_manifest_identity(
            manifest,
            current_repository_root=embedded_root,
            artifact_repository_root=repository_root,
        )
        relocated_identity = {
            **current_identity,
            "history_dataset_id": dataset_id,
            "history_manifest_sha256": manifest_sha256,
        }
        if artifact_identity == relocated_identity:
            if logger is not None:
                logger.info(
                    "Phase 3 joint: accepting resumable artifacts whose history "
                    "identity differs only by repository-root relocation (%s -> %s)",
                    embedded_root,
                    repository_root,
                )
            return artifact_identity
    return dict(current_identity)


def _history_manifest_repository_root(manifest: Mapping[str, Any]) -> Path | None:
    """Recover the repository root embedded by early Phase 3 provenance."""

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    roots: set[Path] = set()
    for key in (
        "history_config",
        "visibility_config",
        "evaluation_config",
        "split_manifest",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            continue
        parts = Path(value).parts
        markers = [
            index
            for index, part in enumerate(parts)
            if part in {"configs", "data"}
        ]
        if markers:
            roots.add(Path(*parts[: markers[-1]]))
    return roots.pop() if len(roots) == 1 else None


def _extract_with_oom_backoff(
    extractor: object,
    history_images: torch.Tensor | None,
    history_padding_mask: torch.Tensor,
    *,
    logger: logging.Logger | None,
    safe_batch_sizes: dict[int, int] | None = None,
) -> torch.Tensor:
    if history_images is None:
        raise ValueError("joint feature precompute requires loaded RGB images")
    real_lengths = (~history_padding_mask).sum(dim=1)
    if torch.unique(real_lengths).numel() != 1:
        raise ValueError("joint feature precompute batches must share one history length")
    history_length = int(real_lengths[0])
    count = int(history_images.shape[0])
    learned_limit = None if safe_batch_sizes is None else safe_batch_sizes.get(history_length)
    if learned_limit is not None and count > learned_limit:
        chunks = []
        for start in range(0, count, learned_limit):
            chunks.append(_extract_with_oom_backoff(
                extractor,
                history_images[start : start + learned_limit],
                history_padding_mask[start : start + learned_limit],
                logger=logger,
                safe_batch_sizes=safe_batch_sizes,
            ))
        return torch.cat(chunks, dim=0)
    try:
        frozen = extractor.extract(history_images, history_padding_mask)
        return frozen.view_features.detach().to("cpu", copy=True)
    except torch.cuda.OutOfMemoryError:
        if count == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = count // 2
        if safe_batch_sizes is not None:
            safe_batch_sizes[history_length] = min(
                safe_batch_sizes.get(history_length, midpoint), midpoint
            )
        if logger is not None:
            logger.warning(
                "Joint feature batch of %d length-%d histories exceeded GPU memory; "
                "using at most %d for this history length",
                count,
                history_length,
                midpoint,
            )
        first = _extract_with_oom_backoff(
            extractor,
            history_images[:midpoint],
            history_padding_mask[:midpoint],
            logger=logger,
            safe_batch_sizes=safe_batch_sizes,
        )
        second = _extract_with_oom_backoff(
            extractor,
            history_images[midpoint:],
            history_padding_mask[midpoint:],
            logger=logger,
            safe_batch_sizes=safe_batch_sizes,
        )
        return torch.cat((first, second), dim=0)


def _validate_manifest(
    manifest: Mapping[str, Any], settings: Phase3JointSettings
) -> None:
    if manifest.get("dataset_type") != "phase3_direct_surface_gain_histories":
        raise ValueError("history manifest does not contain direct surface-gain labels")
    if manifest.get("coverage_target") != settings.coverage_target:
        raise ValueError("history and experiment coverage targets differ")
    if manifest.get("anchor_ordering") != CANONICAL_ORDERING:
        raise ValueError("history dataset uses a different anchor ordering")
    if manifest.get("object_disjoint") is not True:
        raise ValueError("Phase 3 histories must be object-disjoint")
    if manifest.get("rotation_metadata", {}).get("enabled") is not False:
        raise ValueError("Step 17 supports the verified non-rotated dataset only")
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("history manifest must contain train, val, and test")


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


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ValueError(f"{name} has an invalid value")
    return number


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / (1024 ** 2):.1f} MiB"


__all__ = [
    "Phase3JointSettings",
    "load_joint_history_checkpoint",
    "parse_phase3_joint_settings",
    "run_phase3_joint",
]
