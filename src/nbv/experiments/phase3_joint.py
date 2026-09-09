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

from nbv.config import load_config, validate_artifact_path
from nbv.data import HistoryDataset
from nbv.eval.result_schema import write_csv, write_json
from nbv.experiments.phase3_independent import parse_phase3_independent_settings
from nbv.features import VGGTJointExtractor, load_feature_cache
from nbv.features.selection import validate_feature_selection
from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, CANONICAL_ORDERING
from nbv.models import (
    IndependentHistoryGainModel,
    JointHistoryGainModel,
    count_trainable_parameters,
)
from nbv.reproducibility import initialize_run, seed_everything
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
    hidden_dim: int
    dropout: float
    include_anchor_directions: bool
    training: Mapping[str, Any]
    evaluation_batch_size: int
    ndcg_k: int
    preflight_enabled: bool
    preflight_history_lengths: tuple[int, ...]
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
    components = backbone.get("components")
    if not isinstance(components, list) or not components or any(
        not isinstance(value, str) for value in components
    ):
        raise ValueError("backbone.components must be a non-empty string list")
    components = validate_feature_selection("vggt", components)
    if model.get("aggregation") != "masked_mean":
        raise ValueError("Step 17 model.aggregation must be masked_mean")
    include_directions = model.get("include_anchor_directions")
    if not isinstance(include_directions, bool):
        raise TypeError("model.include_anchor_directions must be a boolean")
    dropout = _finite(model.get("dropout"), "model.dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("model.dropout must be below 1")
    required_training = {
        "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
        "weight_decay", "huber_delta", "ranking_weight", "ranking_margin", "patience",
    }
    if set(training) != required_training:
        raise ValueError(f"training must contain exactly {sorted(required_training)}")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "patience"):
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
        hidden_dim=_positive_integer(model.get("hidden_dim"), "model.hidden_dim"),
        dropout=dropout,
        include_anchor_directions=include_directions,
        training=dict(training),
        evaluation_batch_size=_positive_integer(
            evaluation["batch_size"], "evaluation.batch_size"
        ),
        ndcg_k=_positive_integer(evaluation["ndcg_k"], "evaluation.ndcg_k"),
        preflight_enabled=bool(preflight["enabled"]),
        preflight_history_lengths=tuple(int(value) for value in lengths),
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
) -> Path:
    """Train the Step 17 control; held-out comparison remains Step 18."""

    root = Path(repository_root).resolve()
    settings = parse_phase3_joint_settings(config, root)
    active_logger = logger or logging.getLogger(__name__)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    match = _validate_matched_control(settings, root, config)
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
        extractor = VGGTJointExtractor(
            feature_components=settings.feature_components,
            model_id=settings.model_id,
            image_size=settings.image_size,
            layer_index=settings.layer_index,
            expected_feature_dim=settings.expected_feature_dim,
            device=settings.device,
            model_cache_root=settings.model_cache_root,
        )
    model = JointHistoryGainModel(
        extractor,
        settings.expected_feature_dim,
        hidden_dim=settings.hidden_dim,
        dropout=settings.dropout,
        include_anchor_directions=settings.include_anchor_directions,
    )
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
    if joint_parameters != independent_parameters:
        raise ValueError("joint and independent trainable head capacities differ")
    context = initialize_run(config, root)
    preflight = _memory_preflight(model, histories["train"], settings)
    write_json(preflight, context.metrics_dir / "memory_preflight.json")
    active_logger.info(
        "Training joint Phase 3 control on %d histories; %d trainable parameters",
        len(histories["train"]),
        joint_parameters,
    )
    training = dict(settings.training)
    accumulation = int(training.pop("gradient_accumulation_steps"))
    fit = fit_history_model(
        model,
        histories["train"],
        histories["val"],
        device=settings.device,
        seed=seed,
        ndcg_k=settings.ndcg_k,
        logger=active_logger,
        gradient_accumulation_steps=accumulation,
        **training,
    )
    validation = evaluate_history_model(
        model,
        histories["val"],
        batch_size=settings.evaluation_batch_size,
        huber_delta=settings.training["huber_delta"],
        ranking_weight=settings.training["ranking_weight"],
        ranking_margin=settings.training["ranking_margin"],
        ndcg_k=settings.ndcg_k,
        device=settings.device,
    )
    checkpoint_path = context.checkpoint_dir / "best.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": "phase3_joint_history_gain",
        "model": {
            "feature_dim": settings.expected_feature_dim,
            "hidden_dim": settings.hidden_dim,
            "num_anchors": CANONICAL_ANCHOR_COUNT,
            "dropout": settings.dropout,
            "include_anchor_directions": settings.include_anchor_directions,
            "aggregation": model.aggregation,
            "backbone_history_mode": model.backbone_history_mode,
            "anchor_ordering": CANONICAL_ORDERING,
        },
        "backbone": {
            "name": "vggt",
            "model_id": settings.model_id,
            "image_size": settings.image_size,
            "layer_index": settings.layer_index,
            "feature_components": list(settings.feature_components),
            "history_mode": "joint_multiview",
            "padding_strategy": "group_by_real_history_length",
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
    return context.run_dir


def load_joint_history_checkpoint(
    path: str | Path,
    *,
    extractor: object | None = None,
    device: str | torch.device = "cpu",
    model_cache_root: str | Path | None = None,
) -> tuple[JointHistoryGainModel, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("model_type") != "phase3_joint_history_gain":
        raise ValueError("unsupported Phase 3 joint checkpoint")
    fields = payload["model"]
    backbone = payload["backbone"]
    if (
        fields.get("aggregation") != "masked_mean"
        or fields.get("backbone_history_mode") != "joint_multiview"
        or fields.get("anchor_ordering") != CANONICAL_ORDERING
        or backbone.get("history_mode") != "joint_multiview"
        or backbone.get("padding_strategy") != "group_by_real_history_length"
    ):
        raise ValueError("checkpoint does not describe the Step 17 joint contract")
    if extractor is None:
        extractor = VGGTJointExtractor(
            feature_components=backbone["feature_components"],
            model_id=backbone["model_id"],
            image_size=int(backbone["image_size"]),
            layer_index=int(backbone["layer_index"]),
            expected_feature_dim=int(fields["feature_dim"]),
            device=device,
            model_cache_root=model_cache_root,
        )
    model = JointHistoryGainModel(
        extractor,
        int(fields["feature_dim"]),
        hidden_dim=int(fields["hidden_dim"]),
        num_anchors=int(fields["num_anchors"]),
        dropout=float(fields["dropout"]),
        include_anchor_directions=bool(fields["include_anchor_directions"]),
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
        "same_vggt_model_id_and_feature_selection": True,
        "checkpoint_identity_note": (
            "The Step 16 feature cache records facebook/VGGT-1B but no weight "
            "checksum; model ID, preprocessing, layer, and selected component "
            "are matched exactly."
        ),
        "independent_feature_cache": str(cache_path),
        "independent_feature_cache_sha256": _sha256(cache_path),
        "independent_microbatch_size": int(independent.training["batch_size"]),
        "joint_microbatch_size": int(settings.training["batch_size"]),
        "joint_gradient_accumulation_steps": int(
            settings.training["gradient_accumulation_steps"]
        ),
        "effective_batch_size": effective_batch,
        "optimizer": "AdamW",
    }


def _memory_preflight(
    model: JointHistoryGainModel,
    dataset: HistoryDataset,
    settings: Phase3JointSettings,
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
        measurements.append({
            "history_length": length,
            "batch_size": 1,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "prediction_shape": list(prediction.shape),
        })
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


__all__ = [
    "Phase3JointSettings",
    "load_joint_history_checkpoint",
    "parse_phase3_joint_settings",
    "run_phase3_joint",
]
