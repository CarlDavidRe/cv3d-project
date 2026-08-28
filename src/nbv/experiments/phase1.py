"""One-command Phase 1 frozen-feature probe sweep."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nbv.data import NUMDataset
from nbv.features import (
    CachedFeatureDataset,
    FeatureCacheError,
    create_feature_extractor,
    feature_cache_path,
    load_feature_cache,
    save_feature_cache,
    select_feature_components,
    validate_feature_selection,
)
from nbv.models import LightweightProbeHead, count_trainable_parameters
from nbv.reproducibility import RunContext, initialize_run, seed_everything
from nbv.training import evaluate_phase1_probe, fit_phase1_probe


_SPLITS = ("train", "val", "test")
_LOWER_IS_BETTER_TARGETS = frozenset({"PSNR", "SSIM"})
_HIGHER_IS_BETTER_TARGETS = frozenset({"MSE", "LPIPS"})


@dataclass(frozen=True, slots=True)
class ProbeVariant:
    """One backbone and ordered fixed-size feature selection."""

    name: str
    backbone: str
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase1SweepSettings:
    """Validated fields consumed by the Phase 1 runner."""

    variants: tuple[ProbeVariant, ...]
    data_root: Path
    split_manifest: Path
    cache_root: Path
    target_name: str
    target_direction: str
    num_anchors: int
    device: str
    extraction_batch_sizes: Mapping[str, int]
    cache_dtype: torch.dtype
    rebuild_cache: bool
    mask_source_view: bool
    max_samples_per_split: int | None
    hidden_dim: int
    dropout: float
    training: Mapping[str, Any]
    evaluation_batch_size: int
    ndcg_k: int


def run_phase1_sweep(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> Path:
    """Cache, train, evaluate, and compare every configured probe variant."""

    root = Path(repository_root).resolve()
    settings = parse_phase1_sweep_settings(config, root)
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    seed_everything(seed, bool(experiment["deterministic"]))
    context = initialize_run(config, root)
    active_logger = logger or logging.getLogger(__name__)

    caches = _prepare_all_caches(
        settings,
        config,
        logger=active_logger,
    )
    comparison_rows: list[dict[str, Any]] = []
    for variant in settings.variants:
        seed_everything(seed, bool(experiment["deterministic"]))
        active_logger.info("Training Phase 1 variant %s", variant.name)
        split_caches = {
            split: load_feature_cache(
                caches[(variant.name, split)][0],
                expected_metadata=caches[(variant.name, split)][1],
            )
            for split in _SPLITS
        }
        result = _train_and_evaluate_variant(
            variant,
            split_caches,
            settings,
            context,
            seed=seed,
        )
        comparison_rows.append(result)
        active_logger.info(
            "%s: regret=%.4f Spearman=%s NDCG@%d=%.4f",
            variant.name,
            result["normalized_regret_mean"],
            (
                "undefined"
                if result["spearman_mean"] is None
                else f"{result['spearman_mean']:.4f}"
            ),
            settings.ndcg_k,
            result[f"ndcg_at_{settings.ndcg_k}_mean"],
        )
        del split_caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_comparison(context, comparison_rows, ndcg_k=settings.ndcg_k)
    return context.run_dir


def parse_phase1_sweep_settings(
    config: Mapping[str, Any], repository_root: str | Path
) -> Phase1SweepSettings:
    """Validate the Step 7 fields beyond the common config schema."""

    root = Path(repository_root).resolve()
    paths = _mapping(config, "paths")
    phase1 = _mapping(config, "phase1")
    dataset = _mapping(phase1, "num_dataset")
    probe = _mapping(config, "probe")
    cache = _mapping(probe, "feature_cache")
    head = _mapping(probe, "head")
    training = _mapping(probe, "training")
    evaluation = _mapping(probe, "evaluation")

    variants_raw = probe.get("variants")
    if isinstance(variants_raw, (str, bytes)) or not isinstance(
        variants_raw, Sequence
    ):
        raise TypeError("probe.variants must be a list")
    variants: list[ProbeVariant] = []
    names: set[str] = set()
    for index, raw in enumerate(variants_raw):
        if not isinstance(raw, Mapping):
            raise TypeError(f"probe.variants[{index}] must be a mapping")
        name = _safe_name(raw.get("name"), f"probe.variants[{index}].name")
        if name in names:
            raise ValueError(f"Duplicate probe variant name {name!r}")
        backbone = raw.get("backbone")
        if backbone not in ("imagenet_vit", "dinov2", "vggt"):
            raise ValueError(
                f"probe.variants[{index}].backbone must be imagenet_vit, "
                "dinov2, or vggt"
            )
        components = validate_feature_selection(
            backbone, raw.get("components")
        )
        variants.append(ProbeVariant(name, backbone, components))
        names.add(name)
    if not variants:
        raise ValueError("probe.variants must contain at least one variant")

    target_name = _nonempty_string(dataset.get("target_name"), "target_name")
    target_direction = _resolve_target_direction(
        target_name, probe.get("target_direction", "auto")
    )
    dtype_name = cache.get("dtype", "float32")
    cache_dtypes = {"float16": torch.float16, "float32": torch.float32}
    if dtype_name not in cache_dtypes:
        raise ValueError("probe.feature_cache.dtype must be float16 or float32")
    max_samples = probe.get("max_samples_per_split")
    if max_samples is not None:
        max_samples = _positive_integer(
            max_samples, "probe.max_samples_per_split"
        )
    dropout = _number(head.get("dropout", 0.0), "probe.head.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("probe.head.dropout must be in [0, 1)")
    if not isinstance(cache.get("rebuild"), bool):
        raise TypeError("probe.feature_cache.rebuild must be boolean")
    if not isinstance(probe.get("mask_source_view"), bool):
        raise TypeError("probe.mask_source_view must be boolean")

    validated_training = {
        "epochs": _positive_integer(training.get("epochs"), "training.epochs"),
        "batch_size": _positive_integer(
            training.get("batch_size"), "training.batch_size"
        ),
        "learning_rate": _positive_number(
            training.get("learning_rate"), "training.learning_rate"
        ),
        "weight_decay": _nonnegative_number(
            training.get("weight_decay", 0.0), "training.weight_decay"
        ),
        "huber_delta": _positive_number(
            training.get("huber_delta", 1.0), "training.huber_delta"
        ),
        "ranking_weight": _nonnegative_number(
            training.get("ranking_weight", 0.0), "training.ranking_weight"
        ),
        "ranking_margin": _nonnegative_number(
            training.get("ranking_margin", 0.0), "training.ranking_margin"
        ),
        "patience": training.get("patience"),
    }
    if validated_training["patience"] is not None:
        validated_training["patience"] = _positive_integer(
            validated_training["patience"], "training.patience"
        )

    return Phase1SweepSettings(
        variants=tuple(variants),
        data_root=_rooted_path(root, paths.get("data_root"), "paths.data_root"),
        split_manifest=_rooted_path(
            root,
            dataset.get("split_manifest"),
            "phase1.num_dataset.split_manifest",
        ),
        cache_root=_rooted_path(
            root, cache.get("root"), "probe.feature_cache.root"
        ),
        target_name=target_name,
        target_direction=target_direction,
        num_anchors=_positive_integer(
            phase1.get("num_anchors"), "phase1.num_anchors"
        ),
        device=_nonempty_string(probe.get("device", "auto"), "probe.device"),
        extraction_batch_sizes=_extraction_batch_sizes(
            probe.get("extraction_batch_size"), variants
        ),
        cache_dtype=cache_dtypes[dtype_name],
        rebuild_cache=cache["rebuild"],
        mask_source_view=probe["mask_source_view"],
        max_samples_per_split=max_samples,
        hidden_dim=_positive_integer(
            head.get("hidden_dim"), "probe.head.hidden_dim"
        ),
        dropout=dropout,
        training=validated_training,
        evaluation_batch_size=_positive_integer(
            evaluation.get("batch_size"), "probe.evaluation.batch_size"
        ),
        ndcg_k=_positive_integer(
            evaluation.get("ndcg_k", 5), "probe.evaluation.ndcg_k"
        ),
    )


def extract_variant_caches(
    dataset: NUMDataset,
    extractor: Any,
    variants: Sequence[ProbeVariant],
    metadata_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    extraction_batch_size: int,
    num_anchors: int,
    mask_source_view: bool,
    cache_dtype: torch.dtype,
    max_samples: int | None = None,
) -> dict[str, CachedFeatureDataset]:
    """Extract all missing variants for one backbone in a single pass."""

    if not variants:
        raise ValueError("variants must not be empty")
    sample_count = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    feature_batches: dict[str, list[torch.Tensor]] = {
        variant.name: [] for variant in variants
    }
    target_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    source_anchor_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    for start in range(0, sample_count, extraction_batch_size):
        samples = [
            dataset[index]
            for index in range(
                start, min(start + extraction_batch_size, sample_count)
            )
        ]
        images = np.stack([sample.image for sample in samples])
        frozen = extractor.extract(images)
        for variant in variants:
            selected = select_feature_components(
                frozen, variant.components
            ).to(device="cpu", dtype=cache_dtype)
            feature_batches[variant.name].append(selected)
        targets = torch.from_numpy(
            np.stack([sample.target_map for sample in samples])
        ).float()
        if targets.shape[1] != num_anchors:
            raise ValueError(
                f"NUM target width {targets.shape[1]} does not match "
                f"phase1.num_anchors={num_anchors}"
            )
        target_batches.append(targets)
        valid_mask = torch.ones(len(samples), num_anchors, dtype=torch.bool)
        if mask_source_view:
            # NUM maps are source-relative; local anchor zero is the input view.
            valid_mask[:, 0] = False
        mask_batches.append(valid_mask)
        source_anchor_batches.append(
            torch.tensor(
                [sample.source_anchor_id for sample in samples],
                dtype=torch.int64,
            )
        )
        sample_ids.extend(sample.record.sample_id for sample in samples)
        del frozen

    targets = torch.cat(target_batches)
    valid_mask = torch.cat(mask_batches)
    source_anchor_ids = torch.cat(source_anchor_batches)
    return {
        variant.name: CachedFeatureDataset(
            features=torch.cat(feature_batches[variant.name]),
            targets=targets,
            valid_mask=valid_mask,
            sample_ids=tuple(sample_ids),
            source_anchor_ids=source_anchor_ids,
            metadata=dict(metadata_by_variant[variant.name]),
        )
        for variant in variants
    }


def _prepare_all_caches(
    settings: Phase1SweepSettings,
    config: Mapping[str, Any],
    *,
    logger: logging.Logger,
) -> dict[tuple[str, str], tuple[Path, Mapping[str, Any]]]:
    cache_requests: dict[
        tuple[str, str], tuple[Path, Mapping[str, Any]]
    ] = {}
    datasets = {
        split: NUMDataset(
            settings.data_root,
            split=split,
            split_manifest=settings.split_manifest,
            target_name=settings.target_name,
        )
        for split in _SPLITS
    }
    by_backbone: dict[str, list[ProbeVariant]] = {}
    for variant in settings.variants:
        by_backbone.setdefault(variant.backbone, []).append(variant)

    for backbone, variants in by_backbone.items():
        missing: dict[str, list[ProbeVariant]] = {}
        for split, dataset in datasets.items():
            for variant in variants:
                metadata = _cache_metadata(
                    variant,
                    split,
                    dataset,
                    settings,
                    config,
                )
                path = feature_cache_path(
                    settings.cache_root,
                    backbone=backbone,
                    variant=variant.name,
                    split=split,
                    metadata=metadata,
                )
                cache_requests[(variant.name, split)] = (path, metadata)
                if settings.rebuild_cache or not _cache_matches(path, metadata):
                    missing.setdefault(split, []).append(variant)
        if not missing:
            logger.info("Reusing all %s feature caches", backbone)
            continue

        logger.info("Loading frozen %s for missing feature caches", backbone)
        extractor = create_feature_extractor(
            backbone,
            **_extractor_kwargs(backbone, settings, config),
        )
        started = time.perf_counter()
        for split, missing_variants in missing.items():
            logger.info(
                "Extracting %s split for %s: %s",
                split,
                backbone,
                ", ".join(variant.name for variant in missing_variants),
            )
            metadata_by_variant = {
                variant.name: cache_requests[(variant.name, split)][1]
                for variant in missing_variants
            }
            built = extract_variant_caches(
                datasets[split],
                extractor,
                missing_variants,
                metadata_by_variant,
                extraction_batch_size=settings.extraction_batch_sizes[backbone],
                num_anchors=settings.num_anchors,
                mask_source_view=settings.mask_source_view,
                cache_dtype=settings.cache_dtype,
                max_samples=settings.max_samples_per_split,
            )
            for variant_name, cached_dataset in built.items():
                destination = cache_requests[(variant_name, split)][0]
                save_feature_cache(cached_dataset, destination)
                logger.info("Saved feature cache %s", destination)
            del built
        logger.info(
            "Finished %s feature extraction in %.1f seconds",
            backbone,
            time.perf_counter() - started,
        )
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache_requests


def _train_and_evaluate_variant(
    variant: ProbeVariant,
    splits: Mapping[str, CachedFeatureDataset],
    settings: Phase1SweepSettings,
    context: RunContext,
    *,
    seed: int,
) -> dict[str, Any]:
    train = splits["train"]
    validation = splits["val"]
    test = splits["test"]
    head = LightweightProbeHead(
        int(train.features.shape[1]),
        hidden_dim=settings.hidden_dim,
        num_anchors=settings.num_anchors,
        dropout=settings.dropout,
    )
    fit = fit_phase1_probe(
        head,
        train,
        validation,
        device=settings.device,
        seed=seed,
        **settings.training,
    )
    evaluation_kwargs = {
        "batch_size": settings.evaluation_batch_size,
        "target_direction": settings.target_direction,
        "huber_delta": settings.training["huber_delta"],
        "ranking_weight": settings.training["ranking_weight"],
        "ranking_margin": settings.training["ranking_margin"],
        "ndcg_k": settings.ndcg_k,
        "device": settings.device,
    }
    validation_result = evaluate_phase1_probe(
        head, validation, **evaluation_kwargs
    )
    test_result = evaluate_phase1_probe(head, test, **evaluation_kwargs)

    variant_dir = context.run_dir / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "head_state_dict": {
            name: value.detach().cpu()
            for name, value in head.state_dict().items()
        },
        "head": {
            "input_dim": head.input_dim,
            "hidden_dim": head.hidden_dim,
            "num_anchors": head.num_anchors,
            "dropout": head.dropout,
        },
        "variant": {
            "name": variant.name,
            "backbone": variant.backbone,
            "feature_components": list(variant.components),
        },
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.best_validation_loss,
    }
    torch.save(checkpoint, variant_dir / "best.pt")
    (variant_dir / "training_history.json").write_text(
        json.dumps(list(fit.history), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_per_sample_csv(
        variant_dir / "test_per_sample.csv", test_result.per_sample
    )

    trainable_parameters = count_trainable_parameters(head)
    summary = {
        "variant": variant.name,
        "backbone": variant.backbone,
        "feature": "+".join(variant.components),
        "feature_components": list(variant.components),
        "input_dim": head.input_dim,
        "trainable_parameters": trainable_parameters,
        "best_epoch": fit.best_epoch,
        "epochs_completed": fit.epochs_completed,
        "best_validation_loss": fit.best_validation_loss,
        "target_name": settings.target_name,
        "target_direction": settings.target_direction,
        "train_samples": len(train),
        "validation": dict(validation_result.summary),
        "test": dict(test_result.summary),
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "variant": variant.name,
        "backbone": variant.backbone,
        "feature": "+".join(variant.components),
        "input_dim": head.input_dim,
        "trainable_parameters": trainable_parameters,
        "best_epoch": fit.best_epoch,
        "huber_loss": test_result.summary["huber_loss"],
        "normalized_regret_mean": test_result.summary[
            "normalized_regret_mean"
        ],
        "spearman_mean": test_result.summary["spearman_mean"],
        f"ndcg_at_{settings.ndcg_k}_mean": test_result.summary[
            f"ndcg_at_{settings.ndcg_k}_mean"
        ],
    }


def _cache_metadata(
    variant: ProbeVariant,
    split: str,
    dataset: NUMDataset,
    settings: Phase1SweepSettings,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected_records = dataset.records
    if settings.max_samples_per_split is not None:
        selected_records = selected_records[: settings.max_samples_per_split]
    sample_ids = [record.sample_id for record in selected_records]
    return {
        "cache_schema": 1,
        "backbone": variant.backbone,
        "variant": variant.name,
        "feature_components": list(variant.components),
        "feature_pooling": _pooling_metadata(variant.components),
        "backbone_configuration": _backbone_metadata(
            variant.backbone, config
        ),
        "split": split,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "split_manifest_sha256": _file_sha256(settings.split_manifest),
        "target_name": settings.target_name,
        "num_anchors": settings.num_anchors,
        "mask_source_view": settings.mask_source_view,
        "cache_dtype": str(settings.cache_dtype).removeprefix("torch."),
        "extraction_precision": _extraction_precision(
            variant.backbone, settings.device
        ),
        "history_mode": "single_image",
    }


def _backbone_metadata(
    backbone: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    probe = _mapping(config, "probe")
    model = dict(_mapping(probe, backbone))
    if backbone == "imagenet_vit":
        return {
            **model,
            "model_name": "torchvision_vit_b_16",
            "input_resolution": [224, 224],
            "preprocessing": "resize_short_256_center_crop_224_imagenet_norm",
            "layer": "final",
        }
    if backbone == "dinov2":
        return {
            **model,
            "input_resolution": [224, 224],
            "preprocessing": "resize_short_256_center_crop_224_imagenet_norm",
            "layer": "final",
        }
    image_size = int(model.get("image_size", 518))
    return {
        **model,
        "input_resolution": [image_size, image_size],
        "preprocessing": "square_resize_rgb_0_1",
        "history_mode": "single_image",
    }


def _pooling_metadata(components: Sequence[str]) -> dict[str, str]:
    methods = {
        "pooled_patch": "mean",
        "max_pooled_patch": "max",
        "cls_token": "none",
        "pooled_camera": "mean",
        "pooled_register": "mean",
    }
    return {component: methods[component] for component in components}


def _extraction_precision(backbone: str, device: str) -> str:
    if backbone != "vggt":
        return "float32"
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        return "float32"
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    major, _ = torch.cuda.get_device_capability(resolved)
    return "bfloat16" if major >= 8 else "float16"


def _extractor_kwargs(
    backbone: str,
    settings: Phase1SweepSettings,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_config = dict(_mapping(_mapping(config, "probe"), backbone))
    allowed = {
        "imagenet_vit": {"pretrained"},
        "dinov2": {"model_name", "repo_or_dir", "source", "pretrained"},
        "vggt": {"model_id", "image_size", "layer_index"},
    }[backbone]
    unexpected = set(model_config) - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected probe.{backbone} fields: {sorted(unexpected)}"
        )
    return {"device": settings.device, **model_config}


def _cache_matches(path: Path, metadata: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        load_feature_cache(path, expected_metadata=metadata)
    except FeatureCacheError:
        return False
    return True


def _write_comparison(
    context: RunContext,
    rows: Sequence[Mapping[str, Any]],
    *,
    ndcg_k: int,
) -> None:
    columns = (
        "variant",
        "backbone",
        "feature",
        "input_dim",
        "trainable_parameters",
        "best_epoch",
        "huber_loss",
        "normalized_regret_mean",
        "spearman_mean",
        f"ndcg_at_{ndcg_k}_mean",
    )
    csv_path = context.metrics_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in columns} for row in rows
        )
    (context.metrics_dir / "comparison.json").write_text(
        json.dumps(
            {"schema_version": 1, "results": list(rows)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_format_table_value(row.get(column)) for column in columns)
        + " |"
        for row in rows
    ]
    (context.metrics_dir / "comparison.md").write_text(
        "\n".join([header, separator, *body]) + "\n", encoding="utf-8"
    )


def _write_per_sample_csv(
    destination: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        raise ValueError("per-sample rows must not be empty")
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_target_direction(target_name: str, direction: object) -> str:
    if direction in ("higher", "lower"):
        return str(direction)
    if direction != "auto":
        raise ValueError("probe.target_direction must be auto, higher, or lower")
    normalized = target_name.upper()
    if normalized in _LOWER_IS_BETTER_TARGETS:
        return "lower"
    if normalized in _HIGHER_IS_BETTER_TARGETS:
        return "higher"
    raise ValueError(
        f"No automatic target direction is known for {target_name!r}; "
        "set probe.target_direction explicitly"
    )


def _extraction_batch_sizes(
    value: object, variants: Sequence[ProbeVariant]
) -> dict[str, int]:
    backbones = {variant.backbone for variant in variants}
    if isinstance(value, int) and not isinstance(value, bool):
        batch_size = _positive_integer(value, "probe.extraction_batch_size")
        return {backbone: batch_size for backbone in backbones}
    if not isinstance(value, Mapping):
        raise TypeError(
            "probe.extraction_batch_size must be a positive integer or mapping"
        )
    missing = backbones - set(value)
    if missing:
        raise ValueError(
            "probe.extraction_batch_size is missing configured backbones: "
            f"{sorted(missing)}"
        )
    unexpected = set(value) - {"imagenet_vit", "dinov2", "vggt"}
    if unexpected:
        raise ValueError(
            "probe.extraction_batch_size has unknown backbones: "
            f"{sorted(unexpected)}"
        )
    return {
        backbone: _positive_integer(
            value[backbone], f"probe.extraction_batch_size.{backbone}"
        )
        for backbone in backbones
    }


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _rooted_path(root: Path, value: object, name: str) -> Path:
    path = Path(_nonempty_string(value, name))
    return path if path.is_absolute() else root / path


def _safe_name(value: object, name: str) -> str:
    normalized = _nonempty_string(value, name)
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )
    if any(character not in allowed for character in normalized):
        raise ValueError(
            f"{name} may contain only letters, numbers, underscores, and hyphens"
        )
    return normalized


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_number(value: object, name: str) -> float:
    number = _number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return number


def _nonnegative_number(value: object, name: str) -> float:
    number = _number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
