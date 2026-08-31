#!/usr/bin/env python3
"""Overfit a lightweight frozen-feature probe on a deterministic tiny subset."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.data import NUMDataset  # noqa: E402
from nbv.features import (  # noqa: E402
    create_feature_extractor,
    select_feature_components,
    validate_feature_selection,
)
from nbv.logging_utils import configure_logging  # noqa: E402
from nbv.models import LightweightProbeHead, count_trainable_parameters  # noqa: E402
from nbv.reproducibility import initialize_run, seed_everything  # noqa: E402
from nbv.training import fit_probe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/phase1_probe_tiny.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field; may be repeated.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        settings = _probe_settings(config)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    experiment = config["experiment"]
    seed = int(experiment["seed"])
    seed_everything(seed, bool(experiment["deterministic"]))
    context = initialize_run(config, REPOSITORY_ROOT)
    logger = configure_logging(context.log_path, args.verbose)

    dataset_config = config["phase1"]["num_dataset"]
    dataset = NUMDataset(
        _repository_path(config["paths"]["data_root"]),
        split="train",
        split_manifest=_repository_path(dataset_config["split_manifest"]),
        target_name=dataset_config["target_name"],
    )
    sample_count = min(settings["max_samples"], len(dataset))
    logger.info(
        "Extracting %d frozen %s feature vectors once using %s",
        sample_count,
        settings["backbone"],
        "+".join(settings["feature_components"]),
    )
    extractor_kwargs: dict[str, Any] = {
        "device": settings["device"],
        "model_cache_root": _repository_path(
            config["paths"]["model_cache_root"]
        ),
    }
    if settings["backbone"] == "imagenet_vit":
        extractor_kwargs["pretrained"] = config["probe"]["imagenet_vit"][
            "pretrained"
        ]
    elif settings["backbone"] == "dinov2":
        extractor_kwargs["model_name"] = config["probe"]["dinov2"]["model_name"]
    elif settings["backbone"] == "vggt":
        extractor_kwargs.update(config["probe"]["vggt"])
    extractor = create_feature_extractor(settings["backbone"], **extractor_kwargs)
    feature_device = str(extractor.device)

    feature_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    extraction_batch_size = settings["extraction_batch_size"]
    for start in range(0, sample_count, extraction_batch_size):
        samples = [
            dataset[index]
            for index in range(start, min(start + extraction_batch_size, sample_count))
        ]
        images = np.stack([sample.image for sample in samples])
        frozen = extractor.extract(images)
        selected = select_feature_components(
            frozen, settings["feature_components"]
        ).to(device="cpu", dtype=torch.float32)
        feature_batches.append(selected)
        target_batches.append(
            torch.from_numpy(np.stack([sample.target_map for sample in samples]))
        )
        valid_mask = torch.ones(
            len(samples), settings["num_anchors"], dtype=torch.bool
        )
        if settings["mask_source_view"]:
            # Official NUM target maps are source-relative: local anchor zero
            # is the acquired input view, regardless of its global anchor ID.
            valid_mask[:, 0] = False
        mask_batches.append(valid_mask)
        sample_ids.extend(sample.record.sample_id for sample in samples)
        del frozen

    features = torch.cat(feature_batches)
    targets = torch.cat(target_batches).to(dtype=torch.float32)
    valid_mask = torch.cat(mask_batches)
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    head = LightweightProbeHead(
        int(features.shape[1]),
        hidden_dim=settings["hidden_dim"],
        num_anchors=settings["num_anchors"],
        dropout=settings["dropout"],
    )
    trainable_parameters = count_trainable_parameters(head)
    logger.info(
        "Training only the probe head (%d trainable parameters) on %d samples",
        trainable_parameters,
        sample_count,
    )
    training = config["probe"]["training"]
    result = fit_probe(
        head,
        features,
        targets,
        valid_mask,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        huber_delta=training["huber_delta"],
        device=feature_device,
        seed=seed,
    )
    required_reduction = float(training["required_relative_loss_reduction"])
    overfit_succeeded = result.relative_loss_reduction >= required_reduction
    summary = {
        "backbone": settings["backbone"],
        "best_epoch": result.best_epoch,
        "best_loss": result.best_loss,
        "epochs_completed": result.epochs_completed,
        "feature": "+".join(settings["feature_components"]),
        "feature_components": list(settings["feature_components"]),
        "feature_device": feature_device,
        "final_loss": result.final_loss,
        "initial_loss": result.initial_loss,
        "mask_source_view": settings["mask_source_view"],
        "masked_local_anchor_id": 0 if settings["mask_source_view"] else None,
        "num_samples": sample_count,
        "overfit_succeeded": overfit_succeeded,
        "relative_loss_reduction": result.relative_loss_reduction,
        "required_relative_loss_reduction": required_reduction,
        "trainable_parameters": trainable_parameters,
    }
    checkpoint_path = context.checkpoint_dir / "best.pt"
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "head": {
                "input_dim": head.input_dim,
                "hidden_dim": head.hidden_dim,
                "num_anchors": head.num_anchors,
                "dropout": head.dropout,
            },
            "backbone": settings["backbone"],
            "feature": "+".join(settings["feature_components"]),
            "feature_components": list(settings["feature_components"]),
            "sample_ids": sample_ids,
            "summary": summary,
        },
        checkpoint_path,
    )
    summary_path = context.metrics_dir / "tiny_overfit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Tiny-overfit %s: loss %.6g -> %.6g (%.2f%% reduction)",
        "passed" if overfit_succeeded else "failed",
        result.initial_loss,
        result.best_loss,
        100.0 * result.relative_loss_reduction,
    )
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Metrics: %s", summary_path)
    return 0 if overfit_succeeded else 1


def _probe_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    phase1 = _mapping(config, "phase1")
    probe = _mapping(config, "probe")
    head = _mapping(probe, "head")
    training = _mapping(probe, "training")
    backbone = probe.get("backbone")
    if backbone not in ("imagenet_vit", "dinov2", "vggt"):
        raise ValueError("probe.backbone must be imagenet_vit, dinov2, or vggt")
    feature_selection = _mapping(probe, "feature_selection")
    selections = {
        name: validate_feature_selection(name, feature_selection.get(name))
        for name in ("imagenet_vit", "dinov2", "vggt")
    }
    imagenet_vit = _mapping(probe, "imagenet_vit")
    if not isinstance(imagenet_vit.get("pretrained"), bool):
        raise TypeError("probe.imagenet_vit.pretrained must be boolean")
    num_anchors = _positive_integer(phase1.get("num_anchors"), "num_anchors")
    max_samples = _positive_integer(probe.get("max_samples"), "max_samples")
    extraction_batch_size = _positive_integer(
        probe.get("extraction_batch_size"), "extraction_batch_size"
    )
    hidden_dim = _positive_integer(head.get("hidden_dim"), "hidden_dim")
    if not isinstance(probe.get("mask_source_view"), bool):
        raise TypeError("probe.mask_source_view must be boolean")
    if training.get("epochs", 0) < 1 or training.get("batch_size", 0) < 1:
        raise ValueError("probe training epochs and batch_size must be positive")
    required = training.get("required_relative_loss_reduction")
    if not isinstance(required, (int, float)) or isinstance(required, bool):
        raise TypeError("required_relative_loss_reduction must be numeric")
    if not 0.0 <= float(required) <= 1.0:
        raise ValueError("required_relative_loss_reduction must be in [0, 1]")
    return {
        "backbone": backbone,
        "device": probe.get("device", "auto"),
        "dropout": head.get("dropout", 0.0),
        "extraction_batch_size": extraction_batch_size,
        "feature_components": selections[backbone],
        "hidden_dim": hidden_dim,
        "mask_source_view": probe["mask_source_view"],
        "max_samples": max_samples,
        "num_anchors": num_anchors,
    }


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
