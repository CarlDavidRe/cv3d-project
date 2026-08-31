#!/usr/bin/env python3
"""Generate prediction-only figures from a completed Phase 1 experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.config import ConfigError, load_config  # noqa: E402
from nbv.data import NUMDataset, NUMSample  # noqa: E402
from nbv.features import (  # noqa: E402
    FeatureCacheError,
    create_feature_extractor,
    load_feature_cache,
    select_feature_components,
)
from nbv.models import (  # noqa: E402
    LightweightProbeHead,
    PUNUPNet,
    create_pun_transform,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
)
from nbv.visualization import (  # noqa: E402
    Phase1PredictionDiagnostics,
    write_phase1_prediction_svg,
)


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """One generated variant figure and its prediction diagnostics."""

    variant_name: str
    output: Path
    diagnostics: Phase1PredictionDiagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        type=Path,
        help="Completed experiment directory containing config.yaml and variants/.",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        help=(
            "Saved variant to visualize. Repeat to select multiple variants; "
            "omit to generate one plot for every saved variant."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split used only for the qualitative figure.",
    )
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument(
        "--index",
        type=int,
        default=0,
        help="Deterministic split index (default: 0).",
    )
    sample.add_argument(
        "--sample-id",
        help="Stable category/object/source-anchor ID; overrides --index.",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output",
        type=Path,
        help=(
            "SVG destination when exactly one variant is selected. Defaults to "
            "<experiment>/figures/predictions/<split>/<category>/<object>/"
            "<view>/<variant>.svg."
        ),
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for generated SVGs. Defaults to "
            "<experiment>/figures/predictions/."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for optional one-sample feature extraction.",
    )
    parser.add_argument(
        "--extract-missing-features",
        action="store_true",
        help=(
            "Extract one sample in memory if no compatible feature cache exists. "
            "Raw RGB is always extracted in memory; pretrained backbones require "
            "this flag and may need their model checkpoint. No feature cache is "
            "written."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = visualize_experiment_variants(
            args.experiment,
            variant_names=args.variants,
            split=args.split,
            index=args.index,
            sample_id=args.sample_id,
            output=args.output,
            output_dir=args.output_dir,
            device=args.device,
            extract_missing_features=args.extract_missing_features,
        )
    except (ConfigError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Visualization error: {exc}", file=sys.stderr)
        return 1
    for result in results:
        diagnostics = result.diagnostics
        print(f"Figure [{result.variant_name}]: {result.output}")
        print(
            "Prediction: "
            f"top={diagnostics.predicted_anchor_id}, "
            f"target_top={diagnostics.target_anchor_id}, "
            f"regret={diagnostics.normalized_regret:.4f}, "
            f"Spearman={diagnostics.spearman}, "
            f"NDCG={diagnostics.ndcg_at_k:.4f}"
        )
    print(f"Generated {len(results)} prediction figure(s).")
    return 0


def visualize_experiment(
    experiment: str | Path,
    *,
    variant_name: str,
    split: str = "val",
    index: int = 0,
    sample_id: str | None = None,
    output: str | Path | None = None,
    device: str = "auto",
    extract_missing_features: bool = False,
) -> tuple[Path, Phase1PredictionDiagnostics]:
    """Load one saved prediction head and write only its qualitative SVG."""

    result = visualize_experiment_variants(
        experiment,
        variant_names=(variant_name,),
        split=split,
        index=index,
        sample_id=sample_id,
        output=output,
        device=device,
        extract_missing_features=extract_missing_features,
    )[0]
    return result.output, result.diagnostics


def visualize_experiment_variants(
    experiment: str | Path,
    *,
    variant_names: Sequence[str] | None = None,
    split: str = "val",
    index: int = 0,
    sample_id: str | None = None,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    extract_missing_features: bool = False,
) -> tuple[VisualizationResult, ...]:
    """Generate one qualitative SVG per variant in a sample-view directory."""

    experiment_dir = Path(experiment).resolve()
    if not experiment_dir.is_dir():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    config_path = experiment_dir / "config.yaml"
    config = load_config(config_path)
    dataset_config = _mapping(_mapping(config, "phase1"), "num_dataset")
    paths = _mapping(config, "paths")
    probe = _mapping(config, "probe")
    data_root = _rooted_path(paths.get("data_root"), "paths.data_root")
    split_manifest = _rooted_path(
        dataset_config.get("split_manifest"),
        "phase1.num_dataset.split_manifest",
    )
    target_name = _string(dataset_config.get("target_name"), "target_name")
    dataset = NUMDataset(
        data_root,
        split=split,
        split_manifest=split_manifest,
        target_name=target_name,
    )
    sample_index = _resolve_sample_index(dataset, index=index, sample_id=sample_id)
    selected = dataset[sample_index]

    discovered = _discover_variants(experiment_dir, probe)
    names = _select_variants(discovered, variant_names)
    if output is not None and len(names) != 1:
        raise ValueError("--output requires exactly one selected variant")
    figure_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else experiment_dir / "figures" / "predictions"
    )
    results: list[VisualizationResult] = []
    for variant_name in names:
        destination = (
            Path(output).resolve()
            if output is not None
            else _prediction_destination(
                figure_root,
                variant_name=variant_name,
                split=split,
                sample_id=selected.record.sample_id,
            )
        )
        diagnostics = _visualize_loaded_variant(
            experiment_dir,
            config,
            probe,
            selected,
            target_name=target_name,
            variant_name=variant_name,
            split=split,
            destination=destination,
            device=device,
            extract_missing_features=extract_missing_features,
        )
        results.append(VisualizationResult(variant_name, destination, diagnostics))
    return tuple(results)


def _visualize_loaded_variant(
    experiment_dir: Path,
    config: Mapping[str, Any],
    probe: Mapping[str, Any],
    selected: NUMSample,
    *,
    target_name: str,
    variant_name: str,
    split: str,
    destination: Path,
    device: str,
    extract_missing_features: bool,
) -> Phase1PredictionDiagnostics:
    variant_dir = experiment_dir / "variants" / variant_name
    checkpoint_path = variant_dir / "best.pt"
    summary_path = variant_dir / "summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise ValueError(
            f"Variant {variant_name!r} is missing best.pt or summary.json under "
            f"{variant_dir}"
        )
    summary = _read_json_mapping(summary_path)
    target_direction = summary.get("target_direction")
    if target_direction not in ("higher", "lower"):
        raise ValueError("Variant summary has invalid target_direction")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    prediction, feature_source = _predict(
        checkpoint,
        config,
        probe,
        variant_name=variant_name,
        split=split,
        sample_id=selected.record.sample_id,
        image=selected.image,
        device=device,
        extract_missing_features=extract_missing_features,
    )
    valid_mask = np.ones_like(selected.target_map, dtype=np.bool_)
    if bool(probe.get("mask_source_view", True)):
        valid_mask[0] = False
    ndcg_k = int(_mapping(probe, "evaluation").get("ndcg_k", 5))
    if destination.suffix.lower() != ".svg":
        raise ValueError("Visualization output must use the .svg extension")
    diagnostics = write_phase1_prediction_svg(
        selected.target_map,
        prediction,
        valid_mask,
        selected.record.image_path,
        destination,
        title="Phase 1 prediction versus ground truth",
        variant_name=variant_name,
        sample_id=selected.record.sample_id,
        split=split,
        target_name=target_name,
        target_direction=target_direction,
        source_global_anchor_id=selected.source_anchor_id,
        feature_source=feature_source,
        ndcg_k=ndcg_k,
    )
    return diagnostics


def _discover_variants(
    experiment_dir: Path, probe: Mapping[str, Any]
) -> tuple[str, ...]:
    """Find complete saved variants, preferring their configured order."""

    variants_root = experiment_dir / "variants"
    if not variants_root.is_dir():
        raise ValueError(f"Experiment has no variants directory: {variants_root}")
    saved = {
        path.name
        for path in variants_root.iterdir()
        if path.is_dir()
        and (path / "best.pt").is_file()
        and (path / "summary.json").is_file()
    }
    if not saved:
        raise ValueError(f"No complete saved variants found under {variants_root}")

    ordered: list[str] = []
    raw_baselines = probe.get("baselines", ())
    baselines = raw_baselines if isinstance(raw_baselines, list) else []
    raw_variants = probe.get("variants", ())
    variants = raw_variants if isinstance(raw_variants, list) else []
    configured_groups = (
        [
            entry
            for entry in baselines
            if isinstance(entry, Mapping) and entry.get("type") != "pun"
        ],
        variants,
        [
            entry
            for entry in baselines
            if isinstance(entry, Mapping) and entry.get("type") == "pun"
        ],
    )
    for entries in configured_groups:
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name in saved and name not in ordered:
                ordered.append(name)
    ordered.extend(sorted(saved - set(ordered)))
    return tuple(ordered)


def _select_variants(
    discovered: Sequence[str], requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return tuple(discovered)
    if not requested:
        raise ValueError("At least one --variant value is required")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate --variant values are not allowed")
    missing = [name for name in requested if name not in discovered]
    if missing:
        raise ValueError(
            "Requested variant(s) are not complete saved variants: "
            + ", ".join(repr(name) for name in missing)
        )
    return tuple(requested)


def _predict(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    variant_name: str,
    split: str,
    sample_id: str,
    image: np.ndarray,
    device: str,
    extract_missing_features: bool,
) -> tuple[np.ndarray, str]:
    prediction_map = checkpoint.get("prediction_map")
    if isinstance(prediction_map, torch.Tensor):
        return prediction_map.detach().cpu().float().numpy(), "fixed checkpoint map"

    model_config = checkpoint.get("model")
    if isinstance(model_config, Mapping) and model_config.get("type") == "pun_upnet":
        return _predict_pun(checkpoint, image=image, device=device)

    head_config = checkpoint.get("head")
    variant = checkpoint.get("variant")
    state = checkpoint.get("head_state_dict")
    if not isinstance(head_config, Mapping) or not isinstance(variant, Mapping):
        raise ValueError("Learned checkpoint is missing head or variant metadata")
    if not isinstance(state, Mapping):
        raise ValueError("Learned checkpoint is missing head_state_dict")
    if variant.get("name") != variant_name:
        raise ValueError("Checkpoint variant name does not match the request")
    backbone = _string(variant.get("backbone"), "checkpoint backbone")
    raw_components = variant.get("feature_components")
    if not isinstance(raw_components, list) or not all(
        isinstance(value, str) for value in raw_components
    ):
        raise ValueError("Checkpoint feature_components must be a string list")
    components = tuple(raw_components)
    input_dim = int(head_config["input_dim"])
    feature, feature_source = _load_or_extract_feature(
        config,
        probe,
        variant_name=variant_name,
        backbone=backbone,
        components=components,
        split=split,
        sample_id=sample_id,
        image=image,
        input_dim=input_dim,
        device=device,
        extract_missing_features=extract_missing_features,
    )
    head = LightweightProbeHead(
        input_dim,
        hidden_dim=int(head_config["hidden_dim"]),
        num_anchors=int(head_config["num_anchors"]),
        dropout=float(head_config["dropout"]),
    )
    head.load_state_dict(state)
    head.eval()
    with torch.inference_mode():
        prediction = head(feature.to(dtype=torch.float32)).squeeze(0)
    return prediction.detach().cpu().float().numpy(), feature_source


def _predict_pun(
    checkpoint: Mapping[str, Any],
    *,
    image: np.ndarray,
    device: str,
) -> tuple[np.ndarray, str]:
    """Load a complete saved UPNet without downloading pretrained weights."""

    model_config = checkpoint.get("model")
    preprocessing = checkpoint.get("preprocessing")
    official = checkpoint.get("official_checkpoint")
    if not isinstance(model_config, Mapping) or not isinstance(official, Mapping):
        raise ValueError("PUN checkpoint descriptor is missing model metadata")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("PUN checkpoint is missing preprocessing metadata")
    model = PUNUPNet(
        model_name=_string(model_config.get("model_name"), "PUN model_name"),
        num_anchors=int(model_config["num_anchors"]),
        pretrained=False,
    )
    official_path = ensure_pun_checkpoint(
        _string(official.get("path"), "official PUN checkpoint path"),
        expected_sha256=_string(official.get("sha256"), "official PUN SHA-256"),
        download_url=_string(official.get("download_url"), "official PUN URL"),
        download_if_missing=True,
    )
    load_pun_checkpoint(
        model,
        official_path,
        expected_sha256=_string(official.get("sha256"), "official PUN SHA-256"),
    )
    resolved_device = _resolve_device(device)
    model.to(resolved_device).eval()
    transform = create_pun_transform(preprocessing)
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("PUN visualization image must have shape [3, H, W]")
    pixels = np.clip(np.transpose(pixels, (1, 2, 0)) * 255.0, 0, 255)
    pil_image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="RGB")
    inputs = transform(pil_image).unsqueeze(0).to(resolved_device)
    with torch.inference_mode():
        prediction = model(inputs).squeeze(0)
    return (
        prediction.detach().cpu().float().numpy(),
        "saved official-style PUN UPNet checkpoint",
    )


def _load_or_extract_feature(
    config: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    variant_name: str,
    backbone: str,
    components: tuple[str, ...],
    split: str,
    sample_id: str,
    image: np.ndarray,
    input_dim: int,
    device: str,
    extract_missing_features: bool,
) -> tuple[torch.Tensor, str]:
    cache_config = _mapping(probe, "feature_cache")
    cache_root = _rooted_path(
        cache_config.get("root"), "probe.feature_cache.root"
    )
    pattern = cache_root / backbone / variant_name / f"{split}_*.pt"
    configured_backbone = _mapping(probe, backbone)
    dataset_config = _mapping(_mapping(config, "phase1"), "num_dataset")
    configured_target = dataset_config.get("target_name")
    for cache_path in sorted(pattern.parent.glob(pattern.name)):
        try:
            cache = load_feature_cache(cache_path)
        except FeatureCacheError:
            continue
        metadata = cache.metadata
        cached_backbone = metadata.get("backbone_configuration")
        model_matches = isinstance(cached_backbone, Mapping) and all(
            cached_backbone.get(key) == value
            for key, value in configured_backbone.items()
        )
        if (
            metadata.get("backbone") != backbone
            or metadata.get("variant") != variant_name
            or metadata.get("split") != split
            or metadata.get("target_name") != configured_target
            or metadata.get("mask_source_view")
            != bool(probe.get("mask_source_view", True))
            or tuple(metadata.get("feature_components", ())) != components
            or not model_matches
            or cache.features.shape[1] != input_dim
            or sample_id not in cache.sample_ids
        ):
            continue
        selected_index = cache.sample_ids.index(sample_id)
        return (
            cache.features[selected_index : selected_index + 1].float(),
            f"compatible cache {cache_path}",
        )

    if backbone != "raw_rgb" and not extract_missing_features:
        raise ValueError(
            f"No compatible {backbone} cache contains {sample_id!r}. Restore the "
            "experiment feature cache or pass --extract-missing-features to run "
            "one in-memory backbone forward."
        )
    extractor = create_feature_extractor(
        backbone,
        **_extractor_kwargs(config, probe, backbone=backbone, device=device),
    )
    frozen = extractor.extract(np.expand_dims(image, axis=0))
    feature = select_feature_components(frozen, components).cpu().float()
    if feature.shape != (1, input_dim):
        raise ValueError(
            f"Extracted feature shape {tuple(feature.shape)} does not match "
            f"checkpoint input dimension {input_dim}"
        )
    return feature, f"in-memory {backbone} extraction (not cached)"


def _extractor_kwargs(
    config: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    backbone: str,
    device: str,
) -> dict[str, Any]:
    model = dict(_mapping(probe, backbone))
    allowed = {
        "raw_rgb": {"output_size"},
        "imagenet_vit": {"pretrained"},
        "dinov2": {"model_name", "repo_or_dir", "source", "pretrained"},
        "vggt": {"model_id", "image_size", "layer_index"},
    }[backbone]
    unexpected = set(model) - allowed
    if unexpected:
        raise ValueError(f"Unexpected probe.{backbone} fields: {sorted(unexpected)}")
    kwargs: dict[str, Any] = {**model, "device": device}
    if backbone != "raw_rgb":
        paths = _mapping(config, "paths")
        kwargs["model_cache_root"] = _rooted_path(
            paths.get("model_cache_root"), "paths.model_cache_root"
        )
    return kwargs


def _resolve_sample_index(
    dataset: NUMDataset, *, index: int, sample_id: str | None
) -> int:
    if sample_id is not None:
        for candidate, record in enumerate(dataset.records):
            if record.sample_id == sample_id:
                return candidate
        raise ValueError(
            f"Sample {sample_id!r} is not present in split {dataset.split!r}"
        )
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < len(dataset):
        raise ValueError(f"index must be in [0, {len(dataset) - 1}]")
    return index


def _rooted_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return selected


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _prediction_destination(
    figure_root: Path,
    *,
    variant_name: str,
    split: str,
    sample_id: str,
) -> Path:
    """Return one variant file inside a shared object/view directory."""

    sample_parts = tuple(part for part in sample_id.split("/") if part)
    if not sample_parts:
        raise ValueError("sample_id must contain at least one path component")
    components = (split, *sample_parts, variant_name)
    if any(part in (".", "..") or Path(part).name != part for part in components):
        raise ValueError("Prediction output path components must be safe names")
    return figure_root.joinpath(split, *sample_parts, f"{variant_name}.svg")


if __name__ == "__main__":
    raise SystemExit(main())
