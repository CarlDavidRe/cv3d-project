#!/usr/bin/env python3
"""Export compact Phase 1 test predictions for the dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.features import load_feature_cache  # noqa: E402
from nbv.models import (  # noqa: E402
    LightweightProbeHead,
    PUNUPNet,
    create_pun_transform,
    load_pun_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("outputs/phase1/backbone_sweep/seed_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dashboard/data/phase1_test_predictions.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = (REPO_ROOT / args.experiment).resolve()
    manifest_path = REPO_ROOT / "data" / "splits" / "num_v1.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    object_keys = tuple(manifest["splits"]["test"])
    sample_ids = tuple(
        f"{key}/{source_anchor_id}"
        for key in object_keys
        for source_anchor_id in range(48)
    )

    previous_pun: np.ndarray | None = None
    if args.output.is_file():
        with np.load(args.output, allow_pickle=False) as previous:
            previous_names = [str(value) for value in previous["variant_names"]]
            if (
                "pun_upnet" in previous_names
                and previous["predictions"].ndim == 3
                and previous["predictions"].shape[1:] == (len(object_keys), 48)
            ):
                previous_pun = previous["predictions"][
                    previous_names.index("pun_upnet")
                ].astype(np.float32)

    variant_dirs = sorted(
        path
        for path in (experiment / "variants").iterdir()
        if (path / "best.pt").is_file() and (path / "summary.json").is_file()
    )
    predictions: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None

    for variant_dir in variant_dirs:
        checkpoint = torch.load(
            variant_dir / "best.pt", map_location="cpu", weights_only=True
        )
        name = variant_dir.name
        fixed_map = checkpoint.get("prediction_map")
        if isinstance(fixed_map, torch.Tensor):
            predictions[name] = np.broadcast_to(
                fixed_map.numpy(), (len(sample_ids), 48)
            ).copy()
            continue
        if checkpoint.get("model", {}).get("type") == "pun_upnet":
            if previous_pun is None:
                previous_pun = predict_pun(
                    checkpoint, object_keys, batch_size=args.batch_size
                )
            pun_predictions = np.full(
                (len(object_keys), 48, 48), np.nan, dtype=np.float32
            )
            pun_predictions[:, 0, :] = previous_pun
            predictions[name] = pun_predictions.reshape(-1, 48)
            continue

        variant = checkpoint["variant"]
        backbone = variant["backbone"]
        cache_paths = sorted(
            (REPO_ROOT / "data" / "cache" / "features" / backbone / name).glob(
                "test_*.pt"
            )
        )
        if len(cache_paths) != 1:
            raise RuntimeError(f"Expected one test cache for {name}, found {cache_paths}")
        cache = load_feature_cache(cache_paths[0])
        index_by_id = {sample_id: index for index, sample_id in enumerate(cache.sample_ids)}
        selected_indices = [index_by_id[sample_id] for sample_id in sample_ids]
        selected_features = cache.features[selected_indices].float()
        selected_targets = cache.targets[selected_indices].float().numpy()
        if targets is None:
            targets = selected_targets
        elif not np.array_equal(targets, selected_targets):
            raise RuntimeError(f"Target maps disagree in the {name} feature cache")

        head_config = checkpoint["head"]
        head = LightweightProbeHead(
            int(head_config["input_dim"]),
            hidden_dim=int(head_config["hidden_dim"]),
            num_anchors=int(head_config["num_anchors"]),
            dropout=float(head_config["dropout"]),
        )
        head.load_state_dict(checkpoint["head_state_dict"])
        head.eval()
        with torch.inference_mode():
            predictions[name] = head(selected_features).numpy()
        print(f"Exported {name}", flush=True)

    if targets is None:
        raise RuntimeError("No learned feature cache supplied the ground-truth maps")
    missing = [path.name for path in variant_dirs if path.name not in predictions]
    if missing:
        raise RuntimeError(f"Missing predictions for: {missing}")

    variant_names = tuple(path.name for path in variant_dirs)
    stacked_predictions = np.stack([predictions[name] for name in variant_names]).reshape(
        len(variant_names), len(object_keys), 48, 48
    )
    available_source_anchors = np.isfinite(stacked_predictions).all(axis=(1, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(2, dtype=np.int16),
        object_keys=np.asarray(object_keys),
        variant_names=np.asarray(variant_names),
        available_source_anchors=available_source_anchors,
        targets=targets.reshape(len(object_keys), 48, 48).astype(np.float16),
        predictions=stacked_predictions.astype(np.float16),
    )
    print(
        f"Wrote {args.output} with {len(object_keys)} objects and "
        f"{len(variant_names)} variants; PUN retains source anchor 0 only"
    )
    return 0


def predict_pun(
    checkpoint: dict[str, object], object_keys: tuple[str, ...], *, batch_size: int
) -> np.ndarray:
    """Evaluate the official PUN checkpoint in batches for source anchor zero."""
    model_config = checkpoint["model"]
    official = checkpoint["official_checkpoint"]
    model = PUNUPNet(
        model_name=str(model_config["model_name"]),
        num_anchors=int(model_config["num_anchors"]),
        pretrained=False,
    )
    official_path = (
        REPO_ROOT
        / "data"
        / "cache"
        / "models"
        / "pun"
        / str(official["release_name"])
        / "best_vit_regressor.pth"
    )
    load_pun_checkpoint(
        model, official_path, expected_sha256=str(official["sha256"])
    )
    transform = create_pun_transform(checkpoint["preprocessing"])
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(object_keys), batch_size):
            inputs = []
            for object_key in object_keys[start : start + batch_size]:
                image_path = (
                    REPO_ROOT
                    / "data"
                    / "NUM"
                    / object_key
                    / "images"
                    / "viewpoint_0_offset_phi_0.png"
                )
                with Image.open(image_path) as image:
                    inputs.append(transform(image.convert("RGB")))
            batches.append(model(torch.stack(inputs)).numpy())
            print(
                f"Exported PUN objects {start + 1}–"
                f"{min(start + batch_size, len(object_keys))}",
                flush=True,
            )
    return np.concatenate(batches)


if __name__ == "__main__":
    raise SystemExit(main())
