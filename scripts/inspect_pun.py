#!/usr/bin/env python3
"""Run the official pretrained PUN/UPNet on one NUM image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nbv.data import NUMDataset  # noqa: E402
from nbv.models import (  # noqa: E402
    PUN_CHECKPOINT_SHA256,
    PUN_REFERENCE_COMMIT,
    PUN_RELEASE_NAME,
    PUN_REPOSITORY,
    PUNUPNet,
    create_pun_transform,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
    resolve_pun_data_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/NUM"))
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="train"
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/splits/num_v1.json")
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target", choices=("PSNR",), default="PSNR")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument(
        "--model-cache-root", type=Path, default=Path("data/cache/models")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Official checkpoint path (defaults under --model-cache-root)",
    )
    parser.add_argument(
        "--download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-incomplete-object",
        action="store_true",
        help="Permit fewer than 48 source images for a tiny local fixture",
    )
    return parser


def summarize_pun_smoke(
    *,
    sample_id: str,
    source_anchor_id: int,
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: np.ndarray,
    device: torch.device,
    checkpoint: Path,
    data_config: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Validate and summarize one real PUN forward pass."""

    prediction = prediction.detach().to(device="cpu", dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32).reshape(-1)
    if prediction.shape != target_tensor.shape:
        raise ValueError(
            "PUN prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target_tensor.shape)}"
        )
    if prediction.numel() != 48:
        raise ValueError(
            f"PUN smoke prediction must contain 48 values, got {prediction.numel()}"
        )
    if not torch.isfinite(prediction).all():
        raise ValueError("PUN smoke prediction contains NaN or infinite values")

    return {
        "sample_id": sample_id,
        "source_anchor_id": int(source_anchor_id),
        "device": str(device),
        "input_shape": list(image.shape),
        "prediction_shape": list(prediction.shape),
        "prediction_is_finite": True,
        "prediction_min": float(prediction.min().item()),
        "prediction_max": float(prediction.max().item()),
        "predicted_best_local_anchor_id": int(prediction.argmax().item()),
        "official_unmasked_mse_loss": float(
            torch.mean((prediction - target_tensor) ** 2).item()
        ),
        "model": {
            "name": model_name,
            "release": PUN_RELEASE_NAME,
            "source_repository": PUN_REPOSITORY,
            "source_commit": PUN_REFERENCE_COMMIT,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": PUN_CHECKPOINT_SHA256,
        },
        "preprocessing": dict(data_config),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    checkpoint = args.checkpoint or (
        args.model_cache_root
        / "pun"
        / PUN_RELEASE_NAME
        / "best_vit_regressor.pth"
    )
    checkpoint = ensure_pun_checkpoint(
        checkpoint, download_if_missing=args.download_if_missing
    )

    model = PUNUPNet(
        model_name=args.model_name,
        pretrained=False,
        model_cache_root=args.model_cache_root,
    )
    load_pun_checkpoint(model, checkpoint)
    model.requires_grad_(False).to(device).eval()
    data_config = resolve_pun_data_config(model)
    dataset = NUMDataset(
        args.data_root,
        split=args.split,
        split_manifest=args.split_manifest,
        target_name=args.target,
        transform=create_pun_transform(data_config),
        require_complete_objects=not args.allow_incomplete_object,
    )
    sample = dataset[args.sample_index]
    image = torch.as_tensor(sample.image, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(image.to(device))[0]

    summary = summarize_pun_smoke(
        sample_id=sample.record.sample_id,
        source_anchor_id=sample.source_anchor_id,
        image=image,
        prediction=prediction,
        target=sample.target_map,
        device=device,
        checkpoint=checkpoint,
        data_config=data_config,
        model_name=args.model_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
