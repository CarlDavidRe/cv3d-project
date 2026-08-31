#!/usr/bin/env python3
"""Run one frozen backbone on one NUM image and report feature shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from nbv.data import NUMDataset
from nbv.features import create_feature_extractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=("imagenet_vit", "dinov2", "vggt"),
        default="imagenet_vit",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/NUM"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/splits/num_v1.json")
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dinov2-model", default="dinov2_vitb14", help="Official torch.hub name"
    )
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        default=Path("data/cache/models"),
        help="Common root for Torch Hub and Hugging Face model downloads",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = NUMDataset(
        args.data_root,
        split=args.split,
        split_manifest=args.split_manifest,
    )
    sample = dataset[args.sample_index]
    kwargs: dict[str, object] = {
        "device": args.device,
        "model_cache_root": args.model_cache_root,
    }
    if args.backbone == "dinov2":
        kwargs["model_name"] = args.dinov2_model
    elif args.backbone == "vggt":
        kwargs["model_id"] = args.vggt_model

    extractor = create_feature_extractor(args.backbone, **kwargs)
    features = extractor.extract(np.expand_dims(sample.image, axis=0))
    summary = {
        "sample_id": sample.record.sample_id,
        "device": str(extractor.device),
        "features": {
            name: list(tensor.shape) for name, tensor in features.tensors().items()
        },
        "metadata": dict(features.metadata),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
