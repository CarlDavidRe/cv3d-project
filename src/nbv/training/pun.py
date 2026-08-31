"""Streaming common-metric evaluation for the pretrained PUN UPNet."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from nbv.features import CachedFeatureDataset
from nbv.training.phase1 import Phase1EvaluationResult, evaluate_phase1_probe


class PUNImageDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Adapt deterministically transformed NUM samples for UPNet inference."""

    def __init__(
        self,
        dataset: Sequence[Any],
        *,
        num_anchors: int,
        mask_source_view: bool,
        max_samples: int | None = None,
    ) -> None:
        if len(dataset) < 1:
            raise ValueError("PUN dataset must not be empty")
        if isinstance(num_anchors, bool) or not isinstance(num_anchors, int):
            raise TypeError("num_anchors must be an integer")
        if num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if not isinstance(mask_source_view, bool):
            raise TypeError("mask_source_view must be boolean")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer or None")
        self.dataset = dataset
        self.num_anchors = num_anchors
        self.mask_source_view = mask_source_view
        self.sample_count = (
            len(dataset) if max_samples is None else min(max_samples, len(dataset))
        )
        records = getattr(dataset, "records", None)
        self.sample_ids = tuple(
            (
                records[index].sample_id
                if records is not None
                else _sample_id(dataset[index], index)
            )
            for index in range(self.sample_count)
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sample = self.dataset[index]
        image = torch.as_tensor(sample.image, dtype=torch.float32)
        target = torch.as_tensor(sample.target_map, dtype=torch.float32)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("transformed PUN images must have shape [3, H, W]")
        if target.shape != (self.num_anchors,):
            raise ValueError(
                f"PUN targets must have shape [{self.num_anchors}], got "
                f"{tuple(target.shape)}"
            )
        valid_mask = torch.ones(self.num_anchors, dtype=torch.bool)
        if self.mask_source_view:
            valid_mask[0] = False
        return (
            image,
            target,
            valid_mask,
            torch.tensor(int(sample.source_anchor_id), dtype=torch.int64),
        )


def evaluate_pun_upnet(
    model: nn.Module,
    dataset: PUNImageDataset,
    *,
    batch_size: int,
    target_direction: str,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
    ndcg_k: int,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
) -> Phase1EvaluationResult:
    """Evaluate released UPNet predictions through official MSE and common metrics."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    resolved_device = _resolve_device(device)
    model.to(resolved_device).eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    source_anchor_ids: list[Tensor] = []
    weighted_unmasked_mse = 0.0
    sample_count = 0
    with torch.inference_mode():
        for images, batch_targets, valid_mask, batch_source_ids in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ):
            images = images.to(resolved_device, non_blocking=True)
            device_targets = batch_targets.to(resolved_device, non_blocking=True)
            batch_predictions = model(images)
            mse = F.mse_loss(batch_predictions, device_targets)
            weighted_unmasked_mse += float(mse.item()) * images.shape[0]
            sample_count += images.shape[0]
            predictions.append(batch_predictions.detach().cpu().float())
            targets.append(batch_targets.float())
            masks.append(valid_mask)
            source_anchor_ids.append(batch_source_ids)

    cached = CachedFeatureDataset(
        features=torch.cat(predictions),
        targets=torch.cat(targets),
        valid_mask=torch.cat(masks),
        sample_ids=dataset.sample_ids,
        source_anchor_ids=torch.cat(source_anchor_ids),
        metadata={"producer": "official_pretrained_pun_predictions"},
    )
    common = evaluate_phase1_probe(
        nn.Identity(),
        cached,
        batch_size=batch_size,
        target_direction=target_direction,
        huber_delta=huber_delta,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        ndcg_k=ndcg_k,
        device="cpu",
    )
    return Phase1EvaluationResult(
        summary={
            **common.summary,
            "official_unmasked_mse_loss": weighted_unmasked_mse / sample_count,
        },
        per_sample=common.per_sample,
        predictions=common.predictions,
    )


def _sample_id(sample: Any, index: int) -> str:
    record = getattr(sample, "record", None)
    value = getattr(record, "sample_id", None)
    return value if isinstance(value, str) and value else f"sample/{index}"


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved
