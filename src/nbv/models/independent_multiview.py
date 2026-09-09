"""Phase 3 direct-gain control over independently extracted VGGT features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, canonical_anchors
from nbv.models.heads import LightweightProbeHead


class IndependentHistoryGainModel(nn.Module):
    """Masked-mean history aggregation followed by a direct-gain head.

    Every input vector must have been extracted by a single-image VGGT
    forward.  This module is deliberately downstream of the frozen backbone:
    observations can interact only through the permutation-invariant mean.
    Fixed canonical camera directions can be appended before aggregation so
    the control has access to the known observation poses without introducing
    trainable per-anchor embeddings.
    """

    aggregation = "masked_mean"
    backbone_history_mode = "independent_single_image"

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        num_anchors: int = CANONICAL_ANCHOR_COUNT,
        dropout: float = 0.0,
        include_anchor_directions: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(include_anchor_directions, bool):
            raise TypeError("include_anchor_directions must be a boolean")
        if num_anchors != CANONICAL_ANCHOR_COUNT:
            raise ValueError(
                f"Phase 3 uses exactly {CANONICAL_ANCHOR_COUNT} canonical anchors"
            )
        self.feature_dim = feature_dim
        self.include_anchor_directions = include_anchor_directions
        directions = torch.from_numpy(canonical_anchors().directions.copy()).float()
        self.register_buffer("anchor_directions", directions, persistent=True)
        aggregated_dim = feature_dim + (3 if include_anchor_directions else 0)
        self.head = LightweightProbeHead(
            aggregated_dim,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            dropout=dropout,
        )

    @property
    def aggregated_dim(self) -> int:
        return self.head.input_dim

    def aggregate(
        self,
        history_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        """Return one order-invariant vector per padded history batch."""

        self._validate_inputs(
            history_features, history_anchor_ids, history_padding_mask
        )
        values = history_features
        if self.include_anchor_directions:
            safe_ids = history_anchor_ids.clamp_min(0)
            directions = self.anchor_directions[safe_ids]
            values = torch.cat((values, directions.to(values.dtype)), dim=-1)
        keep = (~history_padding_mask).unsqueeze(-1)
        summed = torch.where(keep, values, torch.zeros_like(values)).sum(dim=1)
        counts = keep.sum(dim=1).clamp_min(1).to(values.dtype)
        return summed / counts

    def forward(
        self,
        history_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        pooled = self.aggregate(
            history_features, history_anchor_ids, history_padding_mask
        )
        return self.head(pooled)

    def _validate_inputs(
        self,
        history_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> None:
        if history_features.ndim != 3 or history_features.shape[2] != self.feature_dim:
            raise ValueError(
                "history_features must have shape "
                f"[B, H, {self.feature_dim}], got {tuple(history_features.shape)}"
            )
        expected = history_features.shape[:2]
        if history_anchor_ids.shape != expected or history_anchor_ids.dtype != torch.int64:
            raise ValueError("history_anchor_ids must be int64 with shape [B, H]")
        if history_padding_mask.shape != expected or history_padding_mask.dtype != torch.bool:
            raise ValueError("history_padding_mask must be boolean with shape [B, H]")
        if torch.any(history_padding_mask.all(dim=1)):
            raise ValueError("every history must contain at least one real observation")
        if torch.any(history_anchor_ids[history_padding_mask] != -1):
            raise ValueError("padded history anchor IDs must be -1")
        real_ids = history_anchor_ids[~history_padding_mask]
        if torch.any(real_ids < 0) or torch.any(real_ids >= CANONICAL_ANCHOR_COUNT):
            raise ValueError("real history anchor IDs must be in [0, 47]")
        if not history_features.is_floating_point():
            raise TypeError("history_features must be floating point")
        if not torch.isfinite(history_features).all():
            raise ValueError("history_features must be finite")


def independent_feature_batch(
    history_images: Tensor,
    history_padding_mask: Tensor,
    extractor: object,
    feature_components: Sequence[str],
) -> Tensor:
    """Extract real views as independent batch examples and restore padding.

    The extractor receives a four-dimensional ``[N, C, H, W]`` tensor.  The
    existing :class:`VGGTExtractor` consequently inserts a sequence length of
    one for every item; it never receives the history's view axis.
    """

    from nbv.features.selection import select_feature_components

    if history_images.ndim != 5:
        raise ValueError("history_images must have shape [B, H, C, image_H, image_W]")
    if history_padding_mask.shape != history_images.shape[:2] or history_padding_mask.dtype != torch.bool:
        raise ValueError("history_padding_mask must be boolean with shape [B, H]")
    if torch.any(history_padding_mask.all(dim=1)):
        raise ValueError("every history must contain at least one real observation")
    flat_images = history_images[~history_padding_mask]
    frozen = extractor.extract(flat_images)
    vectors = select_feature_components(frozen, feature_components)
    if vectors.ndim != 2 or vectors.shape[0] != flat_images.shape[0]:
        raise ValueError("independent extractor must return one vector per real view")
    result = torch.zeros(
        (*history_images.shape[:2], vectors.shape[1]),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    result[~history_padding_mask.to(vectors.device)] = vectors
    return result.detach()


__all__ = ["IndependentHistoryGainModel", "independent_feature_batch"]
