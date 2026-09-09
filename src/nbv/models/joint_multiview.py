"""Phase 3 direct-gain model over jointly extracted VGGT histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nbv.geometry.anchors import CANONICAL_ANCHOR_COUNT, canonical_anchors
from nbv.models.heads import LightweightProbeHead


class JointHistoryGainModel(nn.Module):
    """Frozen joint VGGT followed by matched masked pooling and gain head.

    The extractor is intentionally not an ``nn.Module`` child: it owns the
    frozen backbone and always executes it in inference mode. Consequently,
    ``parameters()`` exposes only the trainable direct-gain head, matching the
    independent control's optimization boundary.
    """

    aggregation = "masked_mean"
    backbone_history_mode = "joint_multiview"

    def __init__(
        self,
        extractor: object,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        num_anchors: int = CANONICAL_ANCHOR_COUNT,
        dropout: float = 0.0,
        include_anchor_directions: bool = True,
    ) -> None:
        super().__init__()
        if not callable(getattr(extractor, "extract", None)):
            raise TypeError("extractor must expose extract(images, padding_mask)")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(include_anchor_directions, bool):
            raise TypeError("include_anchor_directions must be a boolean")
        if num_anchors != CANONICAL_ANCHOR_COUNT:
            raise ValueError(
                f"Phase 3 uses exactly {CANONICAL_ANCHOR_COUNT} canonical anchors"
            )
        # Bypass Module.__setattr__ so a third-party extractor can never make
        # its frozen backbone part of the optimizer parameter traversal.
        object.__setattr__(self, "extractor", extractor)
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
        joint_view_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        """Pool history-conditioned per-view vectors without padded entries."""

        self._validate_feature_inputs(
            joint_view_features, history_anchor_ids, history_padding_mask
        )
        values = joint_view_features
        if self.include_anchor_directions:
            safe_ids = history_anchor_ids.clamp_min(0)
            directions = self.anchor_directions[safe_ids]
            values = torch.cat((values, directions.to(values.dtype)), dim=-1)
        keep = (~history_padding_mask).unsqueeze(-1)
        summed = torch.where(keep, values, torch.zeros_like(values)).sum(dim=1)
        counts = keep.sum(dim=1).clamp_min(1).to(values.dtype)
        return summed / counts

    def forward_features(
        self,
        joint_view_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        pooled = self.aggregate(
            joint_view_features, history_anchor_ids, history_padding_mask
        )
        return self.head(pooled.float())

    def forward(
        self,
        history_images: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        self._validate_history_metadata(history_anchor_ids, history_padding_mask)
        frozen = self.extractor.extract(history_images, history_padding_mask)
        features = getattr(frozen, "view_features", None)
        returned_padding = getattr(frozen, "history_padding_mask", None)
        if not isinstance(features, Tensor) or not isinstance(returned_padding, Tensor):
            raise TypeError("joint extractor must return JointVGGTFeatures")
        if not torch.equal(returned_padding.to(history_padding_mask.device), history_padding_mask):
            raise ValueError("joint extractor changed the history padding mask")
        # ``extract`` uses inference_mode. Clone after it returns so trainable
        # LayerNorm/Linear layers receive an ordinary tensor that autograd may
        # save for their parameter gradients.
        head_device = next(self.head.parameters()).device
        features = features.detach().clone().to(head_device)
        return self.forward_features(
            features,
            history_anchor_ids.to(features.device),
            history_padding_mask.to(features.device),
        )

    def _validate_feature_inputs(
        self,
        features: Tensor,
        anchor_ids: Tensor,
        padding: Tensor,
    ) -> None:
        if features.ndim != 3 or features.shape[2] != self.feature_dim:
            raise ValueError(
                "joint_view_features must have shape "
                f"[B, H, {self.feature_dim}], got {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError("joint_view_features must be floating point")
        if not torch.isfinite(features).all():
            raise ValueError("joint_view_features must be finite")
        self._validate_history_metadata(anchor_ids, padding, features.shape[:2])

    @staticmethod
    def _validate_history_metadata(
        anchor_ids: Tensor,
        padding: Tensor,
        expected_shape: tuple[int, int] | torch.Size | None = None,
    ) -> None:
        if anchor_ids.ndim != 2:
            raise ValueError("history_anchor_ids must have shape [B, H]")
        shape = anchor_ids.shape if expected_shape is None else expected_shape
        if tuple(anchor_ids.shape) != tuple(shape) or anchor_ids.dtype != torch.int64:
            raise ValueError("history_anchor_ids must be int64 with shape [B, H]")
        if tuple(padding.shape) != tuple(shape) or padding.dtype != torch.bool:
            raise ValueError("history_padding_mask must be boolean with shape [B, H]")
        if torch.any(padding.all(dim=1)):
            raise ValueError("every history must contain at least one real observation")
        if torch.any(anchor_ids[padding] != -1):
            raise ValueError("padded history anchor IDs must be -1")
        real_ids = anchor_ids[~padding]
        if torch.any(real_ids < 0) or torch.any(real_ids >= CANONICAL_ANCHOR_COUNT):
            raise ValueError("real history anchor IDs must be in [0, 47]")


__all__ = ["JointHistoryGainModel"]
