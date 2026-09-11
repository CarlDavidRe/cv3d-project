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


class PoseConditionedDeepSetsHistoryGainModel(nn.Module):
    """Joint VGGT vectors with nonlinear pose fusion before set pooling.

    The baseline averages ``[feature, direction]`` directly, which is
    equivalent to exposing the head to two unrelated means.  This variant
    first maps every feature/direction pair to a learned element embedding and
    only then applies the same padding-aware, permutation-invariant mean.
    """

    aggregation = "pose_conditioned_deepsets_mean"
    backbone_history_mode = "joint_multiview"
    architecture = "pose_deepsets"

    def __init__(
        self,
        extractor: object,
        feature_dim: int,
        *,
        element_dim: int = 128,
        hidden_dim: int = 128,
        num_anchors: int = CANONICAL_ANCHOR_COUNT,
        dropout: float = 0.0,
        include_anchor_directions: bool = True,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(feature_dim, element_dim, num_anchors)
        if not callable(getattr(extractor, "extract", None)):
            raise TypeError("extractor must expose extract(images, padding_mask)")
        if not isinstance(include_anchor_directions, bool):
            raise TypeError("include_anchor_directions must be a boolean")
        object.__setattr__(self, "extractor", extractor)
        self.feature_dim = feature_dim
        self.element_dim = element_dim
        self.include_anchor_directions = include_anchor_directions
        directions = torch.from_numpy(canonical_anchors().directions.copy()).float()
        self.register_buffer("anchor_directions", directions, persistent=True)
        input_dim = feature_dim + (3 if include_anchor_directions else 0)
        self.element_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, element_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.head = LightweightProbeHead(
            element_dim,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            dropout=dropout,
        )

    def forward_features(
        self,
        joint_view_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        _validate_pooled_feature_inputs(
            joint_view_features,
            history_anchor_ids,
            history_padding_mask,
            self.feature_dim,
        )
        values = joint_view_features.float()
        if self.include_anchor_directions:
            directions = self.anchor_directions[history_anchor_ids.clamp_min(0)]
            values = torch.cat((values, directions.to(values.dtype)), dim=-1)
        encoded = self.element_encoder(values)
        keep = (~history_padding_mask).unsqueeze(-1)
        pooled = torch.where(keep, encoded, torch.zeros_like(encoded)).sum(dim=1)
        pooled = pooled / keep.sum(dim=1).clamp_min(1).to(encoded.dtype)
        return self.head(pooled)

    def forward(
        self,
        history_images: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        frozen = self.extractor.extract(history_images, history_padding_mask)
        features = getattr(frozen, "view_features", None)
        returned_padding = getattr(frozen, "history_padding_mask", None)
        if not isinstance(features, Tensor) or not isinstance(returned_padding, Tensor):
            raise TypeError("joint extractor must return JointVGGTFeatures")
        if not torch.equal(returned_padding.to(history_padding_mask.device), history_padding_mask):
            raise ValueError("joint extractor changed the history padding mask")
        device = next(self.parameters()).device
        return self.forward_features(
            features.detach().clone().to(device),
            history_anchor_ids.to(device),
            history_padding_mask.to(device),
        )


class TokenCandidateAttentionHistoryGainModel(nn.Module):
    """Candidate camera queries attending to pose-tagged joint VGGT tokens."""

    aggregation = "candidate_cross_attention"
    backbone_history_mode = "joint_multiview_spatial_tokens"
    architecture = "token_candidate_attention"

    def __init__(
        self,
        extractor: object,
        feature_dim: int,
        *,
        attention_dim: int = 128,
        attention_heads: int = 4,
        score_hidden_dim: int = 128,
        num_anchors: int = CANONICAL_ANCHOR_COUNT,
        dropout: float = 0.0,
        include_anchor_directions: bool = True,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(feature_dim, attention_dim, num_anchors)
        if not callable(getattr(extractor, "extract", None)):
            raise TypeError("extractor must expose extract(images, padding_mask)")
        if (
            isinstance(attention_heads, bool)
            or not isinstance(attention_heads, int)
            or attention_heads < 1
            or attention_dim % attention_heads
        ):
            raise ValueError("attention_heads must divide attention_dim")
        if (
            isinstance(score_hidden_dim, bool)
            or not isinstance(score_hidden_dim, int)
            or score_hidden_dim < 1
        ):
            raise ValueError("score_hidden_dim must be a positive integer")
        if not isinstance(include_anchor_directions, bool):
            raise TypeError("include_anchor_directions must be a boolean")
        object.__setattr__(self, "extractor", extractor)
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.score_hidden_dim = score_hidden_dim
        self.include_anchor_directions = include_anchor_directions
        directions = torch.from_numpy(canonical_anchors().directions.copy()).float()
        self.register_buffer("anchor_directions", directions, persistent=True)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, attention_dim),
        )
        self.observation_pose_projection = (
            nn.Linear(3, attention_dim, bias=False)
            if include_anchor_directions
            else None
        )
        self.candidate_query = nn.Linear(3, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, 1),
        )

    def forward_features(
        self,
        joint_token_features: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        if joint_token_features.ndim != 4 or joint_token_features.shape[-1] != self.feature_dim:
            raise ValueError(
                "joint_token_features must have shape "
                f"[B, H, K, {self.feature_dim}]"
            )
        _validate_history_metadata(
            history_anchor_ids,
            history_padding_mask,
            joint_token_features.shape[:2],
        )
        if (
            not joint_token_features.is_floating_point()
            or not torch.isfinite(joint_token_features).all()
        ):
            raise ValueError("joint token features must be finite floating-point values")
        batch_size, history_length, token_count, _ = joint_token_features.shape
        tokens = self.token_projection(joint_token_features.float())
        if self.observation_pose_projection is not None:
            directions = self.anchor_directions[history_anchor_ids.clamp_min(0)]
            pose = self.observation_pose_projection(directions).unsqueeze(2)
            tokens = tokens + pose
        tokens = tokens.reshape(batch_size, history_length * token_count, self.attention_dim)
        token_padding = history_padding_mask.unsqueeze(-1).expand(
            batch_size, history_length, token_count
        ).reshape(batch_size, history_length * token_count)
        queries = self.candidate_query(self.anchor_directions).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        attended, _ = self.cross_attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=token_padding,
            need_weights=False,
        )
        return self.score_head(attended + queries).squeeze(-1)

    def forward(
        self,
        history_images: Tensor,
        history_anchor_ids: Tensor,
        history_padding_mask: Tensor,
    ) -> Tensor:
        frozen = self.extractor.extract(history_images, history_padding_mask)
        features = getattr(frozen, "view_features", None)
        returned_padding = getattr(frozen, "history_padding_mask", None)
        if not isinstance(features, Tensor) or not isinstance(returned_padding, Tensor):
            raise TypeError("joint token extractor must return JointVGGTFeatures")
        if not torch.equal(returned_padding.to(history_padding_mask.device), history_padding_mask):
            raise ValueError("joint extractor changed the history padding mask")
        device = next(self.parameters()).device
        return self.forward_features(
            features.detach().clone().to(device),
            history_anchor_ids.to(device),
            history_padding_mask.to(device),
        )


def _validate_model_dimensions(feature_dim: int, inner_dim: int, num_anchors: int) -> None:
    for name, value in (
        ("feature_dim", feature_dim),
        ("inner_dim", inner_dim),
        ("num_anchors", num_anchors),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_anchors != CANONICAL_ANCHOR_COUNT:
        raise ValueError(f"Phase 3 uses exactly {CANONICAL_ANCHOR_COUNT} canonical anchors")


def _validate_history_metadata(
    anchor_ids: Tensor,
    padding: Tensor,
    expected_shape: tuple[int, ...] | torch.Size,
) -> None:
    if tuple(anchor_ids.shape) != tuple(expected_shape) or anchor_ids.dtype != torch.int64:
        raise ValueError("history_anchor_ids must be int64 with shape [B, H]")
    if tuple(padding.shape) != tuple(expected_shape) or padding.dtype != torch.bool:
        raise ValueError("history_padding_mask must be boolean with shape [B, H]")
    if torch.any(padding.all(dim=1)):
        raise ValueError("every history must contain at least one real observation")
    if torch.any(anchor_ids[padding] != -1):
        raise ValueError("padded history anchor IDs must be -1")
    real_ids = anchor_ids[~padding]
    if torch.any(real_ids < 0) or torch.any(real_ids >= CANONICAL_ANCHOR_COUNT):
        raise ValueError("real history anchor IDs must be in [0, 47]")


def _validate_pooled_feature_inputs(
    features: Tensor,
    anchor_ids: Tensor,
    padding: Tensor,
    feature_dim: int,
) -> None:
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError(f"joint_view_features must have shape [B, H, {feature_dim}]")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("joint view features must be finite floating-point values")
    _validate_history_metadata(anchor_ids, padding, features.shape[:2])


__all__ = [
    "JointHistoryGainModel",
    "PoseConditionedDeepSetsHistoryGainModel",
    "TokenCandidateAttentionHistoryGainModel",
]
