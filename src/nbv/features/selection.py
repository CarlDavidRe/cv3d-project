"""Select fixed-size probe inputs from frozen backbone token outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from nbv.features.base import FrozenFeatures


class FeatureSelectionError(ValueError):
    """Raised when a probe requests an invalid or unavailable feature."""


FEATURE_COMPONENTS_BY_BACKBONE = {
    "imagenet_vit": frozenset(
        {"pooled_patch", "max_pooled_patch", "cls_token"}
    ),
    "dinov2": frozenset(
        {"pooled_patch", "max_pooled_patch", "cls_token", "pooled_register"}
    ),
    "vggt": frozenset(
        {
            "pooled_patch",
            "max_pooled_patch",
            "pooled_camera",
            "pooled_register",
        }
    ),
}


def validate_feature_selection(
    backbone: str, components: Sequence[str]
) -> tuple[str, ...]:
    """Validate and normalize one backbone's ordered component list."""

    try:
        allowed = FEATURE_COMPONENTS_BY_BACKBONE[backbone]
    except KeyError as exc:
        raise FeatureSelectionError(f"Unknown backbone {backbone!r}") from exc
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise FeatureSelectionError("feature components must be a list of names")
    normalized = tuple(components)
    if not normalized:
        raise FeatureSelectionError("at least one feature component is required")
    if any(not isinstance(component, str) for component in normalized):
        raise FeatureSelectionError("every feature component must be a string")
    if len(set(normalized)) != len(normalized):
        raise FeatureSelectionError("feature components must not contain duplicates")
    unsupported = [component for component in normalized if component not in allowed]
    if unsupported:
        raise FeatureSelectionError(
            f"Unsupported {backbone} feature components {unsupported}; "
            f"choose from {sorted(allowed)}"
        )
    return normalized


def select_feature_components(
    features: FrozenFeatures, components: Sequence[str]
) -> Tensor:
    """Return ``[B, D_selected]`` by concatenating ordered components.

    Patch and special-token sequences are reduced to one vector so the
    lightweight probe never receives a variable-length token sequence.
    """

    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise FeatureSelectionError("feature components must be a list of names")
    if not components:
        raise FeatureSelectionError("at least one feature component is required")
    if any(not isinstance(component, str) for component in components):
        raise FeatureSelectionError("every feature component must be a string")
    if len(set(components)) != len(components):
        raise FeatureSelectionError("feature components must not contain duplicates")

    selected: list[Tensor] = []
    for component in components:
        if component == "pooled_patch":
            value = features.pooled_patch
        elif component == "max_pooled_patch":
            value = features.patch_tokens.amax(dim=1)
        elif component == "cls_token":
            value = _required_token(features.cls_token, component)
        elif component == "pooled_camera":
            value = _required_token(features.camera_tokens, component).mean(dim=1)
        elif component == "pooled_register":
            value = _required_token(features.register_tokens, component).mean(dim=1)
        else:
            raise FeatureSelectionError(f"Unknown feature component {component!r}")
        selected.append(value)

    result = torch.cat(selected, dim=1)
    if result.ndim != 2 or result.shape[0] != features.batch_size:
        raise FeatureSelectionError("selected features must have shape [B, D]")
    return result.detach()


def _required_token(value: Tensor | None, component: str) -> Tensor:
    if value is None:
        raise FeatureSelectionError(
            f"Selected component {component!r} is not exposed by this model"
        )
    return value
