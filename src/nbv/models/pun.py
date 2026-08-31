"""Official-style PUN UPNet model and image preprocessing.

The reference implementation lives in PUN's ``08-vit-train`` directory.  It
uses a pretrained timm ViT, removes its classification head, and adds one
linear layer that regresses the 48-value Neural Uncertainty Map.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import torch
from torch import Tensor, nn

from nbv.features.model_cache import torch_hub_cache


PUN_REPOSITORY = "https://github.com/ZhangLab-DeepNeuroCogLab/PUN"
PUN_REFERENCE_COMMIT = "aa6f8f4f12154854a4c1867209725c80475af102"
PUN_RELEASE_NAME = "vit_small_patch16_224_PSNR_250425172703"
PUN_CHECKPOINT_FILE_ID = "1vpVFy2LQMjN0jTJ_o1chZ6B4nJNfNVKP"
PUN_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download?"
    f"id={PUN_CHECKPOINT_FILE_ID}&export=download&confirm=t"
)
PUN_CHECKPOINT_SHA256 = (
    "91b2065f7652aac0c84386d4af10cd0c1ae723c049c91e45ccc78722f70907ae"
)


class PUNModelError(RuntimeError):
    """Raised when the official UPNet dependencies or backbone are invalid."""


class PUNUPNet(nn.Module):
    """PUN's ViT backbone followed by a linear uncertainty-map regressor."""

    def __init__(
        self,
        *,
        model_name: str = "vit_small_patch16_224",
        num_anchors: int = 48,
        pretrained: bool = True,
        model_cache_root: str | Path | None = None,
        backbone: nn.Module | None = None,
        model_loader: Callable[[str, bool], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if isinstance(num_anchors, bool) or not isinstance(num_anchors, int):
            raise TypeError("num_anchors must be an integer")
        if num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be boolean")

        if backbone is None:
            loader = model_loader or _load_timm_model
            cache_context = (
                torch_hub_cache(model_cache_root)
                if model_cache_root is not None
                else nullcontext()
            )
            with cache_context:
                backbone = loader(model_name, pretrained)
        feature_dim = _remove_classifier(backbone)

        self.model_name = model_name
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.regressor = nn.Linear(feature_dim, num_anchors)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [B, 3, H, W], got "
                f"{tuple(images.shape)}"
            )
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise PUNModelError(
                "PUN backbone must return [B, feature_dim], got "
                f"{tuple(features.shape)}"
            )
        return self.regressor(features)


def resolve_pun_data_config(model: PUNUPNet) -> dict[str, Any]:
    """Resolve the same timm preprocessing configuration used by PUN."""

    try:
        import timm
    except ImportError as exc:
        raise PUNModelError(
            "PUN requires timm. Install the project requirements before "
            "running the PUN baseline."
        ) from exc
    pretrained_config = getattr(model.backbone, "pretrained_cfg", None)
    if not isinstance(pretrained_config, Mapping):
        raise PUNModelError("PUN backbone does not expose pretrained_cfg")
    resolved = timm.data.resolve_data_config(dict(pretrained_config))
    return _serializable_data_config(resolved)


def create_pun_transform(data_config: Mapping[str, Any]) -> Callable[[Any], Tensor]:
    """Create PUN's deterministic timm image transform from saved metadata."""

    try:
        import timm
    except ImportError as exc:
        raise PUNModelError(
            "PUN requires timm. Install the project requirements before "
            "running or visualizing the PUN baseline."
        ) from exc
    return timm.data.create_transform(**dict(data_config), is_training=False)


def ensure_pun_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str = PUN_CHECKPOINT_SHA256,
    download_url: str = PUN_CHECKPOINT_URL,
    download_if_missing: bool = True,
) -> Path:
    """Verify the released UPNet checkpoint, downloading it atomically if absent."""

    destination = Path(path)
    if destination.is_file():
        _verify_sha256(destination, expected_sha256)
        return destination
    if not download_if_missing:
        raise PUNModelError(
            f"Official PUN checkpoint is missing: {destination}. Download "
            f"{PUN_RELEASE_NAME}/best_vit_regressor.pth from the official "
            "Models link or enable checkpoint download."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with urlopen(download_url, timeout=300) as response:
                    while block := response.read(1024 * 1024):
                        temporary.write(block)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                raise PUNModelError(
                    "Could not download the official PUN checkpoint from "
                    f"{download_url}: {exc}"
                ) from exc
        _verify_sha256(temporary_path, expected_sha256)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_pun_checkpoint(
    model: PUNUPNet,
    checkpoint: str | Path,
    *,
    expected_sha256: str = PUN_CHECKPOINT_SHA256,
) -> None:
    """Load the official plain UPNet state dict after checksum validation."""

    source = Path(checkpoint)
    _verify_sha256(source, expected_sha256)
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PUNModelError(f"Could not load PUN checkpoint {source}: {exc}") from exc
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise PUNModelError("Official PUN checkpoint must be a tensor state dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PUNModelError(
            f"PUN checkpoint does not match {model.model_name}: {exc}"
        ) from exc


def _load_timm_model(model_name: str, pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise PUNModelError(
            "PUN requires timm. Install the project requirements before "
            "running the PUN baseline."
        ) from exc
    return timm.create_model(model_name, pretrained=pretrained)


def _remove_classifier(backbone: nn.Module) -> int:
    """Match the two classifier operations in official ``regress_model.py``."""

    head = getattr(backbone, "head", None)
    feature_dim = getattr(head, "in_features", None)
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise PUNModelError("PUN backbone head must expose integer in_features")
    reset_classifier = getattr(backbone, "reset_classifier", None)
    if not callable(reset_classifier):
        raise PUNModelError("PUN backbone must implement reset_classifier")
    reset_classifier(0)
    return feature_dim


def _serializable_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _verify_sha256(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError(
            "expected_sha256 must contain 64 hexadecimal characters"
        ) from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PUNModelError(f"Could not read PUN checkpoint {path}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise PUNModelError(
            f"PUN checkpoint SHA-256 mismatch for {path}: expected {expected}, "
            f"got {actual}"
        )
