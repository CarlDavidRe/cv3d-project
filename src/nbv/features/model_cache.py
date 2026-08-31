"""Shared locations for downloaded pretrained-model artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from nbv.config import validate_artifact_path


def model_cache_directory(root: str | Path, backend: str) -> Path:
    """Return and create one backend directory below the common model cache."""

    validate_artifact_path(root, "model cache root")
    destination = Path(root) / backend
    destination.mkdir(parents=True, exist_ok=True)
    return destination


@contextmanager
def torch_hub_cache(root: str | Path | None) -> Iterator[None]:
    """Temporarily route Torch Hub and torchvision downloads below ``root``."""

    if root is None:
        yield
        return

    previous = torch.hub.get_dir()
    torch.hub.set_dir(str(model_cache_directory(root, "torch")))
    try:
        yield
    finally:
        torch.hub.set_dir(previous)
