"""RGB acquisition without exposing NUM targets or unacquired image tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from nbv.data.visibility_cache import visibility_cache_path
from nbv.geometry.anchors import canonical_anchors


@dataclass(frozen=True, slots=True)
class AcquiredObservation:
    anchor_id: int
    image_path: str
    rgb: np.ndarray  # uint8 [H, W, 3], full RGB image


class ObservationStore:
    """Evaluator-owned lookup; only acquired snapshots are passed to policies."""

    def __init__(self, object_id: str, image_paths: Mapping[int, str | Path]):
        visibility_cache_path("unused", object_id)
        self.object_id = object_id
        self._paths = {}
        for anchor_id, path in image_paths.items():
            canonical_anchors().by_id(anchor_id)
            path = Path(path).resolve()
            if not path.is_file():
                raise ValueError(f"Missing RGB for anchor {anchor_id}: {path}")
            self._paths[anchor_id] = path
        self._acquired: dict[int, AcquiredObservation] = {}

    @classmethod
    def from_num_object(
        cls, data_root: str | Path, object_id: str, *, require_complete: bool = True
    ) -> ObservationStore:
        visibility_cache_path("unused", object_id)
        directory = Path(data_root) / object_id / "images"
        paths = {}
        for path in sorted(directory.glob("*.png")):
            match = re.fullmatch(r"viewpoint_(\d+)_offset_phi_0\.png", path.name)
            if match:
                anchor_id = int(match[1])
                if anchor_id in paths:
                    raise ValueError(f"Duplicate RGB anchor {anchor_id}: {object_id}")
                paths[anchor_id] = path
        if require_complete and set(paths) != set(range(48)):
            raise ValueError(f"{object_id} must have RGB anchors 0..47; missing={sorted(set(range(48)) - set(paths))}")
        return cls(object_id, paths)

    @property
    def available_mask(self) -> np.ndarray:
        mask = np.zeros(48, dtype=bool)
        mask[list(self._paths)] = True
        return mask

    def acquire(self, anchor_id: int) -> AcquiredObservation:
        """Decode only the requested image, retaining a private cached copy."""
        canonical_anchors().by_id(anchor_id)
        if anchor_id not in self._paths:
            raise ValueError(f"RGB anchor {anchor_id} is unavailable")
        if anchor_id not in self._acquired:
            with Image.open(self._paths[anchor_id]) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            self._acquired[anchor_id] = AcquiredObservation(
                anchor_id, str(self._paths[anchor_id]), rgb
            )
        stored = self._acquired[anchor_id]
        # Copies isolate even policies that deliberately make arrays writable.
        rgb = stored.rgb.copy()
        rgb.setflags(write=False)
        return AcquiredObservation(anchor_id, stored.image_path, rgb)
