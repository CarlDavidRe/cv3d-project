"""Stable representation of the 48 PUN/NUM camera anchors.

The canonical frame is object-centred and right-handed. Directions point from
the object origin toward the camera: +X is right, +Y is up, and anchor 0 is
the +Z reference view. Azimuth is measured from +X toward +Y. Elevation is
measured from the XY plane toward +Z.

The ordering is HEALPix NSIDE=2 RING order after the rotation used by PUN to
place HEALPix pixel 0 on +Z. Keeping the resolved values in a package resource
avoids making ``healpy`` a runtime dependency and prevents version-dependent
regeneration from silently changing target-map semantics.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


CANONICAL_ANCHOR_COUNT = 48
CANONICAL_ORDERING = "pun_healpix_nside2_ring_anchor0_positive_z_v1"
_REQUIRED_COLUMNS = (
    "anchor_id",
    "azimuth_rad",
    "elevation_rad",
    "direction_x",
    "direction_y",
    "direction_z",
)
_UNIT_TOLERANCE = 1e-12


class AnchorValidationError(ValueError):
    """Raised when an anchor definition violates the canonical schema."""


@dataclass(frozen=True, slots=True)
class Anchor:
    """One camera anchor in the canonical object-centred frame."""

    anchor_id: int
    azimuth_rad: float
    elevation_rad: float
    direction: tuple[float, float, float]

    @property
    def polar_angle_rad(self) -> float:
        """Angle down from +Z, matching the theta used in PUN plots."""

        return math.pi / 2 - self.elevation_rad

    def camera_position(self, radius: float = 1.0) -> NDArray[np.float64]:
        """Return the camera centre at ``radius`` from the object origin."""

        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and greater than zero")
        return np.asarray(self.direction, dtype=np.float64) * radius

    @property
    def view_direction(self) -> NDArray[np.float64]:
        """Return the unit direction from the camera toward the object."""

        return -np.asarray(self.direction, dtype=np.float64)

    def camera_to_world(self, radius: float = 1.0) -> NDArray[np.float64]:
        """Return the NUM camera pose, looking at the origin along local -Z.

        Camera Y follows projected world Z away from the poles. At the two
        poles, world +Y remains image-up, matching the released NUM images.
        """

        position = self.camera_position(radius)
        camera_z = np.asarray(self.direction, dtype=np.float64)
        preferred_up = np.array([0.0, 0.0, 1.0])
        camera_x = np.cross(preferred_up, camera_z)
        if np.linalg.norm(camera_x) < _UNIT_TOLERANCE:
            preferred_up = np.array([0.0, 1.0, 0.0])
            camera_x = np.cross(preferred_up, camera_z)
        camera_x /= np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.column_stack((camera_x, camera_y, camera_z))
        transform[:3, 3] = position
        return transform


class AnchorSet(Sequence[Anchor]):
    """Validated anchors whose sequence index is always the anchor ID."""

    def __init__(self, anchors: Iterable[Anchor], ordering: str) -> None:
        self._anchors = tuple(anchors)
        self.ordering = ordering
        _validate_anchors(self._anchors)

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int | slice) -> Anchor | tuple[Anchor, ...]:
        return self._anchors[index]

    def __iter__(self) -> Iterator[Anchor]:
        return iter(self._anchors)

    def by_id(self, anchor_id: int) -> Anchor:
        """Return an anchor, rejecting IDs outside the canonical range."""

        _validate_anchor_id(anchor_id, len(self))
        return self._anchors[anchor_id]

    @cached_property
    def directions(self) -> NDArray[np.float64]:
        """Return a read-only ``[N, 3]`` matrix in canonical ID order."""

        directions = np.asarray(
            [anchor.direction for anchor in self], dtype=np.float64
        )
        directions.setflags(write=False)
        return directions

    def camera_positions(self, radius: float = 1.0) -> NDArray[np.float64]:
        """Return camera centres in canonical ID order."""

        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and greater than zero")
        return self.directions * radius

    @cached_property
    def angular_distance_matrix(self) -> NDArray[np.float64]:
        """Return read-only pairwise great-circle distances in radians."""

        cosine = np.clip(self.directions @ self.directions.T, -1.0, 1.0)
        distances = np.arccos(cosine)
        np.fill_diagonal(distances, 0.0)
        distances.setflags(write=False)
        return distances

    def angular_distance(self, first_id: int, second_id: int) -> float:
        """Return great-circle distance between two anchor IDs in radians."""

        _validate_anchor_id(first_id, len(self))
        _validate_anchor_id(second_id, len(self))
        return float(self.angular_distance_matrix[first_id, second_id])

    def valid_candidate_mask(
        self, acquired_anchor_ids: Iterable[int] = ()
    ) -> NDArray[np.bool_]:
        """Return a mask with acquired anchors excluded from candidacy."""

        mask = np.ones(len(self), dtype=np.bool_)
        for anchor_id in acquired_anchor_ids:
            _validate_anchor_id(anchor_id, len(self))
            mask[anchor_id] = False
        return mask


@lru_cache(maxsize=1)
def canonical_anchors() -> AnchorSet:
    """Load the packaged canonical 48-anchor definition once per process."""

    resource = files("nbv.geometry").joinpath("anchors_v1.csv")
    with as_file(resource) as path:
        return load_anchors(path, ordering=CANONICAL_ORDERING)


def load_anchors(path: str | Path, ordering: str = "external") -> AnchorSet:
    """Load and validate anchors from the canonical CSV schema."""

    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8", newline="")
    except FileNotFoundError as exc:
        raise AnchorValidationError(f"Anchor file does not exist: {source}") from exc

    try:
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(_REQUIRED_COLUMNS):
                raise AnchorValidationError(
                    f"Anchor columns must be exactly {_REQUIRED_COLUMNS}"
                )
            anchors = [
                _parse_anchor(row, row_number)
                for row_number, row in enumerate(reader, 2)
            ]
    except csv.Error as exc:
        raise AnchorValidationError(f"Invalid anchor CSV: {exc}") from exc
    return AnchorSet(anchors, ordering=ordering)


def _parse_anchor(row: dict[str, str], row_number: int) -> Anchor:
    try:
        return Anchor(
            anchor_id=int(row["anchor_id"]),
            azimuth_rad=float(row["azimuth_rad"]),
            elevation_rad=float(row["elevation_rad"]),
            direction=(
                float(row["direction_x"]),
                float(row["direction_y"]),
                float(row["direction_z"]),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise AnchorValidationError(
            f"Invalid numeric value in anchor row {row_number}"
        ) from exc


def _validate_anchors(anchors: tuple[Anchor, ...]) -> None:
    if not anchors:
        raise AnchorValidationError("At least one anchor is required")

    expected_ids = list(range(len(anchors)))
    actual_ids = [anchor.anchor_id for anchor in anchors]
    if actual_ids != expected_ids:
        raise AnchorValidationError(
            "Anchor rows must be ordered with contiguous IDs starting at zero"
        )

    directions = np.asarray(
        [anchor.direction for anchor in anchors], dtype=np.float64
    )
    angles = np.asarray(
        [(anchor.azimuth_rad, anchor.elevation_rad) for anchor in anchors],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(angles)):
        raise AnchorValidationError("Anchor values must be finite")
    if not np.allclose(
        np.linalg.norm(directions, axis=1),
        1.0,
        atol=_UNIT_TOLERANCE,
        rtol=0,
    ):
        raise AnchorValidationError("Every anchor direction must have unit length")
    if np.any(angles[:, 0] < 0) or np.any(angles[:, 0] >= 2 * math.pi):
        raise AnchorValidationError("Azimuth must be in [0, 2*pi)")
    if np.any(angles[:, 1] < -math.pi / 2) or np.any(
        angles[:, 1] > math.pi / 2
    ):
        raise AnchorValidationError("Elevation must be in [-pi/2, pi/2]")

    expected_elevation = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    if not np.allclose(
        angles[:, 1], expected_elevation, atol=_UNIT_TOLERANCE, rtol=0
    ):
        raise AnchorValidationError("Elevation does not match direction")

    away_from_poles = np.linalg.norm(directions[:, :2], axis=1) > _UNIT_TOLERANCE
    expected_azimuth = np.mod(
        np.arctan2(directions[:, 1], directions[:, 0]), 2 * math.pi
    )
    if not np.allclose(
        angles[away_from_poles, 0],
        expected_azimuth[away_from_poles],
        atol=_UNIT_TOLERANCE,
        rtol=0,
    ):
        raise AnchorValidationError("Azimuth does not match direction")

    cosine = directions @ directions.T
    np.fill_diagonal(cosine, -1.0)
    if np.any(cosine > 1.0 - _UNIT_TOLERANCE):
        raise AnchorValidationError("Anchor directions must be unique")


def _validate_anchor_id(anchor_id: int, count: int) -> None:
    if isinstance(anchor_id, bool) or not isinstance(anchor_id, (int, np.integer)):
        raise TypeError("anchor_id must be an integer")
    if anchor_id < 0 or anchor_id >= count:
        raise IndexError(f"anchor_id must be in [0, {count})")
