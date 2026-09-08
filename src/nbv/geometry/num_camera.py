"""Camera calibration for the released NUM RGB images.

PUN's BlenderInterface uses a focal length of (525 / 512) * image width.
Its xyz2pose uses Blender's -Z/Y tracking, aligning camera Y with projected
world Z. At the two poles we pin the roll to the released NUM images.

Sources and pole checks: docs/num_camera_alignment.md.
"""

from __future__ import annotations

import math

import numpy as np

from nbv.geometry.anchors import Anchor


PUN_SOURCE_REVISION = "aa6f8f4f12154854a4c1867209725c80475af102"
NUM_FOCAL_LENGTH_PER_PIXEL = 525.0 / 512.0
NUM_HORIZONTAL_FOV_DEGREES = math.degrees(
    2 * math.atan(0.5 / NUM_FOCAL_LENGTH_PER_PIXEL)
)
CAMERA_CONVENTION = "opengl_blender_track_world_z_num_poles_pixel_centers"


def anchor_camera_to_world(
    anchor: Anchor,
    radius: float,
    *,
    convention: str = CAMERA_CONVENTION,
) -> np.ndarray:
    """Validate cache convention and return the canonical NUM camera pose.

    Pole 0 has identity rotation; pole 46 has diag(-1, 1, -1). Both keep
    world +Y upward in the image. Non-poles match Blender's tracking rotation.
    """
    if convention != CAMERA_CONVENTION:
        raise ValueError(f"Unsupported camera convention: {convention!r}")
    return anchor.camera_to_world(radius)
