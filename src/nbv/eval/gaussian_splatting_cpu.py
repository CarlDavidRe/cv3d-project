"""CPU-only inference for saved 2DGS checkpoints; no training or gsplat import.

Reference conventions: gsplat v1.5.3 Projection2DGSFused.cu and
RasterizeToPixels2DGSFwd.cu. Uses center-z median depth, pixel-center ray/surfel
intersection, the screen-space low-pass filter, and front-to-back alpha.
This NumPy implementation is not certified bit-identical to CUDA (notably ties).
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np

from nbv.eval.gaussian_splatting import GaussianParameters, SplatCameras

RENDERER_VERSION = "numpy_2dgs_median_v1"


def render_depth_cpu(
    model: GaussianParameters, cameras: SplatCameras, resolution: int,
    *, progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return alpha and median z-depth [C,H,W], with bounded image memory."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    means, quats, scales, opacity, _ = (
        value.detach().cpu().numpy() for value in model.values()
    )
    if not all(np.isfinite(a).all() for a in (means, quats, scales, opacity)):
        raise ValueError("checkpoint contains non-finite Gaussian parameters")
    w, x, y, z = quats.T
    tangent = np.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z),
        2*(x*y+w*z), 1-2*(x*x+z*z),
        2*(x*z-w*y), 2*(y*z+w*x),
    ), axis=-1).reshape(-1, 3, 2) * scales[:, None, :2]
    alphas, depths = [], []
    yy, xx = np.mgrid[:resolution, :resolution].astype(np.float32) + 0.5
    for camera_index, (view, intrinsic) in enumerate(zip(
        cameras.world_to_camera, cameras.intrinsics, strict=True
    )):
        rotation = view[:3, :3].astype(np.float32)
        centers = means @ rotation.T + view[:3, 3].astype(np.float32)
        axes = np.einsum("ij,njk->nik", rotation, tangent)
        homography = np.einsum(
            "ij,njk->nik", intrinsic.astype(np.float32),
            np.concatenate((axes, centers[..., None]), axis=-1),
        )
        signature = np.array([1, 1, -1], dtype=np.float32)
        denominator = (homography[:, 2] ** 2 * signature).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            weights = signature / denominator[:, None]
            projected = (weights[:, None] * homography[:, :2] * homography[:, 2:3]).sum(-1)
            variance = projected**2 - (weights[:, None] * homography[:, :2]**2).sum(-1)
            radius = np.ceil(3.33 * np.sqrt(np.maximum(variance, 1e-4)))
        valid = ((centers[:, 2] > 0.01) & (centers[:, 2] < 1e10)
                 & np.isfinite(projected).all(-1) & np.isfinite(radius).all(-1)
                 & (projected + radius > 0).all(-1)
                 & (projected - radius < resolution).all(-1))
        transmission = np.ones((resolution, resolution), dtype=np.float32)
        median = np.zeros_like(transmission)
        done = np.zeros_like(transmission, dtype=bool)
        order = np.flatnonzero(valid)
        order = order[np.argsort(centers[order, 2], kind="stable")]
        for i in order:
            # Match gsplat's 16-pixel tile coverage, including filter tails.
            lower = np.clip(np.floor((projected[i]-radius[i])/16)*16, 0, resolution).astype(int)
            upper = np.clip(np.ceil((projected[i]+radius[i])/16)*16, 0, resolution).astype(int)
            region = np.s_[lower[1]:upper[1], lower[0]:upper[0]]
            t = transmission[region]
            stopped = done[region]
            if not t.size or stopped.all():
                continue
            px, py = xx[region], yy[region]
            matrix = homography[i]
            a = px[..., None] * matrix[2] - matrix[0]
            b = py[..., None] * matrix[2] - matrix[1]
            intersection = np.cross(a, b)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                uv = intersection[..., :2] / intersection[..., 2:]
                rho = np.minimum((uv**2).sum(-1), 2*((px-projected[i, 0])**2 + (py-projected[i, 1])**2))
                alpha = np.minimum(0.999, opacity[i] * np.exp(-0.5*rho))
            contributing = (~stopped & (intersection[..., 2] != 0)
                            & np.isfinite(alpha) & (alpha >= 1/255))
            next_t = t * (1-alpha)
            stopped |= contributing & (next_t <= 1e-4)
            contributing &= ~stopped
            median[region][contributing & (t > 0.5)] = centers[i, 2]
            t[contributing] = next_t[contributing]
        alphas.append(1-transmission)
        depths.append(median)
        if progress is not None:
            progress(camera_index+1, len(cameras.world_to_camera))
    return np.stack(alphas), np.stack(depths)
