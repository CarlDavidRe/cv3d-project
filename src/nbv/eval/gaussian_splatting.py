"""Optional gsplat-backed 2DGS evaluation and qualitative 3DGS rendering.

The two contracts are intentionally different: 2DGS renders and fuses a
surface point cloud for geometric evaluation, while 3DGS is appearance-only.
Neither backend changes the view-selection policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from nbv.eval.reconstruction import point_cloud_metrics
from nbv.geometry.anchors import Anchor
from nbv.geometry.visibility import PerspectiveCamera


SplatBackend = Literal["2dgs", "3dgs"]


@dataclass(frozen=True, slots=True)
class GaussianSplatSettings:
    backend: SplatBackend
    iterations: int = 1_500
    resolution: int = 256
    learning_rate: float = 1e-2
    position_learning_rate: float = 2e-4
    opacity_loss_weight: float = 0.1
    distortion_loss_weight: float = 0.01
    normal_loss_weight: float = 0.05
    alpha_threshold: float = 0.5
    surface_point_count: int = 10_000
    metric_chunk_size: int = 1_024
    render_batch_size: int = 8
    fscore_thresholds: tuple[float, ...] = (0.01, 0.02, 0.10)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.backend not in ("2dgs", "3dgs"):
            raise ValueError("Gaussian backend must be '2dgs' or '3dgs'")
        if self.iterations <= 0 or self.resolution <= 0:
            raise ValueError("iterations and resolution must be positive")
        if self.learning_rate <= 0 or self.position_learning_rate < 0:
            raise ValueError("learning rates must be positive/non-negative")
        if not 0 < self.alpha_threshold < 1:
            raise ValueError("alpha_threshold must lie in (0, 1)")
        if (
            self.surface_point_count <= 0
            or self.metric_chunk_size <= 0
            or self.render_batch_size <= 0
        ):
            raise ValueError("surface, metric, and render batch sizes must be positive")


@dataclass(frozen=True, slots=True)
class SplatCameras:
    """Known NUM cameras represented in gsplat's OpenCV convention."""

    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.camera_to_world)
        if self.camera_to_world.shape != (count, 4, 4):
            raise ValueError("camera_to_world must have shape [C,4,4]")
        if self.world_to_camera.shape != (count, 4, 4):
            raise ValueError("world_to_camera must have shape [C,4,4]")
        if self.intrinsics.shape != (count, 3, 3):
            raise ValueError("intrinsics must have shape [C,3,3]")


class GaussianParameters(nn.Module):
    """Small view-independent-color Gaussian model initialized from VGGT."""

    def __init__(self, means: np.ndarray, colors: np.ndarray, backend: SplatBackend):
        super().__init__()
        points = torch.as_tensor(means, dtype=torch.float32)
        rgb = torch.as_tensor(colors, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        extent = float(torch.linalg.vector_norm(points.max(0).values - points.min(0).values))
        radius = max(extent / math.sqrt(max(len(points), 1)), 1e-4)
        scales = torch.full((len(points), 3), math.log(radius), dtype=torch.float32)
        if backend == "2dgs":
            scales[:, 2] = math.log(max(radius * 0.05, 1e-5))
        quaternions = torch.zeros((len(points), 4), dtype=torch.float32)
        quaternions[:, 0] = 1.0
        self.means = nn.Parameter(points)
        self.log_scales = nn.Parameter(scales)
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(torch.full((len(points),), 2.0))
        self.color_logits = nn.Parameter(torch.logit(rgb))

    def values(self) -> tuple[torch.Tensor, ...]:
        return (
            self.means,
            F.normalize(self.quaternions, dim=-1),
            self.log_scales.exp(),
            self.opacity_logits.sigmoid(),
            self.color_logits.sigmoid(),
        )


def known_num_splat_cameras(
    camera_to_world_opengl: np.ndarray,
    camera: PerspectiveCamera,
) -> SplatCameras:
    """Convert Blender/OpenGL NUM poses to gsplat/OpenCV camera matrices."""

    gl = np.asarray(camera_to_world_opengl, dtype=np.float64)
    if gl.ndim != 3 or gl.shape[1:] != (4, 4):
        raise ValueError("NUM camera poses must have shape [C,4,4]")
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_to_world = gl @ flip
    world_to_camera = np.linalg.inv(camera_to_world)
    intrinsic = np.asarray([
        [camera.focal_x, 0.0, camera.center_x],
        [0.0, camera.focal_y, camera.center_y],
        [0.0, 0.0, 1.0],
    ])
    intrinsics = np.broadcast_to(intrinsic, (len(gl), 3, 3)).copy()
    return SplatCameras(camera_to_world, world_to_camera, intrinsics)


def orbit_camera_poses(
    count: int, radius: float, *, elevation_degrees: float = 20.0
) -> np.ndarray:
    """Create a smooth closed NUM/OpenGL orbit for qualitative rendering."""

    if count < 2 or radius <= 0 or not -89 < elevation_degrees < 89:
        raise ValueError("orbit requires count >= 2, radius > 0 and elevation in (-89,89)")
    elevation = math.radians(elevation_degrees)
    poses = []
    for index in range(count):
        azimuth = 2 * math.pi * index / count
        direction = (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
        poses.append(Anchor(index, azimuth, elevation, direction).camera_to_world(radius))
    return np.stack(poses)


def load_splat_images(
    image_paths: Sequence[str | Path], resolution: int, foreground_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for path in image_paths:
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (resolution, resolution), Image.Resampling.BILINEAR
            )
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        images.append(rgb)
        masks.append(np.min(rgb, axis=-1) < foreground_threshold)
    return np.stack(images), np.stack(masks)


def initialize_point_colors(
    points: np.ndarray,
    images: np.ndarray,
    masks: np.ndarray,
    cameras: SplatCameras,
) -> np.ndarray:
    """Average foreground RGB observations that project onto each VGGT point."""

    count, height, width = len(points), images.shape[1], images.shape[2]
    sums = np.zeros((count, 3), dtype=np.float64)
    hits = np.zeros(count, dtype=np.int32)
    homogeneous = np.concatenate((points, np.ones((count, 1))), axis=1)
    for image, mask, view, intrinsic in zip(
        images, masks, cameras.world_to_camera, cameras.intrinsics, strict=True
    ):
        camera_points = homogeneous @ view.T
        z = camera_points[:, 2]
        projected = camera_points[:, :3] @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
            v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(valid)
        indices = indices[mask[v[indices], u[indices]]]
        sums[indices] += image[v[indices], u[indices]]
        hits[indices] += 1
    colors = np.full((count, 3), 0.5, dtype=np.float32)
    observed = hits > 0
    colors[observed] = (sums[observed] / hits[observed, None]).astype(np.float32)
    return colors


def require_gsplat(backend: SplatBackend):
    if not torch.cuda.is_available():
        raise RuntimeError(f"{backend} requires a CUDA-capable PyTorch environment")
    try:
        if backend == "2dgs":
            from gsplat import rasterization_2dgs

            return rasterization_2dgs
        from gsplat import rasterization

        return rasterization
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Gaussian splatting requires the optional dependency: "
            "pip install 'gsplat>=1.5,<2'"
        ) from exc


def _rasterize_2dgs_training(
    rasterizer: Any,
    values: tuple[torch.Tensor, ...],
    view: torch.Tensor,
    intrinsic: torch.Tensor,
    resolution: int,
    background: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Render RGB and depth required by gsplat's 2DGS distortion loss."""

    values = _expand_2dgs_colors_for_cameras(values, len(view))
    return rasterizer(
        *values,
        view,
        intrinsic,
        resolution,
        resolution,
        backgrounds=background,
        render_mode="RGB+ED",
        distloss=True,
    )


def _expand_2dgs_colors_for_cameras(
    values: tuple[torch.Tensor, ...], camera_count: int
) -> tuple[torch.Tensor, ...]:
    """Work around gsplat >=1.5.3's missing 2DGS color broadcasting.

    Camera-specific ``[C,N,D]`` colors are part of gsplat's public contract and
    work in both affected and unaffected releases.  ``expand`` also preserves
    gradient accumulation into the model's view-independent ``[N,D]`` colors.
    """

    if len(values) != 5:
        raise ValueError("Gaussian rasterizer values must contain five tensors")
    colors = values[-1]
    if colors.ndim != 2:
        raise ValueError("View-independent Gaussian colors must have shape [N,D]")
    expanded_colors = colors.unsqueeze(0).expand(camera_count, -1, -1).contiguous()
    return (*values[:-1], expanded_colors)


def _rasterize_3dgs(
    rasterizer: Any,
    values: tuple[torch.Tensor, ...],
    views: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int,
    background: torch.Tensor,
    render_mode: str,
) -> tuple[torch.Tensor, ...]:
    """Render 3DGS without gsplat's packed-background shape mismatch.

    In affected gsplat releases, packed projected means have shape ``[nnz,2]``;
    the low-level wrapper consequently validates backgrounds as ``[D]`` even
    though the public rasterizer contract requires ``[C,D]``.  Unpacked mode
    retains the camera dimension and works for single- and multi-camera calls.
    """

    return rasterizer(
        *values,
        views,
        intrinsics,
        resolution,
        resolution,
        backgrounds=background,
        render_mode=render_mode,
        rasterize_mode="antialiased",
        packed=False,
    )


def train_gaussian_splats(
    initial_points: np.ndarray,
    image_paths: Sequence[str | Path],
    training_cameras: SplatCameras,
    settings: GaussianSplatSettings,
    *,
    foreground_threshold: float = 245 / 255,
) -> tuple[GaussianParameters, list[dict[str, float]]]:
    """Optimize 2D or 3D splats against one selected view history."""

    rasterizer = require_gsplat(settings.backend)
    torch.manual_seed(settings.seed)
    images, masks = load_splat_images(
        image_paths, settings.resolution, foreground_threshold
    )
    colors = initialize_point_colors(initial_points, images, masks, training_cameras)
    device = torch.device("cuda")
    model = GaussianParameters(initial_points, colors, settings.backend).to(device)
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": settings.position_learning_rate},
        {"params": [model.log_scales, model.quaternions, model.opacity_logits,
                    model.color_logits], "lr": settings.learning_rate},
    ])
    target_images = torch.from_numpy(images).to(device)
    target_masks = torch.from_numpy(masks.astype(np.float32)).to(device)
    views = torch.from_numpy(training_cameras.world_to_camera.astype(np.float32)).to(device)
    intrinsics = torch.from_numpy(training_cameras.intrinsics.astype(np.float32)).to(device)
    white = torch.ones((1, 3), device=device)
    history: list[dict[str, float]] = []
    for step in range(settings.iterations):
        index = step % len(images)
        values = model.values()
        if settings.backend == "2dgs":
            rendered, alpha, normals, surface_normals, distortion, _, _ = (
                _rasterize_2dgs_training(
                    rasterizer,
                    values,
                    views[index : index + 1],
                    intrinsics[index : index + 1],
                    settings.resolution,
                    white,
                )
            )
            normal_valid = alpha[..., 0] > 0.05
            cosine = F.cosine_similarity(normals, surface_normals, dim=-1).abs()
            # Keep the reduction on the GPU, including when no pixels are valid.
            # Boolean indexing and a Python condition would synchronize with the CPU.
            normal_loss = (1 - cosine).masked_fill(~normal_valid, 0).sum() / (
                normal_valid.sum().clamp_min(1)
            )
            regularization = (
                settings.distortion_loss_weight * distortion.mean()
                + settings.normal_loss_weight * normal_loss
            )
        else:
            rendered, alpha, _ = _rasterize_3dgs(
                rasterizer,
                values,
                views[index : index + 1],
                intrinsics[index : index + 1],
                settings.resolution,
                white,
                "RGB",
            )
            regularization = torch.zeros((), device=device)
        color_loss = F.l1_loss(rendered[0, ..., :3], target_images[index])
        opacity_loss = F.l1_loss(alpha[0, ..., 0], target_masks[index])
        loss = color_loss + settings.opacity_loss_weight * opacity_loss + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == settings.iterations:
            history.append({
                "iteration": float(step + 1), "loss": float(loss.detach()),
                "color_l1": float(color_loss.detach()),
                "opacity_l1": float(opacity_loss.detach()),
            })
    return model, history


@torch.inference_mode()
def render_splats(
    model: GaussianParameters,
    cameras: SplatCameras,
    settings: GaussianSplatSettings,
    *,
    include_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rasterizer = require_gsplat(settings.backend)
    device = model.means.device
    views = torch.from_numpy(cameras.world_to_camera.astype(np.float32)).to(device)
    intrinsics = torch.from_numpy(cameras.intrinsics.astype(np.float32)).to(device)
    values = model.values()
    rgb_chunks, alpha_chunks, depth_chunks = [], [], []
    for start in range(0, len(views), settings.render_batch_size):
        batch_views = views[start : start + settings.render_batch_size]
        batch_intrinsics = intrinsics[start : start + settings.render_batch_size]
        white = torch.ones((len(batch_views), 3), device=device)
        if settings.backend == "2dgs":
            batch_values = _expand_2dgs_colors_for_cameras(
                values, len(batch_views)
            )
            rgb, alpha, _, _, _, median_depth, _ = rasterizer(
                *batch_values, batch_views, batch_intrinsics,
                settings.resolution, settings.resolution,
                backgrounds=white, render_mode="RGB", depth_mode="median",
            )
            depth = median_depth[..., 0] if include_depth else None
        else:
            mode = "RGB+ED" if include_depth else "RGB"
            rendered, alpha, _ = _rasterize_3dgs(
                rasterizer,
                values,
                batch_views,
                batch_intrinsics,
                settings.resolution,
                white,
                mode,
            )
            rgb = rendered[..., :3]
            depth = rendered[..., 3] if include_depth else None
        rgb_chunks.append(rgb.clamp(0, 1).cpu())
        alpha_chunks.append(alpha[..., 0].cpu())
        if depth is not None:
            depth_chunks.append(depth.cpu())
    return (
        torch.cat(rgb_chunks).numpy(),
        torch.cat(alpha_chunks).numpy(),
        torch.cat(depth_chunks).numpy() if depth_chunks else None,
    )


def fuse_depth_surfaces(
    depths: np.ndarray,
    alphas: np.ndarray,
    cameras: SplatCameras,
    *,
    alpha_threshold: float,
    point_count: int,
    seed: int,
) -> np.ndarray:
    """Back-project canonical rendered depths and deterministically voxel-fuse."""

    clouds = []
    for depth, alpha, camera_to_world, intrinsic in zip(
        depths, alphas, cameras.camera_to_world, cameras.intrinsics, strict=True
    ):
        rows, columns = np.nonzero(
            (alpha >= alpha_threshold) & np.isfinite(depth) & (depth > 0)
        )
        if not len(rows):
            continue
        z = depth[rows, columns]
        x = (columns + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0] * z
        y = (rows + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1] * z
        camera_points = np.stack((x, y, z), axis=1)
        world = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        clouds.append(world)
    if not clouds:
        raise ValueError("2DGS depth fusion produced no valid surface points")
    points = np.concatenate(clouds)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    voxel = max(extent / 512.0, 1e-6)
    keys = np.floor(points / voxel).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    points = points[np.sort(unique)]
    generator = np.random.default_rng(seed)
    if len(points) > point_count:
        points = points[np.sort(generator.choice(len(points), point_count, replace=False))]
    return points.astype(np.float32)


def evaluate_2dgs_geometry(
    model: GaussianParameters,
    evaluation_cameras: SplatCameras,
    target_points: np.ndarray,
    settings: GaussianSplatSettings,
) -> tuple[np.ndarray, dict[str, float]]:
    if settings.backend != "2dgs":
        raise ValueError("Geometry metrics are deliberately restricted to 2dgs")
    _, alphas, depths = render_splats(
        model, evaluation_cameras, settings, include_depth=True
    )
    assert depths is not None
    surface = fuse_depth_surfaces(
        depths, alphas, evaluation_cameras,
        alpha_threshold=settings.alpha_threshold,
        point_count=settings.surface_point_count,
        seed=settings.seed,
    )
    metrics = point_cloud_metrics(
        surface, target_points,
        chunk_size=settings.metric_chunk_size,
        fscore_thresholds=settings.fscore_thresholds,
        device="cuda",
    )
    return surface, metrics


def save_gaussian_checkpoint(
    path: Path, model: GaussianParameters, settings: GaussianSplatSettings,
    training_history: Sequence[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "settings": asdict(settings),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training_history": list(training_history),
    }, path)


def write_point_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n" f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point in points:
            handle.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def write_render_gallery(path: Path, images: np.ndarray, title: str) -> None:
    """Write a self-contained autoplay/slider gallery for qualitative 3DGS."""

    uris = []
    for array in images:
        buffer = BytesIO()
        Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(buffer, format="PNG")
        uris.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    payload = json.dumps({"title": title, "frames": uris}, separators=(",", ":"))
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Gaussian splat render</title>
<style>body{font:14px system-ui;background:#111827;color:#f8fafc;text-align:center;margin:0}h1{font-size:17px}
img{height:min(78vh,800px);max-width:94vw;image-rendering:auto;background:white}footer{padding:12px}input{width:min(700px,75vw)}</style></head>
<body><h1 id='title'></h1><img id='frame'><footer><button id='play'>Pause</button>
<input id='slider' type='range' min='0' value='0'><span id='number'></span></footer><script>
const data=__DATA__,image=document.getElementById('frame'),slider=document.getElementById('slider');
document.getElementById('title').textContent=data.title;slider.max=data.frames.length-1;let index=0,playing=true;
function show(){image.src=data.frames[index];slider.value=index;document.getElementById('number').textContent=` ${index+1}/${data.frames.length}`}
slider.oninput=()=>{index=Number(slider.value);show()};document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
setInterval(()=>{if(playing){index=(index+1)%data.frames.length;show()}},120);show();</script></body></html>""".replace("__DATA__", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_image_comparison_gallery(
    path: Path,
    references: np.ndarray,
    predictions: np.ndarray,
    title: str,
    *,
    training_view_ids: Sequence[int] = (),
) -> None:
    """Write synchronized GT/render views with side-by-side and overlay modes."""

    if references.shape != predictions.shape or references.ndim != 4:
        raise ValueError("reference and prediction galleries must have matching [V,H,W,3]")

    def encode(array: np.ndarray) -> str:
        buffer = BytesIO()
        Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(
            buffer, format="PNG"
        )
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    training = {int(value) for value in training_view_ids}
    payload = json.dumps({
        "title": title,
        "references": [encode(array) for array in references],
        "predictions": [encode(array) for array in predictions],
        "training": [index in training for index in range(len(references))],
    }, separators=(",", ":"))
    document = """<!doctype html><html><head><meta charset='utf-8'><title>3DGS comparison</title>
<style>*{box-sizing:border-box}body{font:14px system-ui;background:#111827;color:#f8fafc;text-align:center;margin:0}
h1{font-size:17px}.stage{height:min(75vh,800px);display:flex;justify-content:center;gap:18px;padding:8px}.panel{position:relative;height:100%;aspect-ratio:1;background:white}
.panel img{width:100%;height:100%;object-fit:contain}.label{position:absolute;top:8px;left:8px;background:#0f172acc;padding:4px 7px;border-radius:4px}.overlay{display:none}.overlay img{position:absolute;inset:0}.overlay #prediction{opacity:.5}
footer{padding:12px;display:flex;justify-content:center;align-items:center;gap:10px;flex-wrap:wrap}input[type=range]{width:min(600px,60vw)}</style></head>
<body><h1 id='title'></h1><div id='side' class='stage'><div class='panel'><img id='referenceSide'><span class='label'>Ground-truth NUM RGB</span></div><div class='panel'><img id='predictionSide'><span class='label'>3DGS render</span></div></div>
<div id='overlay' class='stage overlay'><div class='panel'><img id='reference'><img id='prediction'><span class='label'>Overlay</span></div></div>
<footer><button id='play'>Pause</button><label>Layout <select id='layout'><option value='side'>Side by side</option><option value='overlay'>Overlay</option></select></label>
<label>Overlay opacity <input id='opacity' type='range' min='0' max='1' step='.05' value='.5'></label><input id='slider' type='range' min='0' value='0'><span id='number'></span></footer><script>
const data=__DATA__,slider=document.getElementById('slider');document.getElementById('title').textContent=data.title;slider.max=data.references.length-1;let index=0,playing=true;
function show(){document.getElementById('referenceSide').src=data.references[index];document.getElementById('predictionSide').src=data.predictions[index];document.getElementById('reference').src=data.references[index];document.getElementById('prediction').src=data.predictions[index];slider.value=index;document.getElementById('number').textContent=`anchor ${index} · ${data.training[index]?'training view':'held-out view'} · ${index+1}/${data.references.length}`}
slider.oninput=()=>{index=Number(slider.value);show()};document.getElementById('layout').onchange=e=>{document.getElementById('side').style.display=e.target.value==='side'?'flex':'none';document.getElementById('overlay').style.display=e.target.value==='overlay'?'flex':'none'};
document.getElementById('opacity').oninput=e=>document.getElementById('prediction').style.opacity=e.target.value;document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
setInterval(()=>{if(playing){index=(index+1)%data.references.length;show()}},350);show();</script></body></html>""".replace("__DATA__", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
