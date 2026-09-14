#!/usr/bin/env python3
"""Refit saved 2DGS/3DGS placement to acquired RGB silhouettes on CPU.

No VGGT inference, Gaussian retraining, cache writes, or ground-truth fitting.
This is a separate silhouette-refined evaluation protocol, not a replacement
for the original camera-only evaluation. Run --help for batch filters.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
from PIL import Image
import torch

from nbv.data.visibility_cache import load_visibility_cache
from nbv.eval.gaussian_splatting import (
    known_num_splat_cameras, GaussianParameters, load_splat_images,
    write_image_comparison_gallery, write_render_gallery,
)
from nbv.eval.gaussian_splatting_cpu import render_depth_cpu
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world
from nbv.geometry.visibility import PerspectiveCamera
from scripts.reevaluate_gaussian_splatting_cpu import (
    OUTPUT_FILES, digest, recover, write_aggregate,
)
from scripts.visualize_reconstruction import validate_object_id

VERSION = "acquired_rendered_silhouette_sim3_v2"
RGB_RENDERER_VERSION = "numpy_3dgs_antialiased_rgb_v1"
RGB_OUTPUT_FILES = ("ground_truth_comparison.html", "turntable.html", "preview.png", "summary.json")


def render_rgb_3dgs_cpu(model, cameras, resolution, *, progress=None,
                        return_alpha=False):
    """Antialiased pinhole EWA splatting with white background, constant RGB.

    Follows gsplat v1.5.3 ProjectionEWA3DGSFused/ RasterizeToPixels3DGSFwd:
    covariance projection, eps2d=.3, tile coverage, front-to-back alpha.
    CPU/CUDA parity is not verified. No spherical harmonics are used by this
    repository's GaussianParameters checkpoints.
    """
    if resolution <= 0:
        raise ValueError("positive resolution is required")
    means, quats, scales, opacity, colors = [x.detach().cpu().numpy() for x in model.values()]
    if not all(np.isfinite(x).all() for x in (means, quats, scales, opacity, colors)):
        raise ValueError("non-finite Gaussian parameters")
    rotation = quaternion_rotation(torch.tensor(quats)).numpy()
    axes = rotation * scales[:, None, :]
    covariance = axes @ axes.transpose(0, 2, 1)
    frames, alphas = [], []
    yy, xx = np.mgrid[:resolution, :resolution].astype(np.float32) + .5
    for camera_index, (view, intrinsic) in enumerate(zip(cameras.world_to_camera, cameras.intrinsics)):
        r = view[:3, :3].astype(np.float32)
        centers = means @ r.T + view[:3, 3].astype(np.float32)
        valid = (centers[:, 2] > .01) & (centers[:, 2] < 1e10) & (opacity >= 1/255)
        indices = np.flatnonzero(valid)
        centers = centers[indices]
        z = centers[:, 2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        uv = centers[:, :2] / z[:, None]
        projected = uv * [fx, fy] + [cx, cy]
        # Clamp only the Jacobian's off-axis terms, not projected centers.
        clipped = np.clip(uv, [-cx/fx-.3*resolution/(2*fx), -cy/fy-.3*resolution/(2*fy)],
                         [(resolution-cx)/fx+.3*resolution/(2*fx), (resolution-cy)/fy+.3*resolution/(2*fy)])
        jacobian = np.zeros((len(indices), 2, 3), dtype=np.float32)
        jacobian[:, 0, 0], jacobian[:, 1, 1] = fx/z, fy/z
        jacobian[:, 0, 2], jacobian[:, 1, 2] = -fx*clipped[:, 0]/z, -fy*clipped[:, 1]/z
        cov_camera = r @ covariance[indices] @ r.T
        cov_unfiltered = jacobian @ cov_camera @ jacobian.transpose(0, 2, 1)
        cov_screen = cov_unfiltered + np.eye(2)*.3
        # Match the training/evaluation wrapper's rasterize_mode='antialiased'.
        compensation = np.sqrt(np.maximum(0., np.linalg.det(cov_unfiltered)/np.linalg.det(cov_screen)))
        effective_opacity = opacity[indices]*compensation
        conic = np.linalg.inv(cov_screen)
        extend = np.minimum(3.33, np.sqrt(np.maximum(0., 2*np.log(np.maximum(effective_opacity*255, 1e-30)))))
        radii = np.ceil(extend[:, None] * np.sqrt(np.diagonal(cov_screen, axis1=1, axis2=2)))
        transmission = np.ones((resolution, resolution), dtype=np.float32)
        rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
        done = np.zeros((resolution, resolution), dtype=bool)
        for j in np.argsort(z, kind="stable"):
            if effective_opacity[j] < 1/255:
                continue
            if (projected[j]+radii[j] <= 0).any() or (projected[j]-radii[j] >= resolution).any():
                continue
            lo = np.clip(np.floor((projected[j]-radii[j])/16)*16, 0, resolution).astype(int)
            hi = np.clip(np.ceil((projected[j]+radii[j])/16)*16, 0, resolution).astype(int)
            region = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
            dx, dy = xx[region]-projected[j, 0], yy[region]-projected[j, 1]
            sigma = .5*(conic[j, 0, 0]*dx*dx + 2*conic[j, 0, 1]*dx*dy + conic[j, 1, 1]*dy*dy)
            alpha = np.minimum(.999, effective_opacity[j]*np.exp(-sigma))
            active = (~done[region]) & (sigma >= 0) & (alpha >= 1/255)
            next_t = transmission[region]*(1-alpha)
            done[region] |= active & (next_t <= 1e-4)
            active &= ~done[region]
            rgb[region] += (transmission[region]*alpha*active)[..., None]*colors[indices[j]]
            transmission[region] = np.where(active, next_t, transmission[region])
        frames.append(np.clip(rgb+transmission[..., None], 0, 1))
        alphas.append(1-transmission)
        if progress:
            progress(camera_index+1, len(cameras.world_to_camera))
    rendered = np.stack(frames)
    return (rendered, np.stack(alphas)) if return_alpha else rendered


def mean_silhouette_iou(alphas: np.ndarray, masks: list[np.ndarray],
                        *, alpha_threshold: float = .5) -> float:
    """Mean per-view IoU between rendered alpha and acquired RGB foreground."""
    rendered = np.asarray(alphas)
    targets = np.asarray(masks, dtype=bool)
    if rendered.shape != targets.shape:
        raise ValueError("rendered alpha and silhouette masks must have matching shapes")
    if not np.isfinite(rendered).all() or not 0 < alpha_threshold < 1:
        raise ValueError("rendered alpha must be finite and threshold must lie in (0, 1)")
    predicted = rendered >= alpha_threshold
    intersection = np.logical_and(predicted, targets).sum(axis=(1, 2))
    union = np.logical_or(predicted, targets).sum(axis=(1, 2))
    return float(np.mean(np.divide(intersection, union, out=np.ones_like(
        intersection, dtype=np.float64), where=union != 0)))


def checkpoint_silhouette_alpha(checkpoint: dict, cameras, resolution: int) -> np.ndarray:
    """Render checkpoint opacity on CPU for placement acceptance validation."""
    backend = checkpoint["settings"]["backend"]
    model = GaussianParameters(
        checkpoint["state_dict"]["means"].numpy(),
        np.full(checkpoint["state_dict"]["means"].shape, .5),
        backend,
    )
    model.load_state_dict(checkpoint["state_dict"])
    if backend == "2dgs":
        alpha, _ = render_depth_cpu(model, cameras, resolution)
        return alpha
    _, alpha = render_rgb_3dgs_cpu(
        model, cameras, resolution, return_alpha=True,
    )
    return alpha


def validate_fitted_checkpoint(checkpoint: dict, fitted: dict, cameras,
                               masks: list[np.ndarray], *, resolution: int,
                               min_iou_improvement: float) -> tuple[dict, dict]:
    """Accept a proxy candidate only when its rendered acquired-mask IoU improves."""
    if not 0 <= min_iou_improvement <= 1:
        raise ValueError("minimum IoU improvement must lie in [0, 1]")
    before = mean_silhouette_iou(
        checkpoint_silhouette_alpha(checkpoint, cameras, resolution), masks,
    )
    candidate = apply_checkpoint_transform(
        checkpoint, fitted["scale"], np.asarray(fitted["quaternion_wxyz"]),
        np.asarray(fitted["translation"]),
    )
    after = before
    gate_accepted = False
    if fitted["accepted"]:
        after = mean_silhouette_iou(
            checkpoint_silhouette_alpha(candidate, cameras, resolution), masks,
        )
        gate_accepted = after >= before + min_iou_improvement
    validation = {
        "metric": "mean_per_view_rendered_mask_iou",
        "alpha_threshold": .5,
        "before": before,
        "candidate": after,
        "candidate_evaluated": bool(fitted["accepted"]),
        "minimum_absolute_improvement": min_iou_improvement,
        "accepted": gate_accepted,
    }
    if gate_accepted:
        fitted["acceptance"] = "proxy_and_rendered_silhouette_iou"
        fitted["rendered_silhouette_validation"] = validation
        return candidate, fitted

    rejected = dict(fitted)
    rejected_candidate = None
    if fitted["accepted"]:
        rejected_candidate = {
            key: rejected[key] for key in (
                "scale", "quaternion_wxyz", "translation", "transform",
            )
        }
        rejected_candidate["silhouette_proxy"] = fitted["silhouette_proxy_after"]
    rejected.update({
        "accepted": False,
        "scale": 1.,
        "quaternion_wxyz": [1., 0., 0., 0.],
        "translation": [0., 0., 0.],
        "transform": np.eye(4).tolist(),
        "silhouette_proxy_after": fitted["silhouette_proxy_before"],
        "acceptance": (
            "identity_retained_by_rendered_silhouette_iou"
            if fitted["accepted"] else "identity_retained_by_proxy"
        ),
        "rendered_silhouette_validation": validation,
    })
    if rejected_candidate is not None:
        rejected["rejected_candidate"] = rejected_candidate
    return checkpoint, rejected


def quaternion_rotation(q: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternions to active rotation matrices."""
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


def apply_checkpoint_transform(checkpoint: dict, scale: float, quaternion: np.ndarray,
                               translation: np.ndarray) -> dict:
    """Transform centers AND covariance axes, preserving colors and opacity."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("repair scale must be finite and positive")
    q = torch.as_tensor(quaternion, dtype=torch.float32)
    t = torch.as_tensor(translation, dtype=torch.float32)
    if q.shape != (4,) or t.shape != (3,) or not torch.isfinite(q).all() or not torch.isfinite(t).all() or q.norm() < 1e-8:
        raise ValueError("invalid repair quaternion or translation")
    q = torch.nn.functional.normalize(q, dim=-1)
    state = {key: value.detach().cpu().clone() for key, value in checkpoint["state_dict"].items()}
    state["means"] = scale * (state["means"] @ quaternion_rotation(q).T) + t
    state["log_scales"] += math.log(scale)
    old = torch.nn.functional.normalize(state["quaternions"], dim=-1)
    scalar = q[0]*old[:, :1] - (q[1:]*old[:, 1:]).sum(-1, keepdim=True)
    vector = q[0]*old[:, 1:] + old[:, :1]*q[1:] + torch.linalg.cross(
        q[1:].expand_as(old[:, 1:]), old[:, 1:], dim=-1,
    )
    state["quaternions"] = torch.cat((scalar, vector), dim=-1)
    return {**checkpoint, "state_dict": state}


def silhouette_targets(masks: list[np.ndarray], count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    targets = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        if len(rows) < 8:
            raise ValueError("an acquired RGB foreground mask has fewer than 8 pixels")
        xy = np.stack(((columns+.5)/mask.shape[1]-.5,
                       (rows+.5)/mask.shape[0]-.5), -1)
        targets.append(xy[rng.choice(len(xy), count, replace=len(xy) < count)])
    return torch.tensor(np.stack(targets), dtype=torch.float32)


def silhouette_loss(world: torch.Tensor, views: torch.Tensor, focal: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    """Symmetric projected-point / foreground distance, in image-width units.

    This is a placement proxy, not a rendered-mask IoU or a geometry metric.
    All acquired cameras contribute equally; no unacquired RGB is loaded.
    """
    camera = torch.einsum("bnj,vkj->bvnk", world, views[:, :3, :3]) + views[None, :, None, :3, 3]
    xy = camera[..., :2] / camera[..., 2:].clamp_min(.05) * focal[None, :, None]
    distances = ((xy[:, :, :, None] - targets[None, :, None])**2).sum(-1)
    return (distances.min(-1).values.mean((-1, -2))
            + distances.min(-2).values.mean((-1, -2))
            + torch.relu(.05-camera[..., 2]).square().mean((-1, -2)))


def fit_similarity(points: np.ndarray, cameras, masks: list[np.ndarray], *,
                   resolution: int, radius: float, steps: int = 200,
                   starts: int = 32, seed: int = 0, progress=None) -> dict:
    """Fit seven placement parameters using RGB only; ground truth is not an input."""
    if len(masks) < 2:
        raise ValueError("at least two acquired views are required; one-view depth/scale is ambiguous")
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("need at least eight finite Gaussian centers")
    rng = np.random.default_rng(seed)
    center = (points.max(0) + points.min(0)) / 2
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    if extent < 1e-8:
        raise ValueError("Gaussian centers have negligible spatial extent")
    selected = rng.choice(len(points), min(256, len(points)), replace=False)
    p = torch.tensor((points[selected]-center)/extent, dtype=torch.float32)
    views = torch.tensor(cameras.world_to_camera, dtype=torch.float32)
    focal = torch.tensor(cameras.intrinsics[:, [0, 1], [0, 1]]/resolution, dtype=torch.float32)
    targets = silhouette_targets(masks, 128, seed)
    q = torch.tensor(rng.normal(size=(starts, 4)), dtype=torch.float32)
    q[0] = torch.tensor([1., 0., 0., 0.])
    q = torch.nn.Parameter(q)
    shift = torch.nn.Parameter(torch.zeros(starts, 3))
    # NUM cameras orbit the object origin. Use this acquisition prior only as
    # an initialization; translation, rotation and scale are all optimized.
    log_size = torch.nn.Parameter(torch.full((starts,), math.log(radius*.6)))
    optimizer = torch.optim.Adam([q, shift, log_size], lr=.035)
    best_loss = torch.full((starts,), float("inf"))
    best_q, best_shift, best_size = q.detach().clone(), shift.detach().clone(), log_size.detach().clone()
    for step in range(steps + 1):
        world = torch.einsum("nj,bkj->bnk", p, quaternion_rotation(q)) * log_size.exp()[:, None, None] + shift[:, None]
        losses = silhouette_loss(world, views, focal, targets)
        if not torch.isfinite(losses).all():
            raise ValueError("non-finite silhouette optimization")
        with torch.no_grad():
            improved = losses < best_loss
            best_loss[improved] = losses[improved]
            best_q[improved], best_shift[improved], best_size[improved] = q[improved], shift[improved], log_size[improved]
        if progress and (step % 50 == 0 or step == steps):
            progress(f"placement {step}/{steps}: silhouette proxy {float(best_loss.min()):.6f}")
        if step == steps:
            break
        optimizer.zero_grad()
        losses.sum().backward()
        optimizer.step()
        with torch.no_grad():
            log_size.clamp_(math.log(radius*.01), math.log(radius*1.8))
            shift.clamp_(-radius*.8, radius*.8)

    # Select on a second, denser deterministic sample, including the original
    # placement. Never force a correction simply because fitting was requested.
    validation = torch.tensor(points[rng.choice(len(points), min(1024, len(points)), replace=False)], dtype=torch.float32)
    validation_targets = silhouette_targets(masks, 384, seed+1)
    with torch.no_grad():
        before = float(silhouette_loss(validation[None], views, focal, validation_targets)[0])
        candidates = []
        for index in range(starts):
            quat = torch.nn.functional.normalize(best_q[index], dim=-1)
            rotation = quaternion_rotation(quat)
            scale = float(best_size[index].exp())/extent
            translation = best_shift[index] - scale * rotation @ torch.tensor(center, dtype=torch.float32)
            fitted = scale * (validation @ rotation.T) + translation
            value = float(silhouette_loss(fitted[None], views, focal, validation_targets)[0])
            candidates.append((value, scale, quat.numpy(), translation.numpy()))
    after, scale, quat, translation = min(candidates, key=lambda item: item[0])
    accepted = after < before * .99
    if not accepted:
        scale, quat, translation, after = 1., np.array([1., 0., 0., 0.]), np.zeros(3), before
    transform = np.eye(4)
    transform[:3, :3] = scale * quaternion_rotation(torch.tensor(quat)).numpy()
    transform[:3, 3] = translation
    return {
        "accepted": accepted, "scale": scale, "quaternion_wxyz": quat.tolist(),
        "translation": translation.tolist(), "transform": transform.tolist(),
        "silhouette_proxy_before": before, "silhouette_proxy_after": after,
        "fitted_to": "acquired_rgb_foreground_only", "ground_truth_used_for_fit": False,
        "limitations": "Similarity placement only; cannot recover missing or deformed geometry. Silhouettes may admit symmetric/ambiguous fits.",
    }


def audit_cameras(summary: dict, prediction_root: Path, known: np.ndarray) -> dict:
    """Relative-rotation disagreement cannot be removed by ANY global Sim(3)."""
    audit = {"original_alignment": summary.get("alignment"), "prediction_cache_available": False}
    recorded = summary.get("prediction_cache")
    if not recorded:
        return audit
    path = prediction_root / summary["object_id"] / Path(recorded).name
    if not path.is_file():
        return audit
    with np.load(path, allow_pickle=False) as data:
        identity = json.loads(str(data["identity_json"].item()))
        if identity["history_anchor_ids"] != summary["history_anchor_ids"]:
            raise ValueError("prediction cache history mismatch")
        predicted = data["camera_to_world"]
    if predicted.shape != known.shape or not np.isfinite(predicted).all():
        raise ValueError("invalid cached predicted cameras")
    known_cv = known @ np.diag([1., -1., -1., 1.])
    errors = []
    for first, second in itertools.combinations(range(len(known)), 2):
        relative_predicted = predicted[first, :3, :3].T @ predicted[second, :3, :3]
        relative_known = known_cv[first, :3, :3].T @ known_cv[second, :3, :3]
        cosine = (np.trace(relative_predicted @ relative_known.T)-1)/2
        errors.append(float(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))))
    audit.update(prediction_cache_available=True, prediction_cache_sha256=digest(path),
                 relative_rotation_errors_degrees=errors,
                 median_relative_rotation_error_degrees=float(np.median(errors)) if errors else None)
    return audit


def repair(source: Path, destination: Path, args) -> str:
    summary = json.loads((source / "summary.json").read_text())
    object_id = validate_object_id(summary["object_id"])
    backend = summary["backend"]
    if backend not in ("2dgs", "3dgs") or "alignment_repair" in summary:
        raise ValueError("input must be an original Gaussian training artifact")
    output_files = OUTPUT_FILES if backend == "2dgs" else RGB_OUTPUT_FILES
    history = summary["history_anchor_ids"]
    if len(history) != summary["acquired_view_count"] or len(set(history)) != len(history):
        raise ValueError("inconsistent acquired history")
    if len(history) < 2:
        return "skipped: one-view placement is underconstrained"
    category, key = object_id.split("/")
    visibility_path = args.visibility_cache_root / category / f"{key}.npz"
    metadata = load_visibility_cache(visibility_path).metadata
    anchors = canonical_anchors()
    poses = np.stack([anchor_camera_to_world(anchors.by_id(i), float(metadata["camera_radius"]),
        convention=metadata.get("camera_convention", CAMERA_CONVENTION)) for i in history])
    audit = audit_cameras(summary, args.prediction_cache_root, poses)
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint["settings"]["backend"] != backend:
        raise ValueError("unsupported checkpoint")
    images = [args.data_root / object_id / "images" / f"viewpoint_{i}_offset_phi_0.png" for i in history]
    masks = []
    for path in images:
        with Image.open(path) as im:
            masks.append(np.asarray(im.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)).min(-1) < 245)
    silhouette_targets(masks, 128, args.seed)  # validate before publishing or fitting
    min_iou_improvement = getattr(args, "min_iou_improvement", .01)
    identity = {"version": VERSION, "steps": args.steps, "starts": args.starts, "seed": args.seed,
                "min_iou_improvement": min_iou_improvement,
                "inputs": {name: digest(path) for name, path in (
                    ("checkpoint", source / "checkpoint.pt"), ("summary", source / "summary.json"),
                    ("visibility", visibility_path))},
                "acquired_rgb_sha256": [digest(path) for path in images], "audit": audit}
    if backend == "2dgs":
        identity["inputs"]["ground_truth"] = digest(source / "ground_truth.ply")
    else:
        # Unacquired RGB is for the comparison gallery only, never placement fitting.
        reference_paths = [args.data_root / object_id / "images" / f"viewpoint_{i.anchor_id}_offset_phi_0.png" for i in anchors]
        identity["reference_rgb_sha256"] = [digest(path) for path in reference_paths]
        identity["renderer"] = RGB_RENDERER_VERSION
    if args.dry_run:
        return f"ready: {len(history)} acquired images; camera audit: {json.dumps(audit)}"
    completed = destination / "summary.json"
    if not args.force and all((destination / name).is_file() for name in (*output_files, "checkpoint.pt")):
        previous = json.loads(completed.read_text())
        if previous.get("alignment_repair", {}).get("identity") == identity:
            return "skipped: matching completed repair"
    cameras = known_num_splat_cameras(poses, PerspectiveCamera(height=64, width=64,
        horizontal_fov_degrees=float(metadata["horizontal_fov_degrees"])))
    fitted = fit_similarity(checkpoint["state_dict"]["means"].numpy(), cameras, masks,
        resolution=64, radius=float(metadata["camera_radius"]), steps=args.steps,
        starts=args.starts, seed=args.seed, progress=lambda message: print(f"  {message}", flush=True))
    repaired, fitted = validate_fitted_checkpoint(
        checkpoint, fitted, cameras, masks, resolution=64,
        min_iou_improvement=min_iou_improvement,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alignment-repair-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        torch.save(repaired, stage / "checkpoint.pt")
        if backend == "2dgs":
            shutil.copyfile(source / "ground_truth.ply", stage / "ground_truth.ply")
            (stage / "summary.json").write_text(json.dumps(summary))
            # Reextract surface visibility after placement changes.
            result = recover(stage, stage, args.visibility_cache_root, force=True)
            recovered = json.loads((stage / "summary.json").read_text())
            recovered["metric_status"] = "silhouette_refined_saved_checkpoint_separate_protocol"
            recovered["cpu_recovery"]["source"] = str(source.resolve())
            recovered["cpu_recovery"]["checkpoint_used"] = str((destination / "checkpoint.pt").resolve())
            recovered.pop("turntable", None)
            for field, filename in (("static", "comparison.png"), ("interactive", "comparison_interactive.html"),
                                    ("combined_point_cloud", "comparison.ply")):
                recovered["ground_truth_comparison"][field] = str((destination / filename).resolve())
        else:
            resolution = checkpoint["settings"]["resolution"]
            all_poses = np.stack([anchor_camera_to_world(i, float(metadata["camera_radius"]),
                convention=metadata.get("camera_convention", CAMERA_CONVENTION)) for i in anchors])
            all_cameras = known_num_splat_cameras(all_poses, PerspectiveCamera(height=resolution, width=resolution,
                horizontal_fov_degrees=float(metadata["horizontal_fov_degrees"])))
            model = GaussianParameters(repaired["state_dict"]["means"].numpy(),
                np.full(repaired["state_dict"]["means"].shape, .5), "3dgs")
            model.load_state_dict(repaired["state_dict"])
            def report(done, total):
                if done == 1 or done % 8 == 0 or done == total:
                    print(f"  RGB cameras {done}/{total}", flush=True)
            renders = render_rgb_3dgs_cpu(model, all_cameras, resolution, progress=report)
            references, _ = load_splat_images(reference_paths, resolution, 245/255)
            title = f"Silhouette-refined CPU 3DGS · {object_id} · {summary['policy']} · {len(history)} views"
            write_image_comparison_gallery(stage / "ground_truth_comparison.html", references, renders,
                                           title, training_view_ids=history)
            write_render_gallery(stage / "turntable.html", renders, title + " · canonical anchors")
            Image.fromarray(np.uint8(renders[history[0]]*255)).save(stage / "preview.png")
            recovered = dict(summary)
            recovered.pop("metrics", None)
            recovered["metric_status"] = "disabled_by_contract_3dgs_is_qualitative_visualization_only"
            recovered["cpu_rendering"] = {"renderer": RGB_RENDERER_VERSION, "cuda_parity_verified": False,
                                         "anchor_ids": [i.anchor_id for i in anchors]}
            recovered["turntable"] = str((destination / "turntable.html").resolve())
            recovered["ground_truth_comparison"] = {
                "type": "canonical_render_vs_corresponding_num_rgb",
                "interactive": str((destination / "ground_truth_comparison.html").resolve()),
                "training_view_ids": history,
                "held_out_view_ids": [i.anchor_id for i in anchors if i.anchor_id not in history],
            }
            result = "complete: regenerated 48 repaired 3DGS RGB views"
        recovered["alignment_repair"] = {"identity": identity, "source": str(source.resolve()), **fitted}
        (stage / "summary.json").write_text(json.dumps(recovered, indent=2, allow_nan=False)+"\n")
        destination.mkdir(parents=True, exist_ok=True)
        # Completion marker last; interrupted repairs are recomputed.
        completed.unlink(missing_ok=True)
        for filename in ("checkpoint.pt", *output_files):
            os.replace(stage / filename, destination / filename)
    return f"{result}; placement {'refined' if fitted['accepted'] else 'retained'}; proxy {fitted['silhouette_proxy_before']:.6f} -> {fitted['silhouette_proxy_after']:.6f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/gaussian_splatting_variant_comparison")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/gaussian_splatting_alignment_repair")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/NUM")
    parser.add_argument("--visibility-cache-root", type=Path, default=ROOT / "data/cache/visibility")
    parser.add_argument("--prediction-cache-root", type=Path, default=ROOT / "data/cache/reconstruction/predictions",
                        help="Optional existing VGGT predictions, used only for camera diagnostics")
    parser.add_argument("--object-id")
    parser.add_argument("--variant", help="For example phase2_pun")
    parser.add_argument("--views", type=int, nargs="+")
    parser.add_argument("--backend", choices=("2dgs", "3dgs", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--starts", type=int, default=32, help="Independent initial rotations for placement fitting")
    parser.add_argument("--min-iou-improvement", type=float, default=.01,
                        help="Minimum absolute rendered silhouette IoU gain required to accept a placement")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and audit cached cameras without writing")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.steps, args.starts, args.threads) <= 0
            or (args.limit is not None and args.limit <= 0) or args.seed < 0
            or not 0 <= args.min_iou_improvement <= 1):
        parser.error(
            "steps, starts, threads, and limit must be positive; seed must be "
            "nonnegative; min-iou-improvement must lie in [0, 1]"
        )
    source_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        parser.error("input and output roots must be separate, non-nested directories")
    for protected in (args.data_root, args.visibility_cache_root, args.prediction_cache_root):
        protected = protected.resolve()
        if protected == output_root or protected in output_root.parents or output_root in protected.parents:
            parser.error("output root must not overlap RGB or cache directories")
    torch.set_num_threads(args.threads)
    sources = []
    for path in sorted(source_root.glob("*/*/*views/phase*/*/summary.json")):
        relative = path.relative_to(source_root)
        if relative.parts[4] not in ("2dgs", "3dgs") or (args.backend != "both" and relative.parts[4] != args.backend):
            continue
        if args.object_id and "/".join(relative.parts[:2]) != args.object_id:
            continue
        if args.variant and relative.parts[3] != args.variant:
            continue
        if args.views and int(relative.parts[2].removesuffix("views")) not in args.views:
            continue
        sources.append(path.parent)
    sources = sources[:args.limit] if args.limit else sources
    if not sources:
        print("No matching Gaussian artifacts found", file=sys.stderr)
        return 1
    failures = 0
    for index, source in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {source.relative_to(source_root)}", flush=True)
        try:
            result = repair(
                source, output_root / source.relative_to(source_root), args,
            )
            print("  " + result, flush=True)
            # Keep the dashboard aggregate usable during long or interrupted batches.
            if (not args.dry_run and source.name == "2dgs"
                    and not result.startswith("skipped")):
                write_aggregate(output_root)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
    if not args.dry_run:
        write_aggregate(output_root)
    print(f"Finished: {len(sources)} histories checked, {failures} failed", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
