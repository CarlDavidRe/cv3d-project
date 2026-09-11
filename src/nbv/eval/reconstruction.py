"""Cached VGGT point-cloud evaluation for closed-loop view histories.

The reconstruction model is deliberately evaluator-owned.  Every selected
policy is scored with the same frozen VGGT point-map backend, so these metrics
compare view selection rather than policy-specific reconstruction systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from nbv.features.base import freeze_module, resolve_device
from nbv.features.model_cache import model_cache_directory
from nbv.geometry.anchors import canonical_anchors
from nbv.geometry.mesh import (
    MESH_CENTERING,
    TriangleMesh,
    load_mesh,
    prepare_visibility_mesh,
    sha256_file,
)
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world


RECONSTRUCTION_CACHE_SCHEMA_VERSION = 1
RECONSTRUCTION_METRIC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReconstructionSettings:
    enabled: bool
    policies: tuple[str, ...]
    view_counts: tuple[int, ...]
    backend: str
    model_id: str
    image_size: int
    point_source: str
    device: str
    confidence_percentile: float
    foreground_threshold: float
    predicted_point_count: int
    ground_truth_point_count: int
    chamfer_chunk_size: int
    fscore_thresholds: tuple[float, ...]
    mesh_root: Path
    mesh_relative_path: str
    mesh_scale: float
    mesh_centering: str
    cache_root: Path
    model_cache_root: Path

    def __post_init__(self) -> None:
        if self.backend != "vggt":
            raise ValueError("reconstruction.backend must be 'vggt'")
        if self.point_source != "point_map":
            raise ValueError("reconstruction.point_source must be 'point_map'")
        for name in (
            "image_size", "predicted_point_count", "ground_truth_point_count",
            "chamfer_chunk_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"reconstruction.{name} must be a positive integer")
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("reconstruction.policies must be non-empty and unique")
        if (
            not self.view_counts
            or len(set(self.view_counts)) != len(self.view_counts)
            or any(type(value) is not int or value <= 0 for value in self.view_counts)
        ):
            raise ValueError("reconstruction.view_counts must be unique positive integers")
        if tuple(sorted(self.view_counts)) != self.view_counts:
            raise ValueError("reconstruction.view_counts must be sorted")
        if not 0.0 <= self.confidence_percentile < 100.0:
            raise ValueError("reconstruction.confidence_percentile must lie in [0, 100)")
        if not 0.0 < self.foreground_threshold <= 1.0:
            raise ValueError("reconstruction.foreground_threshold must lie in (0, 1]")
        if (
            not self.fscore_thresholds
            or any(not math.isfinite(value) or value <= 0 for value in self.fscore_thresholds)
        ):
            raise ValueError("reconstruction.fscore_thresholds must be positive")
        if not math.isfinite(self.mesh_scale) or self.mesh_scale <= 0:
            raise ValueError("reconstruction.mesh_scale must be positive")
        if self.mesh_centering != MESH_CENTERING:
            raise ValueError(f"reconstruction.mesh_centering must be {MESH_CENTERING!r}")


def parse_reconstruction_settings(
    values: Mapping[str, Any] | None,
    paths: Mapping[str, Any],
    repository_root: str | Path,
    *,
    default_policies: Sequence[str],
) -> ReconstructionSettings | None:
    """Parse an optional shared reconstruction-evaluation configuration."""

    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError("reconstruction must be a mapping")
    enabled = values.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("reconstruction.enabled must be boolean")
    if not enabled:
        return None
    root = Path(repository_root).resolve()

    def rooted(value: Any, label: str) -> Path:
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError(f"{label} must be a path")
        path = Path(value)
        return path if path.is_absolute() else root / path

    policies = values.get("policies", list(default_policies))
    view_counts = values.get("view_counts")
    thresholds = values.get("fscore_thresholds", [0.01, 0.02])
    if not isinstance(policies, list) or any(not isinstance(v, str) for v in policies):
        raise TypeError("reconstruction.policies must be a string list")
    if not isinstance(view_counts, list):
        raise TypeError("reconstruction.view_counts must be an integer list")
    if not isinstance(thresholds, list):
        raise TypeError("reconstruction.fscore_thresholds must be a number list")
    unknown = set(policies) - set(default_policies)
    if unknown:
        raise ValueError(f"reconstruction policies are not evaluated: {sorted(unknown)}")
    return ReconstructionSettings(
        enabled=True,
        policies=tuple(policies),
        view_counts=tuple(view_counts),
        backend=str(values.get("backend", "vggt")),
        model_id=str(values.get("model_id", "facebook/VGGT-1B")),
        image_size=int(values.get("image_size", 518)),
        point_source=str(values.get("point_source", "point_map")),
        device=str(values.get("device", "auto")),
        confidence_percentile=float(values.get("confidence_percentile", 50.0)),
        foreground_threshold=float(values.get("foreground_threshold", 245.0 / 255.0)),
        predicted_point_count=int(values.get("predicted_point_count", 10_000)),
        ground_truth_point_count=int(values.get("ground_truth_point_count", 10_000)),
        chamfer_chunk_size=int(values.get("chamfer_chunk_size", 1024)),
        fscore_thresholds=tuple(float(v) for v in thresholds),
        mesh_root=rooted(paths.get("mesh_root"), "paths.mesh_root"),
        mesh_relative_path=str(values.get("mesh_relative_path", "models/model_normalized.ply")),
        mesh_scale=float(values.get("mesh_scale", 2.0)),
        mesh_centering=str(values.get("mesh_centering", MESH_CENTERING)),
        cache_root=rooted(paths.get("reconstruction_cache_root"), "paths.reconstruction_cache_root"),
        model_cache_root=rooted(paths.get("model_cache_root"), "paths.model_cache_root"),
    )


class PointCloudReconstructor(Protocol):
    """Injectable backend contract used by tests and the production VGGT adapter."""

    provenance: Mapping[str, Any]

    def reconstruct(
        self, image_paths: Sequence[str | Path]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return filtered world points [N,3] and OpenCV camera-to-world [S,4,4]."""


class VGGTPointCloudReconstructor:
    """Frozen full-model VGGT point-map adapter with foreground/confidence filtering."""

    def __init__(self, settings: ReconstructionSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        try:
            from vggt.models.vggt import VGGT

            model = VGGT.from_pretrained(
                settings.model_id,
                cache_dir=model_cache_directory(settings.model_cache_root, "huggingface"),
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "VGGT reconstruction requires the official facebookresearch/vggt package"
            ) from exc
        except Exception as exc:
            raise RuntimeError("Could not load the VGGT reconstruction checkpoint") from exc
        self.model = freeze_module(model).to(self.device)
        self.autocast_dtype: torch.dtype | None = None
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.provenance = _vggt_backend_provenance(settings)

    def reconstruct(
        self, image_paths: Sequence[str | Path]
    ) -> tuple[np.ndarray, np.ndarray]:
        images, foreground = _load_images_and_foreground(
            image_paths,
            self.settings.image_size,
            self.settings.foreground_threshold,
        )
        inputs = images.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            prediction = self.model(inputs)
        try:
            points = prediction["world_points"][0].detach().float().cpu().numpy()
            confidence = prediction["world_points_conf"][0].detach().float().cpu().numpy()
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            extrinsic, _ = pose_encoding_to_extri_intri(
                prediction["pose_enc"], inputs.shape[-2:]
            )
            world_to_camera = extrinsic[0].detach().float().cpu().numpy()
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError("VGGT output lacks usable point-map or camera predictions") from exc
        foreground_np = foreground.numpy()
        if foreground_np.shape != confidence.shape:
            masks = F.interpolate(
                foreground.unsqueeze(1).float(),
                size=confidence.shape[-2:],
                mode="nearest",
            )[:, 0].bool().numpy()
        else:
            masks = foreground_np
        valid = masks & np.isfinite(confidence) & np.isfinite(points).all(axis=-1)
        if not valid.any():
            raise ValueError("VGGT reconstruction has no finite foreground points")
        cutoff = float(np.percentile(confidence[valid], self.settings.confidence_percentile))
        selected = points[valid & (confidence >= cutoff)]
        if len(selected) < 3:
            raise ValueError("VGGT confidence filtering retained fewer than three points")
        selected = _deterministic_subsample(
            selected, self.settings.predicted_point_count, _history_seed(image_paths)
        )
        return selected.astype(np.float32), _camera_to_world(world_to_camera)


def evaluate_rollout_reconstruction(
    rollouts: Sequence[Any],
    settings: ReconstructionSettings,
    *,
    reconstructor: PointCloudReconstructor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate configured rollout prefixes and return rows, curves, provenance."""

    selected_rollouts = [
        result for result in rollouts if result.metadata["policy"] in settings.policies
    ]
    if not selected_rollouts:
        raise ValueError("No rollouts match reconstruction.policies")
    active = reconstructor
    backend_provenance = (
        dict(reconstructor.provenance)
        if reconstructor is not None
        else _vggt_backend_provenance(settings)
    )
    metric_device = resolve_device(settings.device)
    rows: list[dict[str, Any]] = []
    prediction_hits = prediction_misses = metric_hits = metric_misses = 0
    gt_cache: dict[str, tuple[np.ndarray, dict[str, Any], Path, bool]] = {}
    for result in selected_rollouts:
        object_id = result.metadata["object_id"]
        visibility_geometry = result.metadata.get("visibility_cache_metadata", {})
        for key, expected in (
            ("mesh_relative_path", settings.mesh_relative_path),
            ("mesh_scale", settings.mesh_scale),
            ("mesh_centering", settings.mesh_centering),
        ):
            if visibility_geometry.get(key) != expected:
                raise ValueError(
                    f"reconstruction {key} differs from rollout visibility geometry"
                )
        if object_id not in gt_cache:
            gt_cache[object_id] = _load_or_create_ground_truth(settings, object_id)
        target_points, target_meta, target_path, target_hit = gt_cache[object_id]
        for count in settings.view_counts:
            indices = np.flatnonzero(result.acquired_view_counts == count)
            if not len(indices):
                continue
            history = result.acquired_anchor_ids[:count].tolist()
            image_paths = result.image_paths[:count]
            prediction_path, prediction_identity = _prediction_cache_location(
                settings, object_id, history, image_paths
            )
            cached = _load_prediction_cache(prediction_path, prediction_identity)
            started = perf_counter()
            if cached is None:
                if active is None:
                    active = VGGTPointCloudReconstructor(settings)
                points, predicted_cameras = active.reconstruct(image_paths)
                points = np.asarray(points, dtype=np.float32)
                predicted_cameras = np.asarray(predicted_cameras, dtype=np.float32)
                _save_prediction_cache(
                    prediction_path, prediction_identity, points, predicted_cameras,
                    dict(active.provenance),
                )
                prediction_misses += 1
                prediction_hit = False
            else:
                points, predicted_cameras, cached_provenance = cached
                backend_provenance = cached_provenance
                prediction_hits += 1
                prediction_hit = True
            known_cameras = _known_cameras(result, history)
            transform, alignment = align_vggt_to_num(
                points, predicted_cameras, known_cameras, target_points
            )
            aligned = _apply_similarity(points, transform)
            metric_path, metric_identity = _metric_cache_location(
                settings, prediction_identity, target_meta, alignment
            )
            metric = _load_metric_cache(metric_path, metric_identity)
            if metric is None:
                metric = point_cloud_metrics(
                    aligned,
                    target_points,
                    chunk_size=settings.chamfer_chunk_size,
                    fscore_thresholds=settings.fscore_thresholds,
                    device=metric_device,
                )
                _save_json_cache(
                    metric_path, {"identity": metric_identity, "metrics": metric}
                )
                metric_misses += 1
                metric_hit = False
            else:
                metric_hits += 1
                metric_hit = True
            rows.append({
                "object_id": object_id,
                "policy": result.metadata["policy"],
                "acquired_view_count": count,
                "history_anchor_ids": history,
                **metric,
                "alignment_mode": alignment["mode"],
                "alignment_scale": alignment["scale"],
                "camera_center_rmse_normalized": alignment["camera_center_rmse_normalized"],
                "predicted_point_count": len(aligned),
                "ground_truth_point_count": len(target_points),
                "prediction_cache_hit": prediction_hit,
                "metric_cache_hit": metric_hit,
                "ground_truth_cache_hit": target_hit,
                "prediction_cache_path": str(prediction_path.resolve()),
                "prediction_cache_sha256": sha256_file(prediction_path),
                "metric_cache_path": str(metric_path.resolve()),
                "evaluation_ms": (perf_counter() - started) * 1000.0,
            })
    curves = reconstruction_curve_rows(rows, settings.policies)
    return rows, curves, {
        "schema_version": RECONSTRUCTION_METRIC_SCHEMA_VERSION,
        "backend": "vggt_shared_across_policies",
        "interpretation": "view_selection_quality_under_a_common_frozen_vggt_reconstructor",
        "settings": _settings_json(settings),
        "backend_provenance": backend_provenance,
        "prediction_cache_hits": prediction_hits,
        "prediction_cache_misses": prediction_misses,
        "metric_cache_hits": metric_hits,
        "metric_cache_misses": metric_misses,
        "ground_truth_cache_paths": {
            object_id: str(value[2].resolve()) for object_id, value in gt_cache.items()
        },
    }


def reconstruction_curve_rows(
    rows: Sequence[Mapping[str, Any]], policies: Sequence[str]
) -> list[dict[str, Any]]:
    curves = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        for count in sorted({int(row["acquired_view_count"]) for row in selected}):
            cohort = [row for row in selected if int(row["acquired_view_count"]) == count]
            curves.append({
                "policy": policy,
                "acquired_view_count": count,
                "object_count": len(cohort),
                **{
                    f"{metric}_mean": float(np.mean([float(row[metric]) for row in cohort]))
                    for metric in _metric_names(cohort[0])
                },
            })
    return curves


def reconstruction_policy_summary(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["policy"] == policy]
    if not selected:
        return {}
    counts = sorted({int(row["acquired_view_count"]) for row in selected})
    final_count = counts[-1]
    final = [row for row in selected if int(row["acquired_view_count"]) == final_count]
    by_count = {
        count: [row for row in selected if int(row["acquired_view_count"]) == count]
        for count in counts
    }
    mean_chamfer = [
        float(np.mean([float(row["chamfer_l1_normalized"]) for row in by_count[count]]))
        for count in counts
    ]
    summary = {
        "reconstruction_object_count": len({row["object_id"] for row in selected}),
        "reconstruction_final_view_count": final_count,
        "final_chamfer_l1_normalized_mean": float(np.mean([
            float(row["chamfer_l1_normalized"]) for row in final
        ])),
        "chamfer_l1_normalized_auc": (
            0.0 if len(counts) == 1
            else float(np.trapz(mean_chamfer, x=np.asarray(counts, dtype=np.float64)))
        ),
    }
    for metric in _metric_names(final[0]):
        summary[f"final_{metric}_mean"] = float(np.mean([
            float(row[metric]) for row in final
        ]))
    return summary


def point_cloud_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int,
    fscore_thresholds: Sequence[float],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Compute diameter-normalized accuracy, completeness, Chamfer-L1 and F-scores."""

    predicted = _points(predicted, "predicted")
    target = _points(target, "target")
    diameter = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("target point cloud must have positive bounding-box diameter")
    resolved = torch.device(device)
    left = torch.from_numpy(predicted.astype(np.float32)).to(resolved)
    right = torch.from_numpy(target.astype(np.float32)).to(resolved)
    accuracy_distances = _nearest_distances(left, right, chunk_size).cpu().numpy()
    completeness_distances = _nearest_distances(right, left, chunk_size).cpu().numpy()
    accuracy = float(np.mean(accuracy_distances) / diameter)
    completeness = float(np.mean(completeness_distances) / diameter)
    result = {
        "accuracy_normalized": accuracy,
        "completeness_normalized": completeness,
        "chamfer_l1_normalized": 0.5 * (accuracy + completeness),
        "ground_truth_diameter": diameter,
    }
    for threshold in fscore_thresholds:
        absolute = float(threshold) * diameter
        precision = float(np.mean(accuracy_distances <= absolute))
        recall = float(np.mean(completeness_distances <= absolute))
        fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        suffix = _threshold_suffix(threshold)
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"fscore_{suffix}"] = fscore
    return result


def align_vggt_to_num(
    points: np.ndarray,
    predicted_camera_to_world: np.ndarray,
    known_camera_to_world: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate Sim(3) from predicted cameras, with a one-view scale fallback."""

    points = _points(points, "points")
    target = _points(target_points, "target_points")
    predicted = np.asarray(predicted_camera_to_world, dtype=np.float64)
    known = np.asarray(known_camera_to_world, dtype=np.float64)
    if predicted.shape != known.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise ValueError("camera arrays must have matching shape [S,4,4]")
    # VGGT uses OpenCV camera axes. NUM poses use Blender/OpenGL axes.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    predicted_gl = predicted @ flip
    candidates = known[:, :3, :3] @ np.swapaxes(predicted_gl[:, :3, :3], 1, 2)
    u, _, vt = np.linalg.svd(candidates.sum(axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    predicted_centers = predicted_gl[:, :3, 3]
    known_centers = known[:, :3, 3]
    pred_centered = predicted_centers - predicted_centers.mean(axis=0)
    known_centered = known_centers - known_centers.mean(axis=0)
    denominator = float(np.sum(pred_centered * pred_centered))
    if len(predicted) >= 2 and denominator > 1e-12:
        rotated = pred_centered @ rotation.T
        scale = float(np.sum(rotated * known_centered) / denominator)
        mode = "camera_pose_sim3"
    else:
        predicted_diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        target_diameter = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
        if predicted_diameter <= 1e-12:
            raise ValueError("single-view predicted point cloud has zero diameter")
        scale = target_diameter / predicted_diameter
        mode = "camera_rotation_center_and_target_diameter_scale_single_view"
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("camera alignment produced a non-positive scale")
    translation = known_centers.mean(axis=0) - scale * (
        predicted_centers.mean(axis=0) @ rotation.T
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    aligned_centers = scale * (predicted_centers @ rotation.T) + translation
    target_diameter = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    rmse = float(np.sqrt(np.mean(np.sum((aligned_centers - known_centers) ** 2, axis=1))))
    return transform, {
        "mode": mode,
        "scale": scale,
        "camera_center_rmse_normalized": rmse / target_diameter,
    }


def _load_or_create_ground_truth(
    settings: ReconstructionSettings, object_id: str
) -> tuple[np.ndarray, dict[str, Any], Path, bool]:
    mesh_path = settings.mesh_root / object_id / settings.mesh_relative_path
    mesh_sha = sha256_file(mesh_path)
    identity = {
        "schema_version": RECONSTRUCTION_CACHE_SCHEMA_VERSION,
        "kind": "ground_truth_surface_sample",
        "object_id": object_id,
        "mesh_sha256": mesh_sha,
        "mesh_scale": settings.mesh_scale,
        "mesh_centering": settings.mesh_centering,
        "point_count": settings.ground_truth_point_count,
        "sampling": "triangle_area_barycentric_pcg64_sha256_seed_v1",
    }
    digest = _mapping_sha256(identity)
    path = settings.cache_root / "ground_truth" / object_id / f"{digest[:20]}.npz"
    cached = _load_array_cache(path, identity, "points")
    if cached is not None:
        return cached.astype(np.float32), identity, path, True
    mesh, _ = prepare_visibility_mesh(
        load_mesh(mesh_path), scale=settings.mesh_scale, centering=settings.mesh_centering
    )
    points = sample_mesh_surface(mesh, settings.ground_truth_point_count, int(digest[:16], 16))
    _save_array_cache(path, identity, points=points.astype(np.float32))
    return points.astype(np.float32), identity, path, False


def sample_mesh_surface(mesh: TriangleMesh, count: int, seed: int) -> np.ndarray:
    triangles = mesh.triangles
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 0
    if not valid.any():
        raise ValueError("mesh has no positive-area triangles")
    triangles = triangles[valid]
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=probabilities)]
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1
    uv[reflected] = 1 - uv[reflected]
    return (
        chosen[:, 0]
        + uv[:, :1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    )


def _prediction_cache_location(
    settings: ReconstructionSettings,
    object_id: str,
    history: Sequence[int],
    image_paths: Sequence[str | Path],
) -> tuple[Path, dict[str, Any]]:
    identity = {
        "schema_version": RECONSTRUCTION_CACHE_SCHEMA_VERSION,
        "kind": "vggt_filtered_point_map",
        "object_id": object_id,
        "history_anchor_ids": list(history),
        "image_sha256": [sha256_file(path) for path in image_paths],
        "backend": settings.backend,
        "model_id": settings.model_id,
        "image_size": settings.image_size,
        "point_source": settings.point_source,
        "confidence_percentile": settings.confidence_percentile,
        "foreground_threshold": settings.foreground_threshold,
        "predicted_point_count": settings.predicted_point_count,
    }
    digest = _mapping_sha256(identity)
    path = settings.cache_root / "predictions" / object_id / f"{digest[:24]}.npz"
    return path, identity


def _metric_cache_location(
    settings: ReconstructionSettings,
    prediction_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    identity = {
        "schema_version": RECONSTRUCTION_METRIC_SCHEMA_VERSION,
        "prediction_id": _mapping_sha256(prediction_identity),
        "target_id": _mapping_sha256(target_identity),
        "alignment": dict(alignment),
        "alignment_definition": "vggt_opencv_to_num_opengl_camera_orientation_sim3_v1",
        "chamfer_definition": "half_mean_bidirectional_euclidean_distance_divided_by_gt_bbox_diameter",
        "chamfer_chunk_size": settings.chamfer_chunk_size,
        "fscore_thresholds_gt_bbox_diameter": list(settings.fscore_thresholds),
    }
    digest = _mapping_sha256(identity)
    object_id = str(prediction_identity["object_id"])
    return settings.cache_root / "metrics" / object_id / f"{digest[:24]}.json", identity


def _load_prediction_cache(
    path: Path, identity: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["identity_json"].item())) != dict(identity):
                return None
            points = payload["points"].copy()
            cameras = payload["camera_to_world"].copy()
            provenance = json.loads(str(payload["provenance_json"].item()))
        _points(points, "cached points")
        if cameras.ndim != 3 or cameras.shape[1:] != (4, 4):
            return None
        if not isinstance(provenance, dict):
            return None
        return points, cameras, provenance
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_prediction_cache(
    path: Path,
    identity: Mapping[str, Any],
    points: np.ndarray,
    camera_to_world: np.ndarray,
    provenance: Mapping[str, Any],
) -> None:
    _save_array_cache(
        path,
        identity,
        points=np.asarray(points, dtype=np.float32),
        camera_to_world=np.asarray(camera_to_world, dtype=np.float32),
        provenance_json=np.asarray(json.dumps(dict(provenance), sort_keys=True)),
    )


def _load_metric_cache(path: Path, identity: Mapping[str, Any]) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != dict(identity):
            return None
        metrics = payload["metrics"]
        if not isinstance(metrics, dict) or not all(math.isfinite(float(v)) for v in metrics.values()):
            return None
        return {str(k): float(v) for k, v in metrics.items()}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_array_cache(
    path: Path, identity: Mapping[str, Any], key: str
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["identity_json"].item())) != dict(identity):
                return None
            value = payload[key].copy()
        _points(value, f"cached {key}")
        return value
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _save_array_cache(path: Path, identity: Mapping[str, Any], **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                identity_json=np.asarray(json.dumps(dict(identity), sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _save_json_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_images_and_foreground(
    paths: Sequence[str | Path], image_size: int, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    masks = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        images.append(tensor)
        masks.append(torch.from_numpy(np.min(rgb, axis=-1) < threshold))
    image_batch = torch.stack(images)
    image_batch = F.interpolate(
        image_batch, size=(image_size, image_size), mode="bilinear",
        align_corners=False, antialias=True,
    )
    mask_batch = F.interpolate(
        torch.stack(masks).unsqueeze(1).float(), size=(image_size, image_size), mode="nearest"
    )[:, 0].bool()
    return image_batch, mask_batch


def _camera_to_world(world_to_camera: np.ndarray) -> np.ndarray:
    matrices = np.asarray(world_to_camera, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 4):
        raise ValueError("VGGT extrinsics must have shape [S,3,4]")
    homogeneous = np.broadcast_to(np.eye(4), (len(matrices), 4, 4)).copy()
    homogeneous[:, :3] = matrices
    return np.linalg.inv(homogeneous)


def _known_cameras(result: Any, history: Sequence[int]) -> np.ndarray:
    metadata = result.metadata["visibility_cache_metadata"]
    radius = float(metadata["camera_radius"])
    convention = metadata.get("camera_convention", CAMERA_CONVENTION)
    anchors = canonical_anchors()
    return np.stack([
        anchor_camera_to_world(anchors.by_id(anchor_id), radius, convention=convention)
        for anchor_id in history
    ])


def _apply_similarity(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _nearest_distances(source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(source), chunk_size):
        chunks.append(torch.cdist(source[start : start + chunk_size], target).min(dim=1).values)
    return torch.cat(chunks)


def _deterministic_subsample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points
    indices = np.random.default_rng(seed).choice(len(points), size=count, replace=False)
    return points[np.sort(indices)]


def _history_seed(paths: Sequence[str | Path]) -> int:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(sha256_file(path)))
    return int.from_bytes(digest.digest()[:8], "big")


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) < 1:
        raise ValueError(f"{name} must have shape [N,3]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _metric_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        key for key in row
        if key in {"accuracy_normalized", "completeness_normalized", "chamfer_l1_normalized"}
        or key.startswith(("precision_", "recall_", "fscore_"))
    )


def _threshold_suffix(value: float) -> str:
    percentage = 100.0 * float(value)
    return f"{percentage:g}pct".replace(".", "p")


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _settings_json(settings: ReconstructionSettings) -> dict[str, Any]:
    result = asdict(settings)
    for key in ("mesh_root", "cache_root", "model_cache_root"):
        result[key] = str(result[key])
    result["policies"] = list(settings.policies)
    result["view_counts"] = list(settings.view_counts)
    result["fscore_thresholds"] = list(settings.fscore_thresholds)
    return result


def _vggt_backend_provenance(settings: ReconstructionSettings) -> dict[str, Any]:
    return {
        "backend": "vggt",
        "model_id": settings.model_id,
        "point_source": settings.point_source,
        "image_size": settings.image_size,
        "preprocessing": "square_bilinear_resize_rgb_0_1",
        "coordinate_output": "vggt_world_points",
        "camera_output": "opencv_camera_to_world_from_pose_encoding",
        "frozen": True,
    }
