#!/usr/bin/env python3
"""Compare NUM mesh projections with real RGB and audit visibility caches.

This is a geometry diagnostic, not policy evaluation or camera fitting.
Objects and anchors are supplied before inspecting the resulting metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbv.config import load_config
from nbv.data import load_object_split, load_visibility_cache, visibility_cache_path
from nbv.eval.result_schema import write_json
from nbv.geometry import (
    PerspectiveCamera, canonical_anchors, load_mesh, render_face_index_map,
    visible_face_union,
)
from nbv.geometry.mesh import prepare_visibility_mesh, sha256_file
from nbv.geometry.num_camera import CAMERA_CONVENTION, anchor_camera_to_world
from nbv.visualization.alignment import (
    best_translation, boundary_distances, mask_extent, shift_mask,
    silhouette_iou, translation_scores,
)
from precompute_visibility import _expected_metadata, _visibility_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/phase2_visibility.yaml")
    parser.add_argument("--object", action="append", required=True, metavar="CATEGORY/OBJECT")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Verify that every supplied object belongs to this split.")
    parser.add_argument("--anchors", type=int, nargs="+", default=[0, 1, 12, 24, 46])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase2/visibility_alignment")
    parser.add_argument("--compare-source-mesh", action="store_true", help="Also diagnose the scaled source mesh before centering; does not affect caches.")
    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    try:
        config = load_config(args.config)
        settings = _visibility_settings(config)
        object_ids = sorted(set(args.object))
        if args.split:
            members = {"/".join(key) for key in load_object_split(resolve(settings["split_manifest"]))[args.split]}
            if set(object_ids) - members:
                raise ValueError(f"Objects must belong to the {args.split} split")
            settings["split"] = args.split
        anchor_ids = tuple(dict.fromkeys(args.anchors))
        for anchor_id in anchor_ids:
            canonical_anchors().by_id(anchor_id)
        destination = args.output_dir.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        report = {
            "camera_convention": CAMERA_CONVENTION, "settings": settings,
            "verified_object_split": args.split,
            "split_manifest_sha256": sha256_file(resolve(settings["split_manifest"])),
            "rgb_foreground_definition": "minimum_RGB_channel < 245 on original pixels",
            "projected_foreground_definition": "Configured face silhouette box-downsampled to RGB size, any covered subpixel",
            "interpretation": "Silhouette agreement is diagnostic, not exact RGB reproduction. Post-hoc translation searches do not modify camera geometry or caches.",
            "boundary_definition": "4-neighbor foreground boundary; pooled bidirectional nearest-boundary Euclidean distances in RGB pixels",
            "translation_search": {"radius_pixels": 8, "positive_axes": "x right, y down", "crop_policy": "exclude shifts losing foreground", "purpose": "diagnostic only; fitted on these images"},
            "mesh_normalization": "Subtract source bounding-box midpoint, then apply configured scale; no RGB fitting",
            "source_mesh_comparison": args.compare_source_mesh,
            "validation_scope": "supplied object/anchor subset; not a full-dataset registration guarantee",
            "objects": [], "failures": [],
        }
        for object_id in object_ids:
            visibility_cache_path("unused", object_id)
            mesh_path = resolve(config["paths"]["mesh_root"]) / object_id / settings["mesh_relative_path"]
            source_mesh = load_mesh(mesh_path)
            mesh, mesh_to_world = prepare_visibility_mesh(
                source_mesh, scale=settings["mesh_scale"], centering=settings["mesh_centering"],
            )
            expected = _expected_metadata(object_id, sha256_file(mesh_path), len(mesh.faces), settings, mesh_to_world)
            cache_path = visibility_cache_path(resolve(config["paths"]["visibility_cache_root"]), object_id)
            cache = None
            if cache_path.exists():
                cache = load_visibility_cache(cache_path, expected_metadata=expected)
                history = [0, 12, 24]
                for target in ("vis", "vis_a"):
                    gains = cache.candidate_gains(history, target=target)
                    differences = [
                        cache.coverage(history + [j], target=target) - cache.coverage(history, target=target)
                        for j in range(48)
                    ]
                    np.testing.assert_allclose(gains, differences, atol=1e-12)
                    np.testing.assert_array_equal(gains[history], 0)
                np.testing.assert_array_equal(
                    visible_face_union(cache.face_visibility, history),
                    visible_face_union(cache.face_visibility, history[::-1]),
                )
            row = {
                "object_id": object_id, "mesh_sha256": expected["mesh_sha256"],
                "cache_path": str(cache_path), "cache_checked": cache is not None,
                "cache_sha256": sha256_file(cache_path) if cache is not None else None,
                "views": [],
                "source_mesh_bounds": source_mesh.bounds.tolist(),
                "mesh_to_world": mesh_to_world.tolist(),
                "prepared_mesh_bounds": mesh.bounds.tolist(),
            }
            panel = Image.new("RGB", ((5 if args.compare_source_mesh else 4) * 224, len(anchor_ids) * 244), "white")
            draw = ImageDraw.Draw(panel)
            for index, anchor_id in enumerate(anchor_ids):
                image_path = resolve(config["paths"]["data_root"]) / object_id / "images" / f"viewpoint_{anchor_id}_offset_phi_0.png"
                with Image.open(image_path) as source:
                    rgb = source.convert("RGB")
                foreground = np.asarray(rgb).min(axis=2) < 245
                camera = PerspectiveCamera(
                    height=settings["render_resolution"][0], width=settings["render_resolution"][1],
                    horizontal_fov_degrees=settings["horizontal_fov_degrees"],
                    near=settings["near"], far=settings["far"],
                )
                pose = anchor_camera_to_world(canonical_anchors()[anchor_id], settings["camera_radius"])
                face_ids = render_face_index_map(mesh, pose, camera, cull_backfaces=settings["cull_backfaces"])
                if cache is not None:
                    visible = np.zeros(len(mesh.faces), bool)
                    visible[np.unique(face_ids[face_ids >= 0])] = True
                    np.testing.assert_array_equal(visible, cache.face_visibility[anchor_id])
                projected = Image.fromarray((face_ids >= 0).astype(np.uint8) * 255)
                downsampled = np.asarray(projected.resize(rgb.size, Image.Resampling.BOX))
                mask = downsampled > 0
                native_camera = PerspectiveCamera(
                    height=rgb.height, width=rgb.width,
                    horizontal_fov_degrees=camera.horizontal_fov_degrees,
                    near=camera.near, far=camera.far,
                )
                native_mask = render_face_index_map(mesh, pose, native_camera, cull_backfaces=settings["cull_backfaces"]) >= 0
                iou = silhouette_iou(mask, foreground)
                shifts = translation_scores(native_mask, foreground, radius=8)
                best = best_translation(shifts)
                shifted = shift_mask(native_mask, best["dx"], best["dy"])
                any_shifts = translation_scores(mask, foreground, radius=8)
                comparison = {}
                if args.compare_source_mesh:
                    source_ids = render_face_index_map(source_mesh.scaled(settings["mesh_scale"]), pose, camera, cull_backfaces=settings["cull_backfaces"])
                    source_image = Image.fromarray((source_ids >= 0).astype(np.uint8) * 255)
                    source_mask = np.asarray(source_image.resize(rgb.size, Image.Resampling.BOX)) > 0
                    comparison = {
                        "source_mesh_iou": silhouette_iou(source_mask, foreground),
                        "source_mesh_boundary": boundary_distances(source_mask, foreground),
                    }
                row["views"].append({
                    "anchor_id": anchor_id, "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path), "rgb_size": list(rgb.size),
                    "silhouette_iou": iou,
                    "native_resolution_iou": silhouette_iou(native_mask, foreground),
                    "majority_subpixel_iou": silhouette_iou(downsampled >= 128, foreground),
                    "rgb_threshold_sweep_native_iou": {
                        str(threshold): silhouette_iou(native_mask, np.asarray(rgb).min(axis=2) < threshold)
                        for threshold in (230, 245, 254)
                    },
                    "rgb_extent": mask_extent(foreground),
                    "native_mesh_extent": mask_extent(native_mask),
                    "native_boundary": boundary_distances(native_mask, foreground),
                    "any_subpixel_boundary": boundary_distances(mask, foreground),
                    "native_translation_scores": shifts,
                    "native_best_translation": best,
                    "native_boundary_after_best_translation": boundary_distances(shifted, foreground),
                    "any_subpixel_translation_scores": any_shifts,
                    "any_subpixel_best_translation": best_translation(any_shifts),
                    **comparison,
                })
                def overlay(value):
                    pixels = np.full((*foreground.shape, 3), 255, np.uint8)
                    pixels[foreground & value] = [70, 170, 80]
                    pixels[foreground & ~value] = [230, 95, 70]
                    pixels[value & ~foreground] = [65, 130, 220]
                    return Image.fromarray(pixels)
                tiles = [rgb, overlay(mask), overlay(native_mask), overlay(shifted)]
                labels = [f"RGB anchor {anchor_id}", f"Any subpixel: {iou:.3f}",
                          f"Native {rgb.width}x{rgb.height}: {silhouette_iou(native_mask, foreground):.3f}",
                          f"Diagnostic shift ({best['dx']},{best['dy']}): {best['iou']:.3f}"]
                if args.compare_source_mesh:
                    tiles.append(overlay(source_mask))
                    labels.append(f"Scaled source: {comparison['source_mesh_iou']:.3f}")
                for column, (tile, label) in enumerate(zip(tiles, labels)):
                    panel.paste(tile.resize((224, 224), Image.Resampling.NEAREST), (column * 224, index * 244 + 20))
                    draw.text((column * 224 + 3, index * 244 + 3), label, fill="black")
                print(f"{object_id} anchor {anchor_id}: IoU={iou:.3f}", flush=True)
            panel_path = destination / f"{object_id.replace('/', '_')}.png"
            panel.save(panel_path)
            row["figure"] = panel_path.name
            report["objects"].append(row)
            write_json(report, destination / "report.json")
        values = [view["silhouette_iou"] for obj in report["objects"] for view in obj["views"]]
        report["silhouette_iou_mean"] = float(np.mean(values))
        views = [view for obj in report["objects"] for view in obj["views"]]
        report["native_resolution_iou_mean"] = float(np.mean([view["native_resolution_iou"] for view in views]))
        report["majority_subpixel_iou_mean"] = float(np.mean([view["majority_subpixel_iou"] for view in views]))
        common_shifts = set.intersection(*[
            {(s["dx"], s["dy"]) for s in view["native_translation_scores"]} for view in views
        ])
        pooled = [{"dx": dx, "dy": dy, "iou": float(np.mean([
            next(s["iou"] for s in view["native_translation_scores"] if (s["dx"], s["dy"]) == (dx, dy))
            for view in views
        ]))} for dx, dy in sorted(common_shifts)]
        report["native_best_shared_translation"] = best_translation(pooled)
        report["native_best_per_view_translation_iou_mean"] = float(np.mean([view["native_best_translation"]["iou"] for view in views]))
        report["any_subpixel_best_per_view_translation_iou_mean"] = float(np.mean([view["any_subpixel_best_translation"]["iou"] for view in views]))
        if args.compare_source_mesh:
            report["source_mesh_iou_mean"] = float(np.mean([view["source_mesh_iou"] for view in views]))
        write_json(report, destination / "report.json")
        print(json.dumps({key: value for key, value in report.items() if key.endswith("_mean")}, indent=2))
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print(f"Alignment audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
