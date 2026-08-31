from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import yaml

from nbv.visualization import (
    prediction_diagnostics,
    write_phase1_prediction_svg,
)
from scripts.visualize_phase1 import visualize_experiment_variants


class Phase1PredictionVisualizationTests(unittest.TestCase):
    def test_prediction_svg_marks_masked_and_best_anchors(self) -> None:
        target = np.arange(48, dtype=np.float32)
        prediction = target.copy()
        prediction[12] = -4.0
        mask = np.ones(48, dtype=np.bool_)
        mask[0] = False
        diagnostics = prediction_diagnostics(
            target,
            prediction,
            mask,
            target_direction="lower",
        )
        self.assertEqual(diagnostics.target_anchor_id, 1)
        self.assertEqual(diagnostics.predicted_anchor_id, 12)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            destination = root / "prediction.svg"
            Image.new("RGB", (12, 10), color=(220, 220, 220)).save(source)
            write_phase1_prediction_svg(
                target,
                prediction,
                mask,
                source,
                destination,
                title="fixture prediction",
                variant_name="fixture",
                sample_id="category/object/0",
                split="val",
                target_name="PSNR",
                target_direction="lower",
                source_global_anchor_id=17,
                feature_source="fixture feature",
            )
            ET.parse(destination)
            document = destination.read_text(encoding="utf-8")

        self.assertIn("data:image/png;base64,", document)
        self.assertIn('data-role="masked" data-anchor-id="0"', document)
        self.assertIn('data-role="target-best" data-anchor-id="1"', document)
        self.assertIn(
            'data-role="predicted-best" data-anchor-id="12"', document
        )
        self.assertIn("Ground-truth top: local anchor 1", document)
        self.assertIn("Predicted top: local anchor 12", document)

    def test_cli_loader_defaults_to_one_figure_per_saved_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "NUM"
            split_path = root / "split.json"
            experiment = root / "experiment"
            object_root = data_root / "category" / "object"
            images = object_root / "images"
            targets = object_root / "uncertainties"
            images.mkdir(parents=True)
            targets.mkdir()
            target = list(range(48))
            for anchor_id in range(48):
                stem = f"viewpoint_{anchor_id}_offset_phi_0"
                Image.new("RGB", (8, 8), color=(anchor_id, 30, 60)).save(
                    images / f"{stem}.png"
                )
                (targets / f"{stem}.json").write_text(
                    json.dumps({"PSNR": target}), encoding="utf-8"
                )
            split_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "splits": {
                            "train": [],
                            "val": ["category/object"],
                            "test": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "experiment": {
                    "phase": "phase1",
                    "name": "fixture",
                    "seed": 0,
                    "deterministic": True,
                },
                "paths": {
                    "data_root": str(data_root),
                    "output_root": str(root / "outputs"),
                    "model_cache_root": str(root / "models"),
                },
                "phase1": {
                    "num_anchors": 48,
                    "num_dataset": {
                        "split_manifest": str(split_path),
                        "target_name": "PSNR",
                    },
                },
                "probe": {
                    "mask_source_view": True,
                    "feature_cache": {"root": str(root / "features")},
                    "evaluation": {"ndcg_k": 5},
                },
            }
            experiment.mkdir()
            (experiment / "config.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            variant = experiment / "variants" / "train_mean_map"
            variant.mkdir(parents=True)
            torch.save(
                {"prediction_map": torch.arange(48, dtype=torch.float32)},
                variant / "best.pt",
            )
            (variant / "summary.json").write_text(
                json.dumps({"target_direction": "lower"}), encoding="utf-8"
            )
            second_variant = experiment / "variants" / "second_fixed_map"
            second_variant.mkdir(parents=True)
            torch.save(
                {"prediction_map": torch.arange(47, -1, -1)},
                second_variant / "best.pt",
            )
            (second_variant / "summary.json").write_text(
                json.dumps({"target_direction": "lower"}), encoding="utf-8"
            )

            results = visualize_experiment_variants(experiment)
            for result in results:
                ET.parse(result.output)
            generated = sorted(
                path.relative_to(experiment)
                for path in experiment.rglob("*")
                if path.is_file()
                and path not in {
                    experiment / "config.yaml",
                    variant / "best.pt",
                    variant / "summary.json",
                    second_variant / "best.pt",
                    second_variant / "summary.json",
                }
            )

        self.assertEqual(
            [result.variant_name for result in results],
            ["second_fixed_map", "train_mean_map"],
        )
        self.assertEqual(results[1].diagnostics.predicted_anchor_id, 1)
        self.assertEqual(
            generated,
            [
                Path(
                    "figures/predictions/val/category/object/0/"
                    "second_fixed_map.svg"
                ),
                Path(
                    "figures/predictions/val/category/object/0/"
                    "train_mean_map.svg"
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
