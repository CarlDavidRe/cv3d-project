from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nbv.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiments"
            / "phase1.yaml"
        )

    def test_checked_in_config_loads(self) -> None:
        config = load_config(self.config_path)
        self.assertEqual(config["experiment"]["phase"], "phase1")
        self.assertEqual(config["paths"]["data_root"], "data/NUM")
        self.assertEqual(
            config["paths"]["model_cache_root"], "data/cache/models"
        )
        self.assertEqual(config["phase1"]["num_anchors"], 48)
        self.assertEqual(
            config["phase1"]["num_dataset"]["target_name"], "PSNR"
        )

    def test_tiny_probe_config_loads_and_supports_training_overrides(self) -> None:
        path = self.config_path.with_name("phase1_probe_tiny.yaml")
        config = load_config(
            path,
            [
                "probe.backbone=vggt",
                "probe.extraction_batch_size=1",
                "probe.feature_selection.vggt=[pooled_camera, pooled_patch]",
            ],
        )

        self.assertEqual(config["probe"]["backbone"], "vggt")
        self.assertEqual(config["probe"]["extraction_batch_size"], 1)
        self.assertEqual(
            config["probe"]["feature_selection"]["vggt"],
            ["pooled_camera", "pooled_patch"],
        )

    def test_existing_fields_can_be_overridden_with_typed_values(self) -> None:
        config = load_config(
            self.config_path,
            ["experiment.seed=7", "experiment.deterministic=false"],
        )
        self.assertEqual(config["experiment"]["seed"], 7)
        self.assertFalse(config["experiment"]["deterministic"])

    def test_unknown_override_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.config_path, ["model.hidden_size=128"])

    def test_step_numbered_output_components_are_rejected(self) -> None:
        for override in (
            "experiment.phase=step5",
            "experiment.name=step_5",
            "paths.output_root=outputs/step-5",
            "paths.model_cache_root=data/step5/models",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ConfigError, "step-numbered"
            ):
                load_config(self.config_path, [override])

    def test_unsafe_run_path_components_are_rejected(self) -> None:
        for override in (
            "experiment.phase=../outside",
            "experiment.name=feature/smoke",
            "experiment.name=FeatureSmoke",
        ):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                load_config(self.config_path, [override])

    def test_non_mapping_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
