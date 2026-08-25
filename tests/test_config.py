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
        self.assertEqual(config["phase1"]["num_anchors"], 48)

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

    def test_non_mapping_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
