from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbv.config import load_config
from nbv.reproducibility import initialize_run, seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_seed_repeats_python_and_numpy_sequences(self) -> None:
        seed_everything(12)
        first = (random.random(), np.random.random())
        seed_everything(12)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_initialize_run_writes_config_and_metadata(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_config(
            repository_root / "configs" / "experiments" / "phase1.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            config["paths"]["output_root"] = temporary_directory
            context = initialize_run(config, repository_root)

            self.assertEqual(context.run_id, "phase1/infrastructure/seed_0")
            self.assertEqual(
                context.run_dir,
                Path(temporary_directory)
                / "phase1"
                / "infrastructure"
                / "seed_0",
            )
            self.assertTrue((context.run_dir / "config.yaml").is_file())
            metadata_path = context.run_dir / "metadata.json"
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 0)
            self.assertIn("git_commit", metadata)
            self.assertTrue(context.checkpoint_dir.is_dir())
            self.assertTrue(context.metrics_dir.is_dir())
            self.assertTrue(context.figure_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
