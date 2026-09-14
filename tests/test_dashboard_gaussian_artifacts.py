from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dashboard.gaussian_artifacts import (
    GAUSSIAN_ALIGNMENT_REPAIR_VERSION,
    is_current_gaussian_repair,
    is_current_gaussian_training,
    summary_file_matches,
)


def summary(backend: str = "2dgs", views: int = 5) -> dict:
    total = 1_500 * views
    return {
        "backend": backend,
        "acquired_view_count": views,
        "settings": {"iterations": total},
        "training_budget": {
            "iterations_per_view": 1_500,
            "view_count": views,
            "total_iterations": total,
        },
        "alignment_repair": {
            "identity": {"version": GAUSSIAN_ALIGNMENT_REPAIR_VERSION},
        },
    }


class DashboardGaussianArtifactTests(unittest.TestCase):
    def test_current_training_requires_exact_default_per_view_budget(self) -> None:
        current = summary()
        self.assertTrue(is_current_gaussian_training(current, "2dgs"))
        current["settings"]["iterations"] = 1_500
        self.assertFalse(is_current_gaussian_training(current, "2dgs"))
        self.assertFalse(is_current_gaussian_training(summary(), "3dgs"))

    def test_current_repair_rejects_old_and_unrepaired_summaries(self) -> None:
        current = summary("3dgs", 10)
        self.assertTrue(is_current_gaussian_repair(current, "3dgs"))
        current["alignment_repair"]["identity"]["version"] = (
            "acquired_silhouette_sim3_v1"
        )
        self.assertFalse(is_current_gaussian_repair(current, "3dgs"))
        current.pop("alignment_repair")
        self.assertFalse(is_current_gaussian_repair(current, "3dgs"))

    def test_summary_file_validation_handles_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text("not json", encoding="utf-8")
            self.assertFalse(summary_file_matches(
                path, backend="2dgs", repaired=True,
            ))
            path.write_text(json.dumps(summary()), encoding="utf-8")
            self.assertTrue(summary_file_matches(
                path, backend="2dgs", repaired=True,
            ))


if __name__ == "__main__":
    unittest.main()
