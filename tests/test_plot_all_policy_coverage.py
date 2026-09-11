from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.plot_all_policy_coverage import (
    _completed_phase3_runs,
    _merge_curves,
    _phase3_policy_specs,
    _read_coverage,
)


class AllPolicyCoveragePlotTests(unittest.TestCase):
    def _run(self, root: Path, name: str, status: str, policy: str) -> Path:
        run = root / name / "seed_0"
        metrics = run / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "phase3_completion.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
        with (metrics / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("policy", "acquired_view_count", "coverage_mean", "object_count"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "policy": policy,
                    "acquired_view_count": 1,
                    "coverage_mean": 0.25,
                    "object_count": 3,
                }
            )
        return run

    def test_discovers_only_completed_runs_and_retains_new_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = self._run(root, "controlled_new_variant", "complete", "vggt_joint_new")
            self._run(root, "controlled_partial", "incomplete", "vggt_joint_partial")

            self.assertEqual(_completed_phase3_runs(root), [completed])
            curves, sizes = _read_coverage(completed / "metrics/coverage.csv")
            merged_curves: dict[str, list[tuple[int, float]]] = {}
            merged_sizes: dict[str, int] = {}
            _merge_curves(merged_curves, merged_sizes, curves, sizes, completed)
            specs = _phase3_policy_specs(set(merged_curves))

            self.assertIn("vggt_joint_new", [policy for policy, _, _, _ in specs])
            self.assertNotIn("vggt_joint_partial", [policy for policy, _, _, _ in specs])

    def test_repeated_identical_control_is_deduplicated(self) -> None:
        curves = {"vggt_independent_history": [(1, 0.25)]}
        sizes = {"vggt_independent_history": 3}
        _merge_curves(curves, sizes, dict(curves), dict(sizes), Path("second-run"))
        self.assertEqual(curves, {"vggt_independent_history": [(1, 0.25)]})

    def test_conflicting_repeated_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting coverage curves"):
            _merge_curves(
                {"vggt_independent_history": [(1, 0.25)]},
                {"vggt_independent_history": 3},
                {"vggt_independent_history": [(1, 0.5)]},
                {"vggt_independent_history": 3},
                Path("different-run"),
            )


if __name__ == "__main__":
    unittest.main()
