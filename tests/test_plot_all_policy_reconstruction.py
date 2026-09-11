from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.plot_all_policy_reconstruction import (
    METRICS,
    _merge_reconstruction,
    _read_reconstruction,
    write_plot,
)
from scripts.plot_all_policy_coverage import PHASE2_POLICIES, _phase3_policy_specs


class AllPolicyReconstructionPlotTests(unittest.TestCase):
    def _csv(self, root: Path, policies: tuple[str, ...]) -> Path:
        path = root / "reconstruction_curves.csv"
        fields = [
            "policy",
            "acquired_view_count",
            "object_count",
            *(metric for metric, _, _ in METRICS),
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for policy in policies:
                for count in (1, 2):
                    writer.writerow({
                        "policy": policy,
                        "acquired_view_count": count,
                        "object_count": 3,
                        "chamfer_l1_normalized_mean": 1.0 / count,
                        "fscore_1pct_mean": count / 10,
                        "fscore_2pct_mean": count / 5,
                        "fscore_10pct_mean": count / 2.5,
                    })
        return path

    def test_reads_merges_and_plots_dynamic_phase3_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2_curves, phase2_sizes = _read_reconstruction(
                self._csv(root, ("random",))
            )
            variant_root = root / "variant"
            variant_root.mkdir()
            phase3_curves, phase3_sizes = _read_reconstruction(
                self._csv(variant_root, ("vggt_joint_future_variant",))
            )
            _merge_reconstruction(
                phase2_curves,
                phase2_sizes,
                phase3_curves,
                phase3_sizes,
                variant_root,
            )
            policies = PHASE2_POLICIES + _phase3_policy_specs(set(phase3_curves))
            missing = [policy for policy, _, _, _ in policies if policy not in phase2_curves]
            output = root / "plot.svg"
            write_plot(output, phase2_curves, phase2_sizes, policies, missing)

            svg = output.read_text(encoding="utf-8")
            self.assertIn("Phase 3 joint future variant (n=3)", svg)
            self.assertIn("Normalized Chamfer-L1 (log-scaled)", svg)
            self.assertIn("F-score @ 2% of object diameter", svg)
            self.assertIn("F-score @ 10% of object diameter", svg)

    def test_legacy_results_without_ten_percent_metric_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.csv"
            fields = [
                "policy",
                "acquired_view_count",
                "object_count",
                *(metric for metric, _, _ in METRICS[:3]),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "policy": "random",
                    "acquired_view_count": 1,
                    "object_count": 3,
                    "chamfer_l1_normalized_mean": 0.5,
                    "fscore_1pct_mean": 0.1,
                    "fscore_2pct_mean": 0.2,
                })
            curves, _ = _read_reconstruction(path)
            self.assertNotIn("fscore_10pct_mean", curves["random"])

    def test_conflicting_repeated_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting reconstruction curves"):
            _merge_reconstruction(
                {"variant": {"metric": [(1, 0.5)]}},
                {"variant": 3},
                {"variant": {"metric": [(1, 0.4)]}},
                {"variant": 3},
                Path("different-run"),
            )


if __name__ == "__main__":
    unittest.main()
