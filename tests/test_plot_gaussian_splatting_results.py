from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.plot_gaussian_splatting_results import (
    METRICS,
    POLICIES,
    complete_views,
    read_results,
    select_primary_object,
    select_transfer_budget,
    select_view_budget_cohort,
    write_transfer_plot,
    write_view_budget_plot,
)


class GaussianSplattingResultsPlotTests(unittest.TestCase):
    def _csv(self, root: Path) -> Path:
        path = root / "recovered_metrics.csv"
        fields = [
            "variant",
            "object_id",
            "acquired_view_count",
            "backend",
            *(metric for metric, _title, _low, _high in METRICS),
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for object_id, views in (("category/a", (2, 5)), ("category/b", (2,))):
                for policy_index, (policy, _label, _color, _group) in enumerate(POLICIES):
                    for view_count in views:
                        writer.writerow({
                            "variant": policy,
                            "object_id": object_id,
                            "acquired_view_count": view_count,
                            "backend": "2dgs",
                            "chamfer_l1_normalized": 0.1 + policy_index / 100,
                            "fscore_2pct": 0.2 + view_count / 100,
                        })
        return path

    def test_selects_balanced_primary_and_transfer_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = read_results(self._csv(Path(temporary)))
            self.assertEqual(complete_views(rows, "category/a"), [2, 5])
            self.assertEqual(select_primary_object(rows), "category/a")
            self.assertEqual(select_view_budget_cohort(rows), (["category/a"], [2, 5]))
            self.assertEqual(select_transfer_budget(rows), (2, ["category/a", "category/b"]))

    def test_writes_both_svg_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = read_results(self._csv(root))
            budget_output = root / "budget.svg"
            transfer_output = root / "transfer.svg"
            write_view_budget_plot(budget_output, rows, ["category/a"])
            write_transfer_plot(transfer_output, rows, 2, ["category/a", "category/b"])

            self.assertIn("Silhouette-refined 2DGS", budget_output.read_text())
            self.assertIn("P3 token attention", budget_output.read_text())
            self.assertIn("2-object", transfer_output.read_text())

    def test_view_budget_selection_prefers_larger_balanced_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = read_results(self._csv(Path(temporary)))
            for policy, _label, _color, _group in POLICIES:
                rows[("category/b", policy, 5)] = {
                    "variant": policy,
                    "object_id": "category/b",
                    "acquired_view_count": "5",
                    "backend": "2dgs",
                    "chamfer_l1_normalized": "0.1",
                    "fscore_2pct": "0.2",
                }
            self.assertEqual(
                select_view_budget_cohort(rows),
                (["category/a", "category/b"], [2, 5]),
            )
            self.assertEqual(
                select_transfer_budget(rows),
                (5, ["category/a", "category/b"]),
            )

    def test_rejects_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._csv(Path(temporary))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("phase2_random,category/a,2,2dgs,0.1,0.2\n")
            with self.assertRaisesRegex(ValueError, "duplicate 2DGS result"):
                read_results(path)


if __name__ == "__main__":
    unittest.main()
