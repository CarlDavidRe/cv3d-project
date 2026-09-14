from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_gaussian_splatting_variants import (
    PHASE2_POLICIES,
    PHASE3_POLICIES,
    Variant,
    discover_variants,
    main,
    normalize_views,
    object_output_root,
    summary_has_training_budget,
    validate_inputs,
    write_index,
)


def write_rows(path: Path, policies: tuple[str, ...], views=(1, 2, 3, 5, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("object_id", "policy", "acquired_view_count"),
        )
        writer.writeheader()
        for policy in policies:
            for view_count in views:
                writer.writerow({
                    "object_id": "category/object",
                    "policy": policy,
                    "acquired_view_count": view_count,
                })


class GaussianSplattingVariantTests(unittest.TestCase):
    def test_resume_requires_matching_per_view_training_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps({"settings": {"iterations": 1500}}))
            self.assertFalse(summary_has_training_budget(path, 1500, 5))
            path.write_text(json.dumps({"training_budget": {
                "iterations_per_view": 1500,
                "view_count": 5,
                "total_iterations": 7500,
            }}))
            self.assertTrue(summary_has_training_budget(path, 1500, 5))
            self.assertFalse(summary_has_training_budget(path, 1000, 5))

    def test_object_output_root_mirrors_num_hierarchy(self) -> None:
        self.assertEqual(
            object_output_root(Path("output"), "category/object"),
            Path("output/category/object"),
        )

    def test_views_are_unique_sorted_and_positive(self) -> None:
        self.assertEqual(normalize_views((10, 1, 5, 1)), (1, 5, 10))
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_views((1, 0))

    def test_discovers_nine_variants_and_deduplicates_independent_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2 = root / "phase2"
            phase3 = root / "phase3"
            phase2_metrics = phase2 / "combined/metrics/reconstruction_per_object.csv"
            controlled = phase3 / "controlled/metrics/reconstruction_per_object.csv"
            deeper = phase3 / "nested/deeper/metrics/reconstruction_per_object.csv"
            write_rows(phase2_metrics, PHASE2_POLICIES)
            write_rows(
                controlled,
                ("vggt_independent_history", "vggt_joint_history"),
            )
            write_rows(
                deeper,
                (
                    "vggt_independent_history",
                    "vggt_joint_pose_deepsets",
                    "vggt_joint_token_attention",
                ),
            )

            variants = discover_variants(phase2, phase3)

            self.assertEqual(len(variants), 9)
            self.assertEqual(len({variant.key for variant in variants}), 9)
            independent = next(
                variant for variant in variants
                if variant.policy == "vggt_independent_history"
            )
            self.assertEqual(independent.metrics, controlled)

    def test_validation_reports_missing_increment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.csv"
            write_rows(metrics, ("random",), views=(1, 2))
            with self.assertRaisesRegex(ValueError, r"phase2_random: \[3\]"):
                validate_inputs(
                    (Variant("phase2", "random", metrics),),
                    "category/object",
                    (1, 2, 3),
                )

    def test_index_links_existing_outputs_and_marks_missing_ones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant = Variant("phase2", "random", root / "metrics.csv")
            target = root / "1views/phase2_random/2dgs/comparison_interactive.html"
            target.parent.mkdir(parents=True)
            target.write_text("fixture", encoding="utf-8")

            write_index(
                root / "index.html", "category/object", (variant,), (1,), "both"
            )

            document = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn(
                'href="1views/phase2_random/2dgs/comparison_interactive.html"',
                document,
            )
            self.assertIn('<span class="missing">3DGS vs GT</span>', document)
            self.assertIn("Each view count is trained independently", document)

    def test_dry_run_builds_all_variant_increment_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2 = root / "phase2"
            phase3 = root / "phase3"
            output = root / "output"
            write_rows(
                phase2 / "combined/metrics/reconstruction_per_object.csv",
                PHASE2_POLICIES,
                views=(1, 3),
            )
            write_rows(
                phase3 / "controlled/metrics/reconstruction_per_object.csv",
                ("vggt_independent_history", "vggt_joint_history"),
                views=(1, 3),
            )
            write_rows(
                phase3 / "pose/deeper/metrics/reconstruction_per_object.csv",
                (
                    "vggt_independent_history",
                    "vggt_joint_pose_deepsets",
                    "vggt_joint_token_attention",
                ),
                views=(1, 3),
            )
            arguments = [
                "evaluate_gaussian_splatting_variants.py",
                "--object-id", "category/object",
                "--views", "1", "3",
                "--phase2-root", str(phase2),
                "--phase3-root", str(phase3),
                "--output-root", str(output),
                "--dry-run",
            ]

            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)

            object_root = output / "category/object"
            payload = json.loads(
                (object_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["variants"]), 9)
            self.assertEqual(len(payload["runs"]), 18)
            self.assertEqual({run["status"] for run in payload["runs"]}, {"dry_run"})
            self.assertTrue((object_root / "index.html").is_file())

    def test_resume_runs_only_the_missing_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2 = root / "phase2"
            phase3 = root / "phase3"
            output = root / "output"
            write_rows(
                phase2 / "combined/metrics/reconstruction_per_object.csv",
                PHASE2_POLICIES,
                views=(1,),
            )
            write_rows(
                phase3 / "controlled/metrics/reconstruction_per_object.csv",
                ("vggt_independent_history", "vggt_joint_history"),
                views=(1,),
            )
            write_rows(
                phase3 / "pose/metrics/reconstruction_per_object.csv",
                ("vggt_joint_pose_deepsets", "vggt_joint_token_attention"),
                views=(1,),
            )
            variant_names = (
                *(f"phase2_{policy}" for policy in PHASE2_POLICIES),
                *(f"phase3_{policy}" for policy in PHASE3_POLICIES),
            )
            for variant_name in variant_names:
                summary = (
                    output / "category/object/1views" / variant_name
                    / "2dgs/summary.json"
                )
                summary.parent.mkdir(parents=True)
                summary.write_text(json.dumps({"training_budget": {
                    "iterations_per_view": 1500,
                    "view_count": 1,
                    "total_iterations": 1500,
                }}), encoding="utf-8")
            arguments = [
                "evaluate_gaussian_splatting_variants.py",
                "--object-id", "category/object",
                "--views", "1",
                "--phase2-root", str(phase2),
                "--phase3-root", str(phase3),
                "--output-root", str(output),
            ]

            with (
                patch("sys.argv", arguments),
                patch(
                    "scripts.evaluate_gaussian_splatting_variants.subprocess.run"
                ) as run,
                redirect_stdout(StringIO()),
            ):
                run.return_value.returncode = 0
                self.assertEqual(main(), 0)

            self.assertEqual(run.call_count, len(variant_names))
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--backend") + 1], "3dgs")


if __name__ == "__main__":
    unittest.main()
