from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.migrate_gaussian_splatting_layout import migrate_layout


class GaussianSplattingMigrationTests(unittest.TestCase):
    def test_moves_object_and_rewrites_absolute_json_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gaussian_splatting_variant_comparison"
            source = root / "category_object"
            summary = source / "1views/phase2_random/3dgs/summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({
                    "artifact": (
                        "/content/project/outputs/"
                        "gaussian_splatting_variant_comparison/"
                        "category_object/1views/result.html"
                    ),
                    "object_id": "category/object",
                }),
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                moved, rewritten = migrate_layout(root, ["category/object"])

            destination = root / "category/object"
            self.assertEqual((moved, rewritten), (1, 1))
            self.assertFalse(source.exists())
            payload = json.loads(
                (destination / summary.relative_to(source)).read_text(encoding="utf-8")
            )
            self.assertIn(
                "/gaussian_splatting_variant_comparison/category/object/",
                payload["artifact"],
            )

    def test_collision_is_detected_before_any_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "category_first"
            collision = root / "category_second"
            first.mkdir()
            collision.mkdir()
            (root / "category/second").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                migrate_layout(root, ["category/first", "category/second"])

            self.assertTrue(first.is_dir())
            self.assertTrue(collision.is_dir())

    def test_dry_run_does_not_move_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "category_object"
            source.mkdir()

            with redirect_stdout(StringIO()):
                moved, rewritten = migrate_layout(
                    root, ["category/object"], dry_run=True
                )

            self.assertEqual((moved, rewritten), (1, 0))
            self.assertTrue(source.is_dir())
            self.assertFalse((root / "category/object").exists())


if __name__ == "__main__":
    unittest.main()
