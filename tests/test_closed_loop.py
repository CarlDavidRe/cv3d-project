from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from nbv.config import load_config
from nbv.data.observation_store import ObservationStore
from nbv.data.visibility_cache import VisibilityCache, load_visibility_cache, save_visibility_cache
from nbv.eval.closed_loop import RolloutConfig, canonical_masked_argmax, replay_rollout, run_rollout
from nbv.eval.result_schema import load_rollout, save_rollout
from nbv.experiments.closed_loop import run_closed_loop_experiment
from nbv.geometry import (
    CAMERA_CONVENTION,
    CANONICAL_ORDERING,
    FACE_VISIBILITY_RENDERER,
    canonical_anchors,
)
from nbv.policies import FarthestPolicy, OraclePolicy, RandomPolicy
from nbv.visualization import write_phase2_rollout_demo


ROOT = Path(__file__).resolve().parents[1]


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.object_id = "category/object"
        self.paths = {}
        for anchor in range(48):
            path = self.root / "NUM" / self.object_id / "images" / f"viewpoint_{anchor}_offset_phi_0.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (5, 4), (anchor, 2, 3)).save(path)
            self.paths[anchor] = path
        visibility = np.zeros((48, 4), bool)
        visibility[0, 0] = True
        visibility[1, 1] = True
        visibility[2, 2] = True
        visibility[3, :3] = True
        # Face 3 is unreachable and must remain in the coverage denominator.
        self.cache = VisibilityCache(visibility, np.array([1., 2., 4., 3.]), np.arange(48), {
            "schema_version": 2, "object_id": self.object_id, "n_faces": 4,
            "anchor_ordering": CANONICAL_ORDERING, "render_resolution": [256, 256],
            "visibility_target": "vis", "visibility_definition": "pun_unoccluded_rasterized_mesh_faces_v1",
            "camera_radius": 2.73,
        })

    def store(self, ids=range(48)):
        return ObservationStore(self.object_id, {a: self.paths[a] for a in ids})

    def test_oracle_matches_brute_force_for_both_targets_and_counts_initial_views(self):
        for target, expected in (("vis", .75), ("vis_a", .7)):
            result = run_rollout(self.cache, self.store(), OraclePolicy(), RolloutConfig(
                initial_anchor_ids=(0,), max_acquired_views=5, coverage_target=target,
            ))
            self.assertEqual(result.acquired_anchor_ids.tolist(), [0, 3, 1, 2, 4])
            self.assertEqual(result.acquired_view_counts.tolist(), [1, 2, 3, 4, 5])
            self.assertAlmostEqual(result.coverage[-1], expected)
            self.assertTrue((np.diff(result.coverage) >= 0).all())
            for step in result.steps:
                history = step["history_anchor_ids"]
                before = self.cache.coverage(history, target=target)
                brute = np.array([self.cache.coverage(history + [a], target=target) - before for a in range(48)])
                index = step["step_index"]
                np.testing.assert_allclose(result.candidate_gains[index], brute, atol=1e-12)
                self.assertAlmostEqual(step["selected_true_gain"], step["coverage_after"] - step["coverage_before"])
                self.assertEqual(step["normalized_regret"], 0.)
                self.assertEqual(step["selected_true_gain"], step["oracle_true_gain"])

    def test_random_is_repeatable_and_independent_of_global_rng_and_other_runs(self):
        policy = RandomPolicy()
        first = run_rollout(self.cache, self.store(), policy, RolloutConfig(seed=7))
        np.random.seed(19)
        run_rollout(self.cache, self.store(), policy, RolloutConfig(seed=8))
        second = run_rollout(self.cache, self.store(), policy, RolloutConfig(seed=7))
        different = run_rollout(self.cache, self.store(), policy, RolloutConfig(seed=9))
        np.testing.assert_array_equal(first.scores, second.scores)
        np.testing.assert_array_equal(first.acquired_anchor_ids, second.acquired_anchor_ids)
        self.assertFalse(np.array_equal(first.scores, different.scores))
        self.assertEqual(len(set(first.acquired_anchor_ids)), 10)

    def test_farthest_uses_max_min_angular_distance_and_canonical_ties(self):
        result = run_rollout(
            self.cache,
            self.store(),
            FarthestPolicy(),
            RolloutConfig(max_acquired_views=10),
        )
        self.assertEqual(
            result.acquired_anchor_ids.tolist(),
            [0, 46, 18, 28, 16, 30, 2, 44, 13, 33],
        )
        distances = canonical_anchors().angular_distance_matrix
        for index, step in enumerate(result.steps):
            expected = distances[:, step["history_anchor_ids"]].min(axis=1)
            np.testing.assert_allclose(result.scores[index], expected, atol=1e-12)
            self.assertEqual(
                step["selected_anchor"],
                canonical_masked_argmax(expected, result.valid_masks[index]),
            )
        self.assertFalse(result.metadata["policy_is_oracle"])
        self.assertEqual(
            result.metadata["policy_score_semantics"],
            "max_min_angular_distance_radians",
        )

    def test_masking_exhaustion_and_multiple_initial_views(self):
        result = run_rollout(self.cache, self.store((0, 1, 2, 3)), OraclePolicy(), RolloutConfig(
            initial_anchor_ids=(2, 0), invalid_anchor_ids=(3,), max_acquired_views=10,
        ))
        self.assertEqual(result.acquired_anchor_ids.tolist(), [2, 0, 1])
        self.assertEqual(result.acquired_view_counts.tolist(), [2, 3])
        self.assertEqual(result.metadata["stop_reason"], "no_valid_candidates")
        self.assertEqual(np.flatnonzero(result.valid_masks[0]).tolist(), [1])
        self.assertIsNone(result.steps[0]["spearman"])

    def test_budget_already_met_and_immediate_exhaustion_have_no_steps(self):
        for config, reason in ((RolloutConfig(max_acquired_views=1), "view_budget_exhausted"),
                               (RolloutConfig(max_acquired_views=3), "no_valid_candidates")):
            result = run_rollout(self.cache, self.store((0,)), OraclePolicy(), config)
            self.assertEqual(result.metadata["stop_reason"], reason)
            self.assertEqual(result.scores.shape, (0, 48))
            self.assertEqual(result.summary()["coverage_auc"], 0.)
            self.assertIsNone(result.summary()["spearman_mean"])
            replay_rollout(result, self.cache, self.store((0,)))

    def test_all_zero_gains_continue_to_budget_with_canonical_ties(self):
        cache = replace(self.cache, face_visibility=np.zeros((48, 4), bool))
        result = run_rollout(cache, self.store(), OraclePolicy(), RolloutConfig(max_acquired_views=48))
        self.assertEqual(result.acquired_anchor_ids.tolist(), list(range(48)))
        self.assertTrue((result.coverage == 0).all())
        self.assertTrue(all(s["normalized_regret"] == s["ndcg_at_5"] == 0 for s in result.steps))
        self.assertTrue(all(s["spearman"] is None for s in result.steps))

    def test_policy_observations_are_acquired_only_and_mutations_are_isolated(self):
        captured = []

        class InspectPolicy:
            name = "inspect"
            is_oracle = False
            score_semantics = "test"

            def score(policy_self, state):
                captured.append(state.acquired_anchor_ids)
                self.assertEqual({f.name for f in fields(state)}, {
                    "object_id", "acquired_observations", "anchor_directions", "camera_to_world",
                    "valid_candidate_mask", "step_index", "seed", "anchor_ordering",
                })
                self.assertFalse(hasattr(state, "__dict__"))
                for observation in state.acquired_observations:
                    self.assertEqual(int(observation.rgb[0, 0, 0]), observation.anchor_id)
                    self.assertIn(f"viewpoint_{observation.anchor_id}_", observation.image_path)
                    observation.rgb.setflags(write=True)
                    observation.rgb[:] = 255
                state.valid_candidate_mask.setflags(write=True)
                state.valid_candidate_mask[:] = True
                state.anchor_directions.setflags(write=True)
                state.anchor_directions[:] = 0
                return -np.arange(48, dtype=float)

        with patch("nbv.data.observation_store.Image.open", wraps=Image.open) as opened:
            result = run_rollout(self.cache, self.store(), InspectPolicy(), RolloutConfig(max_acquired_views=4))
        self.assertEqual(captured, [(0,), (0, 1), (0, 1, 2)])
        self.assertEqual(opened.call_count, 4)
        self.assertEqual(result.acquired_anchor_ids.tolist(), [0, 1, 2, 3])

    def test_nonfinite_valid_scores_and_wrong_shape_fail_but_masked_nonfinite_is_allowed(self):
        valid = np.ones(48, bool)
        valid[0] = False
        scores = np.zeros(48)
        scores[0] = np.nan
        self.assertEqual(canonical_masked_argmax(scores, valid), 1)
        for bad in (np.zeros(47), np.full(48, np.inf), np.full(48, np.nan)):
            with self.assertRaises(ValueError):
                canonical_masked_argmax(bad, valid)
        with self.assertRaises(ValueError):
            canonical_masked_argmax(scores, np.zeros(48, bool))

    def test_saved_rollouts_replay_and_reject_changed_geometry_or_diagnostics(self):
        for policy in (RandomPolicy(), FarthestPolicy(), OraclePolicy()):
            original = run_rollout(self.cache, self.store(), policy)
            path = save_rollout(original, self.root / f"{policy.name}.npz")
            loaded = load_rollout(path)
            replayed = replay_rollout(loaded, self.cache, self.store())
            np.testing.assert_array_equal(original.coverage, replayed.coverage)
            self.assertEqual(original.steps, loaded.steps)
            self.assertEqual(original.image_paths, loaded.image_paths)
            self.assertEqual(original.metadata, loaded.metadata)
            changed = replace(self.cache, face_areas=np.array([2., 2., 4., 3.]))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                replay_rollout(loaded, changed, self.store())
            loaded.candidate_gains[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "candidate_gains"):
                replay_rollout(loaded, self.cache, self.store())

    def test_rollout_demo_embeds_rgb_scores_gains_faces_and_coverage(self):
        result = run_rollout(
            self.cache, self.store(), OraclePolicy(), RolloutConfig(max_acquired_views=4)
        )
        output = write_phase2_rollout_demo(result, self.cache, self.root / "demo.svg")
        ET.parse(output)
        svg = output.read_text()
        self.assertIn("data:image/png;base64,", svg)
        self.assertIn("Aggregated policy scores", svg)
        self.assertIn("Evaluator-only true gains", svg)
        self.assertIn("Accumulated visible-face weight", svg)
        self.assertIn("Coverage trajectory", svg)

    def test_invalid_configuration_and_object_mismatch_fail(self):
        for kwargs in (
            {"initial_anchor_ids": ()}, {"initial_anchor_ids": (0, 0)},
            {"initial_anchor_ids": (48,)}, {"initial_anchor_ids": (True,)},
            {"initial_anchor_ids": (0, 1), "max_acquired_views": 1},
            {"invalid_anchor_ids": (0,)}, {"max_acquired_views": 0},
            {"max_acquired_views": True}, {"seed": -1}, {"coverage_target": "area"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError, IndexError)):
                RolloutConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "initial anchor"):
            run_rollout(self.cache, self.store((1,)), OraclePolicy())
        with self.assertRaisesRegex(ValueError, "object IDs"):
            run_rollout(self.cache, ObservationStore("other/object", self.paths), OraclePolicy())

    def test_num_observation_discovery_ignores_examples_and_needs_no_target_files(self):
        example = self.paths[0].parent / "viewpoint_example_0_offset_phi_0.png"
        example.write_bytes(b"not an image")
        store = ObservationStore.from_num_object(self.root / "NUM", self.object_id)
        self.assertEqual(store.available_mask.sum(), 48)
        self.assertEqual(store.acquire(0).rgb.shape, (4, 5, 3))

    def experiment_config(self):
        config = load_config(ROOT / "configs/experiments/phase2_closed_loop.yaml")
        config["experiment"]["name"] = "geometric_baselines"
        config["phase2"]["evaluation"]["policies"] = [
            "random", "farthest", "oracle"
        ]
        config["paths"].update(data_root=str(self.root / "NUM"), output_root=str(self.root / "outputs"), visibility_cache_root=str(self.root / "cache"))
        split_path = self.root / "split.json"
        split_path.write_text(json.dumps({"schema_version": 1, "splits": {"train": [], "val": [], "test": [self.object_id, "category/missing"]}}))
        config["phase2"]["evaluation"]["split_manifest"] = str(split_path)
        geometry = load_config(ROOT / "configs/experiments/phase2_visibility.yaml")["phase2"]["visibility"]
        metadata = dict(self.cache.metadata, **{key: geometry[key] for key in (
            "mesh_scale", "mesh_centering", "camera_radius", "horizontal_fov_degrees", "render_resolution", "near", "far", "cull_backfaces", "mesh_relative_path",
        )})
        metadata.update(anchor_count=48, renderer=FACE_VISIBILITY_RENDERER, camera_convention=CAMERA_CONVENTION, available_visibility_targets=["vis", "vis_a"])
        cache = replace(self.cache, metadata=metadata)
        save_visibility_cache(cache, self.root / "cache" / f"{self.object_id}.npz")
        return config

    def test_experiment_missing_cache_is_explicit_and_partial_export_replays(self):
        config = self.experiment_config()
        with self.assertRaisesRegex(ValueError, "visibility caches are missing"):
            run_closed_loop_experiment(config, ROOT)
        config["experiment"]["name"] = "partial"
        config["phase2"]["evaluation"]["skip_missing_caches"] = True
        run = run_closed_loop_experiment(config, ROOT)
        summary = json.loads((run / "metrics/summary.json").read_text())
        manifest = json.loads((run / "metrics/visibility_cache_manifest.json").read_text())
        self.assertEqual(summary["evaluated_object_count"], 1)
        self.assertFalse(summary["complete_fixed_split"])
        self.assertEqual(manifest["missing_object_ids"], ["category/missing"])
        self.assertEqual(manifest["evaluated_object_ids"], [self.object_id])
        self.assertEqual(summary["coverage_target"], "vis_a")
        self.assertTrue((run / "metrics/per_step.csv").is_file())
        self.assertTrue((run / "metrics/coverage.csv").is_file())
        completion = json.loads((run / "metrics/phase2_completion.json").read_text())
        self.assertEqual(completion["status"], "pending")
        self.assertFalse(completion["checks"]["complete_fixed_test_split"])
        expected_figures = {
            "coverage": "figures/closed_loop/coverage_curves.svg",
            "per_step": "figures/closed_loop/per_step_policy_quality.svg",
            "summary": "figures/closed_loop/policy_summary.svg",
            "rollout_demo": "figures/closed_loop/rollout_demo.svg",
        }
        self.assertEqual(summary["figures"], expected_figures)
        for relative in expected_figures.values():
            ET.parse(run / relative)
        coverage_svg = (run / expected_figures["coverage"]).read_text()
        self.assertIn('data-policy="random"', coverage_svg)
        self.assertIn('data-policy="farthest"', coverage_svg)
        self.assertIn('data-policy="oracle"', coverage_svg)
        result = load_rollout(run / "rollouts/oracle" / f"{self.object_id}.npz")
        self.assertEqual(result.metadata["visibility_cache_metadata"]["visibility_target"], "vis")
        with self.assertRaisesRegex(ValueError, "not empty"):
            run_closed_loop_experiment(config, ROOT)

    def test_incompatible_available_cache_is_failure_even_when_missing_caches_are_skipped(self):
        config = self.experiment_config()
        config["phase2"]["evaluation"]["skip_missing_caches"] = True
        cache_path = self.root / "cache" / f"{self.object_id}.npz"
        save_visibility_cache(self.cache, cache_path)  # Missing pinned render metadata.
        with self.assertRaisesRegex(ValueError, "Evaluation failed"):
            run_closed_loop_experiment(config, ROOT)
        manifest = json.loads((self.root / "outputs/phase2/geometric_baselines/seed_0/metrics/visibility_cache_manifest.json").read_text())
        self.assertEqual(len(manifest["failures"]), 1)
        self.assertEqual(manifest["evaluated_object_ids"], [])

    def test_experiment_rejects_cache_without_mesh_centering_provenance(self):
        config = self.experiment_config()
        config["phase2"]["evaluation"]["skip_missing_caches"] = True
        cache_path = self.root / "cache" / f"{self.object_id}.npz"
        cache = load_visibility_cache(cache_path)
        cache.metadata.pop("mesh_centering")
        save_visibility_cache(cache, cache_path)
        with self.assertRaisesRegex(ValueError, "Evaluation failed"):
            run_closed_loop_experiment(config, ROOT)
        manifest = json.loads((self.root / "outputs/phase2/geometric_baselines/seed_0/metrics/visibility_cache_manifest.json").read_text())
        self.assertIn("mesh_centering", manifest["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
