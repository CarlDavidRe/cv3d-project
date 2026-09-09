from __future__ import annotations

import math
import unittest

import numpy as np

from nbv.eval import (
    coverage_auc,
    ndcg_at_k,
    normalized_regret,
    reachable_normalized_coverage,
    spearman_rank,
)


class ReachableNormalizedCoverageTests(unittest.TestCase):
    def test_uses_explicit_reachable_ceiling(self) -> None:
        self.assertAlmostEqual(reachable_normalized_coverage(0.7, 0.8), 0.875)
        self.assertEqual(reachable_normalized_coverage(0.0, 0.0), 0.0)
        self.assertEqual(reachable_normalized_coverage(1.0, 1.0), 1.0)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            reachable_normalized_coverage(0.9, 0.8)
        for coverage, ceiling in (
            (-0.1, 0.8),
            (0.5, 1.1),
            (float("nan"), 1.0),
        ):
            with self.subTest(coverage=coverage, ceiling=ceiling):
                with self.assertRaises(ValueError):
                    reachable_normalized_coverage(coverage, ceiling)


class NormalizedRegretTests(unittest.TestCase):
    def test_oracle_selection_has_zero_regret(self) -> None:
        self.assertEqual(normalized_regret([0.1, 0.9, 0.2], [0.2, 1.0, 0.0]), 0.0)

    def test_worst_selection_has_unit_regret_up_to_epsilon(self) -> None:
        result = normalized_regret([0.9, 0.2, 0.1], [0.0, 0.4, 1.0])
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_all_equal_utilities_have_zero_regret(self) -> None:
        self.assertEqual(normalized_regret([3.0, 2.0, 1.0], [0.4, 0.4, 0.4]), 0.0)

    def test_masked_candidates_do_not_affect_selection_or_range(self) -> None:
        result = normalized_regret(
            [100.0, 0.2, 0.9, -np.inf],
            [100.0, 0.0, 1.0, np.nan],
            [False, True, True, False],
        )
        self.assertEqual(result, 0.0)


class SpearmanRankTests(unittest.TestCase):
    def test_perfect_and_reversed_rankings(self) -> None:
        self.assertAlmostEqual(spearman_rank([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_rank([3, 2, 1], [10, 20, 30]), -1.0)

    def test_ties_receive_average_ranks(self) -> None:
        result = spearman_rank([1, 1, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(result, math.sqrt(0.9))

    def test_masked_candidates_are_excluded(self) -> None:
        result = spearman_rank(
            [1, 100, 2, 3], [10, -100, 20, 30], [True, False, True, True]
        )
        self.assertAlmostEqual(result, 1.0)

    def test_constant_or_singleton_ranking_is_undefined(self) -> None:
        self.assertTrue(math.isnan(spearman_rank([1, 1, 1], [1, 2, 3])))
        self.assertTrue(spearman_rank([1], [1]) != spearman_rank([1], [1]))


class NDCGTests(unittest.TestCase):
    def test_perfect_ranking_has_unit_ndcg(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1], [3, 2, 1], k=5), 1.0)

    def test_ndcg_uses_only_valid_candidates(self) -> None:
        result = ndcg_at_k(
            [100, 3, 2, 1], [0, 3, 2, 1], [False, True, True, True], k=2
        )
        self.assertAlmostEqual(result, 1.0)

    def test_predicted_ties_are_index_independent(self) -> None:
        first = ndcg_at_k([1, 1, 0], [3, 1, 0], k=1)
        second = ndcg_at_k([1, 1, 0], [1, 3, 0], k=1)
        self.assertAlmostEqual(first, 2.0 / 3.0)
        self.assertAlmostEqual(first, second)

    def test_all_zero_relevance_has_zero_ndcg(self) -> None:
        self.assertEqual(ndcg_at_k([3, 2, 1], [0, 0, 0]), 0.0)


class CoverageAUCTests(unittest.TestCase):
    def test_default_counts_are_one_view_apart(self) -> None:
        self.assertAlmostEqual(coverage_auc([0.2, 0.5, 0.9]), 1.05)

    def test_explicit_view_counts_define_the_x_axis(self) -> None:
        self.assertAlmostEqual(
            coverage_auc([0.2, 0.5, 0.9], [1, 3, 4]), 1.4
        )

    def test_single_coverage_point_has_zero_area(self) -> None:
        self.assertEqual(coverage_auc([0.5], [3]), 0.0)


class MetricValidationTests(unittest.TestCase):
    def test_ranking_metrics_reject_an_empty_valid_set(self) -> None:
        for metric in (normalized_regret, spearman_rank, ndcg_at_k):
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    metric([1, 2], [2, 1], [False, False])

    def test_ranking_metrics_reject_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            normalized_regret([1, 2], [1])

    def test_coverage_rejects_invalid_curve_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            coverage_auc([0.5, 1.1])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            coverage_auc([0.5, 0.6], [2, 2])


if __name__ == "__main__":
    unittest.main()
