"""Common one-step ranking and closed-loop coverage metrics.

The ranking metrics operate on one candidate set at a time.  This keeps mask
semantics explicit: entries for which ``valid_mask`` is false are removed
before ranking, selection, or normalization.  Callers evaluating a batch
should invoke the functions once per sample and aggregate the returned scalar
values separately.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


_DEFAULT_EPSILON = 1e-12


def reachable_normalized_coverage(
    coverage: float,
    reachable_coverage_ceiling: float,
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> float:
    """Normalize absolute coverage by the surface reachable by eligible views.

    The ceiling is the absolute coverage obtained by taking the union of every
    eligible camera's visibility mask.  A zero ceiling produces zero rather
    than an undefined value; this keeps degenerate all-invisible synthetic
    caches representable without claiming that they cover any surface.
    """

    for name, value in (
        ("coverage", coverage),
        ("reachable_coverage_ceiling", reachable_coverage_ceiling),
        ("epsilon", epsilon),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number")
    coverage = float(coverage)
    ceiling = float(reachable_coverage_ceiling)
    epsilon = float(epsilon)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be finite and lie in [0, 1]")
    if not math.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError(
            "reachable_coverage_ceiling must be finite and lie in [0, 1]"
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and greater than zero")
    if coverage > ceiling + epsilon:
        raise ValueError("coverage cannot exceed the reachable coverage ceiling")
    if ceiling == 0.0:
        return 0.0
    return float(np.clip(coverage / ceiling, 0.0, 1.0))


def normalized_regret(
    predicted_scores: ArrayLike,
    target_utilities: ArrayLike,
    valid_mask: ArrayLike | None = None,
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> float:
    """Return normalized utility regret for the predicted best candidate.

    The selected candidate is the first valid maximum in canonical candidate
    order, matching :func:`numpy.argmax`.  The oracle is the valid candidate
    with maximum target utility.  Regret is zero when all valid utilities are
    equal.
    """

    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(
        epsilon, (int, float, np.integer, np.floating)
    ):
        raise TypeError("epsilon must be a real number")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and greater than zero")

    scores, utilities = _valid_candidate_values(
        predicted_scores, target_utilities, valid_mask
    )
    selected_utility = float(utilities[int(np.argmax(scores))])
    oracle_utility = float(np.max(utilities))
    utility_range = oracle_utility - float(np.min(utilities))
    if utility_range == 0.0:
        return 0.0
    return (oracle_utility - selected_utility) / (utility_range + epsilon)


def spearman_rank(
    predicted_scores: ArrayLike,
    target_utilities: ArrayLike,
    valid_mask: ArrayLike | None = None,
) -> float:
    """Return Spearman's rank correlation over valid candidates.

    Tied values receive their average rank.  The result is ``NaN`` when fewer
    than two candidates are valid or when either input is constant, because
    rank correlation is undefined in those cases.
    """

    scores, utilities = _valid_candidate_values(
        predicted_scores, target_utilities, valid_mask
    )
    if scores.size < 2:
        return float("nan")

    score_ranks = _average_ranks(scores)
    utility_ranks = _average_ranks(utilities)
    centered_scores = score_ranks - np.mean(score_ranks)
    centered_utilities = utility_ranks - np.mean(utility_ranks)
    denominator = float(
        np.linalg.norm(centered_scores) * np.linalg.norm(centered_utilities)
    )
    if denominator == 0.0:
        return float("nan")
    correlation = float(np.dot(centered_scores, centered_utilities) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def ndcg_at_k(
    predicted_scores: ArrayLike,
    target_utilities: ArrayLike,
    valid_mask: ArrayLike | None = None,
    *,
    k: int = 5,
) -> float:
    """Return tie-aware normalized discounted cumulative gain at ``k``.

    Target utilities are used directly as non-negative relevance values; no
    exponential gain transform is applied.  Predicted-score ties receive the
    expected DCG under every ordering of the tied candidates, avoiding an
    arbitrary anchor-index advantage.  If every valid relevance is zero, the
    result is zero.
    """

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores, utilities = _valid_candidate_values(
        predicted_scores, target_utilities, valid_mask
    )
    if np.any(utilities < 0):
        raise ValueError("target_utilities must be non-negative for NDCG")

    cutoff = min(k, scores.size)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    dcg = _tie_aware_dcg(scores, utilities, discounts)
    ideal_relevance = np.sort(utilities)[::-1][:cutoff]
    ideal_dcg = float(np.dot(ideal_relevance, discounts))
    if ideal_dcg == 0.0:
        return 0.0
    return float(np.clip(dcg / ideal_dcg, 0.0, 1.0))


def coverage_auc(
    coverage: ArrayLike,
    acquired_view_counts: ArrayLike | None = None,
) -> float:
    """Integrate surface coverage over the number of acquired views.

    ``coverage[i]`` must be the coverage after ``acquired_view_counts[i]``
    total views, including any initial views.  When counts are omitted,
    consecutive samples are treated as one view apart.  The returned area is
    not divided by the view-count span.
    """

    values = _one_dimensional_finite_array("coverage", coverage)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("coverage values must lie in [0, 1]")

    if acquired_view_counts is None:
        counts = np.arange(values.size, dtype=np.float64)
    else:
        counts = _one_dimensional_finite_array(
            "acquired_view_counts", acquired_view_counts
        )
        if counts.shape != values.shape:
            raise ValueError(
                "coverage and acquired_view_counts must have the same shape"
            )
        if np.any(counts < 0.0):
            raise ValueError("acquired_view_counts must be non-negative")
        if np.any(np.diff(counts) <= 0.0):
            raise ValueError("acquired_view_counts must be strictly increasing")

    if values.size == 1:
        return 0.0
    return float(np.trapz(values, x=counts))


def _valid_candidate_values(
    predicted_scores: ArrayLike,
    target_utilities: ArrayLike,
    valid_mask: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scores = _one_dimensional_array("predicted_scores", predicted_scores)
    utilities = _one_dimensional_array("target_utilities", target_utilities)
    if scores.shape != utilities.shape:
        raise ValueError(
            "predicted_scores and target_utilities must have the same shape"
        )

    if valid_mask is None:
        mask = np.ones(scores.shape, dtype=np.bool_)
    else:
        mask_array = np.asarray(valid_mask)
        if mask_array.ndim != 1 or mask_array.shape != scores.shape:
            raise ValueError("valid_mask must have the same one-dimensional shape")
        if not np.issubdtype(mask_array.dtype, np.bool_):
            raise TypeError("valid_mask must contain boolean values")
        mask = mask_array.astype(np.bool_, copy=False)
    if not np.any(mask):
        raise ValueError("valid_mask must select at least one candidate")

    valid_scores = scores[mask]
    valid_utilities = utilities[mask]
    if not np.all(np.isfinite(valid_scores)):
        raise ValueError("valid predicted_scores must be finite")
    if not np.all(np.isfinite(valid_utilities)):
        raise ValueError("valid target_utilities must be finite")
    return valid_scores, valid_utilities


def _one_dimensional_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


def _one_dimensional_finite_array(
    name: str, values: ArrayLike
) -> NDArray[np.float64]:
    array = _one_dimensional_array(name, values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _tie_aware_dcg(
    scores: NDArray[np.float64],
    relevance: NDArray[np.float64],
    discounts: NDArray[np.float64],
) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_relevance = relevance[order]
    cutoff = discounts.size
    dcg = 0.0

    start = 0
    while start < scores.size and start < cutoff:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        discounted_positions_end = min(end, cutoff)
        mean_relevance = float(np.mean(sorted_relevance[start:end]))
        dcg += mean_relevance * float(
            np.sum(discounts[start:discounted_positions_end])
        )
        start = end
    return dcg
