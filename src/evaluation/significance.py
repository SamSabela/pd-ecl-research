"""Paired significance tests for AUC comparisons.

Implements the inference specified in Section 3.8.2 and Refinement 10: DeLong's
test for the difference between two correlated AUCs, with Holm-Bonferroni
correction across each family of comparisons.

Two models scored on the same evaluation cohort produce correlated AUCs,
because they rank the same borrowers. Treating the two as independent
overstates the variance of their difference, so overlapping marginal
confidence intervals are not evidence that two models are equivalent. DeLong's
method estimates the covariance between the two AUCs from their structural
components, which is what makes a paired test possible without resampling.

The implementation follows the fast midrank formulation of Sun and Xu (2014),
which is O(n log n) and algebraically identical to the estimator of DeLong,
DeLong and Clarke-Pearson (1988). A brute-force O(mn) version is kept in the
test suite as the reference it is checked against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata

__all__ = ["PairedAUCTest", "delong_paired", "holm_adjust"]


@dataclass(frozen=True)
class PairedAUCTest:
    """Result of one paired DeLong comparison of model A against model B."""

    auc_a: float
    auc_b: float
    difference: float
    se: float
    z: float
    p_value: float


def _structural_components(
    y: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midrank structural components for k score vectors on one outcome vector.

    Args:
        y: Boolean outcomes of length N, True for default.
        scores: Array of shape (k, N), one row per model.

    Returns:
        (aucs, v10, v01): AUCs of shape (k,); per-positive components of shape
        (k, m), each the share of negatives ranked below that positive; and
        per-negative components of shape (k, n), each the share of positives
        ranked above that negative. Ties count one half throughout.
    """
    positives = scores[:, y]
    negatives = scores[:, ~y]
    m = positives.shape[1]
    n = negatives.shape[1]
    k = scores.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = rankdata(positives[r])
        ty[r] = rankdata(negatives[r])
        tz[r] = rankdata(np.concatenate([positives[r], negatives[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v10, v01


def delong_paired(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
) -> PairedAUCTest:
    """DeLong's test for the difference between two AUCs on the same observations.

    Args:
        y_true: Binary outcomes, 1 for default.
        prob_a: Model A's predicted probabilities, aligned to y_true.
        prob_b: Model B's predicted probabilities, aligned to y_true.

    Returns:
        The two AUCs, their difference (A minus B), its standard error, the z
        statistic and the two-sided p-value. Where the two models produce
        identical rankings the variance of the difference is zero; the
        difference is then reported as exactly zero with a p-value of one.

    Raises:
        ValueError: If the inputs differ in length or one class is absent.
    """
    y = np.asarray(y_true).astype(bool)
    a = np.asarray(prob_a, dtype=np.float64)
    b = np.asarray(prob_b, dtype=np.float64)
    if not (len(y) == len(a) == len(b)):
        raise ValueError(
            f"inputs must align: y {len(y)}, prob_a {len(a)}, prob_b {len(b)}"
        )
    m = int(y.sum())
    n = len(y) - m
    if m < 2 or n < 2:
        raise ValueError(f"need at least two of each class, got {m} positive, {n} negative")

    aucs, v10, v01 = _structural_components(y, np.vstack([a, b]))
    covariance = np.cov(v10) / m + np.cov(v01) / n
    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    difference = float(aucs[0] - aucs[1])

    if variance <= 1e-18:
        return PairedAUCTest(float(aucs[0]), float(aucs[1]), 0.0, 0.0, 0.0, 1.0)

    se = float(np.sqrt(variance))
    z = difference / se
    p = float(2.0 * norm.sf(abs(z)))
    return PairedAUCTest(float(aucs[0]), float(aucs[1]), difference, se, z, p)


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a family of p-values.

    Controls the family-wise error rate at the nominal level without assuming
    independence between the tests, which matters here because comparisons
    sharing a model or a fold are not independent.

    Args:
        p_values: Raw p-values for one family of comparisons.

    Returns:
        Adjusted p-values in the original order, monotone in the raw ordering
        and capped at one. A comparison is significant at level alpha when its
        adjusted p-value is at most alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    count = len(p)
    adjusted = np.empty(count)
    running = 0.0
    for rank, index in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted
