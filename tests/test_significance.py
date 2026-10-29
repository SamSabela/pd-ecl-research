"""Tests for the paired DeLong test and Holm adjustment.

The decisive test is the agreement with a brute-force implementation written
directly from DeLong, DeLong and Clarke-Pearson (1988). The fast midrank path is
what the study runs; the brute-force path is what it is held to.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import _auc
from src.evaluation.significance import delong_paired, holm_adjust


def _brute_force_delong(y, a, b):
    """O(mn) DeLong straight from the definition, with the psi kernel."""
    y = np.asarray(y).astype(bool)
    out = []
    for s in (a, b):
        pos, neg = s[y], s[~y]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        out.append((psi.mean(), psi.mean(axis=1), psi.mean(axis=0)))
    (auc_a, v10a, v01a), (auc_b, v10b, v01b) = out
    m, n = int(y.sum()), int((~y).sum())
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return auc_a, auc_b, np.sqrt(var)


def _synthetic(n=600, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    signal = y.astype(float)
    a = signal * 0.8 + rng.normal(size=n)
    b = signal * 0.5 + rng.normal(size=n) + 0.3 * a
    if ties:
        a, b = np.round(a, 1), np.round(b, 1)
    return y.astype(int), a, b


class TestAgainstReference:
    @pytest.mark.parametrize("ties", [False, True])
    def test_matches_brute_force_delong(self, ties):
        """The fast path must reproduce the textbook estimator, ties included."""
        y, a, b = _synthetic(ties=ties)
        result = delong_paired(y, a, b)
        auc_a, auc_b, se = _brute_force_delong(y, a, b)
        assert result.auc_a == pytest.approx(auc_a, abs=1e-12)
        assert result.auc_b == pytest.approx(auc_b, abs=1e-12)
        assert result.se == pytest.approx(se, rel=1e-9)

    def test_auc_agrees_with_the_study_metric(self):
        """The AUC reported by the test must equal the one in the results table."""
        y, a, b = _synthetic(n=2000, seed=3, ties=True)
        result = delong_paired(y, a, b)
        assert result.auc_a == pytest.approx(_auc(y, a), abs=1e-12)
        assert result.auc_b == pytest.approx(_auc(y, b), abs=1e-12)

    def test_standard_error_agrees_with_paired_bootstrap(self):
        """An independent check on the covariance term, not only on the algebra."""
        y, a, b = _synthetic(n=3000, seed=7)
        result = delong_paired(y, a, b)
        rng = np.random.default_rng(11)
        diffs = []
        for _ in range(1500):
            idx = rng.integers(0, len(y), len(y))
            diffs.append(_auc(y[idx], a[idx]) - _auc(y[idx], b[idx]))
        assert result.se == pytest.approx(np.std(diffs, ddof=1), rel=0.12)


class TestBehaviour:
    def test_swapping_models_negates_difference_and_keeps_p(self):
        y, a, b = _synthetic()
        ab, ba = delong_paired(y, a, b), delong_paired(y, b, a)
        assert ab.difference == pytest.approx(-ba.difference)
        assert ab.z == pytest.approx(-ba.z)
        assert ab.p_value == pytest.approx(ba.p_value)

    def test_identical_models_are_not_different(self):
        y, a, _ = _synthetic()
        result = delong_paired(y, a, a)
        assert result.difference == 0.0
        assert result.p_value == 1.0

    def test_monotone_transform_is_not_a_different_model(self):
        """AUC is rank-based, so a monotone recalibration changes nothing."""
        y, a, _ = _synthetic()
        result = delong_paired(y, a, 1 / (1 + np.exp(-a)))
        assert result.p_value == 1.0

    def test_informative_model_beats_noise(self):
        rng = np.random.default_rng(5)
        y = (rng.random(5000) < 0.15).astype(int)
        informative = y + rng.normal(scale=0.8, size=5000)
        noise = rng.normal(size=5000)
        result = delong_paired(y, informative, noise)
        assert result.difference > 0
        assert result.p_value < 1e-10

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError, match="align"):
            delong_paired([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3])

    def test_rejects_single_class(self):
        with pytest.raises(ValueError, match="two of each class"):
            delong_paired([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])


class TestHolm:
    def test_worked_example(self):
        """Sorted: 0.005x4=0.02, 0.01x3=0.03, 0.03x2=0.06, 0.04x1 lifted to 0.06."""
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])

    def test_capped_at_one(self):
        assert holm_adjust([0.6, 0.7]).max() == 1.0

    def test_single_test_is_unadjusted(self):
        assert holm_adjust([0.04])[0] == pytest.approx(0.04)

    def test_never_below_raw(self):
        raw = np.random.default_rng(2).random(30)
        assert (holm_adjust(raw) >= raw - 1e-15).all()
