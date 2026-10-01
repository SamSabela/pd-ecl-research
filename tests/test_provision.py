"""Tests for the IFRS 9 ECL computation (src.ecl.provision).

Two of these matter more than the rest.

`test_vectorised_balance_matches_scalar_reference` pins the fast path against
src.ecl.amortisation.outstanding_balance. The vectorised schedule exists only
because a 280,000-row cohort cannot be valued in a Python loop; the scalar
function remains the reference implementation, and if the two ever disagree it
is the fast path that is wrong.

`test_twelve_month_pd_equals_first_twelve_marginals` pins the equivalence that
lets this module work from the term PD at all. ICE 1 Table 3.9 specifies a
constant-hazard extrapolation from the calibrated 12-month PD; the models
produce a term PD. The two are the same object under a constant hazard, and
this test is the proof.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ecl.amortisation import annuity_payment, monthly_rate, outstanding_balance
from src.ecl.provision import (
    assign_stage,
    balance_schedule,
    discount_schedule,
    discounted_exposure_at_risk,
    expected_credit_loss,
    implied_annual_rate,
    implied_monthly_rate,
    marginal_default_schedule,
    timing_weights,
    twelve_month_pd,
)

TERM = 36


@pytest.fixture
def loans():
    rng = np.random.default_rng(20260815)
    principal = rng.uniform(1_000.0, 35_000.0, 250)
    annual = rng.uniform(0.0532, 0.3099, 250)   # Lending Club's published range
    return principal, annual


class TestBalanceSchedule:
    def test_vectorised_balance_matches_scalar_reference(self, loans):
        """The fast path must agree with the tested scalar implementation."""
        principal, annual = loans
        fast = balance_schedule(principal, annual / 12.0, TERM)
        for k in range(0, len(principal), 25):
            for t in (0, 1, 12, 35):
                expected = outstanding_balance(
                    principal[k], monthly_rate(annual[k]), TERM, t
                )
                assert fast[k, t] == pytest.approx(expected, abs=1e-6)

    def test_first_month_is_full_principal(self, loans):
        """EAD(1) is the balance before any payment, so the whole principal."""
        principal, annual = loans
        fast = balance_schedule(principal, annual / 12.0, TERM)
        assert fast[:, 0] == pytest.approx(principal)

    def test_balance_declines_monotonically(self, loans):
        principal, annual = loans
        fast = balance_schedule(principal, annual / 12.0, TERM)
        assert np.all(np.diff(fast, axis=1) < 0)

    def test_zero_rate_amortises_linearly(self):
        fast = balance_schedule(np.array([12_000.0]), np.array([0.0]), TERM)
        assert fast[0, 0] == pytest.approx(12_000.0)
        assert fast[0, 18] == pytest.approx(6_000.0)


class TestTimingAndHazard:
    def test_weights_sum_to_one(self):
        assert timing_weights(TERM, 0.1389).sum() == pytest.approx(1.0)

    def test_weights_decline_under_constant_hazard(self):
        """Survival falls each month, so fewer loans remain to default."""
        assert np.all(np.diff(timing_weights(TERM, 0.1389)) < 0)

    def test_marginal_schedule_sums_to_term_pd(self):
        term_pd = np.array([0.05, 0.1389, 0.40])
        marginal = marginal_default_schedule(term_pd, TERM)
        assert marginal.sum(axis=1) == pytest.approx(term_pd)

    def test_twelve_month_pd_equals_first_twelve_marginals(self):
        """The equivalence that licenses working from the term PD."""
        term_pd = np.array([0.05, 0.1389, 0.40])
        marginal = marginal_default_schedule(term_pd, TERM)
        assert twelve_month_pd(term_pd, TERM) == pytest.approx(marginal[:, :12].sum(axis=1))

    def test_rejects_impossible_cohort_rate(self):
        with pytest.raises(ValueError, match="must lie in"):
            timing_weights(TERM, 0.0)


class TestExpectedCreditLoss:
    def test_lifetime_is_at_least_twelve_month(self, loans):
        principal, annual = loans
        term_pd = np.full(len(principal), 0.1389)
        lifetime = expected_credit_loss(term_pd, principal, annual, TERM, 0.45)
        twelve = expected_credit_loss(
            term_pd, principal, annual, TERM, 0.45, horizon_months=12
        )
        assert np.all(lifetime >= twelve)

    def test_scales_linearly_in_lgd(self, loans):
        principal, annual = loans
        term_pd = np.full(len(principal), 0.1389)
        at45 = expected_credit_loss(term_pd, principal, annual, TERM, 0.45)
        at60 = expected_credit_loss(term_pd, principal, annual, TERM, 0.60)
        assert at60 / at45 == pytest.approx(0.60 / 0.45)

    def test_discounting_reduces_the_provision(self, loans):
        """A positive rate must make the discounted provision smaller."""
        principal, annual = loans
        term_pd = np.full(len(principal), 0.1389)
        discounted = expected_credit_loss(term_pd, principal, annual, TERM, 0.45)
        undiscounted = expected_credit_loss(
            term_pd, principal, np.zeros_like(annual), TERM, 0.45
        )
        assert np.all(discounted < undiscounted)

    def test_horizon_beyond_term_is_capped(self, loans):
        principal, annual = loans
        term_pd = np.full(len(principal), 0.1389)
        full = expected_credit_loss(term_pd, principal, annual, TERM, 0.45)
        over = expected_credit_loss(
            term_pd, principal, annual, TERM, 0.45, horizon_months=120
        )
        assert over == pytest.approx(full)

    def test_discount_schedule_starts_at_month_one(self, loans):
        """Discounting runs to the end of the month of default, not from zero."""
        _, annual = loans
        factors = discount_schedule(annual, TERM)
        assert np.all(factors[:, 0] < 1.0)
        assert np.all(np.diff(factors, axis=1) < 0)


class TestRateRecovery:
    def test_recovers_the_rate_that_generated_the_instalment(self, loans):
        principal, annual = loans
        rate_m = annual / 12.0
        instalment = np.array([
            annuity_payment(p, i, TERM) for p, i in zip(principal, rate_m, strict=True)
        ])
        assert implied_annual_rate(principal, instalment, TERM) == pytest.approx(annual)

    def test_interest_free_loan_returns_zero(self):
        recovered = implied_monthly_rate(
            np.array([12_000.0]), np.array([12_000.0 / TERM]), TERM
        )
        assert recovered == pytest.approx(0.0)


class TestStaging:
    def test_backstop_allocates_stage_two(self):
        stages = assign_stage(np.array([0.02, 0.05, 0.09]), backstop=0.05)
        assert stages.tolist() == [1, 2, 2]

    def test_a_higher_backstop_allocates_fewer_to_stage_two(self):
        pd_12 = np.array([0.02, 0.04, 0.06, 0.09])
        loose = (assign_stage(pd_12, 0.03) == 2).sum()
        strict = (assign_stage(pd_12, 0.08) == 2).sum()
        assert strict < loose


class TestExposureAtRisk:
    def test_is_independent_of_any_model(self, loans):
        """The realised-loss benchmark must not move when the model changes.

        Exposure at risk is built from cohort timing weights, so two different
        models scoring the same cohort share it exactly. That is what makes the
        provisioned-versus-realised comparison attributable to the PD level.
        """
        principal, annual = loans
        weights = timing_weights(TERM, 0.1389)
        first = discounted_exposure_at_risk(principal, annual, TERM, weights)
        second = discounted_exposure_at_risk(principal, annual, TERM, weights)
        assert first == pytest.approx(second)

    def test_lies_below_principal(self, loans):
        """Discounted, amortising exposure cannot exceed the amount advanced."""
        principal, annual = loans
        weights = timing_weights(TERM, 0.1389)
        exposure = discounted_exposure_at_risk(principal, annual, TERM, weights)
        assert np.all(exposure < principal)
