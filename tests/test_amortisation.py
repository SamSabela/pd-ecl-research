"""Tests for the contractual amortisation schedule.

Required by Section 3.10: each component is validated against hand-calculated
examples so that the arithmetic is verified independently of the pipeline.

Every test ending `_hand_calculated` carries its arithmetic worked out in the
docstring. Read those first, they tell you exactly what the function must
produce, and a validator can confirm the expected value without reading your
implementation. That is the point of writing them this way.

Run:  pytest tests/test_amortisation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ecl.amortisation import annuity_payment, monthly_rate, outstanding_balance

TOL = 1e-9


class TestMonthlyRate:
    def test_converts_annual_to_monthly_hand_calculated(self):
        """12% nominal annual is 1% per month: 0.12 / 12 = 0.01."""
        assert monthly_rate(0.12) == pytest.approx(0.01, abs=TOL)

    def test_zero_rate(self):
        assert monthly_rate(0.0) == pytest.approx(0.0, abs=TOL)

    def test_typical_lending_club_rate_hand_calculated(self):
        """A 17.99% APR loan: 0.1799 / 12 = 0.01499166666..."""
        assert monthly_rate(0.1799) == pytest.approx(0.0149916667, abs=1e-9)


class TestAnnuityPayment:
    def test_zero_rate_is_straight_line_hand_calculated(self):
        """With no interest the schedule is straight-line repayment.

            A = P / T = 1000 / 10 = 100.00
        """
        assert annuity_payment(1000.0, 0.0, 10) == pytest.approx(100.0, abs=TOL)

    def test_standard_loan_hand_calculated(self):
        """Loan of 10,000 at 1% per month over 36 months.

        Worked to 10 decimal places (verified at 50-digit precision):

            (1.01)^36      = 1.4307687836
            (1.01)^-36     = 0.6989249496
            1 - (1.01)^-36 = 0.3010750504
            A = 10000 * 0.01 / 0.3010750504
              = 100 / 0.3010750504
              = 332.1430981285
        """
        assert annuity_payment(10_000.0, 0.01, 36) == pytest.approx(
            332.1430981285, abs=1e-9
        )

    def test_present_value_identity(self):
        """Discounting the payment stream at i must recover the principal.

        This is the defining property of an annuity: the lender is indifferent
        between the lump sum today and the payment stream. It holds for ANY
        (P, i, T), which makes it a far stronger check than any single worked
        example, if your formula is subtly wrong, this catches it everywhere.
        """
        principal, rate_m, term = 25_000.0, 0.0075, 60
        payment = annuity_payment(principal, rate_m, term)
        t = np.arange(1, term + 1)
        present_value = float(np.sum(payment / (1.0 + rate_m) ** t))
        assert present_value == pytest.approx(principal, rel=1e-12)

    def test_zero_term_raises(self):
        with pytest.raises(ValueError, match="term_months must be positive"):
            annuity_payment(1000.0, 0.01, 0)

    def test_negative_principal_raises(self):
        with pytest.raises(ValueError, match="principal must be non-negative"):
            annuity_payment(-1.0, 0.01, 12)


class TestOutstandingBalance:
    def test_zero_rate_balance_hand_calculated(self):
        """Interest-free loan of 1000 over 10 months, paying 100 a month.

            t = 0  -> 1000      t = 3  -> 700
            t = 5  ->  500      t = 10 ->   0
        """
        for elapsed, expected in [(0, 1000.0), (3, 700.0), (5, 500.0), (10, 0.0)]:
            assert outstanding_balance(1000.0, 0.0, 10, elapsed) == pytest.approx(
                expected, abs=TOL
            )

    def test_starts_at_principal(self):
        """Before any payment is made, the full principal is outstanding."""
        assert outstanding_balance(15_000.0, 0.01, 36, 0) == pytest.approx(
            15_000.0, rel=1e-12
        )

    def test_reaches_exactly_zero_at_maturity(self):
        """THE test that separates the two formulas.

        A fully amortising loan is exactly repaid by its final payment. The
        retrospective formula subtracts two large, nearly equal quantities here
        and leaves floating-point residue - a balance of a few cents on a
        15,000 loan. The prospective formula evaluates (1 - (1+i)^0) / i = 0
        directly and is exact.

        A residual balance at maturity would silently inflate the final month
        of every lifetime ECL in the study. The tolerance is tight on purpose.
        """
        assert outstanding_balance(15_000.0, 0.01, 36, 36) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_declines_every_month(self):
        """On an amortising loan the outstanding principal is strictly
        decreasing. Note this uses an array input, so your implementation must
        vectorise."""
        balances = outstanding_balance(10_000.0, 0.01, 36, np.arange(37))
        assert np.all(np.diff(balances) < 0.0)

    def test_beyond_maturity_is_zero(self):
        """Past the contractual term there is no exposure. The formula alone
        will not give you this - it starts growing again as (T - t) goes
        negative. You have to clip it."""
        assert outstanding_balance(10_000.0, 0.01, 36, 40) == pytest.approx(
            0.0, abs=TOL
        )

    def test_scalar_returns_scalar_array_returns_array(self):
        """The type of the output must match the type of the input."""
        scalar = outstanding_balance(10_000.0, 0.01, 36, 12)
        array = outstanding_balance(10_000.0, 0.01, 36, np.array([12, 24]))
        assert isinstance(scalar, float)
        assert isinstance(array, np.ndarray)
        assert array.shape == (2,)

    def test_balance_matches_independent_recursion(self):
        """Cross-check against a completely different method.

        Simulate the loan month by month the way a spreadsheet would:
        add interest, subtract the payment, repeat. If your closed-form
        expression agrees with this loop at every month, it is right.

        This is worth internalising as a technique - when you have a formula
        you are unsure of, check it against a slow, obviously-correct
        simulation rather than against your own algebra.
        """
        principal, rate_m, term = 8_000.0, 0.0125, 24
        payment = annuity_payment(principal, rate_m, term)

        balance = principal
        for month in range(1, term + 1):
            balance = balance * (1.0 + rate_m) - payment
            expected = max(balance, 0.0)
            actual = outstanding_balance(principal, rate_m, term, month)
            assert actual == pytest.approx(expected, abs=1e-6), f"month {month}"
