"""Contractual amortisation schedules for exposure-at-default profiling.

Implements Section 3.8.3: exposure at default in month t is the outstanding
principal under the loan's contractual amortisation schedule, computed from the
loan amount, contractual interest rate and term as a standard annuity.

This replaces the proposal's flat assumption that exposure equals the
outstanding balance. It matters because an amortising loan's exposure declines
materially over its life: a 36-month loan in month 30 has almost nothing left
outstanding, and treating it as fully exposed overstates lifetime ECL.

Note on leakage (Table 3.4): the contractual interest rate is used here for
exposure amortisation and discounting only. It is never admitted as a model
input, because it encodes the originator's own risk assessment.

YOUR TASK
---------
Implement the three functions below. Run `pytest tests/test_amortisation.py -v`
until all tests pass. Read the test file first - the worked arithmetic in the
docstrings tells you exactly what each function must produce.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["monthly_rate", "annuity_payment", "outstanding_balance"]


def monthly_rate(annual_rate: float) -> float:
    """Convert a nominal annual rate to a simple monthly rate.

    Use nominal/12 rather than an effective compounding conversion
    ((1+r)^(1/12) - 1). The simple division is the convention used in consumer
    loan amortisation, and it is the convention under which Lending Club
    instalments are quoted. Using the effective conversion here would make your
    computed instalment disagree with the `installment` field in the raw data,
    which is a useful cross-check you will want later.

    Args:
        annual_rate: Nominal annual interest rate as a decimal (0.12 for 12%).

    Returns:
        Monthly periodic rate as a decimal.
    """
    return annual_rate / 12.0


def annuity_payment(principal: float, rate_m: float, term_months: int) -> float:
    """Level monthly instalment for a fully amortising annuity loan.

        A = P * i / (1 - (1 + i)^-T)

    Two things the formula does not tell you, which the tests will:

    1. The zero-rate case. If i = 0 the denominator is 0 and you get a
       ZeroDivisionError (or a numpy warning and a nan). An interest-free loan
       is a legitimate input, and its instalment is simply P / T. Handle it
       explicitly. Use np.isclose(rate_m, 0.0) rather than rate_m == 0.0 -
       exact float equality is fragile.

    2. Input validation. A term of zero or a negative principal are not
       meaningful loans. Raise ValueError with a message that says what was
       wrong and what was received. The tests check the message text, so read
       them for the expected wording.

    Args:
        principal: Original loan amount.
        rate_m: Monthly periodic rate as a decimal.
        term_months: Contractual term in months.

    Returns:
        The level monthly payment.

    Raises:
        ValueError: If term_months is not positive or principal is negative.
    """
    raise NotImplementedError


def outstanding_balance(
    principal: float,
    rate_m: float,
    term_months: int,
    periods_elapsed: int | NDArray[np.int_],
) -> float | NDArray[np.float64]:
    """Outstanding principal after a given number of payments.

    THE DECISION YOU NEED TO MAKE
    -----------------------------
    There are two algebraically identical ways to compute this.

    (a) Retrospective - work forwards from the principal. Grow the original
        balance by interest, subtract the payments made so far with their
        accumulated interest:

            B(t) = P(1+i)^t  -  A * ((1+i)^t - 1) / i

    (b) Prospective - work backwards from what is still owed. Take the present
        value of the payments that remain:

            B(t) = A * (1 - (1+i)^-(T-t)) / i

    They give the same answer in exact arithmetic. They do not give identical
    answers in float64.

    ANSWERED: (b), the prospective form, because (a) suffers catastrophic
    cancellation. At t = T the two terms in (a) are each about 21,461 for a
    15,000 loan and they differ only in the last representable digit, so
    subtracting them destroys every significant figure of a result that should
    be zero.

    Magnitude, honestly: the residual is around 3.6e-12, and at worst 3.7e-9
    across realistic rates and terms. It would not change a single reported
    number in this study. The reason to prefer (b) is that it returns exactly
    zero by construction rather than approximately zero plus a clip - which
    means you can assert exact properties about it and reason about the
    function without holding the surrounding guard rails in your head.

    That is a real but modest benefit. Do not oversell it in the write-up.

    OTHER REQUIREMENTS
    ------------------
    - Must accept a scalar OR a numpy array for periods_elapsed, and return the
      matching type. np.asarray and np.isscalar are your tools.
    - Past maturity the loan is repaid: any t >= term_months returns 0.0.
      Watch out - the formula will not naturally give you this, because
      (T - t) goes negative and the expression starts growing again. Clip it.
    - Handle the zero-rate case, as above.

    Args:
        principal: Original loan amount.
        rate_m: Monthly periodic rate as a decimal.
        term_months: Contractual term in months.
        periods_elapsed: Number of payments made. Scalar or array.

    Returns:
        Outstanding principal, matching the shape of periods_elapsed.
    """
    raise NotImplementedError
