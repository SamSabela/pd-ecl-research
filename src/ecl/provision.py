"""IFRS 9 Expected Credit Loss from model PD estimates (RO2, RQ2).

Implements the specification committed to in ICE Task 1, Tables 3.8 and 3.9,
which supersedes the flat outstanding-balance treatment in Section 3.8.3 of the
June proposal:

    12-month ECL   sum over months 1..12 of PD(t) x LGD x EAD(t) / (1+EIR)^(t/12)
    Lifetime ECL   the same sum over months 1..T
    PD(t)          constant-hazard extrapolation
    EAD(t)         outstanding principal from the contractual annuity schedule
    LGD            45%, with 30% and 60% as sensitivities
    EIR            the loan's contractual interest rate

Three decisions this module makes, each of which is a limitation to state in
the report rather than a result to claim.

Default timing. The model estimates a probability of default over the whole
contractual term. Turning that into a monthly schedule requires a view on when
within the term default occurs. A constant hazard is the weakest assumption
that does the job without estimating a term structure: monthly survival is
s = (1-q)^(1/T) and the marginal probability of defaulting in month t is
s^(t-1)(1-s). This is the same assumption ICE 1 records, and it is equivalent
to extrapolating from a 12-month PD, since 1-(1-q)^(12/T) is precisely the
12-month PD implied by the same hazard. It will understate early-life risk on a
portfolio whose hazard peaks early, which most consumer portfolios do.

Model-independent timing for the realised benchmark. Provisioned loss is
compared against realised loss. If each model's own PD drove the timing
weights, the realised benchmark would move between models and the comparison
would confound level with timing. The timing weights are therefore derived once
per evaluation cohort from its observed default rate and held fixed across
models, so that the only quantity varying across models is the PD level. That
is exactly the isolation RQ2 asks for.

Staging at origination. ICE 1 Table 3.8 allocates Stage 2 on a relative test
(reporting-date PD over origination PD) with an absolute backstop. Every
exposure in this sample is observed at origination, so the relative test is
identically one and cannot bind, and no exposure is credit-impaired, so Stage 3
is empty. The absolute backstop is therefore the operative trigger, applied to
the 12-month PD at the 3%, 5% and 8% levels ICE 1 specifies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "timing_weights",
    "balance_schedule",
    "discount_schedule",
    "twelve_month_pd",
    "marginal_default_schedule",
    "discounted_exposure_at_risk",
    "expected_credit_loss",
    "assign_stage",
    "implied_monthly_rate",
    "implied_annual_rate",
]

# Below this monthly rate the annuity formula is treated as interest-free.
ZERO_RATE_TOLERANCE = 1e-12


def timing_weights(term_months: int, cohort_pd: float) -> NDArray[np.float64]:
    """Normalised distribution of default timing across the contractual term.

    Derived from the cohort's observed default rate rather than from any
    model's prediction, so that the realised-loss benchmark does not move when
    the model changes.

    Args:
        term_months: Contractual term T.
        cohort_pd: Observed default rate over the full term, in (0, 1).

    Returns:
        Array of length T summing to 1, giving the share of defaults occurring
        in each month.

    Raises:
        ValueError: If term_months is not positive or cohort_pd is outside (0, 1).
    """
    if term_months <= 0:
        raise ValueError(f"term_months must be positive, got {term_months}")
    if not 0.0 < cohort_pd < 1.0:
        raise ValueError(f"cohort_pd must lie in (0, 1), got {cohort_pd}")

    survival = (1.0 - cohort_pd) ** (1.0 / term_months)
    hazard = 1.0 - survival
    periods = np.arange(term_months)
    marginal = survival**periods * hazard
    return marginal / marginal.sum()


def balance_schedule(
    principal: NDArray[np.float64],
    rate_m: NDArray[np.float64],
    term_months: int,
) -> NDArray[np.float64]:
    """Outstanding principal at the start of each month, vectorised over loans.

    This is the prospective form used by src.ecl.amortisation.outstanding_balance,
    broadcast across loans so that a 280,000-row cohort can be valued without a
    Python loop. The scalar function remains the reference implementation and
    tests/test_provision.py checks this path against it on a random sample.

    EAD(t) is taken as the balance at the *start* of month t, that is after
    t-1 payments, because that is the amount at risk during the month in which
    default occurs.

    Args:
        principal: Original loan amounts, shape (n,).
        rate_m: Monthly periodic rates, shape (n,).
        term_months: Contractual term T.

    Returns:
        Array of shape (n, T) of outstanding balances.
    """
    principal = np.asarray(principal, dtype=np.float64).reshape(-1, 1)
    rate_m = np.asarray(rate_m, dtype=np.float64).reshape(-1, 1)
    elapsed = np.arange(term_months, dtype=np.float64).reshape(1, -1)
    remaining = term_months - elapsed

    zero = np.abs(rate_m) < ZERO_RATE_TOLERANCE
    safe_rate = np.where(zero, 1.0, rate_m)

    payment = principal * safe_rate / (1.0 - (1.0 + safe_rate) ** (-term_months))
    interest_bearing = payment * (1.0 - (1.0 + safe_rate) ** (-remaining)) / safe_rate
    interest_free = principal * (remaining / term_months)

    balance = np.where(zero, interest_free, interest_bearing)
    return np.clip(balance, 0.0, None)


def discount_schedule(annual_rate: NDArray[np.float64], term_months: int) -> NDArray[np.float64]:
    """Discount factors (1 + EIR)^(-t/12) for months t = 1..T, per loan.

    Discounting runs to the end of the month in which default occurs, matching
    the exponent convention in ICE 1 Table 3.9.
    """
    annual_rate = np.asarray(annual_rate, dtype=np.float64).reshape(-1, 1)
    months = np.arange(1, term_months + 1, dtype=np.float64).reshape(1, -1)
    return (1.0 + annual_rate) ** (-months / 12.0)


def twelve_month_pd(term_pd, term_months: int):
    """Twelve-month PD implied by a term PD under a constant hazard.

    Capped at the remaining term where that is shorter than twelve months, per
    ICE 1 Table 3.9.
    """
    horizon = min(12, term_months)
    return 1.0 - (1.0 - np.asarray(term_pd, dtype=np.float64)) ** (horizon / term_months)


def marginal_default_schedule(term_pd, term_months: int) -> NDArray[np.float64]:
    """Monthly marginal default probabilities implied by each loan's term PD.

    Row t of the result is the probability that the loan survives t-1 months
    and then defaults in month t. Rows sum to the loan's term PD.

    Args:
        term_pd: Predicted probability of default over the term, shape (n,).
        term_months: Contractual term T.

    Returns:
        Array of shape (n, T).
    """
    term_pd = np.clip(np.asarray(term_pd, dtype=np.float64), 1e-12, 1.0 - 1e-12).reshape(-1, 1)
    survival = (1.0 - term_pd) ** (1.0 / term_months)
    hazard = 1.0 - survival
    periods = np.arange(term_months, dtype=np.float64).reshape(1, -1)
    return survival**periods * hazard


def expected_credit_loss(
    term_pd,
    principal,
    annual_rate,
    term_months: int,
    lgd: float,
    horizon_months: int | None = None,
) -> NDArray[np.float64]:
    """Per-loan ECL as the discounted sum of monthly expected losses.

    Args:
        term_pd: Predicted probability of default over the term, shape (n,).
        principal: Original loan amounts, shape (n,).
        annual_rate: Contractual annual interest rates as decimals, shape (n,).
        term_months: Contractual term T.
        lgd: Loss given default as a decimal.
        horizon_months: Months to sum over. None means the full term, which is
            the lifetime ECL; 12 gives the Stage 1 twelve-month ECL.

    Returns:
        Array of shape (n,) of ECL amounts.
    """
    horizon = term_months if horizon_months is None else min(horizon_months, term_months)
    marginal = marginal_default_schedule(term_pd, term_months)[:, :horizon]
    exposure = balance_schedule(principal, np.asarray(annual_rate) / 12.0, term_months)[:, :horizon]
    discount = discount_schedule(annual_rate, term_months)[:, :horizon]
    return lgd * (marginal * exposure * discount).sum(axis=1)


def discounted_exposure_at_risk(
    principal,
    annual_rate,
    term_months: int,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exposure a loan carries at default, discounted and averaged over timing.

    Uses cohort timing weights rather than model-specific ones, so that the
    quantity is identical across models. Provisioned loss for a loan is then
    its predicted PD times LGD times this number, and realised loss is its
    actual default indicator times LGD times the same number. The difference
    between the two is therefore attributable entirely to the PD level.
    """
    exposure = balance_schedule(principal, np.asarray(annual_rate) / 12.0, term_months)
    discount = discount_schedule(annual_rate, term_months)
    return (exposure * discount * weights.reshape(1, -1)).sum(axis=1)


def assign_stage(pd_twelve_month, backstop: float) -> NDArray[np.int_]:
    """Allocate exposures to IFRS 9 stages at origination.

    Stage 3 is empty by construction: the sample contains only loans observed
    at origination, none of which is credit-impaired. The relative test of
    ICE 1 Table 3.8 is identically one at origination and cannot bind, so the
    absolute backstop is the operative trigger.

    Args:
        pd_twelve_month: Twelve-month PD per loan, shape (n,).
        backstop: Absolute PD threshold above which Stage 2 applies.

    Returns:
        Array of stage labels, 1 or 2.
    """
    pd_twelve_month = np.asarray(pd_twelve_month, dtype=np.float64)
    return np.where(pd_twelve_month >= backstop, 2, 1).astype(int)


def implied_monthly_rate(
    principal,
    installment,
    term_months: int,
    iterations: int = 60,
) -> NDArray[np.float64]:
    """Recover the monthly periodic rate from the level instalment.

    The analysis sample carries loan_amnt, installment and term but not the
    contractual rate, which Gate G1 drops because ICE 1 excludes the EIR from
    the feature set. Exclusion from the design matrix is not the same as
    exclusion from the sample, and the ECL translation needs the rate for both
    discounting and amortisation. It is recoverable exactly: the annuity
    identity

        A = P i / (1 - (1+i)^-T)

    is strictly increasing in i, so a bisection on i inverts it uniquely.

    Sixty bisections on a bracket of [0, 0.10] monthly resolve i far below
    float64 precision, and 0.10 monthly is 120 per cent nominal, well above any
    consumer loan. A total repayment at or below principal implies a
    non-positive rate and is returned as zero rather than extrapolated.

    Args:
        principal: Original loan amounts, shape (n,).
        installment: Contractual level monthly payment, shape (n,).
        term_months: Contractual term T.
        iterations: Bisection steps.

    Returns:
        Monthly periodic rates, shape (n,).
    """
    principal = np.asarray(principal, dtype=np.float64)
    installment = np.asarray(installment, dtype=np.float64)

    low = np.zeros_like(principal)
    high = np.full_like(principal, 0.10)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        safe = np.maximum(mid, 1e-12)
        payment = principal * safe / (1.0 - (1.0 + safe) ** (-term_months))
        below = payment < installment
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)

    rate = 0.5 * (low + high)
    interest_free = installment * term_months <= principal
    return np.where(interest_free, 0.0, rate)


def implied_annual_rate(principal, installment, term_months: int) -> NDArray[np.float64]:
    """Nominal annual rate implied by the instalment.

    Uses the nominal/12 convention of src.ecl.amortisation.monthly_rate, so
    that a rate recovered here and fed back through annuity_payment reproduces
    the instalment it came from.
    """
    return 12.0 * implied_monthly_rate(principal, installment, term_months)
