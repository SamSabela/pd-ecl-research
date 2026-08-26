"""Outcome definition, maturity filter and sample construction.

Implements Section 3.4 and Table 3.2. These are the two decisions signed off
at Gate G1, and both address omissions in the June proposal.

Outcome definition (3.4.1)
    The proposal did not state how the binary default indicator would be
    constructed from the raw loan status field, which would have made the
    study non-reproducible. Resolved statuses map to 0 or 1; unresolved
    statuses are excluded rather than silently coerced to non-default.

Maturity filter (3.4.2)
    A loan that has not reached the end of its contractual term cannot be
    observed to have been fully repaid, so young loans are systematically
    over-represented among apparent non-defaults. The filter removes them.

    A caveat on the expected direction, which Section 3.4.2 gets slightly
    wrong. The proposal argues the filter RAISES the observed default rate,
    because unmatured loans masquerade as non-defaults. That holds only if
    unresolved loans are labelled 0 - and Table 3.2 excludes them before this
    filter ever runs. Among the resolved loans that remain, the asymmetry
    reverses: default resolves fast, full repayment usually takes the whole
    term, so young resolved cohorts over-represent DEFAULTS. On real data the
    filter may therefore LOWER the observed rate.

    The filter is necessary either way - an unmatured cohort is not a
    representative sample of outcomes. But do not assume the sign, and if it
    comes out negative, the wording in 3.4.2 needs revising before submission.

YOUR TASK
---------
Implement the three functions marked TODO. Read tests/test_sample.py first -
the worked arithmetic in the test docstrings tells you exactly what each must
produce.

    pytest tests/test_sample.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "OutcomeDefinition",
    "MaturityFilterResult",
    "parse_term_months",
    "assign_outcome",
    "apply_maturity_filter",
]


@dataclass(frozen=True)
class OutcomeDefinition:
    """Mapping from raw loan status to the binary default indicator (Table 3.2).

    Attributes:
        default_statuses: Statuses labelled 1. Resolved adverse outcomes.
        non_default_statuses: Statuses labelled 0. Resolved benign outcomes.
        exclude_credit_policy: Whether to drop "Does not meet the credit
            policy" variants. These were not originated under the standard
            policy, so they are not comparable to the rest of the book.
    """

    default_statuses: tuple[str, ...] = ("Charged Off", "Default")
    non_default_statuses: tuple[str, ...] = ("Fully Paid",)
    exclude_credit_policy: bool = True

    @property
    def resolved_statuses(self) -> set[str]:
        """All statuses carrying an observable outcome."""
        return set(self.default_statuses) | set(self.non_default_statuses)


@dataclass(frozen=True)
class MaturityFilterResult:
    """Outcome of the maturity filter, for the Gate G1 report.

    Attributes:
        frame: The filtered sample.
        rows_before: Row count entering the filter.
        rows_after: Row count leaving it.
        default_rate_before: Observed default rate before filtering.
        default_rate_after: Observed default rate after filtering.
        origination_min: Earliest retained origination date, or None if empty.
        origination_max: Latest retained origination date, or None if empty.
    """

    frame: pd.DataFrame
    rows_before: int
    rows_after: int
    default_rate_before: float
    default_rate_after: float
    origination_min: pd.Timestamp | None
    origination_max: pd.Timestamp | None

    @property
    def bias_avoided(self) -> float:
        """Change in observed default rate attributable to the filter.

        Sign is not assumed - see the module docstring. Report it, do not
        predict it.
        """
        return self.default_rate_after - self.default_rate_before

    def summary(self) -> str:
        """Human-readable summary for the Gate G1 sign-off."""
        window = (
            f"{self.origination_min:%Y-%m} to {self.origination_max:%Y-%m}"
            if self.origination_min is not None
            else "n/a"
        )
        return (
            "Maturity filter\n"
            + "=" * 78
            + f"\n  Rows                 : {self.rows_before:,} -> {self.rows_after:,}"
            f"\n  Origination window   : {window}"
            f"\n  Default rate before  : {self.default_rate_before:.4%}"
            f"\n  Default rate after   : {self.default_rate_after:.4%}"
            f"\n  Change               : {self.bias_avoided:+.4%}"
        )


def parse_term_months(term: pd.Series) -> pd.Series:
    """Parse Lending Club's term field into an integer month count.

    The raw field is a string like " 36 months". Anything unparseable must
    become NA rather than a silent zero - a loan with an unknown term cannot
    be maturity-filtered and must not quietly pass through as term 0.

    TODO: implement.

    Hints:
        - Numeric input should pass through: check pd.api.types.is_numeric_dtype
          first and just .astype("Int64").
        - For strings: .astype("string") then .str.extract(r"(\\d+)", expand=False)
          pulls the digits out.
        - pd.to_numeric(..., errors="coerce") turns failures into NA.
        - Return dtype "Int64" (capital I) - the nullable integer type. Plain
          "int64" cannot hold NA.

    Args:
        term: Raw term column.

    Returns:
        Nullable integer series of months.
    """
    raise NotImplementedError


def assign_outcome(
    frame: pd.DataFrame,
    definition: OutcomeDefinition,
    status_col: str = "loan_status",
    target_col: str = "default_flag",
) -> pd.DataFrame:
    """Construct the binary default indicator and drop unresolved outcomes.

    Unresolved statuses (Current, In Grace Period, Late, Issued) are EXCLUDED,
    not labelled. The event of interest has not been observed for them, and
    coercing them to non-default is precisely the contamination the maturity
    filter also guards against.

    TODO: implement.

    Steps:
        1. Raise KeyError if status_col is absent. Message must contain the
           column name.
        2. Normalise: frame[status_col].astype("string").str.strip()
        3. Identify credit-policy rows: .str.contains("does not meet the credit
           policy", case=False, na=False)
        4. If definition.exclude_credit_policy, drop them. If not, strip the
           prefix so the underlying status is recovered - otherwise the option
           does nothing, because "Does not meet the credit policy. Status:Fully
           Paid" is not equal to "Fully Paid".
        5. Build boolean masks with .isin() against the two status tuples.
        6. Keep only rows where one of them is true.
        7. Return a copy with target_col added as int8: 1 for default, 0 for not.

    Args:
        frame: Raw loan data.
        definition: The outcome mapping.
        status_col: Column holding the raw loan status.
        target_col: Name for the constructed indicator.

    Returns:
        A copy containing only resolved outcomes, with target_col appended.

    Raises:
        KeyError: If the status column is absent.
    """
    raise NotImplementedError


def apply_maturity_filter(
    frame: pd.DataFrame,
    term_cutoffs: dict[int, str],
    origination_col: str = "issue_d",
    term_col: str = "term_months",
    target_col: str = "default_flag",
) -> MaturityFilterResult:
    """Retain only loans whose contractual term had fully elapsed by the cutoff.

    With a data cutoff at end-2018 this admits 36-month loans originated up to
    and including December 2015, and 60-month loans up to December 2013.

    TODO: implement.

    Steps:
        1. Raise KeyError if any of the three required columns is absent.
        2. Parse dates: pd.to_datetime(frame[origination_col], errors="coerce")
        3. Build a keep mask, starting all-False:
               keep = pd.Series(False, index=frame.index)
           For each (term_length, cutoff) in term_cutoffs, OR in the rows that
           match that term AND originated on or before the cutoff.
           Use .fillna(False) on the date comparison - a NaT comparison gives
           NA, which breaks the boolean OR.
        4. A term with no cutoff specified must NOT pass. Starting from all-False
           gives you that for free.
        5. Build and return the MaturityFilterResult, computing both default
           rates and the retained origination window. Guard the empty case:
           .mean() on an empty series is NaN, not 0.

    Args:
        frame: Loans with an outcome already assigned.
        term_cutoffs: {term_months: "YYYY-MM-DD"} latest admissible origination.
        origination_col: Column holding the origination date.
        term_col: Column holding the integer term in months.
        target_col: Column holding the binary default indicator.

    Returns:
        A MaturityFilterResult with the filtered frame and Gate G1 diagnostics.

    Raises:
        KeyError: If a required column is absent.
    """
    raise NotImplementedError
