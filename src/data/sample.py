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

Empirical note (profiled 26 August 2026, 2,260,701 rows)
    Fully Paid   1,076,751   47.6%
    Current        878,317   38.9%   excluded, unresolved
    Charged Off    268,559   11.9%
    Late/Grace      34,252    1.5%   excluded, unresolved
    Credit policy    2,749    0.1%   excluded
    Default             40    0.0%

    Resolved sample 1,345,350 (59.5% of file); raw default rate 19.96%.

    Note that "Default" as a status is vanishingly rare - Lending Club's
    default event is in practice the charge-off. Both are retained in the
    definition because both are resolved adverse outcomes, but the write-up
    should say so rather than leave a reader wondering.
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
    if pd.api.types.is_numeric_dtype(term):
        return term.astype("Int64")
    extracted = term.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


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
    if status_col not in frame.columns:
        raise KeyError(f"status column '{status_col}' not in frame")

    status = frame[status_col].astype("string").str.strip()
    is_credit_policy = status.str.contains(
        "does not meet the credit policy", case=False, na=False
    )

    if definition.exclude_credit_policy:
        keep_policy = ~is_credit_policy
    else:
        # Retaining these rows is only meaningful if the underlying status is
        # recovered from the prefixed label. "Does not meet the credit policy.
        # Status:Fully Paid" is not equal to "Fully Paid", so a plain .isin()
        # would drop it anyway and the option would silently do nothing.
        keep_policy = pd.Series(True, index=frame.index)
        status = status.str.replace(
            r"^Does not meet the credit policy\.\s*Status:\s*", "", regex=True
        ).str.strip()

    is_default = status.isin(definition.default_statuses)
    is_non_default = status.isin(definition.non_default_statuses)
    resolved = (is_default | is_non_default) & keep_policy

    out = frame.loc[resolved].copy()
    out[target_col] = is_default.loc[resolved].astype("int8")
    return out


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
    for column in (origination_col, term_col, target_col):
        if column not in frame.columns:
            raise KeyError(f"required column '{column}' not in frame")

    origination = pd.to_datetime(frame[origination_col], errors="coerce")
    term = frame[term_col].astype("Int64")

    # Start all-False so that a term with no stated cutoff cannot pass. A loan
    # whose maturity rule has not been specified is not admissible evidence.
    keep = pd.Series(False, index=frame.index)
    for term_length, cutoff in term_cutoffs.items():
        matches_term = (term == int(term_length)).fillna(False)
        within_window = (origination <= pd.Timestamp(cutoff)).fillna(False)
        keep |= matches_term & within_window

    filtered = frame.loc[keep].copy()
    retained = origination.loc[keep]

    return MaturityFilterResult(
        frame=filtered,
        rows_before=len(frame),
        rows_after=len(filtered),
        default_rate_before=float(frame[target_col].mean()) if len(frame) else 0.0,
        default_rate_after=float(filtered[target_col].mean()) if len(filtered) else 0.0,
        origination_min=retained.min() if len(filtered) else None,
        origination_max=retained.max() if len(filtered) else None,
    )
