"""Tests for the outcome definition and maturity filter.

These are the two decisions signed off at Gate G1. Every test ending
_hand_calculated carries its arithmetic worked out in the docstring, so a
validator can confirm the expected value without reading your implementation.

Run:  pytest tests/test_sample.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.sample import (
    OutcomeDefinition,
    apply_maturity_filter,
    assign_outcome,
    parse_term_months,
)

TOL = 1e-9


class TestParseTermMonths:
    def test_parses_lending_club_strings(self):
        """The raw field is ' 36 months' - leading space, trailing word."""
        parsed = parse_term_months(pd.Series([" 36 months", "60 months"]))
        assert parsed.tolist() == [36, 60]

    def test_unparseable_becomes_na_not_zero(self):
        """A loan with an unknown term must not silently pass as term 0 - it
        would then match no cutoff and be dropped, which is correct, but a 0
        would be indistinguishable from a genuine data error."""
        parsed = parse_term_months(pd.Series([" 36 months", None, "unknown"]))
        assert parsed.iloc[0] == 36
        assert pd.isna(parsed.iloc[1])
        assert pd.isna(parsed.iloc[2])

    def test_numeric_input_passes_through(self):
        parsed = parse_term_months(pd.Series([36, 60]))
        assert parsed.tolist() == [36, 60]

    def test_returns_nullable_integer_dtype(self):
        """Int64, not int64 - plain int64 cannot hold NA."""
        assert str(parse_term_months(pd.Series([" 36 months"])).dtype) == "Int64"


class TestAssignOutcome:
    @staticmethod
    def _raw() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "loan_status": [
                    "Fully Paid",
                    "Charged Off",
                    "Default",
                    "Current",
                    "Late (31-120 days)",
                    "In Grace Period",
                    "Does not meet the credit policy. Status:Fully Paid",
                ],
                "issue_d": pd.to_datetime(["2014-01-01"] * 7),
            }
        )

    def test_only_resolved_outcomes_retained_hand_calculated(self):
        """Seven rows in. Fully Paid, Charged Off and Default are resolved.
        Current, Late and In Grace Period are not - the event has not been
        observed. The credit-policy row is excluded by default.

        Three rows survive.
        """
        out = assign_outcome(self._raw(), OutcomeDefinition())
        assert len(out) == 3
        assert set(out["loan_status"]) == {"Fully Paid", "Charged Off", "Default"}

    def test_labels_hand_calculated(self):
        """Fully Paid -> 0, Charged Off -> 1, Default -> 1.

        Two of three are defaults, so the observed rate is 2/3 = 0.666667.
        """
        out = assign_outcome(self._raw(), OutcomeDefinition())
        assert out["default_flag"].tolist() == [0, 1, 1]
        assert out["default_flag"].mean() == pytest.approx(2 / 3, abs=TOL)

    def test_target_is_int8(self):
        """A binary flag does not need 64 bits. On 1.3m rows that is 7MB saved
        for nothing."""
        out = assign_outcome(self._raw(), OutcomeDefinition())
        assert out["default_flag"].dtype == "int8"

    def test_credit_policy_excluded_by_default(self):
        out = assign_outcome(self._raw(), OutcomeDefinition())
        assert not out["loan_status"].str.contains("credit policy").any()

    def test_credit_policy_retained_when_disabled_hand_calculated(self):
        """With the flag off, the prefixed row must be RECOVERED, not merely
        left in. "Does not meet the credit policy. Status:Fully Paid" is not
        equal to "Fully Paid", so a naive .isin() would still drop it and the
        option would do nothing.

        Four rows survive, and the recovered one is labelled 0.
        """
        out = assign_outcome(self._raw(), OutcomeDefinition(exclude_credit_policy=False))
        assert len(out) == 4
        assert out["default_flag"].sum() == 2

    def test_other_columns_preserved(self):
        out = assign_outcome(self._raw(), OutcomeDefinition())
        assert "issue_d" in out.columns

    def test_original_frame_not_mutated(self):
        """Return a copy. Mutating the caller's frame is how pipelines develop
        order-dependence that only shows up when you reorder a notebook."""
        raw = self._raw()
        assign_outcome(raw, OutcomeDefinition())
        assert "default_flag" not in raw.columns
        assert len(raw) == 7

    def test_missing_status_column_raises(self):
        with pytest.raises(KeyError, match="loan_status"):
            assign_outcome(pd.DataFrame({"x": [1]}), OutcomeDefinition())

    def test_whitespace_tolerated(self):
        """Real extracts carry stray whitespace."""
        frame = pd.DataFrame({"loan_status": ["  Fully Paid  ", " Charged Off"]})
        out = assign_outcome(frame, OutcomeDefinition())
        assert len(out) == 2
        assert out["default_flag"].tolist() == [0, 1]


class TestMaturityFilter:
    @staticmethod
    def _frame() -> pd.DataFrame:
        """Six loans spanning both terms and both sides of each cutoff."""
        return pd.DataFrame(
            {
                "issue_d": pd.to_datetime(
                    [
                        "2014-06-01",  # 36m, inside  -> keep
                        "2015-12-01",  # 36m, boundary -> keep
                        "2016-06-01",  # 36m, too young -> drop
                        "2013-06-01",  # 60m, inside  -> keep
                        "2014-06-01",  # 60m, too young -> drop
                        "2013-12-01",  # 60m, boundary -> keep
                    ]
                ),
                "term_months": [36, 36, 36, 60, 60, 60],
                "default_flag": [0, 1, 0, 1, 0, 0],
            }
        )

    CUTOFFS = {36: "2015-12-31", 60: "2013-12-31"}

    def test_retention_hand_calculated(self):
        """Cutoffs: 36-month to 2015-12-31, 60-month to 2013-12-31.

        Rows 0, 1, 3 and 5 are retained; rows 2 and 4 are too young.
        Four of six survive.
        """
        result = apply_maturity_filter(self._frame(), self.CUTOFFS)
        assert result.rows_before == 6
        assert result.rows_after == 4
        assert set(result.frame.index) == {0, 1, 3, 5}

    def test_boundary_is_inclusive(self):
        """A 36-month loan originated exactly on 2015-12-31 has just matured
        by the cutoff and must be kept, not dropped."""
        frame = pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2015-12-31"]),
                "term_months": [36],
                "default_flag": [0],
            }
        )
        assert apply_maturity_filter(frame, self.CUTOFFS).rows_after == 1

    def test_default_rates_hand_calculated(self):
        """Before: labels [0,1,0,1,0,0] -> 2 defaults of 6 = 33.3333%.
        After (rows 0,1,3,5): labels [0,1,1,0] -> 2 of 4 = 50.0000%.
        Change: +16.6667 percentage points.
        """
        result = apply_maturity_filter(self._frame(), self.CUTOFFS)
        assert result.default_rate_before == pytest.approx(2 / 6, abs=TOL)
        assert result.default_rate_after == pytest.approx(0.50, abs=TOL)
        assert result.bias_avoided == pytest.approx(0.5 - 2 / 6, abs=TOL)

    def test_origination_window_reported(self):
        """Gate G1 requires the retained window. Earliest kept is 2013-06,
        latest kept is 2015-12."""
        result = apply_maturity_filter(self._frame(), self.CUTOFFS)
        assert result.origination_min == pd.Timestamp("2013-06-01")
        assert result.origination_max == pd.Timestamp("2015-12-01")

    def test_unlisted_term_is_excluded(self):
        """A 12-month term has no cutoff specified. It must not silently pass -
        no maturity rule has been stated for it."""
        frame = pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2010-01-01"]),
                "term_months": [12],
                "default_flag": [0],
            }
        )
        assert apply_maturity_filter(frame, self.CUTOFFS).rows_after == 0

    def test_missing_term_excluded(self):
        """An unparseable term cannot be maturity-checked, so it is dropped."""
        frame = pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2012-01-01"]),
                "term_months": pd.array([None], dtype="Int64"),
                "default_flag": [0],
            }
        )
        assert apply_maturity_filter(frame, self.CUTOFFS).rows_after == 0

    def test_empty_result_does_not_crash(self):
        """Everything filtered out is a legitimate (if alarming) outcome. The
        report must still render rather than raising on an empty mean."""
        frame = pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2018-01-01"]),
                "term_months": [36],
                "default_flag": [0],
            }
        )
        result = apply_maturity_filter(frame, self.CUTOFFS)
        assert result.rows_after == 0
        assert result.origination_min is None
        assert "Maturity filter" in result.summary()

    def test_summary_renders(self):
        assert "Origination window" in apply_maturity_filter(
            self._frame(), self.CUTOFFS
        ).summary()

    def test_missing_column_raises(self):
        frame = self._frame().drop(columns=["term_months"])
        with pytest.raises(KeyError, match="term_months"):
            apply_maturity_filter(frame, self.CUTOFFS)
