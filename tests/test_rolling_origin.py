"""Tests for rolling-origin backtesting (Section 3.4.3, revised).

The row-count split shipped in ICE Task 2 gave the training partition 96 months
and the validation and test partitions four or five months each, because 42% of
the sample was originated in 2015. Over four months every macroeconomic series
is flat, so the planned comparison between feature blocks A, B and C could not
have detected a macroeconomic effect even if one existed.

Rolling-origin backtesting replaces it: the model is refitted at each of several
origination cut-offs and evaluated on the cohort originated in the year that
follows. Three folds give three evaluation years spanning distinct macro states,
which is what the attribution question needs, and it is what a model validation
function would run anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.partition import Fold, RollingOriginResult, rolling_origin_folds


FOLDS = [("2012-12-31", "2013-12-31"),
         ("2013-12-31", "2014-12-31"),
         ("2014-12-31", "2015-12-31")]


def _monthly(start: str = "2010-01-31", periods: int = 72) -> pd.DataFrame:
    """One row per month, 2010-01 to 2015-12 inclusive by default.

    Deterministic outcomes so hand-calculated default rates are exact: every
    third row defaults, giving 24 defaults in 72 rows.
    """
    dates = pd.date_range(start, periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "issue_d": dates,
            "default_flag": [(1 if i % 3 == 0 else 0) for i in range(periods)],
        }
    )


class TestFoldConstruction:
    def test_fold_count_matches_boundaries(self):
        result = rolling_origin_folds(_monthly(), FOLDS)
        assert len(result.folds) == 3
        assert all(isinstance(f, Fold) for f in result.folds)

    def test_hand_calculated_row_counts(self):
        """72 monthly rows, 2010-01 to 2015-12.

        Fold 1 trains on 2010-01 to 2012-12 = 36 months, evaluates on 2013 = 12.
        Fold 2 trains on 2010-01 to 2013-12 = 48 months, evaluates on 2014 = 12.
        Fold 3 trains on 2010-01 to 2014-12 = 60 months, evaluates on 2015 = 12.
        """
        folds = rolling_origin_folds(_monthly(), FOLDS).folds
        assert [f.train_rows for f in folds] == [36, 48, 60]
        assert [f.evaluate_rows for f in folds] == [12, 12, 12]

    def test_fold_names_are_the_evaluation_year(self):
        folds = rolling_origin_folds(_monthly(), FOLDS).folds
        assert [f.name for f in folds] == ["2013", "2014", "2015"]

    def test_index_preserved(self):
        frame = _monthly()
        frame.index = range(5000, 5072)
        fold = rolling_origin_folds(frame, FOLDS).folds[0]
        assert set(fold.train).issubset(set(frame.index))
        assert set(fold.evaluate).issubset(set(frame.index))
        assert fold.train.intersection(fold.evaluate).empty

    def test_shuffled_input_gives_identical_folds(self):
        """Assignment must depend on origination date, not row order."""
        ordered = rolling_origin_folds(_monthly(), FOLDS).folds
        shuffled = rolling_origin_folds(
            _monthly().sample(frac=1.0, random_state=17), FOLDS
        ).folds
        for a, b in zip(ordered, shuffled):
            assert sorted(a.train) == sorted(b.train)
            assert sorted(a.evaluate) == sorted(b.evaluate)


class TestTemporalIntegrity:
    def test_training_strictly_precedes_evaluation(self):
        """The defining property. Without it there is no look-ahead control."""
        frame = _monthly()
        dates = pd.to_datetime(frame["issue_d"])
        for fold in rolling_origin_folds(frame, FOLDS).folds:
            assert dates.loc[fold.train].max() < dates.loc[fold.evaluate].min()

    def test_training_windows_expand(self):
        """Each fold refits on everything the previous fold saw, plus the year
        it was evaluated on. This is an expanding window, not a sliding one:
        discarding early cohorts would throw away the only crisis-era data in
        the sample."""
        folds = rolling_origin_folds(_monthly(), FOLDS).folds
        for earlier, later in zip(folds, folds[1:]):
            assert set(earlier.train).issubset(set(later.train))
            assert set(earlier.evaluate).issubset(set(later.train))

    def test_evaluation_windows_are_disjoint(self):
        folds = rolling_origin_folds(_monthly(), FOLDS).folds
        seen: set = set()
        for fold in folds:
            assert not (seen & set(fold.evaluate))
            seen |= set(fold.evaluate)

    def test_evaluation_windows_are_contiguous(self):
        """No cohort falls between two evaluation windows and goes unscored."""
        folds = rolling_origin_folds(_monthly(), FOLDS).folds
        for earlier, later in zip(folds, folds[1:]):
            gap = (later.evaluate_start - earlier.evaluate_end).days
            assert 0 < gap <= 31


class TestEmbargo:
    def test_embargo_shortens_training_window(self):
        """A 36-month loan originated in December 2014 does not resolve until
        December 2017. Training on it to score a 2015 origination uses an
        outcome that would not have been observable at the time. The embargo
        holds back any training loan whose term had not fully elapsed by the
        start of the evaluation window.

        Fold 1 without embargo trains to 2012-12 (36 rows). With a 12-month
        embargo it trains to 2011-12 (24 rows)."""
        plain = rolling_origin_folds(_monthly(), FOLDS).folds[0]
        held = rolling_origin_folds(_monthly(), FOLDS, embargo_months=12).folds[0]
        assert plain.train_rows == 36
        assert held.train_rows == 24

    def test_embargo_leaves_evaluation_untouched(self):
        plain = rolling_origin_folds(_monthly(), FOLDS).folds[0]
        held = rolling_origin_folds(_monthly(), FOLDS, embargo_months=12).folds[0]
        assert sorted(plain.evaluate) == sorted(held.evaluate)

    def test_zero_embargo_is_the_default(self):
        a = rolling_origin_folds(_monthly(), FOLDS).folds[0]
        b = rolling_origin_folds(_monthly(), FOLDS, embargo_months=0).folds[0]
        assert a.train_rows == b.train_rows

    def test_embargo_emptying_training_raises(self):
        with pytest.raises(ValueError, match="empty training window"):
            rolling_origin_folds(_monthly(), FOLDS, embargo_months=600)


class TestDiagnostics:
    def test_diagnostics_one_row_per_fold(self):
        result = rolling_origin_folds(_monthly(), FOLDS)
        assert len(result.diagnostics) == 3
        assert list(result.diagnostics["fold"]) == ["2013", "2014", "2015"]

    def test_diagnostics_carry_default_rates(self):
        result = rolling_origin_folds(_monthly(), FOLDS)
        for column in ("train_default_rate", "evaluate_default_rate"):
            assert result.diagnostics[column].between(0, 1).all()

    def test_hand_calculated_default_rate(self):
        """Every third row defaults, so any window whose length is a multiple
        of three has a default rate of exactly one third."""
        fold = rolling_origin_folds(_monthly(), FOLDS).folds[0]
        assert fold.train_rows == 36
        assert fold.train_default_rate == pytest.approx(1 / 3)

    def test_summary_renders(self):
        text = rolling_origin_folds(_monthly(), FOLDS).summary()
        assert "Rolling-origin" in text
        assert "2013" in text and "2015" in text


class TestValidation:
    def test_missing_column_raises(self):
        frame = pd.DataFrame({"issue_d": pd.to_datetime(["2013-01-01"])})
        with pytest.raises(KeyError, match="default_flag"):
            rolling_origin_folds(frame, FOLDS)

    def test_unparseable_date_raises(self):
        frame = pd.DataFrame(
            {"issue_d": ["2012-01-01", "not a date"], "default_flag": [0, 1]}
        )
        with pytest.raises(ValueError, match="unparseable origination date"):
            rolling_origin_folds(frame, FOLDS)

    def test_empty_frame_raises(self):
        empty = pd.DataFrame({"issue_d": pd.to_datetime([]), "default_flag": []})
        with pytest.raises(ValueError, match="cannot partition an empty frame"):
            rolling_origin_folds(empty, FOLDS)

    def test_no_folds_raises(self):
        with pytest.raises(ValueError, match="at least one fold"):
            rolling_origin_folds(_monthly(), [])

    def test_non_monotonic_boundaries_raise(self):
        bad = [("2013-12-31", "2014-12-31"), ("2012-12-31", "2013-12-31")]
        with pytest.raises(ValueError, match="chronological order"):
            rolling_origin_folds(_monthly(), bad)

    def test_inverted_fold_raises(self):
        bad = [("2014-12-31", "2013-12-31")]
        with pytest.raises(ValueError, match="must end after"):
            rolling_origin_folds(_monthly(), bad)

    def test_empty_evaluation_window_raises(self):
        """A fold that scores nothing is a specification error, not a fold to
        be silently skipped."""
        bad = [("2016-12-31", "2017-12-31")]
        with pytest.raises(ValueError, match="empty evaluation window"):
            rolling_origin_folds(_monthly(), bad)


class TestResultType:
    def test_result_is_the_declared_type(self):
        assert isinstance(rolling_origin_folds(_monthly(), FOLDS), RollingOriginResult)

    def test_folds_are_iterable_in_order(self):
        result = rolling_origin_folds(_monthly(), FOLDS)
        assert [f.name for f in result.folds] == sorted(f.name for f in result.folds)
