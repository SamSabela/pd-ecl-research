"""Tests for feature preparation (Section 3.5.2).

The central guarantee: every partition produces an IDENTICAL design matrix.
A model trained on one column set cannot score another, and a silent
misalignment is worse than a crash because it produces plausible nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocessing import (
    PreprocessingConfig,
    apply_preprocessor,
    engineer_features,
    fit_preprocessor,
)


def _frame(n: int, seed: int, complete: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, n)
    if not complete:
        values[rng.random(n) < 0.2] = np.nan
    return pd.DataFrame(
        {
            "always_present": rng.normal(0, 1, n),
            "sometimes_missing": values,
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "default_flag": rng.integers(0, 2, n),
        }
    )


class TestColumnStability:
    def test_partitions_yield_identical_columns(self):
        """THE test. Train has missing values in `sometimes_missing`;
        validation does not. Both design matrices must still have exactly the
        same columns in the same order, because the indicator set is fixed at
        fit time from train alone.
        """
        train = _frame(500, seed=1, complete=False)
        validation = _frame(200, seed=2, complete=True)

        state, _ = fit_preprocessor(
            train, ["always_present", "sometimes_missing", "home_ownership"],
            PreprocessingConfig(),
        )
        x_train = apply_preprocessor(train, state)
        x_val = apply_preprocessor(validation, state)

        assert list(x_train.columns) == list(x_val.columns)
        assert x_train.shape[1] == x_val.shape[1]

    def test_indicator_present_when_train_had_missing(self):
        train = _frame(500, seed=1, complete=False)
        state, _ = fit_preprocessor(
            train, ["always_present", "sometimes_missing"], PreprocessingConfig()
        )
        columns = apply_preprocessor(train, state).columns
        assert "sometimes_missing__missing" in columns
        assert "always_present__missing" not in columns

    def test_reported_width_matches_actual(self):
        """The report's column count must equal what apply actually produces."""
        train = _frame(500, seed=1)
        state, report = fit_preprocessor(
            train, ["always_present", "sometimes_missing", "home_ownership"],
            PreprocessingConfig(),
        )
        assert apply_preprocessor(train, state).shape[1] == report.n_output

    def test_unseen_category_falls_through_as_zeros(self):
        """A level absent from train must not create a new column."""
        train = _frame(500, seed=1)
        state, _ = fit_preprocessor(train, ["home_ownership"], PreprocessingConfig())
        other = pd.DataFrame({"home_ownership": ["ANY", "OTHER"]})
        design = apply_preprocessor(other, state)
        expected = [f"home_ownership={lvl}" for lvl in state["levels"]["home_ownership"]]
        assert list(design.columns) == expected
        assert design.to_numpy().sum() == 0

    def test_high_missingness_column_dropped(self):
        rng = np.random.default_rng(3)
        train = pd.DataFrame(
            {
                "mostly_null": np.where(rng.random(500) < 0.9, np.nan, 1.0),
                "ok": rng.normal(0, 1, 500),
            }
        )
        _, report = fit_preprocessor(train, ["mostly_null", "ok"], PreprocessingConfig())
        assert "mostly_null" in report.dropped_missing
        assert report.numeric == ["ok"]

    def test_winsorisation_bounds_come_from_train(self):
        """An extreme value in validation must be clipped to the TRAIN bound,
        not to its own distribution's."""
        train = pd.DataFrame({"x": np.arange(1000, dtype=float)})
        state, _ = fit_preprocessor(train, ["x"], PreprocessingConfig())
        design = apply_preprocessor(pd.DataFrame({"x": [1e9]}), state)
        assert design["x"].iloc[0] <= state["bounds"]["x"][1]


class TestFeatureEngineering:
    def test_credit_history_months_hand_calculated(self):
        """Origination Jan-2015, earliest credit line Jan-2005 = 120 months."""
        frame = pd.DataFrame(
            {
                "earliest_cr_line": ["Jan-2005"],
                "issue_d": pd.to_datetime(["2015-01-01"]),
            }
        )
        out, engineered = engineer_features(frame)
        assert out["credit_history_months"].iloc[0] == 120.0
        assert "credit_history_months" in engineered
        assert "earliest_cr_line" not in out.columns

    def test_emp_length_is_ordinal_not_one_hot(self):
        """'10+ years' must be greater than '2 years'. One-hot would lose that."""
        frame = pd.DataFrame({"emp_length": ["< 1 year", "2 years", "10+ years"]})
        out, _ = engineer_features(frame)
        values = out["emp_length_years"].tolist()
        assert values == [0.5, 2.0, 10.0]
        assert values[2] > values[1] > values[0]
