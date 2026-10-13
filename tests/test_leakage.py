"""Tests for leakage control.

The register and the screen together are the control against the pipeline's
most dangerous failure mode: a model that discriminates almost perfectly
because it was handed the answer. These tests check both lines of defence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.leakage import (
    ColumnClass,
    classify_columns,
    enforce_register,
    screen_for_leakage,
    univariate_auc,
)
from src.utils.config import load_leakage_register

TOL = 1e-9


@pytest.fixture
def register():
    return load_leakage_register()


class TestClassification:
    def test_platform_risk_fields_excluded(self, register):
        """grade and sub_grade encode Lending Club's own risk model, which was
        itself fitted to predict default. Admitting them means benchmarking
        against the platform's scorecard, not estimating PD from borrower
        characteristics. Most published work on this dataset omits this."""
        c = classify_columns(["grade", "sub_grade"], register)
        assert c["grade"] is ColumnClass.POST_ORIGINATION
        assert c["sub_grade"] is ColumnClass.POST_ORIGINATION

    def test_int_rate_is_operational_not_a_feature(self, register):
        """int_rate is needed for ECL discounting and amortisation, so it stays
        in the dataset, but it must never reach the modelling matrix."""
        assert classify_columns(["int_rate"], register)["int_rate"] is ColumnClass.OPERATIONAL

    def test_glob_catches_unlisted_hardship_field(self, register):
        """The register lists hardship_* as a pattern. A hardship field nobody
        enumerated must still be caught, which is the point of the pattern."""
        c = classify_columns(["hardship_never_enumerated", "annual_inc"], register)
        assert c["hardship_never_enumerated"] is ColumnClass.POST_ORIGINATION
        assert c["annual_inc"] is ColumnClass.APPLICATION_TIME

    def test_payment_and_recovery_fields_excluded(self, register):
        c = classify_columns(["total_pymnt", "recoveries", "out_prncp"], register)
        assert all(v is ColumnClass.POST_ORIGINATION for v in c.values())

    def test_identifiers_classified_separately(self, register):
        c = classify_columns(["id", "member_id", "url"], register)
        assert all(v is ColumnClass.IDENTIFIER for v in c.values())

    def test_unknown_column_defaults_to_application_time(self, register):
        """The default is permissive, which is why the screen exists."""
        c = classify_columns(["some_new_bureau_field"], register)
        assert c["some_new_bureau_field"] is ColumnClass.APPLICATION_TIME

    def test_operational_beats_exclusion(self, register):
        """int_rate appears in both platform_risk_assessment and the retained
        list. Operational must win, or it would be dropped and the ECL engine
        would have no discount rate."""
        assert classify_columns(["int_rate"], register)["int_rate"] is ColumnClass.OPERATIONAL


class TestEnforceRegister:
    def test_only_application_time_admitted(self, register):
        frame = pd.DataFrame(
            {
                "annual_inc": [50000.0],
                "dti": [18.0],
                "total_pymnt": [1000.0],
                "recoveries": [0.0],
                "hardship_flag": ["N"],
                "grade": ["B"],
                "id": [1],
                "int_rate": [0.12],
                "default_flag": [0],
            }
        )
        report = enforce_register(frame, register)
        assert report.admitted == ["annual_inc", "dti"]
        assert "int_rate" in report.operational
        for leaked in ("total_pymnt", "recoveries", "hardship_flag", "grade", "id"):
            assert leaked in report.excluded

    def test_target_never_admitted(self, register):
        frame = pd.DataFrame({"annual_inc": [1.0], "default_flag": [0]})
        assert "default_flag" not in enforce_register(frame, register).admitted

    def test_nothing_is_dropped(self, register):
        """enforce_register reports; it does not mutate. A silent drop is not
        an auditable decision."""
        frame = pd.DataFrame({"annual_inc": [1.0], "recoveries": [2.0], "default_flag": [0]})
        enforce_register(frame, register)
        assert list(frame.columns) == ["annual_inc", "recoveries", "default_flag"]

    def test_absent_register_fields_reported(self, register):
        """The register names ~39 fields; a small extract has few of them.
        Harmless, but worth surfacing in case a name is misspelt."""
        frame = pd.DataFrame({"annual_inc": [1.0], "default_flag": [0]})
        assert len(enforce_register(frame, register).register_fields_absent) > 30

    def test_summary_renders(self, register):
        frame = pd.DataFrame({"annual_inc": [1.0], "recoveries": [2.0], "default_flag": [0]})
        assert "Admitted to modelling matrix" in enforce_register(frame, register).summary()


class TestUnivariateAUC:
    def test_perfect_separation_hand_calculated(self):
        """feature 1,2,3,4 with target 0,0,1,1. Positives hold ranks 3 and 4,
        rank sum 7. AUC = (7 - 2*3/2) / (2*2) = 4/4 = 1.0
        """
        assert univariate_auc(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([0, 0, 1, 1])) == (
            pytest.approx(1.0, abs=TOL)
        )

    def test_direction_agnostic(self):
        """Perfect reverse separation is just as much a leakage signal."""
        assert univariate_auc(pd.Series([4.0, 3.0, 2.0, 1.0]), pd.Series([0, 0, 1, 1])) == (
            pytest.approx(1.0, abs=TOL)
        )

    def test_no_signal_hand_calculated(self):
        """Two tied groups, each split evenly between the classes. Every
        positive-negative comparison is a tie, worth 0.5 each.

            ranks in group 1 (value 1.0): all tie at 2.5
            ranks in group 2 (value 2.0): all tie at 6.5
            rank sum over positives = 2.5 + 2.5 + 6.5 + 6.5 = 18
            AUC = (18 - 4*5/2) / (4*4) = 8/16 = 0.5
        """
        feature = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        assert univariate_auc(feature, target) == pytest.approx(0.5, abs=TOL)

    def test_handles_missing_values(self):
        feature = pd.Series([1.0, np.nan, 3.0, 4.0])
        target = pd.Series([0, 1, 1, 1])
        assert not np.isnan(univariate_auc(feature, target))

    def test_degenerate_inputs_return_nan(self):
        assert np.isnan(univariate_auc(pd.Series([1.0, 1.0]), pd.Series([0, 1])))  # constant
        assert np.isnan(univariate_auc(pd.Series([1.0, 2.0]), pd.Series([1, 1])))  # one class
        assert np.isnan(univariate_auc(pd.Series(["a", "b"]), pd.Series([0, 1])))  # non-numeric

    def test_matches_sklearn(self):
        """Cross-check the rank identity against an independent implementation."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(7)
        target = rng.integers(0, 2, 500)
        feature = target * 0.8 + rng.normal(0, 1, 500)
        expected = roc_auc_score(target, feature)
        assert univariate_auc(pd.Series(feature), pd.Series(target)) == pytest.approx(
            expected, abs=1e-9
        )


class TestScreen:
    def test_catches_leak_the_register_missed(self):
        """The screen's whole purpose: a leaked field nobody anticipated."""
        rng = np.random.default_rng(11)
        n = 500
        target = rng.integers(0, 2, n)
        frame = pd.DataFrame(
            {
                "default_flag": target,
                "undocumented_score": target + rng.normal(0, 0.01, n),
                "honest_feature": rng.normal(0, 1, n),
            }
        )
        result = screen_for_leakage(frame, ["undocumented_score", "honest_feature"])
        assert result.flagged == ["undocumented_score"]
        assert result.scores["undocumented_score"] > 0.99
        assert result.scores["honest_feature"] < 0.65

    def test_clean_features_not_flagged(self):
        rng = np.random.default_rng(3)
        n = 400
        frame = pd.DataFrame(
            {
                "default_flag": rng.integers(0, 2, n),
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
            }
        )
        assert screen_for_leakage(frame, ["a", "b"]).flagged == []

    def test_threshold_is_configurable(self):
        rng = np.random.default_rng(5)
        n = 600
        target = rng.integers(0, 2, n)
        frame = pd.DataFrame({"default_flag": target, "weak": target * 0.5 + rng.normal(0, 1, n)})
        assert screen_for_leakage(frame, ["weak"], threshold=0.99).flagged == []
        assert screen_for_leakage(frame, ["weak"], threshold=0.55).flagged == ["weak"]

    def test_non_numeric_dropped_from_scores(self):
        frame = pd.DataFrame({"default_flag": [0, 1, 0, 1], "cat": ["a", "b", "a", "b"]})
        assert "cat" not in screen_for_leakage(frame, ["cat"]).scores.index

    def test_missing_target_raises(self):
        with pytest.raises(KeyError, match="default_flag"):
            screen_for_leakage(pd.DataFrame({"a": [1.0]}), ["a"])

    def test_summary_renders(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {"default_flag": rng.integers(0, 2, 100), "a": rng.normal(0, 1, 100)}
        )
        assert "Univariate AUC leakage screen" in screen_for_leakage(frame, ["a"]).summary()
