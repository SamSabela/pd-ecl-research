"""Tests for out-of-time partitioning (Section 3.4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.partition import Partition, out_of_time_split


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "issue_d": pd.date_range("2012-01-01", periods=n, freq="ME"),
            "default_flag": rng.integers(0, 2, n),
        }
    )


class TestOutOfTimeSplit:
    def test_shares_hand_calculated(self):
        """100 rows at 70/15/15 gives exactly 70, 15 and 15."""
        counts = out_of_time_split(_frame(100)).assignment.value_counts()
        assert counts[Partition.TRAIN] == 70
        assert counts[Partition.VALIDATION] == 15
        assert counts[Partition.TEST] == 15

    def test_partitions_strictly_ordered_in_time(self):
        """The defining property: every training origination precedes every
        validation origination, which precedes every test origination. Without
        this there is no look-ahead control."""
        frame = _frame(100)
        result = out_of_time_split(frame)
        dates = pd.to_datetime(frame["issue_d"])
        assert dates[result.assignment == Partition.TRAIN].max() < (
            dates[result.assignment == Partition.VALIDATION].min()
        )
        assert dates[result.assignment == Partition.VALIDATION].max() < (
            dates[result.assignment == Partition.TEST].min()
        )

    def test_shuffled_input_still_splits_by_time(self):
        """Assignment must depend on origination date, not row order."""
        frame = _frame(100).sample(frac=1.0, random_state=11)
        result = out_of_time_split(frame)
        dates = pd.to_datetime(frame["issue_d"])
        assert dates[result.assignment == Partition.TRAIN].max() < (
            dates[result.assignment == Partition.TEST].min()
        )

    def test_index_preserved(self):
        frame = _frame(50)
        frame.index = range(1000, 1050)
        assert out_of_time_split(frame).assignment.index.tolist() == frame.index.tolist()

    def test_diagnostics_report_default_rate(self):
        result = out_of_time_split(_frame(100))
        assert list(result.diagnostics["partition"]) == [
            Partition.TRAIN, Partition.VALIDATION, Partition.TEST
        ]
        assert result.diagnostics["rows"].sum() == 100
        assert result.diagnostics["default_rate"].between(0, 1).all()

    def test_summary_renders(self):
        assert "Base-rate drift" in out_of_time_split(_frame(100)).summary()

    def test_unparseable_date_raises(self):
        frame = pd.DataFrame(
            {"issue_d": ["2012-01-01", "not a date"], "default_flag": [0, 1]}
        )
        with pytest.raises(ValueError, match="unparseable origination date"):
            out_of_time_split(frame)

    def test_bad_shares_raise(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            out_of_time_split(_frame(10), 0.5, 0.3, 0.3)

    def test_empty_frame_raises(self):
        empty = pd.DataFrame({"issue_d": pd.to_datetime([]), "default_flag": []})
        with pytest.raises(ValueError, match="cannot partition an empty frame"):
            out_of_time_split(empty)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="default_flag"):
            out_of_time_split(pd.DataFrame({"issue_d": pd.to_datetime(["2012-01-01"])}))
