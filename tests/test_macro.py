"""Tests for macroeconomic retrieval and publication-lag alignment.

Section 3.5.3. These guard feature blocks B and C.

The central property is in TestNoLookAhead: for every loan, every attached
macroeconomic value must have been published on or before the loan was
originated. Everything else in this module exists to make that property hold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.macro import (
    add_available_from,
    expand_step_series,
    apply_transform,
    attach_macro,
    build_panel,
    read_fred_csv,
    series_specs,
)


def _config(tmp_path) -> dict:
    """Minimal config carrying one monthly and one quarterly series."""
    return {
        "macro": {
            "apply_publication_lag": True,
            "cache_dir": str(tmp_path),
            "za_series": [
                {
                    "name": "repo_rate",
                    "frequency": "monthly",
                    "source": "local",
                    "file": "sarb_repo_rate.csv",
                    "transform": "none",
                    "publication_lag_months": 0,
                },
                {
                    "name": "gdp_growth_yoy",
                    "frequency": "quarterly",
                    "source": "FRED",
                    "series_id": "NGDPRSAXDCZAQ",
                    "transform": "yoy_4q",
                    "publication_lag_months": 3,
                },
            ],
        }
    }


def _write_csv(path, dates, values, value_header="VALUE"):
    pd.DataFrame({"observation_date": dates, value_header: values}).to_csv(
        path, index=False
    )


class TestSeriesSpecs:
    def test_specs_built_in_config_order(self, tmp_path):
        specs = series_specs(_config(tmp_path), "za")
        assert [s.name for s in specs] == ["repo_rate", "gdp_growth_yoy"]

    def test_fred_url_uses_series_id(self, tmp_path):
        spec = series_specs(_config(tmp_path), "za")[1]
        assert spec.url.endswith("id=NGDPRSAXDCZAQ")

    def test_local_series_has_no_url(self, tmp_path):
        spec = series_specs(_config(tmp_path), "za")[0]
        with pytest.raises(ValueError, match="not a FRED series"):
            _ = spec.url

    def test_cache_filenames(self, tmp_path):
        specs = series_specs(_config(tmp_path), "za")
        assert specs[0].cache_filename == "sarb_repo_rate.csv"
        assert specs[1].cache_filename == "NGDPRSAXDCZAQ.csv"

    def test_unknown_panel_raises(self, tmp_path):
        with pytest.raises(KeyError, match="not configured"):
            series_specs(_config(tmp_path), "uk")

    def test_unknown_transform_raises(self, tmp_path):
        config = _config(tmp_path)
        config["macro"]["za_series"][0]["transform"] = "log_diff"
        with pytest.raises(ValueError, match="unknown transform"):
            series_specs(config, "za")

    def test_unknown_frequency_raises(self, tmp_path):
        config = _config(tmp_path)
        config["macro"]["za_series"][0]["frequency"] = "weekly"
        with pytest.raises(ValueError, match="unknown frequency"):
            series_specs(config, "za")

    def test_fred_series_without_id_raises(self, tmp_path):
        config = _config(tmp_path)
        del config["macro"]["za_series"][1]["series_id"]
        with pytest.raises(ValueError, match="has no series_id"):
            series_specs(config, "za")


class TestReadFredCsv:
    def test_reads_observation_date_header(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(path, ["2013-01-01", "2013-02-01"], [1.0, 2.0])
        frame = read_fred_csv(path, "x")
        assert list(frame.columns) == ["reference_date", "x"]
        assert frame["x"].tolist() == [1.0, 2.0]

    def test_reads_legacy_date_header(self, tmp_path):
        path = tmp_path / "s.csv"
        pd.DataFrame({"DATE": ["2013-01-01"], "FEDFUNDS": [0.14]}).to_csv(
            path, index=False
        )
        assert read_fred_csv(path, "x")["x"].tolist() == [0.14]

    def test_missing_marker_becomes_nan(self, tmp_path):
        """FRED writes '.' for a missing observation. A silent zero here would
        be a fabricated macroeconomic reading."""
        path = tmp_path / "s.csv"
        _write_csv(path, ["2013-01-01", "2013-02-01"], [1.0, "."])
        frame = read_fred_csv(path, "x")
        assert frame["x"].isna().sum() == 1

    def test_sorted_by_date(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(path, ["2013-03-01", "2013-01-01", "2013-02-01"], [3.0, 1.0, 2.0])
        assert read_fred_csv(path, "x")["x"].tolist() == [1.0, 2.0, 3.0]

    def test_wrong_column_count_raises(self, tmp_path):
        path = tmp_path / "s.csv"
        pd.DataFrame({"DATE": ["2013-01-01"], "A": [1], "B": [2]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="expected 2 columns"):
            read_fred_csv(path, "x")

    def test_no_date_column_raises(self, tmp_path):
        path = tmp_path / "s.csv"
        pd.DataFrame({"period": ["2013-01-01"], "value": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="no date column"):
            read_fred_csv(path, "x")


class TestTransforms:
    def test_none_is_identity(self):
        frame = pd.DataFrame(
            {"reference_date": pd.date_range("2013-01-01", periods=3, freq="MS"),
             "x": [1.0, 2.0, 3.0]}
        )
        assert apply_transform(frame, "x", "none")["x"].tolist() == [1.0, 2.0, 3.0]

    def test_yoy_12m_hand_calculated(self):
        """An index rising 100 to 106 over twelve months is 6.0 percent."""
        frame = pd.DataFrame(
            {
                "reference_date": pd.date_range("2013-01-01", periods=13, freq="MS"),
                "x": [100.0] * 12 + [106.0],
            }
        )
        out = apply_transform(frame, "x", "yoy_12m")
        assert len(out) == 1
        assert out["x"].iloc[0] == pytest.approx(6.0)

    def test_yoy_4q_hand_calculated(self):
        """A level rising 200 to 210 over four quarters is 5.0 percent."""
        frame = pd.DataFrame(
            {
                "reference_date": pd.date_range("2013-01-01", periods=5, freq="QS"),
                "x": [200.0] * 4 + [210.0],
            }
        )
        out = apply_transform(frame, "x", "yoy_4q")
        assert out["x"].iloc[0] == pytest.approx(5.0)

    def test_leading_periods_dropped(self):
        frame = pd.DataFrame(
            {"reference_date": pd.date_range("2013-01-01", periods=16, freq="MS"),
             "x": list(range(100, 116))}
        )
        assert len(apply_transform(frame, "x", "yoy_12m")) == 4

    def test_unknown_transform_raises(self):
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-01-01")], "x": [1.0]})
        with pytest.raises(ValueError, match="unknown transform"):
            apply_transform(frame, "x", "log_diff")


class TestAvailableFrom:
    def test_monthly_period_end_plus_lag(self):
        """March 2013 with a one-month lag becomes available 30 April 2013.

        FRED dates a monthly observation at the START of the month, so the
        period end is 31 March, not 1 March."""
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-03-01")]})
        out = add_available_from(frame, "monthly", 1)
        assert out["available_from"].iloc[0] == pd.Timestamp("2013-04-30")

    def test_quarterly_period_end_plus_lag(self):
        """2013 Q1, dated 1 January, ends 31 March. With a three-month lag it
        becomes available 30 June 2013 — after the quarter it describes, which
        is the whole point."""
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-01-01")]})
        out = add_available_from(frame, "quarterly", 3)
        assert out["available_from"].iloc[0] == pd.Timestamp("2013-06-30")

    def test_daily_is_its_own_period_end(self):
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-03-14")]})
        out = add_available_from(frame, "daily", 0)
        assert out["available_from"].iloc[0] == pd.Timestamp("2013-03-14")

    def test_zero_lag_is_period_end(self):
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-03-01")]})
        out = add_available_from(frame, "monthly", 0)
        assert out["available_from"].iloc[0] == pd.Timestamp("2013-03-31")

    def test_apply_lag_false_ignores_lag(self):
        """Provided so the effect of the lag can be measured, not assumed."""
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-01-01")]})
        out = add_available_from(frame, "quarterly", 3, apply_lag=False)
        assert out["available_from"].iloc[0] == pd.Timestamp("2013-03-31")

    def test_unknown_frequency_raises(self):
        frame = pd.DataFrame({"reference_date": [pd.Timestamp("2013-01-01")]})
        with pytest.raises(ValueError, match="unknown frequency"):
            add_available_from(frame, "weekly", 0)


class TestBuildPanel:
    def _populate(self, tmp_path):
        _write_csv(
            tmp_path / "sarb_repo_rate.csv",
            pd.date_range("2012-01-01", periods=36, freq="MS").strftime("%Y-%m-%d"),
            [5.0] * 36,
        )
        _write_csv(
            tmp_path / "NGDPRSAXDCZAQ.csv",
            pd.date_range("2011-01-01", periods=16, freq="QS").strftime("%Y-%m-%d"),
            [float(1000 + 10 * i) for i in range(16)],
        )

    def test_panel_carries_every_series(self, tmp_path):
        self._populate(tmp_path)
        panel = build_panel(_config(tmp_path), "za", cache_dir=tmp_path)
        assert set(panel.columns) == {"available_from", "repo_rate", "gdp_growth_yoy"}

    def test_panel_sorted_and_forward_filled(self, tmp_path):
        self._populate(tmp_path)
        panel = build_panel(_config(tmp_path), "za", cache_dir=tmp_path)
        assert panel["available_from"].is_monotonic_increasing
        tail = panel.dropna(subset=["gdp_growth_yoy"])
        assert tail["repo_rate"].notna().all()

    def test_missing_cache_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not cached"):
            build_panel(_config(tmp_path), "za", cache_dir=tmp_path)


class TestAttachMacro:
    def _panel(self):
        return pd.DataFrame(
            {
                "available_from": pd.to_datetime(
                    ["2013-03-31", "2013-06-30", "2013-09-30"]
                ),
                "gdp_growth_yoy": [1.0, 2.0, 3.0],
            }
        )

    def test_takes_latest_published_value(self):
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-07-15"])})
        out = attach_macro(loans, self._panel())
        assert out["gdp_growth_yoy"].iloc[0] == 2.0

    def test_day_before_release_takes_previous_figure(self):
        """The defining behaviour. A loan written on 29 June cannot use the
        figure released on 30 June."""
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-06-29"])})
        out = attach_macro(loans, self._panel())
        assert out["gdp_growth_yoy"].iloc[0] == 1.0

    def test_release_day_itself_is_available(self):
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-06-30"])})
        out = attach_macro(loans, self._panel())
        assert out["gdp_growth_yoy"].iloc[0] == 2.0

    def test_before_first_release_is_nan(self):
        """Correct rather than convenient: nothing had been published."""
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-01-15"])})
        out = attach_macro(loans, self._panel())
        assert pd.isna(out["gdp_growth_yoy"].iloc[0])

    def test_row_order_preserved(self):
        """merge_asof requires sorted input. If the sort leaked out, the sample
        would be silently reordered and every earlier alignment broken."""
        loans = pd.DataFrame(
            {
                "issue_d": pd.to_datetime(["2013-11-01", "2013-04-01", "2013-08-01"]),
                "loan_id": [1, 2, 3],
            }
        )
        out = attach_macro(loans, self._panel())
        assert out["loan_id"].tolist() == [1, 2, 3]
        assert out["gdp_growth_yoy"].tolist() == [3.0, 1.0, 2.0]

    def test_index_length_unchanged(self):
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-07-15"] * 5)})
        assert len(attach_macro(loans, self._panel())) == 5

    def test_helper_column_removed(self):
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-07-15"])})
        out = attach_macro(loans, self._panel())
        assert "_original_order" not in out.columns
        assert "available_from" not in out.columns

    def test_missing_origination_column_raises(self):
        with pytest.raises(KeyError, match="issue_d"):
            attach_macro(pd.DataFrame({"x": [1]}), self._panel())

    def test_panel_without_available_from_raises(self):
        loans = pd.DataFrame({"issue_d": pd.to_datetime(["2013-07-15"])})
        with pytest.raises(ValueError, match="available_from"):
            attach_macro(loans, pd.DataFrame({"x": [1.0]}))


class TestNoLookAhead:
    """The property the whole module exists to guarantee."""

    def test_no_attached_value_postdates_its_loan(self, tmp_path):
        """For every loan, the panel row it took must have been published on or
        before the loan was originated.

        Verified by attaching the availability date itself as a value column and
        checking it never exceeds the origination date."""
        panel = pd.DataFrame(
            {
                "available_from": pd.date_range("2012-01-31", periods=48, freq="ME"),
                "series": range(48),
            }
        )
        panel["published_on"] = panel["available_from"]

        loans = pd.DataFrame(
            {"issue_d": pd.date_range("2012-01-01", periods=1000, freq="D")}
        )
        out = attach_macro(loans, panel)

        observed = out.dropna(subset=["published_on"])
        assert len(observed) > 0
        assert (observed["published_on"] <= observed["issue_d"]).all()

    def test_lag_strictly_delays_availability(self, tmp_path):
        """With the lag applied, a quarterly figure is never visible earlier
        than it would be without it."""
        frame = pd.DataFrame(
            {"reference_date": pd.date_range("2012-01-01", periods=12, freq="QS")}
        )
        lagged = add_available_from(frame, "quarterly", 3, apply_lag=True)
        unlagged = add_available_from(frame, "quarterly", 3, apply_lag=False)
        assert (lagged["available_from"] > unlagged["available_from"]).all()


class TestExpandStepSeries:
    """A policy rate is published as decisions, not as a monthly series."""

    def _decisions(self):
        return pd.DataFrame(
            {
                "effective_date": ["2013-01-01", "2013-04-15", "2013-09-01"],
                "repo_rate": [5.00, 5.50, 5.25],
            }
        )

    def test_one_row_per_month(self):
        out = expand_step_series(self._decisions(), "2013-01-01", "2013-12-01")
        assert len(out) == 12
        assert list(out.columns) == ["observation_date", "repo_rate"]

    def test_hand_calculated_step_pattern(self):
        """5.00 from January, 5.50 from mid-April, 5.25 from September.

        April takes 5.50 because the rate in force at month END is 5.50 - a
        change effective mid-month is reflected in that month."""
        out = expand_step_series(self._decisions(), "2013-01-01", "2013-12-01")
        assert out["repo_rate"].tolist() == [
            5.00, 5.00, 5.00, 5.50, 5.50, 5.50, 5.50, 5.50,
            5.25, 5.25, 5.25, 5.25,
        ]

    def test_month_start_convention_matches_fred(self):
        out = expand_step_series(self._decisions(), "2013-01-01", "2013-03-01")
        assert out["observation_date"].tolist() == [
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2013-02-01"),
            pd.Timestamp("2013-03-01"),
        ]

    def test_unsorted_decisions_handled(self):
        shuffled = self._decisions().iloc[[2, 0, 1]].reset_index(drop=True)
        assert (
            expand_step_series(shuffled, "2013-01-01", "2013-12-01")["repo_rate"].tolist()
            == expand_step_series(self._decisions(), "2013-01-01", "2013-12-01")[
                "repo_rate"
            ].tolist()
        )

    def test_rate_carries_forward_beyond_last_decision(self):
        out = expand_step_series(self._decisions(), "2013-01-01", "2016-12-01")
        assert out["repo_rate"].iloc[-1] == 5.25

    def test_change_on_the_first_of_a_month(self):
        decisions = pd.DataFrame(
            {"effective_date": ["2013-01-01", "2013-06-01"], "repo_rate": [5.0, 6.0]}
        )
        out = expand_step_series(decisions, "2013-05-01", "2013-06-01")
        assert out["repo_rate"].tolist() == [5.0, 6.0]

    def test_change_on_the_last_day_of_a_month(self):
        decisions = pd.DataFrame(
            {"effective_date": ["2013-01-01", "2013-06-30"], "repo_rate": [5.0, 6.0]}
        )
        out = expand_step_series(decisions, "2013-06-01", "2013-07-01")
        assert out["repo_rate"].tolist() == [6.0, 6.0]

    def test_decision_after_first_month_raises(self):
        """Opening months with no rate in force is a transcription error, not
        a series that should be silently padded with NaN."""
        decisions = pd.DataFrame(
            {"effective_date": ["2013-06-01"], "repo_rate": [5.0]}
        )
        with pytest.raises(ValueError, match="no rate in force"):
            expand_step_series(decisions, "2013-01-01", "2013-12-01")

    def test_empty_decisions_raise(self):
        empty = pd.DataFrame({"effective_date": [], "repo_rate": []})
        with pytest.raises(ValueError, match="decision list is empty"):
            expand_step_series(empty, "2013-01-01", "2013-12-01")

    def test_unparseable_date_raises(self):
        decisions = pd.DataFrame(
            {"effective_date": ["2013-01-01", "not a date"], "repo_rate": [5.0, 6.0]}
        )
        with pytest.raises(ValueError, match="unparseable"):
            expand_step_series(decisions, "2013-01-01", "2013-12-01")

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="repo_rate"):
            expand_step_series(
                pd.DataFrame({"effective_date": ["2013-01-01"]}),
                "2013-01-01",
                "2013-12-01",
            )
