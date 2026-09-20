"""Tests for feature block assembly (Section 3.5.3, Table 3.5).

Block A is the control and must differ from B and C in exactly one respect:
the presence of a macroeconomic panel. These tests pin that down, because an
accidental difference between the blocks would make the A/B/C comparison
measure something other than macroeconomic conditioning.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.blocks import BLOCK_PANELS, assemble_block, block_description


def _config(tmp_path) -> dict:
    return {
        "macro": {
            "apply_publication_lag": True,
            "cache_dir": str(tmp_path),
            "us_series": [
                {
                    "name": "fed_funds_rate",
                    "frequency": "monthly",
                    "source": "FRED",
                    "series_id": "FEDFUNDS",
                    "transform": "none",
                    "publication_lag_months": 0,
                }
            ],
            "za_series": [
                {
                    "name": "za_interbank_rate",
                    "frequency": "monthly",
                    "source": "FRED",
                    "series_id": "IRSTCI01ZAM156N",
                    "transform": "none",
                    "publication_lag_months": 0,
                }
            ],
        },
        "feature_blocks": {
            "A": {"description": "Borrower and credit-bureau features only (control)"},
            "B": {"description": "A + six contemporaneous US macro series (benchmark)"},
            "C": {"description": "A + six South African series (contextualisation)"},
        },
    }


def _cache(tmp_path):
    for series_id in ("FEDFUNDS", "IRSTCI01ZAM156N"):
        pd.DataFrame(
            {
                "observation_date": pd.date_range(
                    "2010-01-01", periods=72, freq="MS"
                ).strftime("%Y-%m-%d"),
                series_id: [1.0 + i / 100 for i in range(72)],
            }
        ).to_csv(tmp_path / f"{series_id}.csv", index=False)


def _loans(index=None):
    return pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2013-05-15", "2014-02-20", "2015-11-03"]),
            "loan_amnt": [10000.0, 15000.0, 8000.0],
            "default_flag": [0, 1, 0],
        },
        index=index,
    )


class TestBlockPanels:
    def test_a_is_the_control(self):
        assert BLOCK_PANELS["A"] is None

    def test_b_and_c_carry_panels(self):
        assert BLOCK_PANELS["B"] == "us"
        assert BLOCK_PANELS["C"] == "za"

    def test_descriptions_come_from_config(self, tmp_path):
        assert "control" in block_description(_config(tmp_path), "A")

    def test_unknown_block_description_raises(self, tmp_path):
        with pytest.raises(KeyError, match="not configured"):
            block_description(_config(tmp_path), "D")


class TestAssembleBlockA:
    def test_returns_loans_unchanged(self, tmp_path):
        loans = _loans()
        out, macro = assemble_block(loans, _config(tmp_path), "A", cache_dir=tmp_path)
        assert macro == []
        pd.testing.assert_frame_equal(out, loans)

    def test_returns_a_copy(self, tmp_path):
        """Mutating the assembled frame must not reach back into the sample."""
        loans = _loans()
        out, _ = assemble_block(loans, _config(tmp_path), "A", cache_dir=tmp_path)
        out.loc[out.index[0], "loan_amnt"] = 999.0
        assert loans["loan_amnt"].iloc[0] == 10000.0

    def test_needs_no_cache(self, tmp_path):
        """Block A must run with no macro data present at all."""
        assemble_block(_loans(), _config(tmp_path), "A", cache_dir=tmp_path)


class TestAssembleBlocksBandC:
    def test_us_panel_attached(self, tmp_path):
        _cache(tmp_path)
        out, macro = assemble_block(_loans(), _config(tmp_path), "B", cache_dir=tmp_path)
        assert macro == ["fed_funds_rate"]
        assert "fed_funds_rate" in out.columns

    def test_za_panel_attached(self, tmp_path):
        _cache(tmp_path)
        out, macro = assemble_block(_loans(), _config(tmp_path), "C", cache_dir=tmp_path)
        assert macro == ["za_interbank_rate"]
        assert "za_interbank_rate" in out.columns

    def test_blocks_differ_only_by_macro_columns(self, tmp_path):
        """The defining property of the design. If B carried any change to the
        borrower columns, the A-to-B difference would not isolate the effect of
        macroeconomic conditioning."""
        _cache(tmp_path)
        config = _config(tmp_path)
        a, _ = assemble_block(_loans(), config, "A", cache_dir=tmp_path)
        b, macro_b = assemble_block(_loans(), config, "B", cache_dir=tmp_path)
        pd.testing.assert_frame_equal(a, b.drop(columns=macro_b))

    def test_row_count_unchanged(self, tmp_path):
        _cache(tmp_path)
        out, _ = assemble_block(_loans(), _config(tmp_path), "B", cache_dir=tmp_path)
        assert len(out) == 3

    def test_sparse_index_preserved(self, tmp_path):
        """Fold membership is held as index labels, so renumbering here would
        silently misassign every loan."""
        _cache(tmp_path)
        loans = _loans(index=[41, 7, 308])
        out, _ = assemble_block(loans, _config(tmp_path), "C", cache_dir=tmp_path)
        assert out.index.tolist() == [41, 7, 308]

    def test_missing_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not cached"):
            assemble_block(_loans(), _config(tmp_path), "B", cache_dir=tmp_path)

    def test_column_collision_raises(self, tmp_path):
        """A macro series named after an existing loan column would overwrite
        it or produce suffixed duplicates. Neither is acceptable silently."""
        _cache(tmp_path)
        loans = _loans()
        loans["fed_funds_rate"] = 0.0
        with pytest.raises(ValueError, match="collide"):
            assemble_block(loans, _config(tmp_path), "B", cache_dir=tmp_path)


class TestValidation:
    def test_unknown_block_raises(self, tmp_path):
        with pytest.raises(KeyError, match="unknown feature block"):
            assemble_block(_loans(), _config(tmp_path), "Z", cache_dir=tmp_path)
