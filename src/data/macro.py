"""Macroeconomic panel retrieval and publication-lag-aware alignment.

Implements Section 3.5.3 and Table 3.6, and supplies feature blocks B and C.

The problem this module exists to solve
    A loan originated in March 2013 could not have been scored using a GDP
    figure for the first quarter of 2013, because that figure did not exist
    until roughly June 2013. Merging macroeconomic series onto loans by
    reference date rather than by release date conditions every early-quarter
    origination on information from its own future. The effect is small per
    observation and systematic across the sample, which is the worst
    combination: it inflates the apparent value of the macro blocks without
    ever looking like an error.

    This is the same failure the maturity filter guards against in the outcome
    dimension and the fold embargo guards against in the training dimension.
    Three different clocks, one discipline: never condition on what was not
    yet knowable.

How it is controlled
    Each observation carries an `available_from` date, computed as the end of
    its reference period plus the series' publication lag. Attachment is then a
    backward `merge_asof`: a loan takes the most recent observation whose
    `available_from` is on or before its origination date. A loan originated
    the day before a release takes the previous figure, which is what a lender
    scoring that application would have had.

Transformations
    FRED serves CPI as an index and GDP as a level. The study needs year-on-year
    rates, so the transformation is declared in config and computed here, where
    it is unit-tested, rather than inherited from a vendor's undocumented
    definition.

Limitation, to be stated in the methodology
    FRED serves revised data. This module controls *when* a figure became
    visible; it cannot recover *what* the figure said before later revision.
    Doing that requires vintage series (ALFRED) for every series and is out of
    scope. The residual look-ahead is therefore revision-shaped, not
    timing-shaped, and is smaller than the error it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "MacroSeriesSpec",
    "FREDGRAPH_URL",
    "series_specs",
    "read_fred_csv",
    "apply_transform",
    "add_available_from",
    "build_panel",
    "attach_macro",
    "expand_step_series",
]

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_TRANSFORMS = {"none", "yoy_12m", "yoy_4q"}
_FREQUENCIES = {"daily", "monthly", "quarterly"}


@dataclass(frozen=True)
class MacroSeriesSpec:
    """One configured macroeconomic series.

    Attributes:
        name: Column name the series takes in the design matrix.
        source: "FRED" or "local".
        frequency: daily, monthly or quarterly. Determines how the reference
            period end is computed.
        transform: none, yoy_12m or yoy_4q.
        publication_lag_months: Months between reference period end and public
            release.
        series_id: FRED identifier, for FRED sources.
        file: Filename under the cache directory, for local sources.
    """

    name: str
    source: str
    frequency: str
    transform: str
    publication_lag_months: int
    series_id: str | None = None
    file: str | None = None

    @property
    def url(self) -> str:
        """Download URL for a FRED series.

        Raises:
            ValueError: If this is not a FRED series.
        """
        if self.source != "FRED" or self.series_id is None:
            raise ValueError(f"series '{self.name}' is not a FRED series")
        return FREDGRAPH_URL.format(series_id=self.series_id)

    @property
    def cache_filename(self) -> str:
        """Filename this series is cached under."""
        if self.source == "local":
            if self.file is None:
                raise ValueError(f"local series '{self.name}' has no file")
            return self.file
        return f"{self.series_id}.csv"


def series_specs(config: dict[str, Any], panel: str) -> list[MacroSeriesSpec]:
    """Build the series specs for one panel.

    Args:
        config: The parsed config.
        panel: "za" or "us".

    Returns:
        One spec per configured series, in config order.

    Raises:
        KeyError: If the panel is not configured.
        ValueError: If a series declares an unknown transform or frequency, or
            a FRED series has no series_id, or a local series has no file.
    """
    key = f"{panel}_series"
    macro = config["macro"]
    if key not in macro:
        raise KeyError(f"macro panel '{panel}' not configured")

    specs: list[MacroSeriesSpec] = []
    for entry in macro[key]:
        transform = entry.get("transform", "none")
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"series '{entry['name']}' has unknown transform '{transform}'"
            )
        if entry["frequency"] not in _FREQUENCIES:
            raise ValueError(
                f"series '{entry['name']}' has unknown frequency '{entry['frequency']}'"
            )
        spec = MacroSeriesSpec(
            name=entry["name"],
            source=entry["source"],
            frequency=entry["frequency"],
            transform=transform,
            publication_lag_months=int(entry["publication_lag_months"]),
            series_id=entry.get("series_id"),
            file=entry.get("file"),
        )
        if spec.source == "FRED" and spec.series_id is None:
            raise ValueError(f"FRED series '{spec.name}' has no series_id")
        if spec.source == "local" and spec.file is None:
            raise ValueError(f"local series '{spec.name}' has no file")
        specs.append(spec)

    return specs


def read_fred_csv(path: str | Path, value_name: str) -> pd.DataFrame:
    """Read a downloaded FRED CSV into a two-column frame.

    FRED serves a date column - named `DATE` historically and
    `observation_date` on newer exports - plus one value column named after the
    series. Missing observations are the literal string ".", which becomes NaN
    rather than a silent zero.

    Args:
        path: Path to the CSV.
        value_name: Name to give the value column.

    Returns:
        A frame with columns `reference_date` and `value_name`, sorted by date,
        with unparseable observations dropped.

    Raises:
        ValueError: If no date column can be identified, or the file has an
            unexpected number of columns.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] != 2:
        raise ValueError(
            f"expected 2 columns in {path}, found {frame.shape[1]}: "
            f"{list(frame.columns)}"
        )

    date_col = next(
        (c for c in frame.columns if c.lower() in {"date", "observation_date"}), None
    )
    if date_col is None:
        raise ValueError(f"no date column found in {path}: {list(frame.columns)}")

    value_col = next(c for c in frame.columns if c != date_col)

    out = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(frame[date_col], errors="coerce"),
            value_name: pd.to_numeric(frame[value_col], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["reference_date"])
        .sort_values("reference_date")
        .reset_index(drop=True)
    )


def apply_transform(frame: pd.DataFrame, value_name: str, transform: str) -> pd.DataFrame:
    """Compute the declared transformation of a retrieved series.

    Percentage changes are computed over a fixed number of periods rather than
    by calendar date, which assumes the series is complete and regularly spaced.
    FRED series are, and a gap would show up as a NaN run in the output rather
    than as a silently wrong number.

    Args:
        frame: Output of `read_fred_csv`, sorted by reference_date.
        value_name: The value column.
        transform: none, yoy_12m or yoy_4q.

    Returns:
        A copy with the value column transformed. The leading periods consumed
        by the lag become NaN and are dropped.

    Raises:
        ValueError: If the transform is unknown.
    """
    if transform == "none":
        return frame.copy()

    periods = {"yoy_12m": 12, "yoy_4q": 4}.get(transform)
    if periods is None:
        raise ValueError(f"unknown transform '{transform}'")

    out = frame.copy()
    out[value_name] = out[value_name].pct_change(periods=periods) * 100.0
    return out.dropna(subset=[value_name]).reset_index(drop=True)


def add_available_from(
    frame: pd.DataFrame,
    frequency: str,
    publication_lag_months: int,
    apply_lag: bool = True,
) -> pd.DataFrame:
    """Compute the date from which each observation was publicly available.

    The reference date FRED reports is the START of the reference period: a
    quarterly observation dated 2013-01-01 covers 2013 Q1, and a monthly
    observation dated 2013-03-01 covers March. Availability is therefore

        period end + publication lag

    not reference date plus lag, which would make a quarterly figure visible
    two months before the quarter it describes had finished.

    Args:
        frame: A frame with a `reference_date` column.
        frequency: daily, monthly or quarterly.
        publication_lag_months: Months from period end to public release.
        apply_lag: When False, availability is the period end itself. Provided
            so the effect of the lag can be measured rather than assumed.

    Returns:
        A copy with an `available_from` column appended.

    Raises:
        ValueError: If the frequency is unknown.
    """
    if frequency not in _FREQUENCIES:
        raise ValueError(f"unknown frequency '{frequency}'")

    reference = pd.to_datetime(frame["reference_date"])

    if frequency == "daily":
        period_end = reference
    elif frequency == "monthly":
        period_end = reference + pd.offsets.MonthEnd(0)
    else:
        period_end = reference + pd.offsets.QuarterEnd(0)

    out = frame.copy()
    lag = publication_lag_months if apply_lag else 0
    out["available_from"] = period_end + pd.DateOffset(months=lag)
    return out


def build_panel(
    config: dict[str, Any],
    panel: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Assemble one macroeconomic panel from cached series files.

    Each series is read, transformed, given an `available_from` date, and then
    all series are combined on that date. Combining on availability rather than
    on reference date is what makes the panel a record of what was knowable
    when, rather than a record of what is known now.

    Args:
        config: The parsed config.
        panel: "za" or "us".
        cache_dir: Directory holding the cached series. Defaults to the
            configured `macro.cache_dir`, resolved by the caller.

    Returns:
        A frame with `available_from` and one column per series, sorted by
        `available_from`, forward-filled so every row carries the latest
        published value of every series as at that date.

    Raises:
        FileNotFoundError: If a cached series file is absent.
    """
    macro = config["macro"]
    apply_lag = bool(macro.get("apply_publication_lag", True))
    directory = Path(cache_dir if cache_dir is not None else macro["cache_dir"])

    combined: pd.DataFrame | None = None

    for spec in series_specs(config, panel):
        path = directory / spec.cache_filename
        if not path.is_file():
            raise FileNotFoundError(
                f"macro series '{spec.name}' not cached at {path}; "
                "run scripts.fetch_macro first"
            )

        frame = read_fred_csv(path, spec.name)
        frame = apply_transform(frame, spec.name, spec.transform)
        frame = add_available_from(
            frame, spec.frequency, spec.publication_lag_months, apply_lag=apply_lag
        )
        frame = frame[["available_from", spec.name]]

        # Two series can publish on the same date, and a series can publish
        # twice for one availability date if the lag collapses two periods.
        # The later reference period wins, which is the last row after sorting.
        frame = frame.drop_duplicates(subset="available_from", keep="last")

        combined = (
            frame
            if combined is None
            else combined.merge(frame, on="available_from", how="outer")
        )

    if combined is None:
        return pd.DataFrame(columns=["available_from"])

    combined = combined.sort_values("available_from").reset_index(drop=True)
    value_columns = [c for c in combined.columns if c != "available_from"]
    combined[value_columns] = combined[value_columns].ffill()
    return combined


def attach_macro(
    loans: pd.DataFrame,
    panel: pd.DataFrame,
    origination_col: str = "issue_d",
) -> pd.DataFrame:
    """Attach the macro panel to loans as at their origination date.

    A backward `merge_asof`: each loan takes the most recent panel row whose
    `available_from` is on or before its origination date. A loan originated
    the day before a release takes the previous figure.

    Row order AND index labels are preserved. `merge_asof` requires both sides
    sorted by the key and discards the index, so the loans are sorted
    internally, then restored to their original order and their original index
    before returning.

    Preserving the index matters as much as preserving the order. A filtered
    sample - 36-month loans only, say - has a sparse index, and silently
    renumbering it would break every fold assignment computed against the
    original labels. The failure would be invisible: the frame would have the
    right length and the right order, and every `.loc` lookup afterwards would
    quietly select the wrong rows.

    Args:
        loans: The analysis sample.
        panel: Output of `build_panel`.
        origination_col: Column holding the origination date.

    Returns:
        A copy of `loans` with one column per macro series appended. Loans
        originated before the first available observation carry NaN, which is
        correct rather than convenient: nothing was published yet.

    Raises:
        KeyError: If the origination column is absent.
        ValueError: If the panel has no `available_from` column.
    """
    if origination_col not in loans.columns:
        raise KeyError(f"required column '{origination_col}' not in frame")
    if "available_from" not in panel.columns:
        raise ValueError("panel must carry an 'available_from' column")

    left = loans.copy()
    left["_original_order"] = range(len(left))
    left["_original_index"] = left.index
    left[origination_col] = pd.to_datetime(left[origination_col])
    left = left.sort_values(origination_col)

    right = panel.copy()
    right["available_from"] = pd.to_datetime(right["available_from"])
    right = right.sort_values("available_from")

    merged = pd.merge_asof(
        left,
        right,
        left_on=origination_col,
        right_on="available_from",
        direction="backward",
    )

    merged = merged.sort_values("_original_order")
    merged.index = merged["_original_index"].to_numpy()
    merged.index.name = loans.index.name
    return merged.drop(
        columns=["_original_order", "_original_index", "available_from"]
    )


def expand_step_series(
    decisions: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    date_col: str = "effective_date",
    value_col: str = "repo_rate",
) -> pd.DataFrame:
    """Expand a policy-rate decision list into a monthly series.

    A policy rate is a step function. It is published as a short list of
    decisions - "from 30 July 2010 the rate is 6.50" - rather than as a monthly
    observation, so the monthly series a macro panel needs has to be derived.

    Deriving it is better than transcribing it. Twenty decisions can each be
    checked against the statement that announced them; a hundred and fifteen
    hand-typed monthly rows cannot, and a single mistyped row would be
    invisible.

    The value assigned to a month is the rate in force at month END. That
    matches how the rest of this module treats monthly series: the reference
    period is the month, and `add_available_from` dates availability at the
    period end. A change effective mid-month is therefore reflected in that
    month, which is what "the rate in force during that month" means to anyone
    reading the series.

    Args:
        decisions: One row per decision, with an effective date and a rate.
            Need not be sorted.
        start: First month to emit.
        end: Last month to emit.
        date_col: Column holding the effective date.
        value_col: Column holding the rate.

    Returns:
        A frame with `observation_date` (month start, matching FRED's
        convention) and `value_col`, one row per month from start to end.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If the decision list is empty, a date is unparseable, or
            the first decision takes effect after the first month requested -
            which would leave the opening months with no rate in force.
    """
    for column in (date_col, value_col):
        if column not in decisions.columns:
            raise KeyError(f"required column '{column}' not in decisions")
    if len(decisions) == 0:
        raise ValueError("decision list is empty")

    effective = pd.to_datetime(decisions[date_col], errors="coerce")
    if effective.isna().any():
        raise ValueError(
            f"{int(effective.isna().sum())} decisions have an unparseable "
            f"'{date_col}'"
        )

    steps = (
        pd.DataFrame({date_col: effective, value_col: decisions[value_col].to_numpy()})
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    if len(months) == 0:
        raise ValueError(f"no months between {start} and {end}")

    month_ends = months + pd.offsets.MonthEnd(0)
    if steps[date_col].iloc[0] > month_ends[0]:
        raise ValueError(
            f"first decision takes effect {steps[date_col].iloc[0]:%Y-%m-%d}, "
            f"after the first month requested ({month_ends[0]:%Y-%m-%d}); "
            "the opening months would have no rate in force"
        )

    out = pd.merge_asof(
        pd.DataFrame({"month_end": month_ends}),
        steps,
        left_on="month_end",
        right_on=date_col,
        direction="backward",
    )

    return pd.DataFrame(
        {"observation_date": months, value_col: out[value_col].to_numpy()}
    )
