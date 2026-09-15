"""Out-of-time partitioning.

Implements Section 3.4.3 and Table 3.3.

The June proposal specified stratified random sampling AND time-aware
partitioning simultaneously, which is not possible: stratifying across
partitions requires drawing from the whole timeline, while a time-aware split
requires that later observations be reserved. The refined design resolves this
in favour of a purely out-of-time split, since that is what delivers the
look-ahead control the proposal claims, and what a model validation function
would expect to see.

Stratification on the outcome is applied within the cross-validation folds
inside the training window, so each fold carries a comparable default rate. It
is not applied across the three partitions, which are separated by time alone.

The default rate in each partition is REPORTED rather than controlled. Base-rate
drift across the origination timeline is itself a finding relevant to the
calibration analysis: a model well calibrated in-sample and poorly calibrated
out-of-time may be responding to a genuine shift in the population rather than
to a modelling defect, and telling those apart requires knowing the drift is
there.

Revision, 12 September 2026
    `out_of_time_split` produced the ICE Task 2 figures and is retained for that
    reason, but it is no longer the study's partitioning scheme. Because 42% of
    the sample was originated in 2015, splitting by row count gave the training
    partition 96 months and the evaluation partitions four and five. Over four
    months every macroeconomic series is flat, so the planned comparison between
    feature blocks A, B and C could not have detected a macroeconomic effect
    whether or not one existed.

    `rolling_origin_folds` replaces it with three expanding-window refit steps,
    each evaluated on the following year. See the concept reference for the
    reasoning, including the two-clock problem the embargo addresses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Partition",
    "PartitionResult",
    "out_of_time_split",
    "Fold",
    "RollingOriginResult",
    "rolling_origin_folds",
]


class Partition:
    """Partition labels."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class PartitionResult:
    """Assignment plus the diagnostics Gate G1 needs.

    Attributes:
        assignment: Partition label per row, aligned to the input index.
        boundaries: The two origination dates at which the splits fall.
        diagnostics: Per-partition rows, share, date range and default rate.
    """

    assignment: pd.Series
    boundaries: tuple[pd.Timestamp, pd.Timestamp]
    diagnostics: pd.DataFrame

    def summary(self) -> str:
        """Human-readable summary for the Gate G1 sign-off."""
        lines = [
            "Out-of-time partition",
            "=" * 78,
            f"  Train/validation boundary : {self.boundaries[0]:%Y-%m-%d}",
            f"  Validation/test boundary  : {self.boundaries[1]:%Y-%m-%d}",
            "",
            f"  {'partition':<12}{'rows':>11}{'share':>9}{'from':>10}{'to':>10}"
            f"{'default rate':>15}",
            "  " + "-" * 68,
        ]
        for row in self.diagnostics.itertuples(index=False):
            lines.append(
                f"  {row.partition:<12}{row.rows:>11,}{row.share:>9.1%}"
                f"{row.date_min:>10}{row.date_max:>10}{row.default_rate:>15.4%}"
            )
        rates = self.diagnostics["default_rate"]
        lines.append("  " + "-" * 68)
        lines.append(
            f"  Base-rate drift, train to test: {float(rates.iloc[-1] - rates.iloc[0]):+.4%}"
        )
        lines.append(
            "  Reported, not controlled. Drift bears on how calibration decay in"
        )
        lines.append("  the out-of-time test partition should be read.")
        return "\n".join(lines)


def out_of_time_split(
    frame: pd.DataFrame,
    train_share: float = 0.70,
    validation_share: float = 0.15,
    test_share: float = 0.15,
    origination_col: str = "issue_d",
    target_col: str = "default_flag",
) -> PartitionResult:
    """Split the sample by origination time, earliest first.

    Partitions are contiguous blocks of the origination timeline: the earliest
    `train_share` of observations become the training set, the next
    `validation_share` the validation set, and the latest `test_share` the
    out-of-time test set.

    Ties are broken by row position rather than by date, so a single origination
    month spanning a boundary is divided deterministically instead of being
    assigned wholesale to one side. The reported boundary dates are the actual
    origination dates at which each split falls.

    Args:
        frame: The analysis sample.
        train_share: Share of observations in the training partition.
        validation_share: Share in the validation partition.
        test_share: Share in the out-of-time test partition.
        origination_col: Column holding the origination date.
        target_col: Column holding the binary default indicator.

    Returns:
        A PartitionResult carrying the assignment and its diagnostics.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If shares do not sum to 1, the frame is empty, or any
            origination date is unparseable.
    """
    for column in (origination_col, target_col):
        if column not in frame.columns:
            raise KeyError(f"required column '{column}' not in frame")

    total_share = train_share + validation_share + test_share
    if not np.isclose(total_share, 1.0):
        raise ValueError(f"shares must sum to 1.0, got {total_share}")
    if len(frame) == 0:
        raise ValueError("cannot partition an empty frame")

    origination = pd.to_datetime(frame[origination_col], errors="coerce")
    if origination.isna().any():
        raise ValueError(
            f"{int(origination.isna().sum())} rows have an unparseable origination "
            "date; resolve before partitioning"
        )

    # perm[k] is the row index of the k-th earliest origination. Inverting it
    # once gives each row its position on the timeline.
    n = len(frame)
    perm = np.lexsort((np.arange(n), origination.to_numpy(dtype="datetime64[ns]")))
    position = np.empty(n, dtype=np.int64)
    position[perm] = np.arange(n)

    train_cut = int(round(n * train_share))
    validation_cut = int(round(n * (train_share + validation_share)))

    labels = np.full(n, Partition.TEST, dtype=object)
    labels[position < train_cut] = Partition.TRAIN
    labels[(position >= train_cut) & (position < validation_cut)] = Partition.VALIDATION
    assignment = pd.Series(labels, index=frame.index, name="partition")

    sorted_dates = origination.sort_values().to_numpy()
    boundaries = (
        pd.Timestamp(sorted_dates[min(train_cut, n - 1)]),
        pd.Timestamp(sorted_dates[min(validation_cut, n - 1)]),
    )

    rows = []
    for name in (Partition.TRAIN, Partition.VALIDATION, Partition.TEST):
        mask = assignment == name
        dates = origination.loc[mask]
        rows.append(
            {
                "partition": name,
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "date_min": f"{dates.min():%Y-%m}" if mask.any() else "-",
                "date_max": f"{dates.max():%Y-%m}" if mask.any() else "-",
                "default_rate": (
                    float(frame.loc[mask, target_col].mean()) if mask.any() else 0.0
                ),
            }
        )

    return PartitionResult(
        assignment=assignment, boundaries=boundaries, diagnostics=pd.DataFrame(rows)
    )


# -----------------------------------------------------------------------------
# Rolling-origin backtesting - Section 3.4.3, revised 12 September 2026
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One refit-and-evaluate step of the rolling-origin backtest.

    Attributes:
        name: Label for the fold, taken from the evaluation year.
        train: Index labels of the training cohort.
        evaluate: Index labels of the evaluation cohort.
        train_end: Latest origination date admitted to training, inclusive.
            Already pulled back by the embargo, if one is in force.
        evaluate_start: Earliest origination date in the evaluation cohort.
        evaluate_end: Latest origination date in the evaluation cohort.
        train_rows: Size of the training cohort.
        evaluate_rows: Size of the evaluation cohort.
        train_default_rate: Observed default rate in training.
        evaluate_default_rate: Observed default rate in evaluation.
    """

    name: str
    train: pd.Index
    evaluate: pd.Index
    train_end: pd.Timestamp
    evaluate_start: pd.Timestamp
    evaluate_end: pd.Timestamp
    train_rows: int
    evaluate_rows: int
    train_default_rate: float
    evaluate_default_rate: float


@dataclass
class RollingOriginResult:
    """All folds, plus the diagnostics table the methodology chapter needs."""

    folds: list[Fold]
    embargo_months: int
    diagnostics: pd.DataFrame

    def summary(self) -> str:
        """Human-readable summary for the sign-off."""
        lines = [
            "Rolling-origin backtest",
            "=" * 78,
            f"  Folds          : {len(self.folds)}",
            f"  Embargo        : {self.embargo_months} months",
            "",
            f"  {'fold':<8}{'train':>10}{'to':>10}{'dr':>9}"
            f"{'eval':>10}{'window':>18}{'dr':>9}",
            "  " + "-" * 68,
        ]
        for f in self.folds:
            window = f"{f.evaluate_start:%Y-%m} to {f.evaluate_end:%Y-%m}"
            lines.append(
                f"  {f.name:<8}{f.train_rows:>10,}{f'{f.train_end:%Y-%m}':>10}"
                f"{f.train_default_rate:>9.2%}"
                f"{f.evaluate_rows:>10,}{window:>18}"
                f"{f.evaluate_default_rate:>9.2%}"
            )
        lines.append("  " + "-" * 68)
        lines.append("  Training windows expand; evaluation windows are disjoint and")
        lines.append("  contiguous. Every cohort after the first cut-off is scored once.")
        return "\n".join(lines)


def rolling_origin_folds(
    frame: pd.DataFrame,
    fold_boundaries: Sequence[tuple[str, str]],
    origination_col: str = "issue_d",
    target_col: str = "default_flag",
    embargo_months: int = 0,
) -> RollingOriginResult:
    """Build expanding-window folds for out-of-time backtesting.

    Each fold trains on every loan originated on or before `train_end` and is
    evaluated on the loans originated after `train_end` and on or before
    `evaluate_end`. The training window expands from fold to fold: the cohort
    evaluated in one fold becomes training data in the next, which is how a
    model validation function would actually rebuild the model each year.

    An expanding window is used rather than a sliding one because the early
    cohorts are the only crisis-era observations in the sample. Discarding them
    to keep the window a fixed width would remove the only stress period the
    macroeconomic feature blocks have to work with.

    The embargo
        A 36-month loan originated in December 2014 does not resolve until
        December 2017. Training on its outcome in order to score a January 2015
        origination therefore uses information that did not exist at the time.
        Origination-time splitting alone does not catch this, and almost no
        published work on this dataset does catch it.

        `embargo_months` holds back any training loan whose contractual term had
        not fully elapsed by the start of the evaluation window, by pulling the
        training cut-off back by that many months. On a book restricted to
        36-month loans this is exact: set it to 36 and every training outcome
        was observable before the evaluation cohort was written.

        The default is 0, which is the conventional origination-time split and
        what the main results use. Setting 36 is the strict deployability check
        and belongs in the robustness section, because on this sample it costs
        roughly 91% of the training data.

    Args:
        frame: The analysis sample.
        fold_boundaries: Ordered (train_end, evaluate_end) date pairs, each
            parseable by pandas. Both bounds are inclusive.
        origination_col: Column holding the origination date.
        target_col: Column holding the binary default indicator.
        embargo_months: Months to pull the training cut-off back by.

    Returns:
        A RollingOriginResult carrying the folds and their diagnostics.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If the frame is empty, no folds are given, boundaries are
            not in chronological order, a fold ends before it starts, an
            origination date is unparseable, or any fold has an empty training
            or evaluation window.
    """
    for column in (origination_col, target_col):
        if column not in frame.columns:
            raise KeyError(f"required column '{column}' not in frame")

    if len(frame) == 0:
        raise ValueError("cannot partition an empty frame")

    if not fold_boundaries:
        raise ValueError("at least one fold is required")

    origination = pd.to_datetime(frame[origination_col], errors="coerce")
    if origination.isna().any():
        raise ValueError(
            f"{int(origination.isna().sum())} rows have an unparseable origination "
            "date; resolve before partitioning"
        )
        
    boundaries = [
        (pd.Timestamp(train_end), pd.Timestamp(evaluate_end))
        for train_end, evaluate_end in fold_boundaries
    ]
    for train_end, evaluate_end in boundaries:
        if evaluate_end <= train_end:
            raise ValueError(
                f"a fold must end after its training cut-off; got cut-off "
                f"{train_end:%Y-%m-%d} and end {evaluate_end:%Y-%m-%d}"
            )

    for (earlier_end, _), (later_end, _) in zip(boundaries, boundaries[1:]):
        if later_end <= earlier_end:
            raise ValueError(
                f"fold boundaries must be given in chronological order; training "
                f"cut-off {later_end:%Y-%m-%d} does not follow {earlier_end:%Y-%m-%d}"
            )
    
    folds: list[Fold] = []

    for train_end, evaluate_end in boundaries:
        cutoff = train_end - pd.DateOffset(months=embargo_months)

        train_mask = origination <= cutoff
        evaluate_mask = (origination > train_end) & (origination <= evaluate_end)

        if not train_mask.any():
            raise ValueError(
                f"fold ending {evaluate_end:%Y} has an empty training window "
                f"at cut-off {cutoff:%Y-%m-%d}"
            )
        if not evaluate_mask.any():
            raise ValueError(
                f"fold ending {evaluate_end:%Y} has an empty evaluation window "
                f"covering {train_end:%Y-%m-%d} to {evaluate_end:%Y-%m-%d}"
            )