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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Partition", "PartitionResult", "out_of_time_split"]


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
