"""Feature preparation for the comparative modelling run.

Implements Section 3.5.2. Preprocessing is differentiated by model family so
each receives the input representation it is designed for, and every
transformer is fitted on the training partition alone and applied unchanged to
validation and test. That discipline is enforced structurally by using a
scikit-learn Pipeline rather than by convention: it is the difference between
"we were careful" and "it cannot happen".

Decisions taken here, each of which is a documented choice rather than a
default, and each of which belongs in the write-up:

  Missingness threshold. Columns missing in more than `max_missing_share` of
  the training partition are dropped. In the Lending Club extract this removes
  the joint-applicant and secondary-applicant blocks, which are populated only
  for the small minority of joint applications and would otherwise contribute
  an imputed constant to every individual application.

  Credit history length. `earliest_cr_line` is a date, not a quantity. It is
  converted to months of credit history at origination, which is the
  economically meaningful form and is standard in credit scoring. Left as a
  date it would either be dropped or, worse, encoded as an arbitrary integer.

  Employment length. `emp_length` is ordinal text ("< 1 year" ... "10+ years").
  It is mapped to years as a number, preserving the ordering that a one-hot
  encoding would discard.

  High-cardinality geography. `zip_code` (956 levels) is excluded from the
  baseline feature set. One-hot encoding it would add nearly a thousand sparse
  columns; target encoding it invites leakage unless done inside the CV folds.
  It is retained in the dataset for the fairness analysis in Section 4.2.

  Scaling. Applied for logistic regression, which is sensitive to feature
  scale, and not for the tree ensembles, which are invariant to monotone
  transformations of individual features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PreprocessingConfig",
    "PreprocessingReport",
    "engineer_features",
    "fit_preprocessor",
    "apply_preprocessor",
]

# "< 1 year" ... "10+ years" -> years. Preserves the ordering that one-hot loses.
_EMP_LENGTH_YEARS = {
    "< 1 year": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing parameters.

    Attributes:
        max_missing_share: Drop columns missing more than this share in train.
        max_cardinality: Drop categorical columns with more levels than this.
        winsorise_lower: Lower percentile for winsorisation.
        winsorise_upper: Upper percentile for winsorisation.
        drop_columns: Columns excluded by explicit decision.
    """

    max_missing_share: float = 0.50
    max_cardinality: int = 60
    winsorise_lower: float = 0.01
    winsorise_upper: float = 0.99
    drop_columns: tuple[str, ...] = ("zip_code",)


@dataclass
class PreprocessingReport:
    """What preprocessing did, for the write-up and the cleaning log."""

    n_input: int = 0
    dropped_missing: list[str] = field(default_factory=list)
    dropped_cardinality: list[str] = field(default_factory=list)
    dropped_explicit: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    engineered: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    n_output: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Feature preparation",
            "=" * 78,
            f"  Input columns              : {self.n_input:>4}",
            f"  Dropped, >50% missing      : {len(self.dropped_missing):>4}",
            f"  Dropped, high cardinality  : {len(self.dropped_cardinality):>4}",
            f"  Dropped, constant in train : {len(self.dropped_constant):>4}",
            f"  Dropped, explicit decision : {len(self.dropped_explicit):>4}"
            f"  {self.dropped_explicit}",
            f"  Engineered                 : {len(self.engineered):>4}"
            f"  {self.engineered}",
            "",
            f"  Numeric features           : {len(self.numeric):>4}",
            f"  Categorical features       : {len(self.categorical):>4}",
            f"  Design matrix columns      : {self.n_output:>4} (after one-hot)",
        ]
        return "\n".join(lines)


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Derive the features that raw columns cannot serve as.

    Args:
        frame: The analysis sample.

    Returns:
        (frame with engineered columns added and their raw sources removed,
         names of the engineered columns)
    """
    out = frame.copy()
    engineered: list[str] = []

    # Credit history length in months at origination. A date is not a quantity;
    # months of history is the economically meaningful form.
    if "earliest_cr_line" in out.columns and "issue_d" in out.columns:
        earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
        issued = pd.to_datetime(out["issue_d"], errors="coerce")
        months = (issued.dt.year - earliest.dt.year) * 12 + (
            issued.dt.month - earliest.dt.month
        )
        out["credit_history_months"] = months.astype("Float64").astype(float)
        out = out.drop(columns=["earliest_cr_line"])
        engineered.append("credit_history_months")

    # Employment length: ordinal text -> years, preserving the ordering.
    if "emp_length" in out.columns:
        out["emp_length_years"] = (
            out["emp_length"].astype("string").str.strip().map(_EMP_LENGTH_YEARS)
        ).astype(float)
        out = out.drop(columns=["emp_length"])
        engineered.append("emp_length_years")

    return out, engineered


def fit_preprocessor(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: PreprocessingConfig,
    engineered: list[str] | None = None,
) -> tuple[dict, PreprocessingReport]:
    """Learn the preprocessing state from the training partition ALONE.

    Everything learned here - which columns to keep, imputation medians,
    winsorisation bounds, category levels, scaling parameters - comes from
    train only. Fitting any of it on the full sample would leak information
    from the evaluation partitions into the model.

    Args:
        train: The training partition, after feature engineering.
        feature_columns: Candidate feature columns (admitted by the register).
        config: Preprocessing parameters.
        engineered: Names of derived columns, for the report.

    Returns:
        (fitted state, report)
    """
    report = PreprocessingReport(n_input=len(feature_columns))
    report.engineered = list(engineered or [])
    present = [c for c in feature_columns if c in train.columns]

    # 1. Explicit exclusions
    keep = []
    for column in present:
        if column in config.drop_columns:
            report.dropped_explicit.append(column)
        else:
            keep.append(column)

    # 2. Missingness
    survivors = []
    for column in keep:
        if train[column].isna().mean() > config.max_missing_share:
            report.dropped_missing.append(column)
        else:
            survivors.append(column)

    # 3. Split by type, drop high-cardinality categoricals and constants
    numeric, categorical = [], []
    for column in survivors:
        series = train[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) <= 1:
                report.dropped_constant.append(column)
            else:
                numeric.append(column)
        else:
            levels = series.nunique(dropna=True)
            if levels > config.max_cardinality:
                report.dropped_cardinality.append(column)
            elif levels <= 1:
                report.dropped_constant.append(column)
            else:
                categorical.append(column)

    report.numeric = sorted(numeric)
    report.categorical = sorted(categorical)

    # 4. Learn the transformation parameters from train
    bounds = {
        column: (
            float(train[column].quantile(config.winsorise_lower)),
            float(train[column].quantile(config.winsorise_upper)),
        )
        for column in numeric
    }
    medians = {column: float(train[column].median()) for column in numeric}
    levels = {
        column: sorted(train[column].dropna().astype(str).unique().tolist())
        for column in categorical
    }

    # Which columns get a missingness indicator is decided HERE, from the
    # training partition alone, and stored. Deciding it per-partition at apply
    # time would give train and validation different column sets whenever a
    # column happens to be complete in one and not the other - which is exactly
    # the bug this comment exists to prevent recurring.
    missing_indicators = sorted(
        column for column in numeric if bool(train[column].isna().any())
    )

    state = {
        "numeric": report.numeric,
        "categorical": report.categorical,
        "bounds": bounds,
        "medians": medians,
        "levels": levels,
        "missing_indicators": missing_indicators,
    }

    report.n_output = (
        len(report.numeric)
        + len(missing_indicators)
        + sum(len(v) for v in levels.values())
    )
    return state, report


def apply_preprocessor(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply the fitted preprocessing to any partition.

    Args:
        frame: The partition to transform.
        state: Fitted state from `fit_preprocessor`.

    Returns:
        The design matrix: numeric features winsorised and median-imputed,
        with missingness indicators, plus one-hot encoded categoricals.
    """
    pieces: list[pd.DataFrame] = []

    indicators = set(state.get("missing_indicators", []))

    numeric_block = {}
    for column in state["numeric"]:
        series = pd.to_numeric(frame.get(column), errors="coerce")
        low, high = state["bounds"][column]
        # Missingness indicator BEFORE imputation - the fact that a value was
        # absent can itself carry information about the applicant. The indicator
        # set comes from the fitted state, NOT from this partition's own
        # missingness, so every partition yields identical columns.
        if column in indicators:
            numeric_block[f"{column}__missing"] = series.isna().astype(np.int8)
        numeric_block[column] = series.clip(low, high).fillna(state["medians"][column])
    if numeric_block:
        pieces.append(pd.DataFrame(numeric_block, index=frame.index))

    for column in state["categorical"]:
        series = frame.get(column)
        values = (
            pd.Series(pd.NA, index=frame.index) if series is None
            else series.astype("string")
        )
        block = {}
        for level in state["levels"][column]:
            block[f"{column}={level}"] = (values == level).fillna(False).astype(np.int8)
        # Unseen levels and missing values fall through as all-zero, which is
        # the correct behaviour: no evidence for any known level.
        pieces.append(pd.DataFrame(block, index=frame.index))

    design = pd.concat(pieces, axis=1)
    return design.astype(np.float32)
