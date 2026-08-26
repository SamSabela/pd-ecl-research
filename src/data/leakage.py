"""Leakage control: enforce the register, then screen for what it missed.

Implements Section 3.5.1 and Table 3.4.

The Lending Club file records the full life of each loan, not merely its state
at origination. Fields describing payments received, outstanding principal,
recoveries, settlement activity and hardship arrangements are populated after
origination and are direct consequences of the outcome being predicted. A model
given access to them will discriminate almost perfectly and will have learned
nothing about credit risk.

Because such a result looks like success, this is the single most dangerous
failure mode in the pipeline, and the control is specified before any modelling
begins.

Two controls, in order:

  1. The register (config/leakage_register.yaml). Every column is classified as
     application-time, post-origination, identifier, or operational. Only
     application-time fields reach the modelling matrix.

  2. The screen. Univariate AUC of every admitted feature against the outcome;
     anything above 0.90 is flagged for manual review. A single feature that
     separates defaulters that cleanly in credit data is almost always leakage.

The register can only exclude what was anticipated. The screen catches what
was not. Neither alone is sufficient.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ColumnClass",
    "LeakageReport",
    "ScreenResult",
    "classify_columns",
    "enforce_register",
    "univariate_auc",
    "screen_for_leakage",
]


class ColumnClass(StrEnum):
    """Classification assigned to every raw column, per Section 3.5.1."""

    APPLICATION_TIME = "application_time"
    POST_ORIGINATION = "post_origination"
    IDENTIFIER = "identifier"
    OPERATIONAL = "operational"  # used for ECL/partitioning, never a model input


def _explicit_fields(register: dict[str, Any]) -> dict[str, str]:
    """Map each explicitly listed excluded field to its reason category."""
    reasons: dict[str, str] = {}
    for category, spec in register.get("exclusions", {}).items():
        for name in spec.get("fields") or []:
            reasons[name] = category
    return reasons


def _patterns(register: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (glob pattern, category) pairs, e.g. ("hardship_*", ...)."""
    out: list[tuple[str, str]] = []
    for category, spec in register.get("exclusions", {}).items():
        for pattern in spec.get("field_patterns") or []:
            out.append((pattern, category))
    return out


def _operational_fields(register: dict[str, Any]) -> set[str]:
    """Fields retained for ECL or partitioning but never admitted as features."""
    return {
        entry["field"]
        for entry in register.get("retained_with_justification") or []
        if entry.get("excluded_from_modelling", False)
    }


def _category_for(
    column: str,
    reasons: dict[str, str],
    patterns: list[tuple[str, str]],
) -> str | None:
    """Reason category for a column, by exact name then by glob."""
    if column in reasons:
        return reasons[column]
    for pattern, category in patterns:
        if fnmatch.fnmatch(column, pattern):
            return category
    return None


def classify_columns(
    columns: list[str],
    register: dict[str, Any],
) -> dict[str, ColumnClass]:
    """Classify every column against the register.

    This is the operational form of the data dictionary required by Section
    3.10. Columns the register does not mention default to application-time,
    which is precisely why the univariate screen exists as a second line of
    defence: the default is permissive, so something must catch mistakes.

    Args:
        columns: All column names in the raw file.
        register: The parsed leakage register.

    Returns:
        Mapping of column name to classification.
    """
    reasons = _explicit_fields(register)
    patterns = _patterns(register)
    operational = _operational_fields(register)

    result: dict[str, ColumnClass] = {}
    for column in columns:
        if column in operational:
            result[column] = ColumnClass.OPERATIONAL
            continue

        category = _category_for(column, reasons, patterns)
        if category is None:
            result[column] = ColumnClass.APPLICATION_TIME
        elif category == "identifiers_and_free_text":
            result[column] = ColumnClass.IDENTIFIER
        else:
            result[column] = ColumnClass.POST_ORIGINATION
    return result


@dataclass
class LeakageReport:
    """Result of applying the register to an actual dataframe."""

    admitted: list[str]
    excluded: dict[str, str]      # column -> reason category
    operational: list[str]
    register_fields_absent: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for the Gate G1 sign-off."""
        lines = [
            "Leakage register enforcement",
            "=" * 78,
            f"  Admitted to modelling matrix : {len(self.admitted):>4}",
            f"  Excluded as leakage          : {len(self.excluded):>4}",
            f"  Retained operationally only  : {len(self.operational):>4}",
            "",
        ]
        counts: dict[str, int] = {}
        for reason in self.excluded.values():
            counts[reason] = counts.get(reason, 0) + 1
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {reason:<36} {count:>4}")
        if self.register_fields_absent:
            lines.append(
                f"\n  Register lists {len(self.register_fields_absent)} field(s) not present "
                "in this extract (harmless, but worth confirming the names are right)"
            )
        return "\n".join(lines)


def enforce_register(
    frame: pd.DataFrame,
    register: dict[str, Any],
    target_col: str = "default_flag",
) -> LeakageReport:
    """Apply the register and report what would be admitted.

    Nothing is dropped here. The report is produced so the exclusion can be
    reviewed and signed off at Gate G1 before it is acted on — a silent drop
    is not an auditable decision.

    Args:
        frame: The raw or partially cleaned dataframe.
        register: The parsed leakage register.
        target_col: Outcome column, never admitted as a feature.

    Returns:
        A LeakageReport.
    """
    columns = [c for c in frame.columns if c != target_col]
    classification = classify_columns(columns, register)
    reasons = _explicit_fields(register)
    patterns = _patterns(register)

    admitted, operational = [], []
    excluded: dict[str, str] = {}

    for column, column_class in classification.items():
        if column_class is ColumnClass.OPERATIONAL:
            operational.append(column)
        elif column_class in (ColumnClass.POST_ORIGINATION, ColumnClass.IDENTIFIER):
            excluded[column] = _category_for(column, reasons, patterns) or "unspecified"
        else:
            admitted.append(column)

    absent = sorted(set(reasons) - set(frame.columns))

    return LeakageReport(
        admitted=sorted(admitted),
        excluded=excluded,
        operational=sorted(operational),
        register_fields_absent=absent,
    )


def univariate_auc(feature: pd.Series, target: pd.Series) -> float:
    """AUC of a single feature against a binary outcome.

    Computed via the Mann-Whitney U identity on ranks rather than by sweeping
    thresholds: it handles ties correctly and needs no sklearn dependency for
    what is fundamentally a rank statistic.

    Folded to [0.5, 1.0], because a feature that predicts perfectly in reverse
    is exactly as much of a leakage signal as one that predicts forwards.

    Args:
        feature: Numeric feature values. Non-numeric input returns NaN.
        target: Binary outcome, 0 or 1.

    Returns:
        AUC in [0.5, 1.0], or NaN if the feature is non-numeric or constant,
        or the outcome has only one class among non-missing rows.
    """
    if not pd.api.types.is_numeric_dtype(feature):
        return float("nan")

    valid = feature.notna() & target.notna()
    x = feature.loc[valid].to_numpy(dtype=np.float64)
    y = target.loc[valid].to_numpy()

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0 or x.size == 0 or np.all(x == x[0]):
        return float("nan")

    ranks = pd.Series(x).rank(method="average").to_numpy()
    auc = (float(np.sum(ranks[y == 1])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(auc, 1.0 - auc)


@dataclass
class ScreenResult:
    """Result of the automated univariate AUC screen."""

    scores: pd.Series
    flagged: list[str]
    threshold: float

    def summary(self) -> str:
        """Human-readable summary, listing anything needing manual review."""
        lines = [
            "Univariate AUC leakage screen",
            "=" * 78,
            f"  Threshold         : {self.threshold:.2f}",
            f"  Features screened : {len(self.scores):,}",
            f"  Flagged           : {len(self.flagged)}",
        ]
        if self.flagged:
            lines.append("\n  MANUAL REVIEW REQUIRED before modelling proceeds:")
            for feature in self.flagged:
                lines.append(f"      {feature:<44} AUC = {self.scores[feature]:.4f}")
        else:
            lines.append("\n  Nothing exceeded the threshold.")

        lines.append("\n  Highest univariate AUC:")
        for feature, score in self.scores.sort_values(ascending=False).head(12).items():
            lines.append(f"      {feature:<44} {score:.4f}")
        return "\n".join(lines)


def screen_for_leakage(
    frame: pd.DataFrame,
    features: list[str],
    target_col: str = "default_flag",
    threshold: float = 0.90,
) -> ScreenResult:
    """Run the automated univariate AUC screen over admitted features.

    Args:
        frame: Dataframe containing features and outcome.
        features: Columns admitted by the register.
        target_col: The binary outcome column.
        threshold: Flag any feature whose univariate AUC exceeds this.

    Returns:
        A ScreenResult listing every score and every flagged feature.

    Raises:
        KeyError: If the target column is absent.
    """
    if target_col not in frame.columns:
        raise KeyError(f"target column '{target_col}' not in frame")

    target = frame[target_col]
    scores = {
        name: univariate_auc(frame[name], target) for name in features if name in frame.columns
    }
    series = pd.Series(scores, dtype=float).dropna()
    flagged = sorted(series[series > threshold].index.tolist())

    return ScreenResult(scores=series, flagged=flagged, threshold=threshold)
