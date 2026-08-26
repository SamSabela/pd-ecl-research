"""Cleaning log.

Implements the cleaning-log artefact in Section 3.10: every filtering and
transformation step is recorded with before-and-after row and column counts, so
that the path from the raw file to the analysis sample is fully auditable.

This exists because "we dropped some rows" is not a reproducible statement. At
Gate G1 the sample definition has to be signed off, and signing off means being
able to account for every row that left the dataset and why.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["CleaningStep", "CleaningLog"]


@dataclass(frozen=True)
class CleaningStep:
    """One recorded filtering or transformation step."""

    step: str
    description: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    timestamp_utc: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        """Rows dropped by this step."""
        return self.rows_before - self.rows_after

    @property
    def rows_removed_pct(self) -> float:
        """Rows dropped as a share of rows entering the step."""
        return self.rows_removed / self.rows_before if self.rows_before else 0.0

    @property
    def cols_removed(self) -> int:
        """Columns dropped by this step."""
        return self.cols_before - self.cols_after


class CleaningLog:
    """Append-only record of the path from raw file to analysis sample.

    Usage:
        log = CleaningLog()
        df = log.record("maturity", "Retain matured loans", before, after,
                        default_rate_before=0.16, default_rate_after=0.15)
        print(log.summary())
        log.write("outputs/logs/cleaning.jsonl")
    """

    def __init__(self) -> None:
        """Initialise an empty log."""
        self._steps: list[CleaningStep] = []

    def record(
        self,
        step: str,
        description: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
        **details: Any,
    ) -> pd.DataFrame:
        """Record a step and return the resulting frame unchanged.

        Returns `after` so the call can be chained inline, which makes it
        harder to perform a filtering step and forget to log it.

        Args:
            step: Short machine-readable step name.
            description: What the step does and why.
            before: Frame entering the step.
            after: Frame leaving the step.
            **details: Step-specific figures worth preserving, e.g. the default
                rate before and after a maturity filter.

        Returns:
            The `after` frame, unmodified.
        """
        self._steps.append(
            CleaningStep(
                step=step,
                description=description,
                rows_before=len(before),
                rows_after=len(after),
                cols_before=before.shape[1],
                cols_after=after.shape[1],
                timestamp_utc=datetime.now(UTC).isoformat(),
                details=details,
            )
        )
        return after

    @property
    def steps(self) -> list[CleaningStep]:
        """All recorded steps, in order."""
        return list(self._steps)

    def write(self, path: str | Path) -> None:
        """Write the log as JSON Lines.

        Args:
            path: Destination path. Parent directories are created.
        """
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            for step in self._steps:
                handle.write(json.dumps(asdict(step)) + "\n")

    def to_frame(self) -> pd.DataFrame:
        """Render the log as a table, suitable for the Gate G1 report."""
        return pd.DataFrame(
            [
                {
                    "step": s.step,
                    "description": s.description,
                    "rows_before": s.rows_before,
                    "rows_after": s.rows_after,
                    "rows_removed": s.rows_removed,
                    "rows_removed_pct": s.rows_removed_pct,
                    "cols_before": s.cols_before,
                    "cols_after": s.cols_after,
                }
                for s in self._steps
            ]
        )

    def summary(self) -> str:
        """Human-readable summary for console output and Gate G1 sign-off."""
        if not self._steps:
            return "Cleaning log is empty."

        lines = ["Cleaning log", "=" * 78]
        for s in self._steps:
            lines.append(
                f"  {s.step:<26} {s.rows_before:>10,} -> {s.rows_after:>10,}"
                f"  ({s.rows_removed:>9,} removed, {s.rows_removed_pct:>6.2%})"
            )
            if s.cols_removed:
                lines.append(f"  {'':<26} columns {s.cols_before} -> {s.cols_after}")
            for key, value in s.details.items():
                shown = f"{value:.4%}" if isinstance(value, float) else f"{value:,}"
                lines.append(f"  {'':<26}   {key}: {shown}")

        first, last = self._steps[0], self._steps[-1]
        retained = last.rows_after / first.rows_before if first.rows_before else 0.0
        lines.append("-" * 78)
        lines.append(
            f"  {'NET':<26} {first.rows_before:>10,} -> {last.rows_after:>10,}"
            f"  ({retained:.2%} retained)"
        )
        return "\n".join(lines)
