"""Feature block assembly.

Implements Section 3.5.3 and Table 3.5. Three blocks, one control and two
treatments:

    A   borrower and credit-bureau attributes only
    B   A plus the six United States macroeconomic series
    C   A plus the six South African macroeconomic series

The design is what makes the macroeconomic claim testable. Comparing A against
C alone would confound two questions - does macroeconomic conditioning help at
all, and does *South African* conditioning help - because any improvement could
come from either. Block B is the control for the first question, so that the
B-to-C difference isolates the second.

It also converts the study's most attackable feature into a measured quantity.
An examiner encountering South African indicators merged onto United States
loans will object to the merge; the answer is that the merge was performed
against a matched benchmark panel and the difference reported, so the objection
is already in the results table.

Nothing here decides whether the macro blocks help. It assembles the inputs so
that the question can be asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.macro import attach_macro, build_panel, series_specs

__all__ = ["BLOCK_PANELS", "block_description", "assemble_block"]

# Which macroeconomic panel each block carries. None is the control.
BLOCK_PANELS: dict[str, str | None] = {"A": None, "B": "us", "C": "za"}


def block_description(config: dict[str, Any], block: str) -> str:
    """Return the configured description of a feature block.

    Args:
        config: The parsed config.
        block: "A", "B" or "C".

    Returns:
        The description from `feature_blocks` in config.

    Raises:
        KeyError: If the block is not configured.
    """
    blocks = config.get("feature_blocks", {})
    if block not in blocks:
        raise KeyError(f"feature block '{block}' not configured")
    return str(blocks[block]["description"])


def assemble_block(
    loans: pd.DataFrame,
    config: dict[str, Any],
    block: str,
    cache_dir: str | Path | None = None,
    origination_col: str = "issue_d",
) -> tuple[pd.DataFrame, list[str]]:
    """Attach the macroeconomic panel a feature block calls for.

    Block A returns the loans unchanged, which is the point of a control: it
    must differ from B and C in exactly one respect.

    The panel is attached to the whole sample before any fold is taken. That is
    safe because attachment is row-wise and depends only on each loan's own
    origination date - no statistic is computed across rows, so nothing crosses
    the train/evaluate boundary. Fitting the preprocessor, by contrast, happens
    per fold on training rows alone.

    Args:
        loans: The analysis sample.
        config: The parsed config.
        block: "A", "B" or "C".
        cache_dir: Directory holding the cached macro series. Defaults to the
            configured `macro.cache_dir`.
        origination_col: Column holding the origination date.

    Returns:
        (frame with macro columns attached, names of the columns added). The
        list is empty for block A.

    Raises:
        KeyError: If the block is unknown.
        FileNotFoundError: If a required macro series is not cached.
    """
    if block not in BLOCK_PANELS:
        raise KeyError(
            f"unknown feature block '{block}'; expected one of "
            f"{sorted(BLOCK_PANELS)}"
        )

    panel_name = BLOCK_PANELS[block]
    if panel_name is None:
        return loans.copy(), []

    panel = build_panel(config, panel_name, cache_dir=cache_dir)
    macro_columns = [spec.name for spec in series_specs(config, panel_name)]

    collisions = sorted(set(macro_columns) & set(loans.columns))
    if collisions:
        raise ValueError(
            f"macro series {collisions} collide with existing loan columns; "
            "rename the series in config before attaching"
        )

    attached = attach_macro(loans, panel, origination_col=origination_col)
    return attached, macro_columns
