"""Build the rolling-origin folds and write their diagnostics.

Runs the partitioning scheme adopted on 12 September 2026 against the analysis
sample and produces the fold table that Section 3.4.3 reports.

Two passes are made over the same folds:

    embargo 0   the conventional origination-time split, comparable to the
                published literature, and the basis for the main results
    embargo 36  the strict deployability check, in which every training loan
                had resolved before its evaluation cohort was originated

The gap between the two is a finding in its own right: on this sample the
embargo removes roughly 91% of the training data, because origination volume
grew so fast that stepping back three years empties the book. Every published
AUC on this dataset is obtained under an information advantage no lender had.

Run:
    python -m scripts.build_folds
    python -m scripts.build_folds --all-terms    # ignore the 36-month scope
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.partition import rolling_origin_folds
from src.utils.config import load_config, resolve_path, set_global_seed


def build_fold_boundaries(config: dict) -> list[tuple[str, str]]:
    """Read the fold boundaries out of config into the shape the API wants.

    `rolling_origin_folds` takes an ordered sequence of (train_end,
    evaluate_end) pairs. The config stores them as a list of mappings, because
    YAML keys document themselves and bare pairs do not.

    Args:
        config: The parsed config.

    Returns:
        Ordered (train_end, evaluate_end) string pairs.

    Raises:
        KeyError: If the rolling-origin block or a fold key is absent.
    """
    folds = config["partition"]["rolling_origin"]["folds"]
    return [(fold["train_end"], fold["evaluate_end"]) for fold in folds]


def load_sample(config: dict, term_months: int | None) -> pd.DataFrame:
    """Load the analysis sample, optionally restricted to one contractual term.

    Args:
        config: The parsed config.
        term_months: Retain only this term. None keeps every term, which
            reintroduces the confound described in the term_scope config
            comment and is intended only for comparison.

    Returns:
        The sample, with issue_d parsed.
    """
    path = resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    frame = pd.read_parquet(path)
    frame["issue_d"] = pd.to_datetime(frame["issue_d"])

    if term_months is not None:
        before = len(frame)
        frame = frame.loc[frame["term_months"] == term_months].copy()
        share = len(frame) / before if before else 0.0
        print(
            f"  Term scope           : {term_months}-month only, "
            f"{len(frame):,} of {before:,} retained ({share:.1%})"
        )
    else:
        print(f"  Term scope           : all terms, {len(frame):,} rows")

    return frame


def main() -> None:
    """Build the folds at both embargo settings and write the diagnostics."""
    parser = argparse.ArgumentParser(description="Build rolling-origin folds")
    parser.add_argument(
        "--all-terms",
        action="store_true",
        help="Ignore the 36-month term scope and use every term",
    )
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["seed"])

    rolling = config["partition"]["rolling_origin"]
    boundaries = build_fold_boundaries(config)
    term = None if args.all_terms else config["sample"]["term_scope"]["primary_term_months"]

    print("=" * 78)
    print("Rolling-origin fold construction")
    print("=" * 78)

    frame = load_sample(config, term)
    print(f"  Observed default rate: {frame['default_flag'].mean():.4%}")
    print(f"  Folds requested      : {len(boundaries)}")
    print()

    settings = {
        "main": rolling["embargo_months"],
        "robustness": rolling["embargo_months_robustness"],
    }

    tables = resolve_path(config["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    collected = []

    for label, embargo in settings.items():
        result = rolling_origin_folds(frame, boundaries, embargo_months=embargo)
        print(result.summary())
        print()

        diagnostics = result.diagnostics.copy()
        diagnostics.insert(0, "setting", label)
        diagnostics.insert(1, "embargo_months", embargo)
        collected.append(diagnostics)

    combined = pd.concat(collected, ignore_index=True)
    out_path = tables / "fold_diagnostics.csv"
    combined.to_csv(out_path, index=False)

    main_rows = combined.loc[combined["setting"] == "main", "train_rows"]
    strict_rows = combined.loc[combined["setting"] == "robustness", "train_rows"]
    loss = 1.0 - strict_rows.sum() / main_rows.sum()

    print("=" * 78)
    print(f"  Training data removed by the embargo: {loss:.1%}")
    print(f"  Diagnostics written to             : {out_path.relative_to(resolve_path('.'))}")
    print("=" * 78)


if __name__ == "__main__":
    main()