"""Expand the SARB repo rate decision list into the monthly series.

A policy rate is published as a short list of decisions, not as a monthly
observation. This script reads that list, expands it to one row per month
across the origination window, and writes the result where the macro panel
expects to find it.

Doing it this way means the only hand-transcribed artefact is the decision
list - roughly twenty rows, each checkable against the MPC statement that
announced it. A hand-typed monthly series would be a hundred and fifteen rows
in which a single mistyped digit would be invisible.

Input, which you create:
    data/external/sarb_mpc_decisions.csv

        effective_date,repo_rate
        2007-01-01,9.00
        2007-06-08,9.50
        ...

    The first row must take effect on or before the first month of the window,
    so that every month has a rate in force. Give the rate in force at the
    start of the window as the opening row, then one row per MPC change.

Output:
    data/external/macro/sarb_repo_rate.csv

Source: South African Reserve Bank, Monetary Policy Committee statements.
Cite the SARB in the methodology, not FRED - FRED carries the interbank call
money rate, which is not the policy rate.

Run:
    python -m scripts.build_repo_rate
    python -m scripts.build_repo_rate --start 2006-01 --end 2016-12
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.macro import expand_step_series
from src.data.provenance import log_retrieval
from src.utils.config import load_config, resolve_path

DECISIONS_FILENAME = "sarb_mpc_decisions.csv"
OUTPUT_FILENAME = "sarb_repo_rate.csv"


def main() -> None:
    """Expand the decision list and write the monthly series."""
    parser = argparse.ArgumentParser(description="Build the SARB repo rate series")
    parser.add_argument("--start", default="2006-01", help="First month, YYYY-MM")
    parser.add_argument("--end", default="2016-12", help="Last month, YYYY-MM")
    args = parser.parse_args()

    config = load_config()
    external = resolve_path(config["paths"]["data_external"])
    cache = resolve_path(config["macro"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    decisions_path = external / DECISIONS_FILENAME
    if not decisions_path.is_file():
        raise SystemExit(
            f"decision list not found at {decisions_path}\n"
            "Create it with columns effective_date,repo_rate - one row per MPC "
            "change, plus an opening row giving the rate in force at the start "
            "of the window. See this script's docstring."
        )

    decisions = pd.read_csv(decisions_path)
    series = expand_step_series(
        decisions, f"{args.start}-01", f"{args.end}-01"
    )

    out_path = cache / OUTPUT_FILENAME
    series.to_csv(out_path, index=False)

    log_retrieval(
        path=out_path,
        source_url="https://www.resbank.co.za/en/home/what-we-do/monetary-policy",
        licence="South African Reserve Bank, public statistical release",
        log_path=resolve_path(config["paths"]["provenance_log"]),
        notes=(
            f"Expanded from {len(decisions)} MPC decisions in {DECISIONS_FILENAME}; "
            f"monthly {args.start} to {args.end}; value is the rate in force at "
            "month end"
        ),
    )

    changes = series["repo_rate"].diff().fillna(0) != 0

    print("=" * 78)
    print("SARB repo rate series")
    print("=" * 78)
    print(f"  Decisions read  : {len(decisions)}")
    print(f"  Months written  : {len(series)}  ({args.start} to {args.end})")
    print(f"  Changes in range: {int(changes.sum())}")
    low = series["repo_rate"].min()
    high = series["repo_rate"].max()
    print(f"  Range           : {low:.2f} to {high:.2f} percent")
    print(f"  Written to      : {out_path.relative_to(resolve_path('.'))}")
    print()
    print("  Month-by-month, where the rate changed:")
    for row in series.loc[changes].itertuples(index=False):
        print(f"      {row.observation_date:%Y-%m}   {row.repo_rate:>6.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
