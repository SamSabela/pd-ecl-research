"""Gate G1: sample definition sign-off.

Runs Steps 1-4 end to end on the real extract and produces every artefact
required to close Gate G1:

    - retained row count and origination window
    - observed default rate, before and after the maturity filter
    - leakage register enforcement report
    - univariate AUC screen over admitted features
    - out-of-time partition diagnostics with per-partition default rates
    - cleaning log with before/after counts for every step
    - the analysis sample, written to data/processed/

Memory: the file is 2.26m rows x 151 columns. Rather than load it whole, only
the columns needed for sample construction and the leakage screen are read,
and the outcome and maturity filters are applied chunk by chunk so peak usage
stays bounded.

Gate G1 is a hard stop. Step 5 does not begin until this report exists and has
been reviewed.

Run:
    python -m scripts.gate_g1
    python -m scripts.gate_g1 --rows 200000     # quick pass while iterating
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.cleaning_log import CleaningLog
from src.data.leakage import classify_columns, enforce_register, screen_for_leakage
from src.data.partition import out_of_time_split
from src.data.provenance import read_provenance_log, verify_file
from src.data.sample import (
    OutcomeDefinition,
    apply_maturity_filter,
    assign_outcome,
    parse_term_months,
)
from src.utils.config import load_config, load_leakage_register, resolve_path, set_global_seed

CHUNK_ROWS = 200_000
STRUCTURAL = ["loan_status", "term", "issue_d"]


def verify_inputs(config: dict) -> None:
    """Confirm the raw extract still matches its recorded checksum.

    A silent change to an input file between profiling and modelling produces
    irreproducible results without ever raising an error. This is the guard.
    """
    log_path = resolve_path(config["paths"]["provenance_log"])
    records = read_provenance_log(log_path)
    if not records:
        print("  WARNING: no provenance records found - cannot verify inputs")
        return

    seen: set[str] = set()
    for record in reversed(records):
        if record.filename in seen:
            continue
        seen.add(record.filename)
        path = Path(record.path)
        if not path.is_file():
            print(f"  WARNING: {record.filename} no longer at recorded path")
            continue
        ok = verify_file(path, record.sha256)
        print(f"  {record.filename:35} {'VERIFIED' if ok else 'CHECKSUM MISMATCH'}")
        if not ok:
            raise SystemExit("raw data has changed since it was logged - stopping")


def build_sample(
    path: Path,
    usecols: list[str],
    definition: OutcomeDefinition,
    max_rows: int | None,
) -> tuple[pd.DataFrame, int]:
    """Stream the extract, applying the outcome definition chunk by chunk.

    The maturity filter is applied afterwards on the concatenated survivors,
    because its before/after default rates must be computed over the whole
    resolved population rather than per chunk.

    Args:
        path: Path to the gzipped extract.
        usecols: Columns to read.
        definition: The outcome mapping (Table 3.2).
        max_rows: Row cap for a quick pass, or None.

    Returns:
        (resolved sample with default_flag and term_months, raw rows read)
    """
    kept: list[pd.DataFrame] = []
    raw_rows = 0

    reader = pd.read_csv(
        path, chunksize=CHUNK_ROWS, usecols=usecols, low_memory=False, compression="gzip"
    )
    for chunk in reader:
        raw_rows += len(chunk)
        chunk["term_months"] = parse_term_months(chunk["term"])
        kept.append(assign_outcome(chunk, definition))
        print(f"  ... {raw_rows:,} rows read", end="\r", flush=True)
        if max_rows is not None and raw_rows >= max_rows:
            break

    print(f"  ... {raw_rows:,} rows read" + " " * 20)
    sample = pd.concat(kept, ignore_index=True)
    sample["issue_d"] = pd.to_datetime(sample["issue_d"], format="%b-%Y", errors="coerce")
    return sample, raw_rows


def main() -> None:
    """Run the Gate G1 pipeline and print the sign-off report."""
    parser = argparse.ArgumentParser(description="Gate G1 sample definition sign-off")
    parser.add_argument("--rows", type=int, default=None, help="Row cap for a quick pass")
    parser.add_argument(
        "--file", type=str, default="data/raw/accepted_2007_to_2018Q4.csv.gz"
    )
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    set_global_seed(config["seed"])

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"extract not found: {path}")

    print("=" * 78)
    print("GATE G1 - SAMPLE DEFINITION SIGN-OFF")
    print("=" * 78)
    print(f"Config version : {config['meta']['config_version']}")
    print(f"Seed           : {config['seed']}")
    print(f"Extract        : {path.name}")
    if args.rows:
        print(f"Mode           : SAMPLE, first {args.rows:,} rows")
    print()

    print("Verifying raw inputs against provenance log")
    print("-" * 78)
    verify_inputs(config)
    print()

    # ---- decide which columns to read ---------------------------------------
    header = pd.read_csv(path, nrows=0, compression="gzip")
    all_columns = header.columns.tolist()
    classification = classify_columns(all_columns, register)
    admitted = [c for c, k in classification.items() if k.value == "application_time"]
    usecols = sorted(set(STRUCTURAL) | set(admitted))

    print(f"Columns in extract : {len(all_columns)}")
    print(f"Reading            : {len(usecols)} (structural + admitted)")
    print()

    log = CleaningLog()

    # ---- Steps 1-2: outcome definition (Table 3.2) --------------------------
    print("Building sample")
    print("-" * 78)
    definition = OutcomeDefinition(
        default_statuses=tuple(config["sample"]["outcome"]["default_statuses"]),
        non_default_statuses=tuple(config["sample"]["outcome"]["non_default_statuses"]),
        exclude_credit_policy=config["sample"]["outcome"]["exclude_credit_policy"],
    )
    resolved, raw_rows = build_sample(path, usecols, definition, args.rows)

    log.record(
        "outcome_definition",
        "Retain resolved outcomes only (Table 3.2); label Charged Off/Default as 1",
        pd.DataFrame(index=range(raw_rows), columns=usecols),
        resolved,
        default_rate=float(resolved["default_flag"].mean()),
    )
    print()

    # ---- Step 3: maturity filter (Section 3.4.2) ----------------------------
    cutoffs = {
        int(k): v for k, v in config["sample"]["maturity_filter"]["term_cutoffs"].items()
    }
    maturity = apply_maturity_filter(resolved, cutoffs)
    sample = log.record(
        "maturity_filter",
        "Retain only loans whose contractual term had fully elapsed by the cutoff",
        resolved,
        maturity.frame,
        default_rate_before=maturity.default_rate_before,
        default_rate_after=maturity.default_rate_after,
        change=maturity.bias_avoided,
    )
    print(maturity.summary())
    print()
    direction = "LOWERED" if maturity.bias_avoided < 0 else "RAISED"
    print(f"  The filter {direction} the observed default rate.")
    if maturity.bias_avoided < 0:
        print("  Section 3.4.2 predicts the opposite. That prediction assumes")
        print("  unresolved loans are labelled 0, but Table 3.2 excludes them first.")
        print("  Among resolved loans the asymmetry reverses: default resolves fast,")
        print("  full repayment takes the whole term, so young cohorts over-represent")
        print("  defaults. THE WORDING IN 3.4.2 NEEDS REVISING BEFORE SUBMISSION.")
    print()

    # ---- Step 4: leakage enforcement (Table 3.4) ----------------------------
    report = enforce_register(sample, register)
    print(report.summary())
    print()

    numeric_admitted = [
        c for c in report.admitted
        if c in sample.columns and pd.api.types.is_numeric_dtype(sample[c])
    ]
    screen = screen_for_leakage(
        sample, numeric_admitted,
        threshold=config["leakage"]["univariate_auc_flag_threshold"],
    )
    print(screen.summary())
    print()

    # ---- Out-of-time partition (Section 3.4.3) ------------------------------
    partition = out_of_time_split(
        sample,
        train_share=config["partition"]["train_share"],
        validation_share=config["partition"]["validation_share"],
        test_share=config["partition"]["test_share"],
    )
    sample = sample.assign(partition=partition.assignment)
    print(partition.summary())
    print()

    # ---- artefacts ----------------------------------------------------------
    print(log.summary())
    print()

    log.write(resolve_path(config["paths"]["cleaning_log"]))
    processed = resolve_path(config["paths"]["data_processed"])
    processed.mkdir(parents=True, exist_ok=True)
    out_path = processed / "analysis_sample_v1.parquet"
    sample.to_parquet(out_path, index=False)

    tables = resolve_path(config["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    partition.diagnostics.to_csv(tables / "partition_diagnostics.csv", index=False)
    screen.scores.sort_values(ascending=False).to_csv(tables / "univariate_auc_screen.csv")
    log.to_frame().to_csv(tables / "cleaning_log.csv", index=False)

    # ---- gate decision ------------------------------------------------------
    blockers = []
    if screen.flagged:
        blockers.append(
            f"{len(screen.flagged)} feature(s) exceeded the {screen.threshold:.2f} "
            f"univariate AUC threshold: {', '.join(screen.flagged)}"
        )

    print("=" * 78)
    print("GATE G1 DECISION")
    print("=" * 78)
    print(f"  Raw rows            : {raw_rows:,}")
    print(f"  Analysis sample     : {len(sample):,} loans")
    print(
        f"  Origination window  : {maturity.origination_min:%Y-%m} to "
        f"{maturity.origination_max:%Y-%m}"
    )
    print(f"  Default rate        : {sample['default_flag'].mean():.4%}")
    print(f"  Features admitted   : {len(report.admitted)}")
    print(f"  Sample written to   : {out_path.relative_to(resolve_path('.'))}")
    print()
    if blockers:
        print("  STATUS: BLOCKED - resolve before Step 5 begins")
        for issue in blockers:
            print(f"    - {issue}")
    else:
        print("  STATUS: CLEAR - Step 5 (EDA) may begin")
    print()
    print("  MANUAL REVIEW STILL REQUIRED:")
    print(f"    {len(report.admitted)} columns are admitted to the modelling matrix.")
    print("    The register excludes what was anticipated; confirm each admitted")
    print("    column really is known at origination. See data_dictionary.csv.")
    print("=" * 78)


if __name__ == "__main__":
    main()
