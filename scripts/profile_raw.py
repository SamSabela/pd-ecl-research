"""Step 2: profile the raw Lending Club extract and emit the data dictionary.

Gate G1 deliverable. Produces:

    outputs/tables/data_dictionary.csv   every column, dtype, missingness,
                                         cardinality, and leakage classification
    outputs/tables/loan_status_counts.csv
    outputs/tables/term_counts.csv
    outputs/tables/origination_by_year.csv

Reads the gzipped file in chunks so it never holds the whole extract in memory.
The accepted file is roughly 2.2 million rows by 150 columns; loaded whole in
pandas that is several gigabytes, and there is no need for it.

Run:
    python -m scripts.profile_raw
    python -m scripts.profile_raw --rows 200000      # quick pass while iterating
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.leakage import classify_columns
from src.utils.config import load_config, load_leakage_register, resolve_path

CHUNK_ROWS = 200_000

# Columns we need full distributions for. Everything else is summarised
# structurally (dtype, missingness, cardinality) without loading it whole.
KEY_COLUMNS = ["loan_status", "term", "issue_d", "loan_amnt", "int_rate"]


def profile_columns(path: Path, max_rows: int | None) -> pd.DataFrame:
    """Stream the file and accumulate per-column statistics.

    Args:
        path: Path to the gzipped CSV.
        max_rows: Stop after roughly this many rows. None reads everything.

    Returns:
        One row per column: dtype, non-null count, null count, cardinality
        estimate and a sample value.
    """
    total_rows = 0
    non_null: dict[str, int] = {}
    distinct: dict[str, set] = {}
    dtypes: dict[str, str] = {}
    example: dict[str, object] = {}

    reader = pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False, compression="gzip")

    for chunk in reader:
        total_rows += len(chunk)
        for column in chunk.columns:
            series = chunk[column]
            non_null[column] = non_null.get(column, 0) + int(series.notna().sum())
            dtypes.setdefault(column, str(series.dtype))

            # Cap the distinct set so a high-cardinality column cannot blow up
            # memory. Anything past 1000 is "high cardinality" for our purposes.
            bucket = distinct.setdefault(column, set())
            if len(bucket) < 1000:
                bucket.update(series.dropna().unique()[:1000])

            if column not in example:
                first = series.dropna()
                if len(first):
                    example[column] = first.iloc[0]

        print(f"  ... {total_rows:,} rows", end="\r", flush=True)
        if max_rows is not None and total_rows >= max_rows:
            break

    print(f"  ... {total_rows:,} rows profiled" + " " * 20)

    rows = []
    for column in dtypes:
        filled = non_null.get(column, 0)
        cardinality = len(distinct.get(column, set()))
        rows.append(
            {
                "column": column,
                "dtype": dtypes[column],
                "non_null": filled,
                "null": total_rows - filled,
                "null_pct": (total_rows - filled) / total_rows if total_rows else 0.0,
                "cardinality": f">={cardinality}" if cardinality >= 1000 else str(cardinality),
                "example": str(example.get(column, ""))[:40],
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["total_rows"] = total_rows
    return frame


def key_distributions(path: Path, max_rows: int | None) -> dict[str, pd.DataFrame]:
    """Load only the key columns and compute the distributions Gate G1 needs.

    Args:
        path: Path to the gzipped CSV.
        max_rows: Row cap, or None for everything.

    Returns:
        Mapping of table name to frame.
    """
    frame = pd.read_csv(
        path,
        usecols=lambda c: c in KEY_COLUMNS,
        nrows=max_rows,
        low_memory=False,
        compression="gzip",
    )

    status = (
        frame["loan_status"].value_counts(dropna=False).rename_axis("loan_status")
        .reset_index(name="count")
    )
    status["share"] = status["count"] / status["count"].sum()

    term = (
        frame["term"].value_counts(dropna=False).rename_axis("term")
        .reset_index(name="count")
    )

    issued = pd.to_datetime(frame["issue_d"], format="%b-%Y", errors="coerce")
    by_year = (
        issued.dt.year.value_counts(dropna=True).rename_axis("origination_year")
        .reset_index(name="count").sort_values("origination_year")
    )

    return {
        "loan_status_counts": status,
        "term_counts": term,
        "origination_by_year": by_year,
        "_issued": pd.DataFrame({"issue_d": issued}),
    }


def main() -> None:
    """Profile the raw extract and write the Gate G1 tables."""
    parser = argparse.ArgumentParser(description="Profile the raw Lending Club extract")
    parser.add_argument("--rows", type=int, default=None, help="Row cap for a quick pass")
    parser.add_argument(
        "--file",
        type=str,
        default="data/raw/accepted_2007_to_2018Q4.csv.gz",
        help="Path to the gzipped extract",
    )
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"extract not found: {path}")

    tables_dir = resolve_path(config["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 2 - RAW DATA PROFILE")
    print("=" * 78)
    print(f"File   : {path}  ({path.stat().st_size / 1e6:.0f} MB gzipped)")
    if args.rows:
        print(f"Mode   : SAMPLE, first {args.rows:,} rows")
    print()

    print("Profiling columns...")
    profile = profile_columns(path, args.rows)
    total_rows = profile.attrs["total_rows"]

    print("Loading key columns...")
    dists = key_distributions(path, args.rows)
    issued = dists.pop("_issued")["issue_d"]

    # ---- leakage classification -------------------------------------------
    classification = classify_columns(profile["column"].tolist(), register)
    profile["classification"] = profile["column"].map(lambda c: classification[c].value)

    profile = profile.sort_values(["classification", "column"]).reset_index(drop=True)
    profile.to_csv(tables_dir / "data_dictionary.csv", index=False)
    for name, frame in dists.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    # ---- report ------------------------------------------------------------
    print()
    print(f"Rows    : {total_rows:,}")
    print(f"Columns : {len(profile)}")
    print()

    print("Column classification (Table 3.4)")
    print("-" * 78)
    for label, count in profile["classification"].value_counts().items():
        print(f"  {label:<22} {count:>4}")
    print()

    print("Loan status distribution")
    print("-" * 78)
    for row in dists["loan_status_counts"].itertuples(index=False):
        print(f"  {str(row.loan_status):<52} {row.count:>9,}  {row.share:>6.2%}")
    print()

    print("Contractual term")
    print("-" * 78)
    for row in dists["term_counts"].itertuples(index=False):
        print(f"  {str(row.term):<52} {row.count:>9,}")
    print()

    print("Origination window")
    print("-" * 78)
    print(f"  Earliest : {issued.min():%Y-%m}")
    print(f"  Latest   : {issued.max():%Y-%m}")
    print(f"  Unparsed : {int(issued.isna().sum()):,}")
    print()

    print("Most incomplete columns")
    print("-" * 78)
    worst = profile.nlargest(10, "null_pct")
    for row in worst.itertuples(index=False):
        print(f"  {row.column:<42} {row.null_pct:>7.2%} missing   [{row.classification}]")
    print()

    print("Written to outputs/tables/")
    print("-" * 78)
    for name in ["data_dictionary", *dists]:
        print(f"  {name}.csv")
    print()
    print("NEXT: review data_dictionary.csv. Every column classified")
    print("'application_time' is admitted to the modelling matrix - confirm each")
    print("one really is known at origination before Gate G1 sign-off.")
    print("=" * 78)


if __name__ == "__main__":
    main()
