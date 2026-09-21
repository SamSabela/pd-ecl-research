"""Is block C's embargo-36 advantage macroeconomic conditioning, or a clock?

At embargo 36 the South African panel adds about +0.006 AUC to every tree
model while adding nothing to logistic regression, and the US panel subtracts.
A genuine macroeconomic risk relationship should reach the linear model too.
A series that merely indexes time would not: only a model that can partition
the timeline can exploit it.

usdzar is a daily series carried at zero publication lag, so inside a single
evaluation year it takes a near-unique value per origination date. That makes
it a candidate date stamp rather than a macroeconomic signal.

Four arms settle it, all at embargo 36:

    A                 borrower features only, the control
    A + month index   a pure origination-month counter, which carries no
                      macroeconomic content whatsoever
    C                 the South African panel as run
    C minus usdzar    the same panel with the daily series removed

If the month index alone reproduces C's gain, the gain is a clock. If removing
usdzar destroys the gain, the clock is specifically that series. If C keeps its
advantage over both, the effect survives the challenge and is worth reporting.

The script also prints within-year variance retention for both panels, which is
the mechanism in one table: a series that keeps most of its variation inside a
single evaluation year is able to separate loans within that year, and AUC is
computed within exactly that window.

Run:
    python -m scripts.diagnose_macro_clock
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.base import clone

from scripts.run_experiment import build_models, candidate_features
from src.data.partition import rolling_origin_folds
from src.evaluation.metrics import compute_metrics
from src.features.blocks import assemble_block
from src.features.preprocessing import (
    PreprocessingConfig,
    apply_preprocessor,
    engineer_features,
    fit_preprocessor,
)
from src.utils.config import (
    load_config,
    load_leakage_register,
    resolve_path,
    set_global_seed,
)

PLACEBO = "origination_month_index"

# Each arm is (block, mutation). The mutation is what distinguishes the arm
# from its block's default assembly, and it is the only thing that differs.
ARMS: dict[str, tuple[str, str | None]] = {
    "A  borrower only": ("A", None),
    "A + month index": ("A", "placebo"),
    "C  SA macro": ("C", None),
    "C  minus usdzar": ("C", "drop_usdzar"),
}


def add_month_index(frame: pd.DataFrame, origination_col: str = "issue_d") -> pd.DataFrame:
    """Add a counter of months since the first origination in the sample.

    This is the placebo. It is known at application time, so it is a legitimate
    feature, but it carries no information about the economy - only about when
    the loan was written. Anything it can predict, a date stamp can predict.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[origination_col])
    base = dates.min()
    out[PLACEBO] = (
        (dates.dt.year - base.year) * 12 + (dates.dt.month - base.month)
    ).astype(float)
    return out


def variance_retention(
    frame: pd.DataFrame, columns: list[str], folds: list
) -> pd.DataFrame:
    """Share of a series' full-period variation that survives inside one year.

    A value near 1.0 means the series varies as much within a single evaluation
    year as it does across the whole sample period, so it can rank loans inside
    that year. A value near 0.0 means it is effectively constant within the
    year and cannot.
    """
    rows = []
    for column in columns:
        full_sd = frame[column].std()
        if not full_sd:
            continue
        per_fold = {
            fold.name: float(frame.loc[fold.evaluate, column].std() / full_sd)
            for fold in folds
        }
        rows.append(
            {
                "series": column,
                **per_fold,
                "mean": sum(per_fold.values()) / len(per_fold),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Placebo test for the block C gain")
    parser.add_argument("--embargo", type=int, default=36)
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    seed = config["seed"]
    set_global_seed(seed)

    rolling = config["partition"]["rolling_origin"]
    boundaries = [(f["train_end"], f["evaluate_end"]) for f in rolling["folds"]]
    cache_dir = resolve_path(config["macro"]["cache_dir"])

    sample_path = (
        resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    )
    sample = pd.read_parquet(sample_path)
    term = config["sample"]["term_scope"]["primary_term_months"]
    sample = sample.loc[sample["term_months"] == term].copy()
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])
    sample, engineered = engineer_features(sample)
    candidates = candidate_features(sample, register, engineered)

    models = build_models(seed)

    print("=" * 78)
    print("IS THE BLOCK C GAIN MACROECONOMIC, OR A CLOCK?")
    print("=" * 78)
    print(f"  Embargo : {args.embargo} months")
    print(f"  Seed    : {seed}")
    print()

    # Mechanism first: how much of each series' variation survives the year.
    print("-" * 78)
    print("WITHIN-YEAR VARIANCE RETENTION  (1.00 = varies as much inside one")
    print("evaluation year as across the whole period, so it can rank within it)")
    print("-" * 78)
    for block, label in (("B", "US panel"), ("C", "SA panel")):
        frame, macro_columns = assemble_block(sample, config, block, cache_dir=cache_dir)
        folds = rolling_origin_folds(
            frame, boundaries, embargo_months=args.embargo
        ).folds
        table = variance_retention(frame, macro_columns, folds)
        print(f"\n  {label}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()

    # Then the arms.
    results: list[dict] = []
    for arm, (block, mutation) in ARMS.items():
        frame, macro_columns = assemble_block(sample, config, block, cache_dir=cache_dir)
        arm_candidates = candidates + macro_columns

        if mutation == "placebo":
            frame = add_month_index(frame)
            arm_candidates = arm_candidates + [PLACEBO]
        elif mutation == "drop_usdzar":
            frame = frame.drop(columns=["usdzar"])
            arm_candidates = [c for c in arm_candidates if c != "usdzar"]

        print("-" * 78)
        print(f"ARM  {arm}")
        print("-" * 78)

        folds = rolling_origin_folds(
            frame, boundaries, embargo_months=args.embargo
        ).folds
        for fold in folds:
            train = frame.loc[fold.train]
            evaluate = frame.loc[fold.evaluate]

            state, _ = fit_preprocessor(
                train, arm_candidates, PreprocessingConfig(), engineered
            )
            x_train = apply_preprocessor(train, state)
            x_eval = apply_preprocessor(evaluate, state)
            y_train = train["default_flag"].to_numpy()
            y_eval = evaluate["default_flag"].to_numpy()

            for model_name, prototype in models.items():
                model = clone(prototype)
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_eval)[:, 1]
                metrics = compute_metrics(
                    y_eval, probabilities, model_name, fold.name, 0.0
                )
                results.append(
                    {"arm": arm, "fold": fold.name, "model": model_name,
                     "auc": metrics.auc}
                )
            print(f"  fold {fold.name}: train {len(train):,}  "
                  f"evaluate {len(evaluate):,}  {x_train.shape[1]} design columns")
        print()

    frame_results = pd.DataFrame(results)
    pivot = frame_results.pivot_table(
        index="model", columns="arm", values="auc", aggfunc="mean"
    )[list(ARMS)]

    print("=" * 78)
    print("MEAN AUC ACROSS FOLDS")
    print("=" * 78)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

    control = pivot["A  borrower only"]
    deltas = pivot.drop(columns=["A  borrower only"]).sub(control, axis=0)
    print("CHANGE AGAINST THE BORROWER-ONLY CONTROL")
    print(deltas.to_string(float_format=lambda v: f"{v:+.4f}"))
    print()

    tables = resolve_path(config["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"macro_clock_diagnostic_embargo{args.embargo}.csv"
    frame_results.to_csv(out_path, index=False)
    print(f"  {len(results)} cells written to "
          f"{out_path.relative_to(resolve_path('.'))}")
    print("=" * 78)


if __name__ == "__main__":
    main()