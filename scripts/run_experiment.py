"""Run the full experiment grid: model families x folds x feature blocks.

Implements Research Objective 1 (do machine learning methods discriminate
better than a regulatory benchmark?) and the macroeconomic half of Research
Objective 2 (does macroeconomic conditioning help, and does South African
conditioning help specifically?).

The grid

    5 model families  x  3 rolling-origin folds  x  3 feature blocks  =  45 fits

Every cell sees the identical fold, the identical seed and an equal, untuned
configuration, so what varies across a row is the model family and what varies
across a column is the information available to it. That is what makes the
comparison a comparison rather than a collection of numbers.

What is fitted where

    The macroeconomic panel is attached to the whole sample once per block,
    before folds are taken. Attachment is row-wise and depends only on each
    loan's own origination date, so no statistic crosses the train/evaluate
    boundary.

    The preprocessor is fitted per fold, on training rows alone. Imputation
    medians, winsorisation bounds, category levels and scaling all come from
    the training cohort of that fold and are applied unchanged to its
    evaluation cohort. Fitting once on the whole sample would leak the
    evaluation distribution into every model.

Hyperparameters are fixed, not tuned. The equal computational budget across
families is what makes the comparison fair, and it has not been spent yet - the
tuning run is a later step. Relative standings here are indicative.

Run:
    python -m scripts.run_experiment
    python -m scripts.run_experiment --blocks A          one block only
    python -m scripts.run_experiment --sample 50000      subsample training rows
    python -m scripts.run_experiment --embargo 36        strict deployability
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.base import clone

from src.data.leakage import classify_columns
from src.data.partition import rolling_origin_folds
from src.evaluation.metrics import bootstrap_auc_ci, compute_metrics
from src.features.blocks import assemble_block, block_description
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


def build_models(seed: int) -> dict:
    """Instantiate the model families at fixed, untuned defaults.

    Class weighting is deliberately not applied. The methodology specifies
    weighted loss for imbalance, but at a 13.9% base rate the imbalance is
    moderate, and weighting inflates predicted probabilities - which would
    corrupt the calibration metrics this run reports, and calibration is what
    Expected Credit Loss depends on. Weighting is revisited alongside the
    calibration step, where it belongs.

    Args:
        seed: Random seed from config.

    Returns:
        Mapping of display name to unfitted estimator. Families whose native
        libraries are unavailable are omitted with a warning rather than
        aborting the run.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict = {
        # Scaling matters for the linear model and not for the trees, so it is
        # attached to this pipeline rather than applied globally.
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=seed,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    except Exception as exc:  # ImportError, or a missing native library
        print(f"  xgboost unavailable - skipping  ({type(exc).__name__})")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    except Exception as exc:
        print(f"  lightgbm unavailable - skipping  ({type(exc).__name__})")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, allow_writing_files=False,
        )
    except Exception as exc:
        print(f"  catboost unavailable - skipping  ({type(exc).__name__})")

    return models


def candidate_features(sample: pd.DataFrame, register: dict, engineered: list[str]) -> list[str]:
    """Columns admitted by the leakage register, plus engineered features.

    Args:
        sample: The analysis sample after feature engineering.
        register: The parsed leakage register.
        engineered: Names of derived columns.

    Returns:
        Sorted candidate feature names, excluding the outcome and any
        partition bookkeeping.
    """
    classification = classify_columns(sample.columns.tolist(), register)
    admitted = [c for c, k in classification.items() if k.value == "application_time"]
    admitted = [c for c in admitted if c != "default_flag"] + list(engineered)
    return sorted(set(admitted) - {"partition", "term_months"})


def run_cell(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train,
    x_eval: pd.DataFrame,
    y_eval,
    fold_name: str,
    block: str,
    seed: int,
    replicates: int,
) -> dict:
    """Fit one model on one fold under one feature block, and score it.

    Returns:
        A flat record suitable for a tidy results table.
    """
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started

    probabilities = model.predict_proba(x_eval)[:, 1]
    metrics = compute_metrics(y_eval, probabilities, model_name, fold_name, elapsed)
    low, high = bootstrap_auc_ci(y_eval, probabilities, replicates=replicates, seed=seed)

    return {
        "block": block,
        "fold": fold_name,
        "model": model_name,
        "n_train": len(x_train),
        "n_evaluate": len(x_eval),
        "n_features": x_train.shape[1],
        "base_rate": metrics.base_rate,
        "predicted_rate": metrics.predicted_rate,
        "auc": metrics.auc,
        "auc_ci_low": low,
        "auc_ci_high": high,
        "gini": metrics.gini,
        "ks": metrics.ks,
        "brier": metrics.brier,
        "brier_reliability": metrics.brier_reliability,
        "brier_resolution": metrics.brier_resolution,
        "brier_uncertainty": metrics.brier_uncertainty,
        "ece": metrics.ece,
        "fit_seconds": elapsed,
    }


def main() -> None:
    """Run the grid and write the tidy results table."""
    parser = argparse.ArgumentParser(description="Run the full experiment grid")
    parser.add_argument("--blocks", default="A,B,C", help="Comma-separated blocks")
    parser.add_argument("--sample", type=int, default=None, help="Subsample training rows")
    parser.add_argument("--embargo", type=int, default=None, help="Override embargo months")
    parser.add_argument("--replicates", type=int, default=200, help="Bootstrap replicates")
    parser.add_argument(
        "--all-terms", action="store_true", help="Ignore the 36-month term scope"
    )
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    seed = config["seed"]
    set_global_seed(seed)

    rolling = config["partition"]["rolling_origin"]
    embargo = args.embargo if args.embargo is not None else rolling["embargo_months"]
    boundaries = [(f["train_end"], f["evaluate_end"]) for f in rolling["folds"]]
    blocks = [b.strip().upper() for b in args.blocks.split(",") if b.strip()]

    sample_path = resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    if not sample_path.is_file():
        raise SystemExit(f"analysis sample not found: {sample_path}\nRun scripts.gate_g1 first.")

    print("=" * 78)
    print("EXPERIMENT GRID - model families x folds x feature blocks")
    print("=" * 78)
    print(f"  Config version : {config['meta']['config_version']}")
    print(f"  Seed           : {seed}")
    print(f"  Embargo        : {embargo} months")
    print(f"  Blocks         : {', '.join(blocks)}")
    print("  Test partition : NOT TOUCHED - reserved for Gate G3")
    print()

    sample = pd.read_parquet(sample_path)
    if not args.all_terms:
        term = config["sample"]["term_scope"]["primary_term_months"]
        before = len(sample)
        sample = sample.loc[sample["term_months"] == term].copy()
        print(f"  Term scope     : {term}-month only, {len(sample):,} of {before:,} "
              f"({len(sample) / before:.1%})")
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])
    sample, engineered = engineer_features(sample)
    candidates = candidate_features(sample, register, engineered)
    print(f"  Borrower features admitted: {len(candidates)}")
    print()

    models = build_models(seed)
    print(f"  Model families : {len(models)}  {list(models)}")
    print()

    records: list[dict] = []
    cache_dir = resolve_path(config["macro"]["cache_dir"])

    for block in blocks:
        print("=" * 78)
        print(f"BLOCK {block} - {block_description(config, block)}")
        print("=" * 78)

        frame, macro_columns = assemble_block(
            sample, config, block, cache_dir=cache_dir
        )
        block_candidates = candidates + macro_columns
        if macro_columns:
            missing = frame[macro_columns].isna().all(axis=1).sum()
            print(f"  Macro series attached : {len(macro_columns)}  {macro_columns}")
            print(f"  Rows with no macro    : {missing:,} "
                  f"(originated before the first release)")
        print()

        result = rolling_origin_folds(frame, boundaries, embargo_months=embargo)

        for fold in result.folds:
            train = frame.loc[fold.train]
            evaluate = frame.loc[fold.evaluate]
            if args.sample and len(train) > args.sample:
                train = train.sample(n=args.sample, random_state=seed)

            state, report = fit_preprocessor(
                train, block_candidates, PreprocessingConfig(), engineered
            )
            x_train = apply_preprocessor(train, state)
            x_eval = apply_preprocessor(evaluate, state)

            if list(x_train.columns) != list(x_eval.columns):
                raise SystemExit(
                    f"block {block} fold {fold.name}: design matrices disagree, "
                    f"train {x_train.shape[1]} columns, evaluate {x_eval.shape[1]}"
                )

            y_train = train["default_flag"].to_numpy()
            y_eval = evaluate["default_flag"].to_numpy()

            print(f"  Fold {fold.name}: train {len(train):,} "
                  f"({fold.train_default_rate:.2%})  "
                  f"evaluate {len(evaluate):,} ({fold.evaluate_default_rate:.2%})  "
                  f"{x_train.shape[1]} design columns")

            for model_name, prototype in models.items():
                record = run_cell(
                    model_name, clone(prototype), x_train, y_train,
                    x_eval, y_eval, fold.name, block, seed, args.replicates,
                )
                records.append(record)
                print(f"      {model_name:<22} AUC {record['auc']:.4f} "
                      f"[{record['auc_ci_low']:.4f}, {record['auc_ci_high']:.4f}]   "
                      f"KS {record['ks']:.4f}   ECE {record['ece']:.4f}   "
                      f"{record['fit_seconds']:>6.1f}s")
            print()

    results = pd.DataFrame(records)
    tables = resolve_path(config["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    suffix = "" if embargo == rolling["embargo_months"] else f"_embargo{embargo}"
    out_path = tables / f"experiment_results{suffix}.csv"
    results.to_csv(out_path, index=False)

    print("=" * 78)
    print("MEAN AUC ACROSS FOLDS")
    print("=" * 78)
    pivot = results.pivot_table(
        index="model", columns="block", values="auc", aggfunc="mean"
    ).sort_values(blocks[0], ascending=False)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"  {len(records)} cells written to "
          f"{out_path.relative_to(resolve_path('.'))}")
    print("=" * 78)


if __name__ == "__main__":
    main()
