"""SHAP attributions for the best ensemble, set against logistic regression.

Implements the interpretability stream of Section 3.8.5 and answers RQ4: what
does adopting a machine learning PD model cost in explainability, and can that
cost be mitigated?

For each fold, on block A, the script fits XGBoost and logistic regression on
the identical training cohort and compares how each one assigns importance.

    XGBoost         Exact TreeSHAP values (Lundberg and Lee, 2017), computed by
                    XGBoost's own implementation via pred_contribs. This is the
                    algorithm shap.TreeExplainer runs, without depending on the
                    shap package keeping pace with the pinned XGBoost release.
                    Importance is mean absolute SHAP over a seeded sample of
                    evaluation loans; direction is the sign of the rank
                    correlation between a feature's value and its SHAP value.

    Logistic        Coefficients on the standardised scale the pipeline fits,
    regression      so magnitudes are comparable across features. Importance is
                    the absolute coefficient; direction is its sign.

Three questions are then answered in numbers rather than asserted.

    Agreement   Do the two models rank features alike (Spearman), share their
                top ten, and agree on the direction of each top feature?
    Stability   Does the XGBoost explanation hold across the three folds, or
                does it move with the training cohort? An explanation that
                changes when the data does is weak evidence for a governance
                file, however well it renders.
    Leakage     Where do installment and loan_amnt rank? Section 5.5 shows that
                their ratio carries the excluded contractual rate. A tree model
                can learn that ratio through interactions; a linear model can
                only approximate it.

Run:
    python -m scripts.interpretability
    python -m scripts.interpretability --drop installment --tag no_installment
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

from scripts.run_experiment import build_models, candidate_features
from src.data.partition import rolling_origin_folds
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

TOP_K = 10
WATCH = ("installment", "loan_amnt")


def tree_shap(model, x: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP contributions from a fitted XGBClassifier, bias column dropped."""
    import xgboost as xgb

    contributions = model.get_booster().predict(
        xgb.DMatrix(x, feature_names=list(x.columns)), pred_contribs=True
    )
    return contributions[:, :-1]


def direction(values: np.ndarray, contributions: np.ndarray) -> float:
    """Sign of the rank correlation between a feature and its SHAP value."""
    if np.nanstd(values) == 0 or np.nanstd(contributions) == 0:
        return np.nan
    rho = spearmanr(values, contributions).statistic
    return float(np.sign(rho)) if np.isfinite(rho) else np.nan


def explain_fold(fold_name, x_train, y_train, x_eval, models, n_explain, seed):
    """Fit both models on one fold and return a per-feature comparison table."""
    xgb_model = clone(models["XGBoost"]).fit(x_train, y_train)
    lr_model = clone(models["Logistic Regression"]).fit(x_train, y_train)

    rng = np.random.default_rng(seed)
    take = rng.choice(len(x_eval), size=min(n_explain, len(x_eval)), replace=False)
    x_sample = x_eval.iloc[np.sort(take)]
    shap_values = tree_shap(xgb_model, x_sample)

    coefficients = lr_model.named_steps["clf"].coef_[0]
    features = list(x_train.columns)
    table = pd.DataFrame({
        "fold": fold_name,
        "feature": features,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "shap_direction": [
            direction(x_sample[f].to_numpy(), shap_values[:, i])
            for i, f in enumerate(features)
        ],
        "lr_coef_std": coefficients,
    })
    table["lr_direction"] = np.sign(table["lr_coef_std"])
    table["rank_shap"] = table["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    table["rank_lr"] = table["lr_coef_std"].abs().rank(ascending=False, method="min").astype(int)
    return table.sort_values("rank_shap").reset_index(drop=True)


def summarise(table: pd.DataFrame) -> dict:
    top_shap = set(table.nsmallest(TOP_K, "rank_shap")["feature"])
    top_lr = set(table.nsmallest(TOP_K, "rank_lr")["feature"])
    head = table[table["feature"].isin(top_shap)].dropna(subset=["shap_direction"])
    total = table["mean_abs_shap"].sum()
    return {
        "fold": table["fold"].iloc[0],
        "n_features": len(table),
        "spearman_shap_vs_lr": spearmanr(
            table["mean_abs_shap"], table["lr_coef_std"].abs()
        ).statistic,
        "top10_overlap": len(top_shap & top_lr),
        "top10_direction_agreement": float(
            (head["shap_direction"] == head["lr_direction"]).mean()
        ),
        "top10_share_of_shap": float(
            table.nsmallest(TOP_K, "rank_shap")["mean_abs_shap"].sum() / total
        ),
    }


def stability(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = sorted(tables)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ja = tables[a].set_index("feature")
            jb = tables[b].set_index("feature")
            common = ja.index.intersection(jb.index)
            top_a = set(ja.loc[common].nsmallest(TOP_K, "rank_shap").index)
            top_b = set(jb.loc[common].nsmallest(TOP_K, "rank_shap").index)
            rows.append({
                "fold_a": a, "fold_b": b, "n_common": len(common),
                "spearman_shap": spearmanr(
                    ja.loc[common, "mean_abs_shap"], jb.loc[common, "mean_abs_shap"]
                ).statistic,
                "top10_overlap": len(top_a & top_b),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP against logistic regression, block A")
    parser.add_argument("--n-explain", type=int, default=5000, help="Evaluation loans explained")
    parser.add_argument("--drop", default="", help="Comma-separated features to withhold")
    parser.add_argument("--tag", default="", help="Suffix for the output filenames")
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    seed = config["seed"]
    set_global_seed(seed)

    rolling = config["partition"]["rolling_origin"]
    boundaries = [(f["train_end"], f["evaluate_end"]) for f in rolling["folds"]]
    sample = pd.read_parquet(
        resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    )
    term = config["sample"]["term_scope"]["primary_term_months"]
    sample = sample.loc[sample["term_months"] == term].copy()
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])
    sample, engineered = engineer_features(sample)
    drop = tuple(d.strip() for d in args.drop.split(",") if d.strip())
    candidates = candidate_features(sample, register, engineered, drop=drop)

    models = build_models(seed)
    if "XGBoost" not in models:
        raise SystemExit("xgboost is required for this script")

    cache_dir = resolve_path(config["macro"]["cache_dir"])
    frame, _ = assemble_block(sample, config, "A", cache_dir=cache_dir)
    embargo = rolling["embargo_months"]
    folds = rolling_origin_folds(frame, boundaries, embargo_months=embargo).folds

    print("=" * 78)
    print("INTERPRETABILITY - XGBoost TreeSHAP against logistic regression, block A")
    print("=" * 78)
    print(f"  Seed {seed}   loans explained per fold {args.n_explain:,}"
          + (f"   withheld: {', '.join(drop)}" if drop else ""))
    print()

    tables: dict[str, pd.DataFrame] = {}
    for fold in folds:
        train = frame.loc[fold.train]
        evaluate = frame.loc[fold.evaluate]
        state, _ = fit_preprocessor(train, candidates, PreprocessingConfig(), engineered)
        x_train = apply_preprocessor(train, state)
        x_eval = apply_preprocessor(evaluate, state)
        y_train = train["default_flag"].to_numpy()
        print(f"  Fold {fold.name}: train {len(train):,}, {x_train.shape[1]} design columns ...")
        tables[str(fold.name)] = explain_fold(
            str(fold.name), x_train, y_train, x_eval, models, args.n_explain, seed
        )

    importance = pd.concat(tables.values(), ignore_index=True)
    summary = pd.DataFrame([summarise(t) for t in tables.values()])
    stable = stability(tables)

    suffix = f"_{args.tag}" if args.tag else ""
    out = resolve_path(config["paths"]["tables"])
    importance.to_csv(out / f"shap_importance{suffix}.csv", index=False)
    summary.to_csv(out / f"shap_summary{suffix}.csv", index=False)
    stable.to_csv(out / f"shap_stability{suffix}.csv", index=False)

    print()
    latest = tables[max(tables)]
    print(f"TOP {TOP_K} BY MEAN |SHAP| - fold {max(tables)}   (direction: +1 raises PD)")
    print("-" * 78)
    print(latest.head(TOP_K)[["feature", "mean_abs_shap", "shap_direction",
                              "rank_lr", "lr_coef_std", "lr_direction"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("AGREEMENT WITH LOGISTIC REGRESSION, by fold")
    print("-" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("STABILITY OF THE XGBOOST EXPLANATION ACROSS FOLDS")
    print("-" * 78)
    print(stable.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("WATCHED FEATURES (Section 5.5, the ratio that carries the contractual rate)")
    print("-" * 78)
    watched = importance[importance["feature"].isin(WATCH)]
    if watched.empty:
        print("  neither feature is in the design matrix (withheld)")
    else:
        print(watched[["fold", "feature", "rank_shap", "rank_lr", "lr_coef_std"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print()
    print(f"  Written: outputs/tables/shap_importance{suffix}.csv, "
          f"shap_summary{suffix}.csv, shap_stability{suffix}.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
