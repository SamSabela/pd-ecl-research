"""Paired significance tests across the experiment grid.

Implements the inference in Section 3.8.2 and Refinement 10, which the
preliminary results deferred: every AUC comparison the report makes is tested
with DeLong's paired test on identical evaluation loans, and each family of
comparisons is Holm-Bonferroni corrected.

Three families are tested, matching the three questions the report asks of the
grid.

    models      Within each feature block and fold, every pair of model
                families. Block A is the headline comparison for RQ1; blocks B
                and C are reported because RQ1 is posed with macroeconomic
                inputs included. Holm-corrected within each block (30 tests).

    blocks      Within each model family and fold, block B against A, block C
                against A, and block C against B. The last is the incremental
                South African contribution measured against the correctly
                specified United States benchmark, as Refinement 6 specifies.
                Holm-corrected across all 45 tests.

    pooled      The three folds are disjoint evaluation cohorts, so their z
                statistics are independent and can be combined by Stouffer's
                method into one test per model pair. Block A only. Holm across
                the 10 pairs.

Before testing anything, the script recomputes every AUC from the saved
predictions and checks it against the frozen results table. A re-run that does
not reproduce the reported numbers would make the tests answer a question about
a different set of models, so a mismatch is reported loudly.

Run:
    python -m scripts.run_experiment --save-predictions --tag repro
    python -m scripts.significance
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation.significance import delong_paired, holm_adjust
from src.utils.config import load_config, resolve_path

MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
BLOCK_CONTRASTS = [("B", "A"), ("C", "A"), ("C", "B")]
REPRODUCTION_TOLERANCE = 5e-4


def load_inputs(predictions_path, frozen_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(predictions_path)
    frozen = pd.read_csv(frozen_path)
    for frame in (predictions, frozen):
        frame["fold"] = frame["fold"].astype(str)
        frame["block"] = frame["block"].astype(str)
        frame["model"] = frame["model"].astype(str)
    return predictions, frozen


def cell(predictions: pd.DataFrame, block: str, fold: str, model: str) -> pd.Series:
    """One model's probabilities on one fold, indexed by loan."""
    rows = predictions[
        (predictions["block"] == block)
        & (predictions["fold"] == fold)
        & (predictions["model"] == model)
    ]
    if rows.empty:
        raise SystemExit(f"no predictions for block {block}, fold {fold}, {model}")
    return rows.set_index("row_id")[["y", "p"]]


def paired(a: pd.DataFrame, b: pd.DataFrame):
    """Align two cells on loan identity and test them."""
    joined = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if len(joined) != len(a) or len(joined) != len(b):
        raise SystemExit(
            f"cells do not cover the same loans: {len(a):,} vs {len(b):,}, "
            f"{len(joined):,} in common"
        )
    if not (joined["y_a"] == joined["y_b"]).all():
        raise SystemExit("outcomes disagree for the same loan - predictions are misaligned")
    return delong_paired(joined["y_a"], joined["p_a"], joined["p_b"]), len(joined)


def check_reproduction(predictions: pd.DataFrame, frozen: pd.DataFrame) -> float:
    """Largest absolute gap between recomputed and reported AUC across the grid."""
    worst = 0.0
    for _, row in frozen.iterrows():
        c = cell(predictions, row["block"], row["fold"], row["model"])
        recomputed = delong_paired(c["y"], c["p"], c["p"]).auc_a
        worst = max(worst, abs(recomputed - row["auc"]))
    return worst


def record(family, fold, block_a, block_b, model_a, model_b, result, n) -> dict:
    return {
        "family": family, "fold": fold,
        "block_a": block_a, "block_b": block_b,
        "model_a": model_a, "model_b": model_b,
        "n_evaluate": n,
        "auc_a": result.auc_a, "auc_b": result.auc_b,
        "difference": result.difference, "se": result.se,
        "z": result.z, "p_raw": result.p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired DeLong tests across the grid")
    parser.add_argument("--predictions", default="outputs/predictions/predictions_repro.parquet")
    parser.add_argument("--frozen", default="outputs/tables/experiment_results.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    config = load_config()
    predictions, frozen = load_inputs(resolve_path(args.predictions), resolve_path(args.frozen))
    folds = sorted(frozen["fold"].unique())
    blocks = sorted(frozen["block"].unique())
    models = [m for m in MODEL_ORDER if m in set(frozen["model"])]

    print("=" * 78)
    print("PAIRED SIGNIFICANCE - DeLong test, Holm-Bonferroni within each family")
    print("=" * 78)
    print(f"  Seed {config['seed']}   folds {folds}   blocks {blocks}   alpha {args.alpha}")
    print(f"  Predictions : {args.predictions}  ({len(predictions):,} rows)")
    print()

    worst = check_reproduction(predictions, frozen)
    verdict = "REPRODUCES" if worst <= REPRODUCTION_TOLERANCE else "DOES NOT REPRODUCE"
    print(f"  Reproduction check: largest AUC gap against {args.frozen} = {worst:.2e}  "
          f"-> {verdict}")
    if worst > REPRODUCTION_TOLERANCE:
        print("  WARNING: the re-run differs from the reported results. The tests below")
        print("  describe the re-run, not the numbers in the report. Investigate first.")
    print()

    rows: list[dict] = []

    # family 1: model pairs within each block
    for block in blocks:
        for fold in folds:
            for model_a, model_b in combinations(models, 2):
                result, n = paired(
                    cell(predictions, block, fold, model_a),
                    cell(predictions, block, fold, model_b),
                )
                rows.append(record(f"models_{block}", fold, block, block,
                                   model_a, model_b, result, n))

    # family 2: block contrasts within each model
    for model in models:
        for fold in folds:
            for block_a, block_b in BLOCK_CONTRASTS:
                if block_a not in blocks or block_b not in blocks:
                    continue
                result, n = paired(
                    cell(predictions, block_a, fold, model),
                    cell(predictions, block_b, fold, model),
                )
                rows.append(record("blocks", fold, block_a, block_b,
                                   model, model, result, n))

    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _family, index in table.groupby("family").groups.items():
        table.loc[index, "p_holm"] = holm_adjust(table.loc[index, "p_raw"])
    table["significant"] = table["p_holm"] <= args.alpha

    # family 3: block A model pairs pooled across the independent folds
    block_a = table[table["family"] == "models_A"]
    pooled = (
        block_a.groupby(["model_a", "model_b"], sort=False)
        .agg(mean_difference=("difference", "mean"),
             folds_a_ahead=("difference", lambda d: int((d > 0).sum())),
             z_sum=("z", "sum"), k=("z", "size"))
        .reset_index()
    )
    pooled["z_stouffer"] = pooled["z_sum"] / np.sqrt(pooled["k"])
    pooled["p_raw"] = 2.0 * norm.sf(pooled["z_stouffer"].abs())
    pooled["p_holm"] = holm_adjust(pooled["p_raw"])
    pooled["significant"] = pooled["p_holm"] <= args.alpha
    pooled = pooled.drop(columns=["z_sum", "k"])

    tables = resolve_path(config["paths"]["tables"])
    table.to_csv(tables / "significance_delong.csv", index=False)
    pooled.to_csv(tables / "significance_delong_pooled.csv", index=False)

    # ---- console summary
    pd.set_option("display.width", 140)
    fmt = {"difference": "{:+.4f}", "mean_difference": "{:+.4f}", "se": "{:.4f}",
           "z": "{:+.2f}", "z_stouffer": "{:+.2f}", "p_raw": "{:.2e}", "p_holm": "{:.2e}"}

    def show(frame, columns):
        out = frame[columns].copy()
        for column, spec in fmt.items():
            if column in out:
                out[column] = out[column].map(spec.format)
        print(out.to_string(index=False))

    print("BLOCK A - model pairs pooled across folds (Stouffer), Holm over 10 pairs")
    print("-" * 78)
    show(pooled, ["model_a", "model_b", "mean_difference", "folds_a_ahead",
                  "z_stouffer", "p_holm", "significant"])
    print()

    print("BLOCK A - model pairs by fold, Holm over 30 tests")
    print("-" * 78)
    show(block_a, ["fold", "model_a", "model_b", "difference", "z", "p_holm", "significant"])
    print()

    contrasts = table[table["family"] == "blocks"]
    print("FEATURE BLOCKS - contrasts within each model, Holm over 45 tests")
    print("-" * 78)
    show(contrasts, ["fold", "model_a", "block_a", "block_b",
                     "difference", "z", "p_holm", "significant"])
    print()
    summary = contrasts.assign(
        direction=np.where(contrasts["difference"] > 0, "gain", "loss")
    ).groupby(["block_a", "block_b", "direction"])["significant"].agg(["size", "sum"])
    print("  Contrast counts (size = cells, sum = significant after Holm):")
    print(summary.to_string())
    print()
    print(f"  Written: outputs/tables/significance_delong.csv ({len(table)} tests)")
    print(f"           outputs/tables/significance_delong_pooled.csv ({len(pooled)} pairs)")
    print("=" * 78)


if __name__ == "__main__":
    main()
