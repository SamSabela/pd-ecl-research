"""ICE Task 2: preliminary comparative model run.

Trains the model families named in the proposal, plus two later gradient
boosting implementations, on the Gate G1 analysis sample and compares them on
discrimination and calibration.

WHAT THIS IS NOT
----------------
This is a PRELIMINARY run, and three things distinguish it from the study's
final comparison at Steps 7-8:

  1. The TEST PARTITION IS NOT TOUCHED. Models are trained on `train` and
     evaluated on `validation`. Section 3.8.2 commits to scoring the test
     partition exactly once, after every specification decision is frozen at
     Gate G3. Reporting test-set numbers now would spend that once, and every
     subsequent decision would be contaminated by having seen them.

  2. Hyperparameters are FIXED at reasonable defaults, not tuned. The study
     specifies 60 trials per model per feature block under an equal
     computational budget (Section 3.6); equal budget is what makes the
     comparison fair, and it has not been spent yet. Relative standings here
     are indicative, not final.

  3. Only FEATURE BLOCK A (borrower and credit-bureau features) is used. Blocks
     B and C require the macroeconomic panels, which are Step 3 and not yet
     retrieved. The A/B/C comparison that addresses RO1 follows.

Run:
    python -m scripts.ice2_baseline
    python -m scripts.ice2_baseline --sample 50000     # quick pass
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.leakage import classify_columns
from src.evaluation.metrics import bootstrap_auc_ci, compute_metrics, metrics_table
from src.features.preprocessing import (
    PreprocessingConfig,
    apply_preprocessor,
    engineer_features,
    fit_preprocessor,
)
from src.utils.config import load_config, load_leakage_register, resolve_path, set_global_seed

warnings.filterwarnings("ignore", category=UserWarning)

# Validated categorical palette, slots 1-5 (light mode).
PALETTE = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
    "LightGBM": "#eda100",
    "CatBoost": "#e87ba4",
}


def build_models(seed: int) -> dict:
    """Instantiate the model families at fixed, untuned defaults.

    Note on class weighting. The methodology specifies class-weighted loss
    functions for imbalance. At a 14.8% base rate the imbalance is moderate,
    and weighting inflates predicted probabilities - which would corrupt the
    calibration metrics this run reports, and calibration is what ECL depends
    on. Models are therefore fitted unweighted here, with weighting revisited
    at Step 7 alongside the calibration step where it belongs.

    Args:
        seed: Random seed from config.

    Returns:
        Mapping of display name to unfitted estimator.
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
    except Exception as exc:  # ImportError, or a missing native library
        print(f"  lightgbm unavailable - skipping  ({type(exc).__name__})")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, allow_writing_files=False,
        )
    except Exception as exc:  # ImportError, or a missing native library
        print(f"  catboost unavailable - skipping  ({type(exc).__name__})")

    return models


def plot_roc(results: dict, y_val: np.ndarray, path: Path) -> None:
    """ROC curves for every model on the validation partition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for name, (probs, metrics) in results.items():
        order = np.argsort(probs)[::-1]
        y_sorted = y_val[order]
        tpr = np.cumsum(y_sorted == 1) / max((y_val == 1).sum(), 1)
        fpr = np.cumsum(y_sorted == 0) / max((y_val == 0).sum(), 1)
        step = max(len(fpr) // 4000, 1)  # thin the curve; 100k points is noise
        ax.plot(
            np.r_[0, fpr[::step], 1], np.r_[0, tpr[::step], 1],
            color=PALETTE.get(name, "#52514e"), linewidth=2.0,
            label=f"{name}  (AUC {metrics.auc:.4f})", solid_capstyle="round",
        )

    ax.plot([0, 1], [0, 1], color="#c9c8c2", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xlabel("False positive rate", fontsize=10, color="#52514e")
    ax.set_ylabel("True positive rate", fontsize=10, color="#52514e")
    ax.set_title(
        "ROC curves, validation partition (feature block A)",
        fontsize=11, color="#0b0b0b", pad=12, loc="left",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results: dict, y_val: np.ndarray, path: Path, bins: int = 10) -> None:
    """Reliability diagram: predicted probability against observed frequency.

    The plot that matters most for this study. ECL is computed from probability
    magnitudes, so a model that ranks well but sits off the diagonal produces
    provisions that are systematically wrong.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for name, (probs, metrics) in results.items():
        edges = np.quantile(probs, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        index = np.digitize(probs, edges[1:-1], right=False)
        xs, ys = [], []
        for b in range(bins):
            mask = index == b
            if mask.sum() == 0:
                continue
            xs.append(float(np.mean(probs[mask])))
            ys.append(float(np.mean(y_val[mask])))
        ax.plot(
            xs, ys, marker="o", markersize=5, linewidth=2.0,
            color=PALETTE.get(name, "#52514e"),
            markeredgecolor="#fcfcfb", markeredgewidth=1.0,
            label=f"{name}  (ECE {metrics.ece:.4f})",
        )

    lim = max(
        max(np.mean(y_val) * 2.2, 0.05),
        *(float(np.quantile(p, 0.99)) for p, _ in results.values()),
    )
    ax.plot([0, lim], [0, lim], color="#c9c8c2", linewidth=1.0,
            linestyle=(0, (4, 4)), zorder=0, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=10, color="#52514e")
    ax.set_ylabel("Observed default frequency", fontsize=10, color="#52514e")
    ax.set_title(
        "Reliability diagram, validation partition",
        fontsize=11, color="#0b0b0b", pad=12, loc="left",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the preliminary comparison and emit tables and figures."""
    parser = argparse.ArgumentParser(description="ICE Task 2 preliminary model run")
    parser.add_argument("--sample", type=int, default=None, help="Subsample training rows")
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    seed = config["seed"]
    set_global_seed(seed)

    sample_path = resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    if not sample_path.is_file():
        raise SystemExit(f"analysis sample not found: {sample_path}\nRun scripts.gate_g1 first.")

    print("=" * 78)
    print("ICE TASK 2 - PRELIMINARY COMPARATIVE MODEL RUN")
    print("=" * 78)
    print(f"Config version : {config['meta']['config_version']}")
    print(f"Seed           : {seed}")
    print("Feature block  : A (borrower and credit-bureau features only)")
    print("Evaluated on   : VALIDATION partition")
    print("Test partition : NOT TOUCHED - reserved for Gate G3")
    print()

    sample = pd.read_parquet(sample_path)
    sample, engineered = engineer_features(sample)

    classification = classify_columns(sample.columns.tolist(), register)
    candidates = [c for c, k in classification.items() if k.value == "application_time"]
    candidates = [c for c in candidates if c != "default_flag"] + engineered
    candidates = sorted(set(candidates) - {"partition"})

    train = sample[sample["partition"] == "train"]
    validation = sample[sample["partition"] == "validation"]
    if args.sample:
        train = train.sample(n=min(args.sample, len(train)), random_state=seed)

    print(f"Train      : {len(train):,} rows, default rate {train['default_flag'].mean():.4%}")
    print(f"Validation : {len(validation):,} rows, default rate "
          f"{validation['default_flag'].mean():.4%}")
    print()

    state, report = fit_preprocessor(train, candidates, PreprocessingConfig(), engineered)
    print(report.summary())
    print()

    print("Building design matrices...")
    x_train = apply_preprocessor(train, state)
    x_val = apply_preprocessor(validation, state)
    y_train = train["default_flag"].to_numpy()
    y_val = validation["default_flag"].to_numpy()
    if list(x_train.columns) != list(x_val.columns):
        raise SystemExit(
            "design matrices disagree: "
            f"train {x_train.shape[1]} columns, validation {x_val.shape[1]}. "
            "The preprocessor must produce identical columns for every partition."
        )
    print(f"  train {x_train.shape}   validation {x_val.shape}  (columns match)")
    print()

    models = build_models(seed)
    results: dict = {}
    records = []

    print("Training")
    print("-" * 78)
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        probs = model.predict_proba(x_val)[:, 1]
        metrics = compute_metrics(y_val, probs, name, "validation", elapsed)
        low, high = bootstrap_auc_ci(y_val, probs, replicates=200, seed=seed)
        results[name] = (probs, metrics)
        records.append((metrics, low, high))
        print(
            f"  {name:<22} AUC {metrics.auc:.4f} [{low:.4f}, {high:.4f}]"
            f"   KS {metrics.ks:.4f}   Brier {metrics.brier:.4f}"
            f"   ECE {metrics.ece:.4f}   {elapsed:>6.1f}s"
        )
    print()

    table = metrics_table([m for m, _, _ in records])
    table["auc_ci_low"] = [low for _, low, _ in records]
    table["auc_ci_high"] = [high for _, _, high in records]
    table = table.sort_values("auc", ascending=False)

    tables_dir = resolve_path(config["paths"]["tables"])
    figures_dir = resolve_path(config["paths"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "ice2_model_comparison.csv", index=False)

    plot_roc(results, y_val, figures_dir / "ice2_roc_curves.png")
    plot_calibration(results, y_val, figures_dir / "ice2_calibration.png")

    print("Performance summary, validation partition")
    print("-" * 78)
    print(f"  {'Model':<22}{'AUC':>8}{'Gini':>8}{'KS':>8}{'Brier':>9}{'ECE':>8}{'fit(s)':>9}")
    print("  " + "-" * 70)
    for row in table.itertuples(index=False):
        print(
            f"  {row.model:<22}{row.auc:>8.4f}{row.gini:>8.4f}{row.ks:>8.4f}"
            f"{row.brier:>9.4f}{row.ece:>8.4f}{row.fit_seconds:>9.1f}"
        )
    print()

    best = table.iloc[0]
    worst = table.iloc[-1]
    print(f"  Spread, best to worst AUC : {best.auc - worst.auc:.4f}")
    print(f"  Base rate (validation)    : {y_val.mean():.4%}")
    print()
    print("Written to")
    print("-" * 78)
    print("  outputs/tables/ice2_model_comparison.csv")
    print("  outputs/figures/ice2_roc_curves.png")
    print("  outputs/figures/ice2_calibration.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
