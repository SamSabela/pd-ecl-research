"""Produce the figures for the results and discussion chapters.

Four figures, numbered and named as the supervisor requires. Each answers one
question, and the order is the order the argument runs in:

    Figure 1  ROC curves by fold              do the families separate?
    Figure 2  Reliability diagrams by fold    do they predict the right level?
    Figure 3  Level error by fold and block   does macro conditioning help?
    Figure 4  Within-year feature variation   why AUC cannot answer Figure 3

Figures 1 and 2 need per-loan predicted probabilities, which the results table
does not carry, so block A is refitted here - fifteen fits, under a minute. The
seed and the fold definitions are identical to scripts.run_experiment, so the
AUCs printed here reproduce the ones in experiment_results.csv exactly. A
mismatch means something in the pipeline is not deterministic and is worth
stopping for.

Figures 3 and 4 read the results table and the sample respectively, and refit
nothing.

Run:
    python -m scripts.build_figures
    python -m scripts.build_figures --figures 3,4     skip the refit
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

# Validated categorical palette. Five model slots and three block slots, both
# checked with the data-viz validator for colour-vision separation. Three of
# these sit below 3:1 contrast on a white surface, so every series carries a
# visible label, and the model series also carry distinct dash patterns - the
# yellow/aqua pair separates poorly under tritanopia and colour alone would not
# be enough.
MODEL_COLOURS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
    "LightGBM": "#eda100",
    "CatBoost": "#e87ba4",
}
MODEL_DASHES = {
    "Logistic Regression": (None, None),
    "Random Forest": (6, 2),
    "XGBoost": (2, 2),
    "LightGBM": (8, 2, 2, 2),
    "CatBoost": (1, 2),
}
BLOCK_COLOURS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
BLOCK_MARKERS = {"A": "o", "B": "s", "C": "^"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def style(ax) -> None:
    """Recessive grid and axes, no chartjunk."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def roc_points(y: np.ndarray, p: np.ndarray, grid: int = 200) -> tuple:
    """False and true positive rates on a fixed grid of thresholds.

    Computed on a grid rather than at every distinct score, because with
    280,000 observations the full curve is a quarter of a million points and
    renders as a solid band.
    """
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    idx = np.linspace(0, len(tpr) - 1, grid).astype(int)
    return np.r_[0.0, fpr[idx]], np.r_[0.0, tpr[idx]]


def reliability_points(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple:
    """Mean predicted probability and observed frequency per decile of score."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-12
    which = np.digitize(p, edges[1:-1])
    mean_p, obs = [], []
    for b in range(bins):
        mask = which == b
        if mask.sum() >= 50:
            mean_p.append(p[mask].mean())
            obs.append(y[mask].mean())
    return np.array(mean_p), np.array(obs)


def fit_block_a(config: dict, register: dict) -> dict:
    """Refit block A across the folds, returning predictions per fold.

    Returns:
        {fold_name: {"y": array, "models": {name: (probabilities, auc)}}}
    """
    seed = config["seed"]
    set_global_seed(seed)

    rolling = config["partition"]["rolling_origin"]
    boundaries = [(f["train_end"], f["evaluate_end"]) for f in rolling["folds"]]

    path = resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    sample = pd.read_parquet(path)
    term = config["sample"]["term_scope"]["primary_term_months"]
    sample = sample.loc[sample["term_months"] == term].copy()
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])
    sample, engineered = engineer_features(sample)
    candidates = candidate_features(sample, register, engineered)

    frame, _ = assemble_block(sample, config, "A")
    result = rolling_origin_folds(
        frame, boundaries, embargo_months=rolling["embargo_months"]
    )
    models = build_models(seed)
    predictions: dict = {}

    for fold in result.folds:
        train = frame.loc[fold.train]
        evaluate = frame.loc[fold.evaluate]
        state, _ = fit_preprocessor(
            train, candidates, PreprocessingConfig(), engineered
        )
        x_train = apply_preprocessor(train, state)
        x_eval = apply_preprocessor(evaluate, state)
        y_train = train["default_flag"].to_numpy()
        y_eval = evaluate["default_flag"].to_numpy()

        entry: dict = {"y": y_eval, "models": {}}
        for name, prototype in models.items():
            model = clone(prototype)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_eval)[:, 1]
            metrics = compute_metrics(y_eval, probabilities, name, fold.name)
            entry["models"][name] = (probabilities, metrics.auc)
            print(f"    {fold.name}  {name:<22} AUC {metrics.auc:.4f}")
        predictions[fold.name] = entry

    return predictions


def figure_1(predictions: dict, path) -> None:
    """ROC curves by fold. Discrimination is a ranking question, so the form is
    a curve over the full threshold range rather than a single number."""
    folds = sorted(predictions)
    fig, axes = plt.subplots(1, len(folds), figsize=(4.0 * len(folds), 4.1), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, fold in zip(axes, folds, strict=True):
        style(ax)
        ax.plot([0, 1], [0, 1], color="#c9c8c3", linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        for name, (probabilities, auc) in predictions[fold]["models"].items():
            fpr, tpr = roc_points(predictions[fold]["y"], probabilities)
            ax.plot(
                fpr, tpr, color=MODEL_COLOURS[name], linewidth=2, zorder=3,
                dashes=MODEL_DASHES[name] if MODEL_DASHES[name][0] else (1, 0),
                label=f"{name}  ({auc:.4f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Fold {fold}", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("False positive rate", fontsize=9, color=MUTED)
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=MUTED)

    axes[0].set_ylabel("True positive rate", fontsize=9, color=MUTED)
    fig.suptitle(
        "Figure 1. ROC curves by evaluation fold, feature block A",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_2(predictions: dict, path) -> None:
    """Reliability diagrams by fold. Calibration is what Expected Credit Loss
    depends on, so it gets equal billing with discrimination rather than a
    footnote."""
    folds = sorted(predictions)
    fig, axes = plt.subplots(1, len(folds), figsize=(4.0 * len(folds), 4.1), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, fold in zip(axes, folds, strict=True):
        style(ax)
        y = predictions[fold]["y"]
        ax.plot([0, 0.4], [0, 0.4], color="#c9c8c3", linewidth=1,
                linestyle=(0, (4, 4)), zorder=1, label="Perfect calibration")
        ax.axhline(y.mean(), color="#c9c8c3", linewidth=1, zorder=1)
        for name, (probabilities, _) in predictions[fold]["models"].items():
            mean_p, observed = reliability_points(y, probabilities)
            ax.plot(mean_p, observed, color=MODEL_COLOURS[name], linewidth=1.8,
                    marker="o", markersize=4, zorder=3,
                    dashes=MODEL_DASHES[name] if MODEL_DASHES[name][0] else (1, 0),
                    label=name)
        top = max(0.36, y.mean() * 2.6)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(f"Fold {fold}   observed {y.mean():.2%}", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("Mean predicted probability", fontsize=9, color=MUTED)
        ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=MUTED)

    axes[0].set_ylabel("Observed default frequency", fontsize=9, color=MUTED)
    fig.suptitle(
        "Figure 2. Reliability diagrams by evaluation fold, feature block A",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_3(results: pd.DataFrame, path) -> None:
    """Predicted minus observed default rate, every model, every block.

    One dot per model rather than a mean, because the mean would hide that the
    blocks disagree in fold 2013 and agree in 2014 and 2015. A grouped bar of
    averages would have told a cleaner story than the data supports.
    """
    results = results.copy()
    results["level_error"] = results["predicted_rate"] - results["base_rate"]
    folds = sorted(results["fold"].unique())
    blocks = ["A", "B", "C"]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axvline(0, color=MUTED, linewidth=1.2, zorder=2)

    offsets = {"A": 0.22, "B": 0.0, "C": -0.22}
    for f_i, fold in enumerate(folds):
        for block in blocks:
            subset = results[(results["fold"] == fold) & (results["block"] == block)]
            ax.scatter(
                subset["level_error"], [f_i + offsets[block]] * len(subset),
                s=46, color=BLOCK_COLOURS[block], marker=BLOCK_MARKERS[block],
                edgecolor=SURFACE, linewidth=1.4, zorder=3,
                label=f"Block {block}" if f_i == 0 else None,
            )

    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"Fold {f}" for f in folds], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(folds) - 0.4)
    ax.set_xlabel(
        "Predicted minus observed default rate     "
        "(left = under-predicts, right = over-predicts)",
        fontsize=9, color=MUTED,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=MUTED, ncol=3)
    fig.suptitle(
        "Figure 3. Calibration level error by fold and feature block, five model families",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_4(config: dict, path) -> None:
    """How much of each feature's variation survives inside one evaluation year.

    This is the explanation for Figure 3's null discrimination result. AUC
    ranks loans within the evaluation set; a feature that is nearly constant
    across that set cannot change the ranking, however well it predicts the
    level.
    """
    sample_path = resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    sample = pd.read_parquet(sample_path)
    term = config["sample"]["term_scope"]["primary_term_months"]
    sample = sample.loc[sample["term_months"] == term].copy()
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])

    frame, macro_columns = assemble_block(sample, config, "B")
    frame["year"] = frame["issue_d"].dt.year

    borrower = [c for c in ("fico_range_low", "dti", "annual_inc", "loan_amnt", "open_acc")
                if c in frame.columns]

    rows = []
    for column, kind in [(c, "Macroeconomic") for c in macro_columns] + [
        (c, "Borrower") for c in borrower
    ]:
        overall = frame[column].std()
        within = frame.loc[frame["year"] == 2015, column].std()
        if overall and np.isfinite(overall):
            rows.append({"feature": column, "kind": kind, "retained": within / overall})

    table = pd.DataFrame(rows).sort_values("retained")

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(table) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)

    colours = {"Macroeconomic": "#eb6834", "Borrower": "#2a78d6"}
    positions = range(len(table))
    ax.barh(
        list(positions), table["retained"], height=0.62,
        color=[colours[k] for k in table["kind"]],
        edgecolor=SURFACE, linewidth=1.5, zorder=3,
    )
    for pos, value in enumerate(table["retained"]):
        ax.text(value + 0.015, pos, f"{value:.0%}", va="center", fontsize=8, color=MUTED)

    ax.axvline(1.0, color="#c9c8c3", linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(table["feature"], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(
        "Standard deviation within 2015 as a share of standard deviation across 2007-2015",
        fontsize=9, color=MUTED,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours["Borrower"]),
        plt.Rectangle((0, 0), 1, 1, color=colours["Macroeconomic"]),
    ]
    ax.legend(handles, ["Borrower feature", "Macroeconomic feature"],
              loc="lower right", fontsize=8, frameon=False, labelcolor=MUTED)
    fig.suptitle(
        "Figure 4. Variation surviving within a single evaluation year",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    """Build the requested figures."""
    parser = argparse.ArgumentParser(description="Build the results figures")
    parser.add_argument("--figures", default="1,2,3,4", help="Comma-separated figure numbers")
    args = parser.parse_args()
    wanted = {int(n) for n in args.figures.split(",") if n.strip()}

    config = load_config()
    register = load_leakage_register()
    figures = resolve_path(config["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    tables = resolve_path(config["paths"]["tables"])

    print("=" * 78)
    print("FIGURES")
    print("=" * 78)

    if wanted & {1, 2}:
        print("  Refitting block A for per-loan predictions...")
        predictions = fit_block_a(config, register)
        print()
        if 1 in wanted:
            figure_1(predictions, figures / "fig01_roc_by_fold.png")
            print("  Figure 1  fig01_roc_by_fold.png")
        if 2 in wanted:
            figure_2(predictions, figures / "fig02_reliability_by_fold.png")
            print("  Figure 2  fig02_reliability_by_fold.png")

    if 3 in wanted:
        results_path = tables / "experiment_results.csv"
        if not results_path.is_file():
            raise SystemExit(
                f"{results_path} not found. Run scripts.run_experiment first."
            )
        figure_3(pd.read_csv(results_path), figures / "fig03_level_error_by_block.png")
        print("  Figure 3  fig03_level_error_by_block.png")

    if 4 in wanted:
        figure_4(config, figures / "fig04_within_year_variation.png")
        print("  Figure 4  fig04_within_year_variation.png")

    print("=" * 78)


if __name__ == "__main__":
    main()
