"""Discrimination and calibration metrics.

Implements Section 3.8.2 and Table 3.7.

Discrimination is measured by AUC, the Gini coefficient and the
Kolmogorov-Smirnov statistic - the standard measures across the benchmark
credit-scoring literature and in practitioner validation reports.

Calibration is measured separately, by the Brier score, its three-way
decomposition, and Expected Calibration Error. Keeping the two apart matters
for this study specifically: ECL depends on probability MAGNITUDES, not only on
ranking. A model that ranks obligors perfectly but predicts 5% when the truth
is 15% will produce provisions that are wrong by a factor of three while
scoring an excellent AUC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = ["ModelMetrics", "compute_metrics", "bootstrap_auc_ci", "metrics_table"]


@dataclass(frozen=True)
class ModelMetrics:
    """Discrimination and calibration for one model on one partition."""

    model: str
    partition: str
    n: int
    n_default: int
    base_rate: float
    predicted_rate: float
    auc: float
    gini: float
    ks: float
    brier: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    ece: float
    fit_seconds: float = 0.0


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity on ranks.

    Handles ties correctly and needs no sklearn dependency for what is
    fundamentally a rank statistic.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ks(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max vertical gap between the two CDFs.

    The largest separation achievable between defaulters and non-defaulters at
    any single cut-off, which is why practitioners use it alongside AUC.
    """
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = np.cumsum(y_sorted == 1) / n_pos
    cum_neg = np.cumsum(y_sorted == 0) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def _brier_decomposition(
    y: np.ndarray, p: np.ndarray, bins: int = 10
) -> tuple[float, float, float, float]:
    """Brier score and its reliability-resolution-uncertainty decomposition.

        Brier = reliability - resolution + uncertainty

    reliability  how far predicted probabilities sit from observed frequencies
                 within each bin. LOWER is better; this is calibration.
    resolution   how far bin frequencies sit from the overall base rate.
                 HIGHER is better; this is discrimination.
    uncertainty  the irreducible variance of the outcome itself, base(1-base).
                 Identical for every model on the same data.

    The decomposition is what lets you say WHY one model's Brier beat another's
    rather than only that it did.
    """
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    uncertainty = base * (1.0 - base)

    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.digitize(p, edges[1:-1], right=False)

    reliability = resolution = 0.0
    n = len(y)
    for b in range(bins):
        mask = index == b
        size = int(mask.sum())
        if size == 0:
            continue
        observed = float(np.mean(y[mask]))
        predicted = float(np.mean(p[mask]))
        reliability += size * (predicted - observed) ** 2
        resolution += size * (observed - base) ** 2
    return brier, reliability / n, resolution / n, uncertainty


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error: mean absolute gap, weighted by bin size.

    Summarises the reliability diagram as a single number, so calibration can
    be tabulated alongside AUC rather than only plotted.
    """
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.digitize(p, edges[1:-1], right=False)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return total / len(y)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model: str,
    partition: str,
    fit_seconds: float = 0.0,
) -> ModelMetrics:
    """Compute the full metric set for one model on one partition.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities of default.
        model: Model family name.
        partition: Partition name.
        fit_seconds: Training wall-clock time, for the computational-cost note.

    Returns:
        A ModelMetrics record.
    """
    y = np.asarray(y_true).astype(np.int8)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1 - 1e-12)

    auc = _auc(y, p)
    brier, reliability, resolution, uncertainty = _brier_decomposition(y, p)

    return ModelMetrics(
        model=model,
        partition=partition,
        n=len(y),
        n_default=int(y.sum()),
        base_rate=float(y.mean()),
        predicted_rate=float(p.mean()),
        auc=auc,
        gini=2.0 * auc - 1.0,
        ks=_ks(y, p),
        brier=brier,
        brier_reliability=reliability,
        brier_resolution=resolution,
        brier_uncertainty=uncertainty,
        ece=_ece(y, p),
        fit_seconds=fit_seconds,
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    replicates: int = 200,
    alpha: float = 0.05,
    seed: int = 20260815,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for AUC.

    Reported so that differences between models can be judged against sampling
    noise rather than compared as point estimates. On a validation partition of
    this size the interval is narrow, which is itself worth stating: small AUC
    gaps are real, and the question then becomes whether they are MATERIAL, not
    whether they are significant.

    The full study uses the DeLong test for paired AUC comparisons and 2,000
    stratified paired bootstrap replicates (Section 3.8.2). This is the
    preliminary version.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        replicates: Bootstrap replicates.
        alpha: Two-sided significance level.
        seed: Seed for reproducibility.

    Returns:
        (lower, upper) bounds of the interval.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    n = len(y)

    scores = np.empty(replicates)
    for i in range(replicates):
        idx = rng.integers(0, n, n)
        scores[i] = _auc(y[idx], p[idx])
    return (
        float(np.nanquantile(scores, alpha / 2)),
        float(np.nanquantile(scores, 1 - alpha / 2)),
    )


def metrics_table(records: list[ModelMetrics]) -> pd.DataFrame:
    """Assemble metric records into a table for the report."""
    return pd.DataFrame([asdict(r) for r in records])
