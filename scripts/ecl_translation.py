"""Translate model PD estimates into IFRS 9 ECL (RO2, RQ2).

Implements ICE Task 1, Tables 3.8 and 3.9. The computation itself lives in
src.ecl.provision; this script refits the experiment grid so that the ECL
figures come from exactly the same fits as the discrimination figures in
scripts.run_experiment, and writes a tidy table.

What the output answers

    RQ2 asks what differences in PD estimates across model types do to
    portfolio provisioning. The comparison is constructed so that the only
    quantity varying across models is the PD level: the exposure path, the
    discounting and the default-timing weights are identical across models
    within a fold. The gap between provisioned and realised loss is therefore
    the calibration error of the discussion, expressed in currency.

Run:
    python -m scripts.ecl_translation
    python -m scripts.ecl_translation --blocks A
    python -m scripts.ecl_translation --embargo 36
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone

from scripts.run_experiment import build_models, candidate_features
from src.data.partition import rolling_origin_folds
from src.ecl.provision import (
    assign_stage,
    discounted_exposure_at_risk,
    expected_credit_loss,
    implied_annual_rate,
    timing_weights,
    twelve_month_pd,
)
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

# ICE 1 Table 3.8: absolute backstop levels for the Stage 2 trigger.
BACKSTOPS = (0.03, 0.05, 0.08)
# ICE 1 Table 3.9: LGD base case and sensitivities.
LGD_LEVELS = (0.30, 0.45, 0.60)
BASE_LGD = 0.45


def effective_interest_rate(frame: pd.DataFrame, term_months: int) -> pd.Series:
    """Contractual annual rate per loan, as a decimal, indexed like the frame.

    ICE 1 Table 3.9 requires the EIR for discounting and amortisation while
    excluding it from the feature set. Gate G1 drops the column entirely, which
    conflates the two: a column models may not see is not the same as a column
    the accounting does not need.

    Where int_rate survives it is used directly. Otherwise the rate is
    recovered exactly from the level instalment, since the annuity identity is
    strictly increasing in the rate and therefore invertible. Computed once for
    the whole sample rather than per fitted cell, because a bisection across
    600,000 rows is not something to repeat 45 times.
    """
    if "int_rate" in frame.columns:
        rate = pd.to_numeric(frame["int_rate"], errors="coerce")
        if rate.median() > 1.0:          # carried as a percentage
            rate = rate / 100.0
        return rate.fillna(rate.median())

    if "installment" not in frame.columns:
        raise SystemExit(
            "neither int_rate nor installment is present in the analysis "
            "sample, so the contractual rate required by ICE 1 Table 3.9 "
            "cannot be recovered. Re-run scripts.gate_g1 retaining int_rate."
        )

    principal = frame["loan_amnt"].to_numpy(dtype=float)
    instalment = pd.to_numeric(frame["installment"], errors="coerce")
    instalment = instalment.fillna(instalment.median()).to_numpy(dtype=float)
    recovered = implied_annual_rate(principal, instalment, term_months)

    # Lending Club's published floor is 5.32%. A materially lower recovered
    # rate means instalment and principal disagree for that row - a missing
    # instalment, or a loan_amnt that differs from the amount the instalment
    # was priced on. Report it rather than discounting at a rate that cannot
    # be real.
    implausible = int((recovered < 0.04).sum())
    if implausible:
        share = implausible / len(recovered)
        print(f"  WARNING: {implausible:,} rows ({share:.3%}) recovered an "
              f"annual rate below 4%, which is under Lending Club's published "
              f"floor. Their instalment and principal are inconsistent.")
    return pd.Series(recovered, index=frame.index)


def cell_record(
    evaluate: pd.DataFrame,
    probabilities: np.ndarray,
    actual: np.ndarray,
    term: int,
    weights: np.ndarray,
    eir: pd.Series,
) -> dict:
    """Provisioned and realised loss for one fitted cell."""
    principal = evaluate["loan_amnt"].to_numpy(dtype=float)
    rates = eir.loc[evaluate.index].to_numpy(dtype=float)

    # Identical across models within a fold, so model differences in the
    # provision are attributable to the PD level alone.
    dear = discounted_exposure_at_risk(principal, rates, term, weights)
    pd_12 = twelve_month_pd(probabilities, term)

    record = {
        "n_evaluate": len(evaluate),
        "exposure": float(principal.sum()),
        "mean_pd_term": float(probabilities.mean()),
        "mean_pd_12m": float(pd_12.mean()),
        "actual_default_rate": float(actual.mean()),
    }

    for lgd in LGD_LEVELS:
        tag = f"{int(round(lgd * 100))}"
        lifetime = float(expected_credit_loss(
            probabilities, principal, rates, term, lgd).sum())
        realised = float((actual * lgd * dear).sum())
        record[f"ecl_lifetime_lgd{tag}"] = lifetime
        record[f"loss_realised_lgd{tag}"] = realised
        record[f"ecl_error_lgd{tag}"] = lifetime - realised
        record[f"ecl_error_pct_lgd{tag}"] = (
            (lifetime - realised) / realised * 100.0 if realised else float("nan")
        )

    # Staged variant at the base LGD, across the three ICE 1 backstops.
    twelve = expected_credit_loss(
        probabilities, principal, rates, term, BASE_LGD, horizon_months=12)
    lifetime_each = expected_credit_loss(
        probabilities, principal, rates, term, BASE_LGD)
    for backstop in BACKSTOPS:
        stage = assign_stage(pd_12, backstop)
        staged = np.where(stage == 2, lifetime_each, twelve)
        tag = f"{int(round(backstop * 100)):02d}"
        record[f"ecl_staged_bs{tag}"] = float(staged.sum())
        record[f"stage2_share_bs{tag}"] = float((stage == 2).mean())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PD estimates into IFRS 9 ECL")
    parser.add_argument("--blocks", default="A,B,C", help="Comma-separated blocks")
    parser.add_argument("--embargo", type=int, default=None, help="Override embargo months")
    args = parser.parse_args()

    config = load_config()
    register = load_leakage_register()
    seed = config["seed"]
    set_global_seed(seed)

    rolling = config["partition"]["rolling_origin"]
    embargo = args.embargo if args.embargo is not None else rolling["embargo_months"]
    boundaries = [(f["train_end"], f["evaluate_end"]) for f in rolling["folds"]]
    blocks = [b.strip().upper() for b in args.blocks.split(",") if b.strip()]
    cache_dir = resolve_path(config["macro"]["cache_dir"])

    sample_path = (
        resolve_path(config["paths"]["data_processed"]) / "analysis_sample_v1.parquet"
    )
    if not sample_path.is_file():
        raise SystemExit(f"analysis sample not found: {sample_path}")

    sample = pd.read_parquet(sample_path)
    term = config["sample"]["term_scope"]["primary_term_months"]
    sample = sample.loc[sample["term_months"] == term].copy()
    sample["issue_d"] = pd.to_datetime(sample["issue_d"])
    sample, engineered = engineer_features(sample)
    candidates = candidate_features(sample, register, engineered)
    models = build_models(seed)
    eir = effective_interest_rate(sample, term)
    source = "int_rate column" if "int_rate" in sample.columns else (
        "recovered from installment by inverting the annuity identity")

    print("=" * 78)
    print("ECL TRANSLATION - IFRS 9 three-stage impairment (RO2, RQ2)")
    print("=" * 78)
    print("  Specification   : ICE 1 Tables 3.8 and 3.9")
    print(f"  Seed            : {seed}")
    print(f"  Embargo         : {embargo} months")
    print(f"  Term            : {term} months")
    print(f"  LGD             : {', '.join(f'{v:.0%}' for v in LGD_LEVELS)}  (base 45%)")
    print("  EAD(t)          : contractual annuity amortisation schedule")
    print(f"  EIR source      : {source}")
    print(f"  EIR range       : {eir.min():.2%} to {eir.max():.2%}, "
          f"median {eir.median():.2%}")
    print("  Discounting     : loan contractual rate, (1+EIR)^(-t/12)")
    print("  PD(t)           : constant hazard from the term PD")
    print(f"  Stage 2 trigger : absolute backstop at "
          f"{', '.join(f'{v:.0%}' for v in BACKSTOPS)} of the 12-month PD")
    print("  Stage 3         : empty - no exposure is credit-impaired at origination")
    print()

    records: list[dict] = []
    for block in blocks:
        frame, macro_columns = assemble_block(sample, config, block, cache_dir=cache_dir)
        block_candidates = candidates + macro_columns
        result = rolling_origin_folds(frame, boundaries, embargo_months=embargo)

        print("-" * 78)
        print(f"BLOCK {block}")
        print("-" * 78)

        for fold in result.folds:
            train = frame.loc[fold.train]
            evaluate = frame.loc[fold.evaluate]

            state, _ = fit_preprocessor(
                train, block_candidates, PreprocessingConfig(), engineered
            )
            x_train = apply_preprocessor(train, state)
            x_eval = apply_preprocessor(evaluate, state)
            y_train = train["default_flag"].to_numpy()
            y_eval = evaluate["default_flag"].to_numpy()

            # Cohort timing, fixed across models within the fold.
            weights = timing_weights(term, float(y_eval.mean()))

            for model_name, prototype in models.items():
                model = clone(prototype)
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_eval)[:, 1]
                record = cell_record(
                    evaluate, probabilities, y_eval, term, weights, eir
                )
                record.update({"block": block, "fold": fold.name, "model": model_name})
                records.append(record)
                print(f"  {fold.name}  {model_name:<22} "
                      f"ECL {record['ecl_lifetime_lgd45']:>13,.0f}  "
                      f"realised {record['loss_realised_lgd45']:>13,.0f}  "
                      f"error {record['ecl_error_pct_lgd45']:>7.2f}%  "
                      f"S2 {record['stage2_share_bs05']:.1%}")
        print()

    results = pd.DataFrame(records)
    lead = ["block", "fold", "model"]
    results = results[lead + [c for c in results.columns if c not in lead]]

    tables = resolve_path(config["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    suffix = "" if embargo == rolling["embargo_months"] else f"_embargo{embargo}"
    out_path = tables / f"ecl_translation{suffix}.csv"
    results.to_csv(out_path, index=False)

    print("=" * 78)
    print("PORTFOLIO ECL BY MODEL, LIFETIME BASIS, LGD 45%, SUMMED ACROSS FOLDS")
    print("=" * 78)
    summary = results.groupby(["block", "model"]).agg(
        ecl=("ecl_lifetime_lgd45", "sum"),
        realised=("loss_realised_lgd45", "sum"),
        exposure=("exposure", "sum"),
    )
    summary["error"] = summary["ecl"] - summary["realised"]
    summary["error_pct"] = summary["error"] / summary["realised"] * 100.0
    summary["coverage_pct"] = summary["ecl"] / summary["exposure"] * 100.0
    print(summary.to_string(float_format=lambda v: f"{v:,.2f}"))
    print()

    if "A" in summary.index.get_level_values("block"):
        base = summary.loc["A"]
        spread = base["ecl"].max() - base["ecl"].min()
        print(f"  Spread in provisioned ECL across model families, block A: {spread:,.0f}")
        print(f"  As a share of the smallest provision: "
              f"{spread / base['ecl'].min() * 100:.2f}%")
        print()

    print(f"  {len(records)} cells written to {out_path.relative_to(resolve_path('.'))}")
    print("=" * 78)


if __name__ == "__main__":
    main()
