# Probability of default and IFRS 9 expected credit loss

A reproducible comparison of logistic regression against four tree-ensemble families for
probability-of-default estimation, and the translation of those probabilities into IFRS 9
Expected Credit Loss provisions.

The study is deliberately built around three questions the applied literature tends to skip:
whether the metrics used to choose a model answer the question an impairment charge actually
asks, what honest look-ahead control costs on a term product, and whether macroeconomic
features behave as conditioning or as something else wearing its clothes.

Postgraduate Diploma in Data Analytics research project, The Independent Institute of
Education. Supervisor: Prof. Meera K. Joseph. The accompanying report is under examination;
this repository contains the code, configuration and results tables behind it.

---

## What the study found

**Machine learning beats logistic regression by less than the literature implies.** Across
three rolling-origin folds on 618,584 thirty-six-month loans, the best gradient-boosting family
reaches a mean AUC of 0.6830 against logistic regression's 0.6758. All three boosting
implementations cluster within 0.0025 AUC of one another, and the fold-level bootstrap intervals
overlap in every fold. Random Forest finishes last on all three discrimination measures and
worst on calibration.

**Loan-level macroeconomic conditioning shifts the level, it does not improve the ranking.**
Adding six United States or six South African macroeconomic series changes mean AUC by at most
0.0033, and that change is negative. The same features move the predicted *level* by a
near-constant amount, positive in 14 of 15 model-by-fold cells. Macroeconomic variables are
near-constant within an origination cohort by construction, so they cannot separate two
borrowers originated on the same day. This is loan-level empirical support for the
portfolio-level scalar architecture the IFRS 9 literature already adopts.

**Honest look-ahead control costs 91.3% of the training data and inverts the model ranking.**
A 36-month loan's outcome is not observable until 36 months after origination. Imposing an
embargo that removes training rows whose outcome was unknown at the cut-off reduces the largest
training cohort from 335,558 rows to 29,096, costs every family 0.06 to 0.08 AUC, and moves
Random Forest from last place to first. The ranking of model families is a property of the
training regime, not of the models.

**An apparent macroeconomic gain turned out to be a clock.** Under that embargo the South
African panel appeared to add about +0.006 AUC to every tree model. A placebo carrying nothing
but a count of months since the first origination reproduces the gain and exceeds it for
XGBoost and CatBoost, while logistic regression gains nothing from any arm. Applied work
reporting a small macroeconomic AUC gain without a temporal placebo cannot distinguish
conditioning from a date stamp.

**A univariate leakage screen cannot catch leakage carried in a ratio.** The register excludes
the contractual rate, grade and sub-grade as platform risk assessment. But `installment` and
`loan_amnt` are both admitted, and on a fixed-term sample their ratio is a monotone function of
the excluded rate. Each is individually indistinguishable from random — univariate AUC 0.5019
and 0.5156, far below the 0.90 screening threshold — while the ratio reaches **0.6558**.
Withholding the instalment costs between 0.0014 and 0.0053 AUC and does not change the ranking,
so the exclusion holds in substance; but screening features one at a time is structurally blind
to this class of leakage.

**The lender's own price is a harder benchmark than logistic regression.** That same implied
rate achieves 0.6558 on its own, against 0.6830 for the best model on 97 features. The
incremental discriminatory value of the entire modelling exercise over the price already being
charged is **0.0272 AUC**.

**Every model under-provisions.** Translating the probabilities into Expected Credit Loss at
45% loss given default, over a contractual annuity exposure schedule discounted at each loan's
own rate, every family provisions below realised loss — by 4.80% for XGBoost to 8.91% for
logistic regression on $6.94bn of exposure. Model choice moves the size of the shortfall, not
its direction. The spread across families is $8,595,240, or 4.52% of the smallest provision.
Expected Calibration Error and provisioning error disagree about which model is worst, because
the first is unsigned and unweighted while the second is signed and exposure-weighted.

---

## Reproducing the results

Python 3.14. From a clean checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The raw data is not redistributed here (see Data below). Place the Lending Club accepted-loans
CSV under `data/raw/`, then run the pipeline in order:

```bash
python -m scripts.gate_g1               # sample construction and the leakage screen
python -m scripts.fetch_macro           # macroeconomic panels from FRED, cached locally
python -m scripts.run_experiment        # the 45-cell grid
python -m scripts.run_experiment --embargo 36          # look-ahead robustness
python -m scripts.diagnose_macro_clock                 # the temporal placebo
python -m scripts.run_experiment --drop installment --tag no_installment
python -m scripts.ecl_translation       # IFRS 9 ECL from the fitted probabilities
python -m scripts.build_figures         # all six figures
```

Every run reads its seed, partition boundaries, term scope and macroeconomic series definitions
from `config/config.yaml`, and writes a dated artefact to `outputs/`. Results are deterministic
under the configured seed.

```bash
pytest          # the test suite
ruff check .    # lint
```

---

## Layout

| Path | Contents |
|---|---|
| `src/data/` | Leakage register and classification, rolling-origin partitioning, macroeconomic panel construction with publication-lag alignment, retrieval provenance |
| `src/features/` | Feature block assembly (A, B, C) and the fit-on-train preprocessing state |
| `src/evaluation/` | Discrimination and calibration metrics, bootstrap intervals |
| `src/ecl/` | Annuity amortisation and the IFRS 9 ECL provisioning computation |
| `src/utils/` | Configuration loading, path resolution, seeding |
| `scripts/` | Executable entry points, one per pipeline stage |
| `tests/` | Unit tests, including the ones that pin the fast paths against reference implementations |
| `config/` | The single configuration file and the leakage register |
| `outputs/tables/` | Results tables produced by the scripts |
| `docs/figures/` | The frozen figures as cited by the report |

Two design decisions are worth knowing before reading the code.

**Leakage control is explicit and two-layered.** A YAML register names 39 excluded fields plus
glob patterns, and a univariate screen flags any admitted numeric feature exceeding 0.90 AUC
against the outcome. Of 151 raw columns, 44 are classified post-origination, 6 identifier and
3 operational. The limits of that approach are documented above and in the report.

**The fast paths are pinned against reference implementations.** `src/ecl/provision.py` computes
a vectorised amortisation schedule because a 280,000-row cohort cannot be valued in a Python
loop, and `tests/test_provision.py` checks it against the scalar `outstanding_balance` in
`src/ecl/amortisation.py`, which remains the reference. Where the two disagree, the fast path is
wrong.

---

## Data

The modelling data is the publicly available Lending Club accepted-loans dataset, used under the
terms of its Kaggle distribution. It is **not** redistributed in this repository; `data/` is
ignored by git apart from directory placeholders.

Macroeconomic series are retrieved from the Federal Reserve Economic Database and cached under
`data/external/macro/`. Each series is aligned to loan origination dates under a publication-lag
rule, so no value enters a design matrix before the date on which it was actually released.
The South African policy rate is represented by the interbank call rate (`IRSTCI01ZAM156N`)
rather than the SARB repurchase rate, which is not available as a machine-readable series for
the full window; the substitution is named honestly in the configuration and the report.

The portfolio is a United States retail book, used as a transferability proxy. The components
examined — the comparison protocol, the discrimination and calibration framework, the ECL
translation and the stress-testing approach — transfer to the corporate domain; borrower-level
feature semantics do not, and no claim is made that they do.

---

## Status and caveats

This is coursework under examination, not production code, and three things are outstanding.
Hyperparameters are fixed and untuned, so the comparison measures relative behaviour under an
equal budget rather than each family's ceiling. No paired significance testing has been run, so
no AUC difference reported here should be treated as established. And discrimination is measured
over twelve-month origination cohorts, which the placebo shows rewards any feature that orders
loans by date; a month-by-month evaluation is the planned correction.

---

## Author

Kamogelo Samkelo Sabela (ST10517781) · [github.com/SamSabela](https://github.com/SamSabela)
