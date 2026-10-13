# Probability of Default and IFRS 9 Expected Credit Loss

This repository contains the code, configuration and results tables for a comparative study of
logistic regression and four tree-ensemble families in the estimation of the Probability of
Default (PD), together with the translation of those probabilities into Expected Credit Loss
(ECL) provisions under International Financial Reporting Standard 9 (IFRS 9).

The study is organised around three questions that the applied credit-risk literature tends to
leave unexamined. The first is whether the metrics conventionally used to select a PD model
answer the question that an impairment charge actually poses. The second is what a defensible
look-ahead control costs on a fixed-term product. The third is whether macroeconomic variables,
when supplied as loan-level features, function as conditioning information or as something else
operating in its place.

Research project for the Postgraduate Diploma in Data Analytics, The Independent Institute of
Education. Supervisor: Prof. Meera K. Joseph. The accompanying research report is under
examination; this repository holds the code, configuration and results tables behind it.

---

## Findings

**The margin by which machine learning outperforms logistic regression is narrower than the
benchmark literature implies.** Across three rolling-origin folds on 618,584 thirty-six-month
loans, the strongest gradient-boosting family attains a mean AUC of 0.6830 against 0.6758 for
logistic regression. All three boosting implementations fall within 0.0025 AUC of one another,
and the fold-level bootstrap intervals overlap in every fold. Random Forest ranks last on all
three discrimination measures and worst on calibration.

**Loan-level macroeconomic conditioning shifts the predicted level without improving the
ranking.** The addition of six United States or six South African macroeconomic series changes
mean AUC by at most 0.0033, and that change is negative. The same features move the predicted
level by a near-constant amount, positive in 14 of the 15 model-by-fold cells. Macroeconomic
variables are near-constant within an origination cohort by construction, and they therefore
cannot separate two borrowers originated on the same day. This constitutes loan-level empirical
support for the portfolio-level scalar architecture that the IFRS 9 literature has already
adopted.

**A defensible look-ahead control removes 91.3% of the training data and inverts the model
ranking.** The outcome of a thirty-six-month loan is not observable until thirty-six months
after origination. An embargo that withholds training rows whose outcome was unknown at the
training cut-off reduces the largest training cohort from 335,558 rows to 29,096, costs every
family between 0.06 and 0.08 AUC, and moves Random Forest from last position to first. The
ranking of model families is therefore a property of the training regime rather than of the
models themselves.

**An apparent macroeconomic gain proved on inspection to be a temporal proxy.** Under that
embargo the South African panel appeared to contribute approximately +0.006 AUC to every tree
model. A placebo carrying nothing other than a count of months since the first origination
reproduces that gain and exceeds it for XGBoost and CatBoost, while logistic regression gains
nothing from any arm. Applied work that reports a small macroeconomic AUC improvement without a
temporal placebo cannot distinguish genuine conditioning from a date stamp.

**A univariate leakage screen cannot detect leakage that is carried in a ratio.** The leakage
register excludes the contractual interest rate, the grade and the sub-grade as platform risk
assessment. Both `installment` and `loan_amnt` are nonetheless admitted, and on a fixed-term
sample their ratio is a monotone function of the excluded rate. Each is individually
indistinguishable from random, with univariate AUC of 0.5019 and 0.5156 respectively, far below
the 0.90 screening threshold, whereas the ratio attains 0.6558. Withholding the instalment costs
between 0.0014 and 0.0053 AUC and does not alter the ranking, so the exclusion holds in
substance; screening features one at a time is nevertheless structurally blind to this class of
leakage.

**The lender's own price constitutes a more demanding benchmark than logistic regression.** That
same implied rate attains 0.6558 unaided, against 0.6830 for the best model fitted on 97
features. The incremental discriminatory value of the entire modelling exercise, measured against
the price the lender was already charging, is therefore 0.0272 AUC.

**Every model family under-provisions relative to realised loss.** Translating the fitted
probabilities into Expected Credit Loss at 45% loss given default, over a contractual annuity
exposure schedule discounted at each loan's own rate, every family provisions below realised
loss, by 4.80% for XGBoost through to 8.91% for logistic regression on $6.94 billion of exposure.
Model choice moves the magnitude of the shortfall rather than its direction. The spread across
families is $8,595,240, or 4.52% of the smallest provision. Expected Calibration Error and
provisioning error disagree as to which model performs worst, because the first is unsigned and
unweighted while the second is signed and exposure-weighted.

---

## Reproducing the results

Python 3.14. From a clean checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The raw data is not redistributed here, for the reasons set out under Data below. Place the
Lending Club accepted-loans CSV under `data/raw/`, then run the pipeline in the following order:

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

Each run reads its seed, partition boundaries, term scope and macroeconomic series definitions
from `config/config.yaml`, and writes a dated artefact to `outputs/`. Results are deterministic
under the configured seed.

```bash
pytest          # the test suite
ruff check .    # lint
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | Leakage register and classification, rolling-origin partitioning, macroeconomic panel construction with publication-lag alignment, retrieval provenance |
| `src/features/` | Feature block assembly (A, B, C) and the fit-on-train preprocessing state |
| `src/evaluation/` | Discrimination and calibration metrics, bootstrap intervals |
| `src/ecl/` | Annuity amortisation and the IFRS 9 ECL provisioning computation |
| `src/utils/` | Configuration loading, path resolution, seeding |
| `scripts/` | Executable entry points, one for each pipeline stage |
| `tests/` | Unit tests, including those that pin the fast paths against reference implementations |
| `config/` | The single configuration file and the leakage register |
| `outputs/tables/` | Results tables produced by the scripts |
| `docs/figures/` | The frozen figures as cited by the report |

Two design decisions are worth noting before reading the code.

**Leakage control is explicit and operates at two levels.** A YAML register names 39 excluded
fields together with glob patterns, and a univariate screen flags any admitted numeric feature
exceeding 0.90 AUC against the outcome. Of 151 raw columns, 44 are classified as
post-origination, 6 as identifier and 3 as operational. The limitations of that approach are
documented in the findings above and in the report.

**The fast paths are pinned against reference implementations.** `src/ecl/provision.py` computes
a vectorised amortisation schedule, because a cohort of 280,000 rows cannot be valued in a Python
loop, and `tests/test_provision.py` checks it against the scalar `outstanding_balance` function
in `src/ecl/amortisation.py`, which remains the reference. Where the two disagree, the fast path
is taken to be wrong.

---

## Data

The modelling data is the publicly available Lending Club accepted-loans dataset, used under the
terms of its Kaggle distribution. It is not redistributed in this repository, and `data/` is
ignored by git apart from directory placeholders.

Macroeconomic series are retrieved from the Federal Reserve Economic Database and cached under
`data/external/macro/`. Each series is aligned to loan origination dates under a publication-lag
rule, so that no value enters a design matrix before the date on which it was actually released.
The South African policy rate is represented by the interbank call rate (`IRSTCI01ZAM156N`)
rather than by the SARB repurchase rate, which is not available as a machine-readable series for
the full observation window; the substitution is recorded in the configuration and disclosed in
the report.

The portfolio is a United States retail book, employed as a transferability proxy. The components
examined, namely the comparison protocol, the discrimination and calibration framework, the ECL
translation and the stress-testing approach, transfer to the corporate domain; borrower-level
feature semantics do not, and no claim is made to the contrary.

---

## Status and limitations

This is coursework under examination rather than production code, and three matters remain
outstanding. First, hyperparameters are fixed and untuned, so the comparison measures relative
behaviour under an equal budget rather than the ceiling attainable by each family. Second, no
paired significance testing has yet been conducted, and no AUC difference reported here should
therefore be treated as established. Third, discrimination is measured over twelve-month
origination cohorts, which the placebo shows to reward any feature that orders loans by date; a
month-by-month evaluation is the planned correction.

---

## Author

Kamogelo Samkelo Sabela (ST10517781), [github.com/SamSabela](https://github.com/SamSabela)
