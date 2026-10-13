# Figures for the research report

These are the frozen figures as submitted. They are copies rather than the live
renders, because `outputs/figures/` is deliberately untracked, being rewritten on
every run, and a submitted figure must not be able to drift from the report that
cites it.

The live versions are regenerated with:

    python -m scripts.build_figures

Figure 5 additionally requires `outputs/tables/macro_clock_diagnostic_embargo36.csv`,
which is produced by `python -m scripts.diagnose_macro_clock`. Figure 6 requires
`outputs/tables/ecl_translation.csv`, produced by `python -m scripts.ecl_translation`.

| File | Figure | Shows |
|---|---|---|
| `fig01_roc_by_fold.png` | Figure 1 | ROC curves by model family, one panel per rolling-origin fold |
| `fig02_reliability_by_fold.png` | Figure 2 | Reliability curves by model family, one panel per fold |
| `fig03_level_error_by_block.png` | Figure 3 | Mean predicted minus actual default rate, by feature block and fold |
| `fig04_within_year_variation.png` | Figure 4 | Within-year variance retention, borrower features against both macroeconomic panels |
| `fig05_clock_placebo.png` | Figure 5 | The placebo test, an origination-month counter against the South African panel, embargo 36 |
| `fig06_provisioning_error.png` | Figure 6 | Provisioning error by fold and model family, block A, LGD 45% |

## How the figures carry the argument

Figures 3 to 5 carry the argument of Research Objective 2, and they are intended
to be read in that order.

Figure 3 establishes that the block B effect is a near-uniform positive shift in
predicted level rather than macroeconomic conditioning, being positive in 14 of
the 15 model-by-fold cells.

Figure 4 explains why no macroeconomic series can do more than that. Borrower
features retain between 93% and 100% of their variation within a single
evaluation year, whereas every macroeconomic series retains 63% or less, falling
to 10% for the United States federal funds rate. A series that is near-constant
within the year cannot separate two borrowers originated on the same day,
although it can still order loans by origination date, which is the possibility
Figure 5 tests. The figure also shows the South African panel retaining more
within-year variation than the United States panel at almost every rank, which
is a difference in temporal resolution rather than in economic relevance to a
United States loan book.

Figure 5 settles the question. Under a 36-month embargo the South African panel
appears to contribute approximately +0.006 AUC to every tree model. A bare count
of months since the first origination in the sample, carrying no economic content
whatsoever, reproduces that gain and, for XGBoost and CatBoost, exceeds it.
Logistic regression gains nothing from any arm, which is the signature of a
temporal proxy rather than of a macroeconomic relationship. The apparent South
African effect is therefore a clock, and it must not be reported as conditioning.

Figure 6 extends the argument to the provisioning outcome. It deliberately reuses
the visual grammar of Figure 3, being a dot plot of signed error by fold with a
zero rule, so that the two can be read against one another. Every family
under-provisions in 2014 and 2015 and over-provides in 2013. Model choice moves
the magnitude of the shortfall rather than its direction, and the sign reverses
between folds exactly as Figure 3 shows.
