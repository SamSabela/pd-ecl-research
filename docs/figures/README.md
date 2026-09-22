# Figures for the research report

These are the frozen figures as submitted. They are copies, not the live
renders: `outputs/figures/` is deliberately untracked because it is rewritten
on every run, and a submitted figure must not be able to drift from the report
that cites it.

Regenerate the live versions with:

    python -m scripts.build_figures

Figure 5 additionally requires `outputs/tables/macro_clock_diagnostic_embargo36.csv`,
produced by `python -m scripts.diagnose_macro_clock`.

| File | Figure | Shows |
|---|---|---|
| `fig01_roc_by_fold.png` | Figure 1 | ROC curves by model family, one panel per rolling-origin fold |
| `fig02_reliability_by_fold.png` | Figure 2 | Reliability curves by model family, one panel per fold |
| `fig03_level_error_by_block.png` | Figure 3 | Mean predicted minus actual default rate, by feature block and fold |
| `fig04_within_year_variation.png` | Figure 4 | Within-year variance retention: borrower features against both macroeconomic panels |
| `fig05_clock_placebo.png` | Figure 5 | The placebo test - an origination-month counter against the South African panel, embargo 36 |

Figures 3 to 5 carry the argument of Research Objective 2, and they have to be
read in that order.

Figure 3 shows that the block B effect is a near-uniform positive shift in
predicted level rather than macroeconomic conditioning, positive in 14 of 15
cells.

Figure 4 shows why no macroeconomic series can do more than that. Borrower
features retain 93-100% of their variation inside a single evaluation year;
every macroeconomic series retains 63% or less, falling to 10% for the US
federal funds rate. A series that is near-constant within the year cannot
separate two borrowers originated on the same day. It can still order loans by
origination date, which is what Figure 5 tests. The figure also shows the
South African panel retaining more within-year variation than the US panel at
almost every rank - a difference in temporal resolution, not in economic
relevance to a US loan book.

Figure 5 settles it. Under a 36-month embargo the South African panel appears
to add about +0.006 AUC to every tree model. A bare count of months since the
first origination in the sample - carrying no economic content at all -
reproduces that gain and, for XGBoost and CatBoost, exceeds it. Logistic
regression gains nothing from any arm, which is the signature of a temporal
proxy rather than a macroeconomic relationship. The apparent South African
effect is a clock and must not be reported as conditioning.
