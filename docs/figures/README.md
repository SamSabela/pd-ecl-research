# Figures for the research report

These are the frozen figures as submitted. They are copies, not the live
renders: `outputs/figures/` is deliberately untracked because it is rewritten
on every run, and a submitted figure must not be able to drift from the report
that cites it.

Regenerate the live versions with:

    python -m scripts.build_figures

| File | Figure | Shows |
|---|---|---|
| `fig01_roc_by_fold.png` | Figure 1 | ROC curves by model family, one panel per rolling-origin fold |
| `fig02_reliability_by_fold.png` | Figure 2 | Reliability curves by model family, one panel per fold |
| `fig03_level_error_by_block.png` | Figure 3 | Mean predicted minus actual default rate, by feature block and fold |
| `fig04_within_year_variation.png` | Figure 4 | Within-year variance retention, borrower features against macroeconomic series |

Figure 3 carries the central result of Research Objective 2: the block B effect
is a near-uniform positive shift in predicted level, not macroeconomic
conditioning, and it is positive in 14 of 15 cells.
