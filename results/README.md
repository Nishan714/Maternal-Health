# Verified Results

This directory contains result artifacts corresponding to the historical
experiment preserved in `notebooks/final_experiment.ipynb`.

The repository does **not** rerun the published experiment automatically.
The tables below were recorded from the historical notebook outputs used
for the published study.

## Main results

The selected Random Forest achieved:

| Metric | Test-set result |
|---|---:|
| Accuracy | 0.9793 |
| Weighted Precision | 0.9713 |
| Weighted Recall | 0.9793 |
| Weighted F1 | 0.9752 |
| Weighted OvR ROC-AUC | 0.9954 |

## Cross-validation

Random Forest achieved a mean weighted F1 of **0.9598 ± 0.0078**
over stratified 5-fold cross-validation.

Gradient Boosting achieved **0.9594 ± 0.0049**.

The paired comparison was not statistically significant
(t = 0.1977, p = 0.8529, df = 4).

## Important limitation

The `Unknown` target class contained only 3 samples in the held-out
test set and received zero precision, recall, and F1. Therefore the
high overall weighted metrics should not be interpreted as evidence
that all target classes are reliably detected.

## Historical implementation note

The published paper describes mean imputation for numerical variables,
whereas the preserved notebook uses median imputation. The historical
notebook is intentionally retained rather than silently rewritten.
This discrepancy is documented in the repository documentation.
