# Statistical Comparison

The preserved historical notebook compared Random Forest and Gradient
Boosting using paired fold-level weighted F1 scores from the same
5-fold stratified cross-validation procedure.

- Random Forest mean weighted F1: 0.9598
- Gradient Boosting mean weighted F1: 0.9594
- Paired t-statistic: 0.1977
- Degrees of freedom: 4
- p-value: 0.8529

The difference was not statistically significant. The published study
therefore selected Random Forest based on its competitive performance,
stability, and interpretability rather than claiming statistically
significant superiority over Gradient Boosting.

Final held-out test-set results for the selected Random Forest:

- Accuracy: 0.9793
- Weighted precision: 0.9713
- Weighted recall: 0.9793
- Weighted F1: 0.9752
- Weighted one-vs-rest ROC-AUC: 0.9954

The historical notebook also shows that the Unknown class had only
3 samples in the held-out test set and received zero precision, recall,
and F1. This is an important limitation of the experiment.
