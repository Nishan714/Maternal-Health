\# Methodology



\## Problem



The study investigates machine-learning-based classification of maternal health risk using clinical and demographic attributes.



\## Target



The historical implementation encodes the `Risk Level` target into three classes:



\- High

\- Low

\- Unknown



The Unknown class corresponds to missing target labels in the supplied dataset.



\## Data split



The notebook uses:



\- 80% training data

\- 20% held-out test data

\- stratified splitting

\- `random\_state=42`



The resulting test set contains 241 observations.



\## Preprocessing



The historical notebook uses:



\- numerical imputation

\- standardization

\- categorical preprocessing where applicable

\- SMOTE inside the cross-validation pipeline



An important documentation discrepancy exists: the paper describes numerical mean imputation, while the historical notebook uses median imputation.



The notebook is preserved as the historical implementation rather than modified retrospectively.



\## Model comparison



Ten classifiers are evaluated using stratified five-fold cross-validation and weighted F1-score.



Random Forest achieves the highest mean weighted F1-score by a very small margin over Gradient Boosting.



\## Statistical comparison



The notebook performs a paired t-test comparing Random Forest and Gradient Boosting fold-level scores.



The reported result is:



`t = 0.1977, p = 0.8529, df = 4`



This does not indicate a statistically significant difference.



\## Final evaluation



The selected Random Forest model is fitted on the training set and evaluated on the held-out test set.



Reported test accuracy is 97.93%, with weighted F1-score of 97.52%.



Because the Unknown class contains only three test observations, class-specific interpretation is important.

