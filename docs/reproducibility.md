\# Reproducibility



\## Historical implementation



The main implementation is:



`notebooks/final\_experiment.ipynb`



This notebook preserves the historical experiment associated with the published paper.



\## Environment



The requirements file specifies:



`scikit-learn==1.5.0`



along with the principal packages used by the notebook.



Exact reproduction should be performed in a clean environment using the documented dependency versions.



\## Randomness



The historical implementation uses `random\_state=42` for the main stochastic components, including:



\- train/test splitting

\- cross-validation

\- SMOTE

\- Random Forest

\- Gradient Boosting

\- AdaBoost

\- XGBoost

\- MLP



\## Historical paper/code discrepancy



The paper describes numerical mean imputation.



The historical notebook uses:



`SimpleImputer(strategy='median')`



This discrepancy is intentionally documented rather than silently corrected.



The repository therefore represents the historical implementation, not a newly reconstructed experiment.



\## Reported outputs



The notebook contains the historical experiment outputs used to support the published results, including:



\- cross-validation model comparison

\- paired t-test

\- held-out test metrics

\- ROC-AUC

\- Random Forest feature importance

\- permutation importance

\- SHAP analyses



\## SHAP limitation



The historical notebook successfully performs global SHAP analyses but its multiclass waterfall-plot section does not execute successfully because of the shape of the multiclass SHAP output.



This is retained as part of the historical notebook rather than being presented as a fully successful visualization.



\## Reproduction status



Exact environment-level reproduction should be considered \*\*not yet independently verified\*\* until the notebook is executed successfully in an environment matching the required dependency versions.

