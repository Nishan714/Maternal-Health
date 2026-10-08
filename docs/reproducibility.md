# Reproducibility

## Historical experiment

This repository preserves the implementation used for the published study:

**Maternal Health Risk Classification Using Machine Learning: A Study Based on Clinical Data from Bangladesh**

The preserved notebook is:

`notebooks/final_experiment.ipynb`

The notebook contains the historical experiment, including preprocessing, model comparison, cross-validation, final evaluation, statistical comparison, and interpretability analysis.

## Important: this is a historical research artifact

The repository intentionally does **not** silently rewrite the original notebook to match current software versions or retrospectively modify the published experiment.

The published paper is the primary reference for the study's reported methodology and results.

The notebook is the primary implementation record.

## Dataset

The raw dataset is not included in this repository.

See:

`data/README.md`

for the original dataset source and attribution information.

## Historical file path

The preserved notebook originally loaded the dataset using a Google Colab-style path:

`/content/Final.csv`

Therefore, the notebook is not expected to run unchanged on every local machine.

For reproduction, download the original dataset and modify only the input path required for the local environment.

## Environment

The historical notebook explicitly installed:

- scikit-learn 1.5.0
- imbalanced-learn
- XGBoost
- SHAP
- NumPy
- pandas
- matplotlib
- seaborn
- SciPy

Only the scikit-learn version was explicitly pinned in the historical notebook.

The repository therefore does **not** claim byte-for-byte reproducibility across arbitrary future package versions.

## Experimental procedure

The historical implementation performs:

1. Dataset loading and inspection.
2. Missing target labels represented as `Unknown`.
3. Stratified 80/20 train/test split.
4. Numerical preprocessing and standardization.
5. Categorical preprocessing.
6. SMOTE within the model-evaluation pipeline.
7. Stratified 5-fold cross-validation.
8. Comparison of ten machine-learning classifiers.
9. Selection of Random Forest.
10. Final evaluation on the held-out test set.
11. ROC-AUC analysis.
12. Gini feature importance.
13. Permutation importance.
14. SHAP analysis.

## Published-paper / notebook discrepancy

An important historical discrepancy is documented here rather than hidden.

The published paper describes **mean imputation** for numerical variables.

The preserved notebook uses **median imputation** for numerical variables.

The historical notebook has intentionally been retained rather than retrospectively altered. This repository therefore does not claim that the notebook and the published methodological description are identical in this respect.

## Reproduction expectations

A researcher attempting reproduction should:

1. Obtain the dataset from the original Mendeley source.
2. Verify the downloaded dataset against the published dataset description.
3. Use an environment compatible with the historical dependencies.
4. Adapt the notebook's dataset path to the local environment.
5. Run the notebook from the beginning.
6. Compare the resulting outputs with the verified historical results stored under `results/`.

Small numerical differences may occur when package versions, numerical libraries, or execution environments differ.

## Scope

This repository is intended as a research companion to the published paper.

It is not presented as a clinical software package or a clinically validated decision-support system.
