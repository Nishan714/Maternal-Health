# Maternal Health Risk Classification Using Machine Learning

Research companion repository for the paper:

**Maternal Health Risk Classification Using Machine Learning: A Study Based on Clinical Data from Bangladesh**

Published in the *2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)*.

## Overview

This repository contains the historical experimental implementation associated with the published study on maternal health risk classification using machine learning.

The study investigates whether commonly available maternal-health attributes can be used to classify maternal health risk into High, Low, and Unknown categories.

The repository is intended to support transparency and reproducibility of the reported research rather than clinical deployment.

## Research workflow

The historical experiment follows this workflow:

1. Load the maternal-health dataset.
2. Separate predictors and the `Risk Level` target.
3. Encode the target classes.
4. Create an 80/20 stratified train-test split.
5. Apply preprocessing within a machine-learning pipeline.
6. Apply SMOTE to the training folds.
7. Compare ten machine-learning classifiers using stratified five-fold cross-validation.
8. Select Random Forest based on cross-validation performance and interpretability.
9. Evaluate the selected model on the held-out test set.
10. Analyze model predictions using feature importance, permutation importance, and SHAP.

## Dataset

The experiment uses a dataset containing 1,205 maternal-health records and 11 predictor variables:

- Age
- Systolic BP
- Diastolic
- BS
- Body Temp
- BMI
- Previous Complications
- Preexisting Diabetes
- Gestational Diabetes
- Mental Health
- Heart Rate

The target variable is `Risk Level`.

The dataset contains missing target labels. In the historical implementation, these missing labels are represented as an `Unknown` class for the experiment.

The raw dataset is **not included in this repository**. See [`data/README.md`](data/README.md).

## Models evaluated

The historical notebook evaluates:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbours
- Gaussian Naive Bayes
- Gradient Boosting
- AdaBoost
- XGBoost
- Artificial Neural Network / MLP

Random Forest was selected as the final model.

## Reported results

The published experiment reports:

| Metric | Random Forest |
|---|---:|
| Test accuracy | 97.93% |
| Weighted precision | 97.13% |
| Weighted recall | 97.93% |
| Weighted F1-score | 97.52% |
| Weighted ROC-AUC | ~0.995 |

The Unknown class contains only three test samples and receives zero precision, recall, and F1-score. Therefore, the high weighted metrics should not be interpreted as evidence of equally strong performance across all classes.

The five-fold cross-validation weighted F1-score for Random Forest was approximately **0.9598 ± 0.0078**.

A paired comparison between Random Forest and Gradient Boosting produced `t(4) = 0.20, p = 0.853`, indicating no statistically significant difference between their fold-level scores.

## Interpretability

The historical notebook includes:

- Random Forest Gini feature importance
- Permutation importance
- SHAP global summary analysis
- SHAP bar analysis

These analyses are exploratory model-interpretability analyses and should not be interpreted as causal clinical evidence.

## Important reproducibility note

The published paper describes numerical mean imputation, whereas the historical notebook uses median imputation.

This repository preserves the historical notebook rather than silently changing the experiment.

See [`docs/reproducibility.md`](docs/reproducibility.md) for details.

## Repository structure

```text
Maternal-Health/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── final_experiment.ipynb
├── data/
│   └── README.md
├── docs/
│   ├── methodology.md
│   └── reproducibility.md
└── results/
    ├── figures/
    ├── tables/
    └── README.md



