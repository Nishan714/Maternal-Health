# Maternal Health Risk Classification Using Explainable Machine Learning

An interpretable multi-class maternal health risk classification framework developed using real-world clinical data from Bangladesh.  

**Accepted at IEEE QPAIN 2026.**

---

## Overview

Maternal mortality remains a persistent public health challenge in low- and middle-income countries. Early and transparent risk assessment is essential for timely clinical intervention.

This study proposes an explainable machine learning framework for multi-class maternal risk stratification using routine antenatal clinical indicators. Importantly, missing risk labels were treated as an explicit **“Unknown” class**, preserving real-world clinical uncertainty instead of applying potentially biased imputation.

---

## Dataset

- Source: Kaggle – Maternal Health Risk Dataset  
- Total Samples: 1,205  
- Features: 11 demographic and clinical attributes  
- Target Classes:
  - High Risk (0)
  - Low Risk (1)
  - Unknown (2)

The extremely small “Unknown” category reflects incomplete clinical assessments rather than random noise and was retained to ensure methodological transparency.

---

## Methodology

### 1. Data Preprocessing
- Mean imputation for numerical features  
- Feature standardization  
- One-hot encoding for categorical variables  
- Stratified 80/20 train-test split  

### 2. Class Imbalance Handling
- SMOTE applied exclusively to the training set  
- Stratified 5-fold cross-validation  
- Weighted F1-score used as primary evaluation metric  

### 3. Model Benchmarking (10 Classifiers)

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- XGBoost  
- Support Vector Machine  
- K-Nearest Neighbors  
- Gaussian Naive Bayes  
- Multi-Layer Perceptron  

Models were evaluated using stratified cross-validation to ensure stable comparison across imbalanced class distributions.

---

## Results

Random Forest demonstrated superior and stable performance:

- **Accuracy:** 97.93%  
- **Weighted F1-score:** 97.52%  
- **ROC-AUC (One-vs-Rest):** ≈ 0.99  

A paired t-test comparing Random Forest and Gradient Boosting revealed no statistically significant difference (p = 0.853), although Random Forest was selected due to its stability and interpretability advantages.

---

## Model Interpretability

To ensure clinical transparency and usability:

- Tree-based Feature Importance (Gini importance)  
- Permutation Importance  
- SHAP (TreeExplainer) analysis  

Dominant predictors:

- BMI  
- Blood Sugar (BS)  
- Heart Rate  
- Preexisting Diabetes  

These indicators align with established maternal risk factors and support cost-effective early screening strategies in resource-constrained environments.

---

## Repository Structure
Maternal-Health-Risk-Classification/
│
├── train.py
├── requirements.txt
├── README.md
└── results/


---

## Reproducibility
 
python train.py --data path_to_dataset.csv

Dataset must be downloaded separately from Kaggle and placed locally.

---

## Citation

If you use this work, please cite:

Mitra, N., et al.  
*Maternal Health Risk Classification Using Machine Learning: A Study Based on Clinical Data from Bangladesh.*  
IEEE QPAIN 2026.

---

## Status

Accepted for presentation at IEEE QPAIN 2026.






