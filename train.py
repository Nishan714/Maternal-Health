# train.py

import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib


def load_data(path):
    df = pd.read_csv(path)

    # Treat missing target as "Unknown"
    df["Risk Level"] = df["Risk Level"].fillna("Unknown")

    target_mapping = {"High": 0, "Low": 1, "Unknown": 2}
    df["Risk Level"] = df["Risk Level"].map(target_mapping)

    X = df.drop("Risk Level", axis=1)
    y = df["Risk Level"]

    return X, y


def build_preprocessor(X):

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def get_models():

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    return models


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    print("\nLoading dataset...")
    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    preprocessor = build_preprocessor(X)

    models = get_models()

    print("\nPerforming 5-Fold Cross-Validation...\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():

        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted"
        )

        results[name] = scores.mean()
        print(f"{name}: Mean Weighted F1 = {scores.mean():.4f}")

    # Select best model
    best_model_name = max(results, key=results.get)
    print(f"\nBest Model: {best_model_name}")

    final_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", models[best_model_name])
    ])

    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)

    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(final_pipeline, "maternal_health_model.pkl")
    print("\nModel saved as maternal_health_model.pkl")


if __name__ == "__main__":
    main()