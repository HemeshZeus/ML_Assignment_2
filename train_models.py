"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset (sklearn built-in: 30 features, 569 instances, binary classification
-> satisfies the assignment's >=12 features and >=500 instances requirement).

Models trained:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Gaussian Naive Bayes
    5. Random Forest Classifier (Ensemble)

Outputs:
    - model/*.joblib          -> trained model files
    - model/scaler.joblib      -> fitted StandardScaler (used by app.py)
    - model/feature_names.joblib -> list of feature column names
    - test_data.csv            -> held-out test set (features + true label)
    - Console printout of all 6 metrics per model
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
MODEL_DIR = "model"
TEST_CSV_PATH = "test_data.csv"


def load_data():
    """Load the dataset and return features (X), target (y), and column names."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    return X, y


def get_models():
    """Return a dict of {model_name: unfitted estimator}."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test):
    """Compute the 6 required evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)

    # AUC needs probability / decision scores for the positive class
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def filename_for(model_name: str) -> str:
    """Turn a model display name into a safe filename."""
    return (
        model_name.lower()
        .replace(" (ensemble)", "")
        .replace(" ", "_")
        + ".joblib"
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (helps Logistic Regression / kNN); tree-based models are
    # scale-invariant so this is safe to apply uniformly.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = get_models()

    print(f"{'Model':<25}{'Accuracy':>10}{'AUC':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'MCC':>10}")
    print("-" * 87)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics

        print(
            f"{name:<25}"
            f"{metrics['Accuracy']:>10.4f}"
            f"{metrics['AUC']:>10.4f}"
            f"{metrics['Precision']:>12.4f}"
            f"{metrics['Recall']:>10.4f}"
            f"{metrics['F1']:>10.4f}"
            f"{metrics['MCC']:>10.4f}"
        )

        # Save the trained model
        joblib.dump(model, os.path.join(MODEL_DIR, filename_for(name)))

    # Save the scaler and feature names so app.py can reproduce preprocessing
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))

    # Save the held-out test set (features + true label) for the Streamlit app
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print(f"\nSaved {len(models)} trained models to '{MODEL_DIR}/'")
    print(f"Saved test set ({len(test_df)} rows) to '{TEST_CSV_PATH}'")

    # Optional: dump a metrics summary CSV for convenience when writing README
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("metrics_summary.csv")
    print("Saved metrics summary to 'metrics_summary.csv'")


if __name__ == "__main__":
    main()
