"""
app.py
------
Streamlit web app to demonstrate 5 pre-trained classification models on a
user-uploaded test dataset (intended to be test_data.csv produced by
train_models.py).

Features:
    a. CSV upload (test data only)
    b. Model selection dropdown
    c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    d. Confusion matrix + classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

MODEL_DIR = "model"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

st.set_page_config(page_title="ML Classification Demo", layout="wide")


@st.cache_resource
def load_artifacts(model_filename: str):
    """Load a trained model plus the shared scaler and feature list."""
    model = joblib.load(os.path.join(MODEL_DIR, model_filename))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, feature_names


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🔬 ML Classification Model Demo")
    st.write(
        "Upload the test dataset (CSV), pick a trained model, and view "
        "its evaluation metrics and confusion matrix."
    )

    # --- a. Dataset upload -------------------------------------------------
    st.sidebar.header("1. Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv", type=["csv"]
    )

    # --- b. Model selection --------------------------------------------
    st.sidebar.header("2. Select Model")
    model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

    if uploaded_file is None:
        st.info("👈 Upload `test_data.csv` from the sidebar to get started.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    if TARGET_COL not in df.columns:
        st.error(
            f"Uploaded CSV must contain a '{TARGET_COL}' column with the true "
            "labels (as produced by train_models.py)."
        )
        st.stop()

    try:
        model, scaler, feature_names = load_artifacts(MODEL_FILES[model_name])
    except FileNotFoundError:
        st.error(
            "Trained model files not found. Run `train_models.py` first "
            "so the `model/` folder is populated."
        )
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df[TARGET_COL]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_scaled)[:, 1]
    else:
        y_score = model.decision_function(X_scaled)

    # --- c. Evaluation metrics ------------------------------------------
    st.subheader(f"📊 Evaluation Metrics — {model_name}")
    metrics = compute_metrics(y_true, y_pred, y_score)

    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    # --- d. Confusion matrix + classification report ---------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

    st.success("Prediction and evaluation complete ✅")


if __name__ == "__main__":
    main()
