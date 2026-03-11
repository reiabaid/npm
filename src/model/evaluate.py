"""
Model Evaluation
Evaluates the trained model's performance with various metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Any


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model: The trained model with a predict method.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        A dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def print_evaluation_report(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Print a detailed evaluation report.

    Args:
        model: The trained model.
        X_test: Test feature matrix.
        y_test: True labels.
    """
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("SCOPE MODEL EVALUATION REPORT")
    print("=" * 60)
    print()

    metrics = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()
