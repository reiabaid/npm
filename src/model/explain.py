"""
Model Explainability
Provides SHAP-based explanations for model predictions.
"""

import shap
import numpy as np
import pandas as pd
from typing import Any, Optional


def get_shap_explainer(model: Any, X_background: Optional[np.ndarray] = None) -> shap.Explainer:
    """Create a SHAP explainer for the model.

    Args:
        model: The trained model.
        X_background: Background dataset for the explainer.

    Returns:
        A SHAP Explainer object.
    """
    if X_background is not None:
        return shap.Explainer(model, X_background)
    return shap.Explainer(model)


def explain_prediction(explainer: shap.Explainer, X: np.ndarray,
                       feature_names: Optional[list] = None) -> dict:
    """Generate SHAP explanations for predictions.

    Args:
        explainer: A SHAP Explainer object.
        X: The feature matrix to explain.
        feature_names: Optional list of feature names.

    Returns:
        A dictionary containing SHAP values and base values.
    """
    shap_values = explainer(X)

    result = {
        "shap_values": shap_values.values,
        "base_values": shap_values.base_values,
        "data": shap_values.data,
    }

    if feature_names:
        result["feature_names"] = feature_names

    return result


def get_top_features(shap_values: np.ndarray, feature_names: list,
                     top_n: int = 10) -> list[dict]:
    """Get the top contributing features based on mean absolute SHAP values.

    Args:
        shap_values: SHAP values array.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        A list of dictionaries with feature names and importance scores.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    top_features = []
    for idx in sorted_indices:
        top_features.append({
            "feature": feature_names[idx],
            "importance": float(mean_abs_shap[idx]),
        })

    return top_features


def explain_single_prediction(explainer: shap.Explainer, X_single: np.ndarray,
                               feature_names: list) -> list[dict]:
    """Explain a single prediction with per-feature contributions.

    Args:
        explainer: A SHAP Explainer object.
        X_single: A single sample's features (1D or 2D array).
        feature_names: List of feature names.

    Returns:
        A list of dictionaries with feature contributions sorted by absolute impact.
    """
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    shap_values = explainer(X_single)
    contributions = []

    for i, name in enumerate(feature_names):
        contributions.append({
            "feature": name,
            "value": float(X_single[0, i]),
            "shap_value": float(shap_values.values[0, i]),
        })

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return contributions
