"""
SCOPE – Day 5: SHAP Explainability + Final Test Evaluation
===========================================================
"""

import os
import sys
import warnings
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from src.model.preprocess import build_preprocessor  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

NUMERICAL_FEATURES = [
    "days_since_created", "days_since_last_update", "num_versions",
    "release_velocity", "num_maintainers", "description_length",
    "stargazers_count", "forks_count", "open_issues_count",
    "subscribers_count", "contributor_count", "days_since_last_commit",
]
BINARY_FEATURES = ["has_postinstall", "license_is_standard", "has_github_repo"]
ALL_FEATURES    = NUMERICAL_FEATURES + BINARY_FEATURES

# Human-readable label mapping
FEATURE_LABELS: dict[str, str] = {
    "days_since_created":      "Package age (days)",
    "days_since_last_update":  "Days since last update",
    "num_versions":            "Number of published versions",
    "release_velocity":        "Release velocity (versions/day)",
    "num_maintainers":         "Number of maintainers",
    "description_length":      "Description length (chars)",
    "stargazers_count":        "GitHub stars",
    "forks_count":             "GitHub forks",
    "open_issues_count":       "Open GitHub issues",
    "subscribers_count":       "GitHub subscribers / watchers",
    "contributor_count":       "Total contributors",
    "days_since_last_commit":  "Days since last commit",
    "has_postinstall":         "Has postinstall script",
    "license_is_standard":     "Uses a standard licence",
    "has_github_repo":         "Has linked GitHub repo",
}

def get_shap_explainer(model):
    """Initialize and return a SHAP TreeExplainer for the given model."""
    return shap.TreeExplainer(model)

def explain_single_prediction(explainer, X_scaled, feature_names):
    """
    Get SHAP explanations for a single scaled sample.
    Returns a list of dicts: [{'feature': name, 'shap_value': val}, ...]
    Sorted by absolute impact descending.
    """
    shap_values = explainer.shap_values(X_scaled)
    
    # shap_values[1] is for the 'suspicious' class (class 1)
    if isinstance(shap_values, list):
        sample_shap = shap_values[1][0]
    else:
        # For some models, it returns a single array for binary class
        sample_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    results = []
    for name, val in zip(feature_names, sample_shap):
        results.append({
            "feature": name,
            "shap_value": float(val)
        })
    
    results.sort(key=lambda t: abs(t["shap_value"]), reverse=True)
    return results

def get_explanation(shap_vals_single, feature_names=None):
    names = feature_names if feature_names is not None else ALL_FEATURES
    results = []
    for name, val in zip(names, shap_vals_single):
        display = FEATURE_LABELS.get(name, name)
        direction = "toward suspicious" if val >= 0 else "toward healthy"
        human = f"{display}: {val:+.4f} {direction}"
        results.append((name, float(val), human))
    results.sort(key=lambda t: abs(t[1]), reverse=True)
    return results

def generate_health_score_text(pred_proba, shap_vals_single, feature_names=None, top_n=4):
    if pred_proba >= 0.85: risk_level = "CRITICAL"
    elif pred_proba >= 0.60: risk_level = "HIGH"
    elif pred_proba >= 0.35: risk_level = "MEDIUM"
    else: risk_level = "LOW"

    explanation = get_explanation(shap_vals_single, feature_names)
    top_factors = [human for (_, _, human) in explanation[:top_n]]
    lead_factor = explanation[0][2] if explanation else "no dominant factor"
    risk_pct = f"{pred_proba * 100:.1f}%"
    summary  = f"{risk_level} risk ({risk_pct}) — driven by: {lead_factor}"

    return {
        "score":       pred_proba,
        "risk_level":  risk_level,
        "risk_pct":    risk_pct,
        "top_factors": top_factors,
        "summary":     summary,
    }

def main():
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

    print("SECTION 1 – Rebuilding data splits")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at {DATASET_PATH}.")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        joblib.dump(preprocessor, preprocessor_path)

    X_test_transformed = preprocessor.transform(X_test)

    tuned_path = os.path.join(MODELS_DIR, "rf_tuned.pkl")
    baseline_path = os.path.join(MODELS_DIR, "rf_baseline.pkl")
    
    if os.path.exists(tuned_path):
        best_model = joblib.load(tuned_path)
    elif os.path.exists(baseline_path):
        best_model = joblib.load(baseline_path)
    else:
        from sklearn.ensemble import RandomForestClassifier
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Use SMOTE for training if needed
        sm = SMOTE(random_state=42)
        X_train_transformed = preprocessor.transform(X_train)
        X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)
        best_model.fit(X_train_res, y_train_res)
        joblib.dump(best_model, baseline_path)

    joblib.dump(ALL_FEATURES, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print("Initialising shap.TreeExplainer …")
    explainer = get_shap_explainer(best_model)
    shap_values = explainer.shap_values(X_test_transformed)
    shap_suspicious = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot
    shap.summary_plot(shap_suspicious, X_test_transformed, feature_names=ALL_FEATURES, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_plot.png"))
    plt.close()

    # Final eval
    y_test_pred = best_model.predict(X_test_transformed)
    print(classification_report(y_test, y_test_pred))

    joblib.dump(best_model, os.path.join(MODELS_DIR, "rf_final.pkl"))
    print("Final artefacts saved.")

if __name__ == "__main__":
    main()
